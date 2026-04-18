"""Single-agent LLM solver for CVRP.

One Claude agent, no personas. Iteratively proposes a solution, validator
checks it, critique feeds back. Repeats up to ``max_rounds`` rounds and
returns the best valid solution seen.

Prompt-cacheable system prompt + disk cache of LLM responses keyed on the
full prompt. Tests inject a stub ``LLMCaller`` so this module never needs
network access in CI.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

from src.models import Instance, Route, Solution
from src.validator import compute_cost, validate

Representation = Literal["coords", "matrix"]

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_ROUNDS = 3
DEFAULT_MAX_TOKENS = 8000
DEFAULT_CACHE_DIR = Path(".cache") / "single_agent"

SYSTEM_PROMPT = """\
You are an expert vehicle routing solver. Given a Capacitated Vehicle Routing \
Problem (CVRP) instance, propose a set of routes that visits every non-depot \
customer exactly once, respects vehicle capacity on every route, and \
minimises the total Euclidean tour distance (each route starts and ends at \
the depot implicitly — do not list the depot in the route).

Return only JSON matching the provided schema: {"routes": [[customer_id, ...], ...]}.
Customer ids are the integer ids given in the problem. The depot id must NOT \
appear in any route. Every non-depot customer id must appear in exactly one route.

Think carefully about clustering, distance, and capacity before answering. If \
the user provides critique of a prior attempt, incorporate it in the next one.
"""

ROUTES_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "routes": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
            },
        }
    },
    "required": ["routes"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# LLM caller abstraction — real implementation uses the Anthropic SDK;
# tests substitute a stub that returns canned responses.
# ---------------------------------------------------------------------------


class LLMCaller(Protocol):
    def __call__(self, system: str, messages: list[dict], model: str) -> list[list[int]]:
        """Return parsed ``routes`` list."""
        ...


def anthropic_llm_caller(client=None, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> LLMCaller:
    """Real LLM caller using the Anthropic SDK.

    Imported lazily so the module is importable without the ``anthropic``
    package installed (e.g. in CI for unit tests that use a stub).
    """
    import anthropic  # noqa: PLC0415

    if client is None:
        client = anthropic.Anthropic()

    def _call(system: str, messages: list[dict], model: str) -> list[list[int]]:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": ROUTES_JSON_SCHEMA,
                }
            },
        )
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        return data["routes"]

    return _call


# ---------------------------------------------------------------------------
# Problem serialisation
# ---------------------------------------------------------------------------


def _serialise_coords(instance: Instance) -> str:
    depot = instance.depot
    lines = [
        f"Instance: {instance.name}",
        f"Vehicle capacity: {instance.capacity}",
        f"Depot: id={depot.id} at ({depot.x:g}, {depot.y:g})",
        "Customers (id, x, y, demand):",
    ]
    for c in instance.non_depot_customers:
        lines.append(f"  {c.id}: ({c.x:g}, {c.y:g}) demand={c.demand}")
    return "\n".join(lines)


def _serialise_matrix(instance: Instance) -> str:
    ids = [c.id for c in instance.customers]
    header = [
        f"Instance: {instance.name}",
        f"Vehicle capacity: {instance.capacity}",
        f"Depot id: {instance.depot_id}",
        "Demands (id: demand):",
        "  " + ", ".join(f"{c.id}:{c.demand}" for c in instance.customers),
        "Pairwise Euclidean distances (i -> j: dist), rounded to 2 dp:",
    ]
    lines: list[str] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            lines.append(f"  {a}->{b}: {instance.distance(a, b):.2f}")
    return "\n".join(header + lines)


def _problem_message(instance: Instance, representation: Representation) -> str:
    body = (
        _serialise_coords(instance) if representation == "coords" else _serialise_matrix(instance)
    )
    return f"{body}\n\nPropose a valid minimum-distance set of routes as JSON."


# ---------------------------------------------------------------------------
# Disk cache for LLM responses
# ---------------------------------------------------------------------------


def _cache_key(model: str, system: str, messages: list[dict]) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "messages": messages},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_read(cache_dir: Path, key: str) -> list[list[int]] | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["routes"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _cache_write(cache_dir: Path, key: str, routes: list[list[int]]) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{key}.json").write_text(json.dumps({"routes": routes}))
    except OSError:
        pass  # cache is best-effort; never fail the solve on write error.


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------


@dataclass
class _Attempt:
    routes: list[list[int]]
    solution: Solution
    errors: list[str]
    cost: float


def _build_solution(instance: Instance, routes: list[list[int]]) -> Solution:
    return Solution(
        instance_name=instance.name,
        routes=[Route(customer_ids=list(r)) for r in routes if r],
    )


def _critique(attempt: _Attempt) -> str:
    if attempt.errors:
        bullets = "\n".join(f"- {e}" for e in attempt.errors)
        return (
            "Your previous solution was invalid. Issues:\n"
            f"{bullets}\n"
            "Please propose a new solution that fixes these issues."
        )
    return (
        f"Your previous solution was valid with total cost {attempt.cost:.2f}. "
        "Can you find a cheaper one? Remember every route's demand must stay "
        "within capacity, and every customer must appear exactly once."
    )


def single_agent(
    instance: Instance,
    *,
    llm: LLMCaller | None = None,
    model: str = DEFAULT_MODEL,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    representation: Representation = "coords",
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> Solution:
    """Run the single-agent refinement loop and return the best valid solution.

    Raises ``RuntimeError`` if no valid solution is produced after ``max_rounds``.
    """
    if llm is None:
        llm = anthropic_llm_caller()

    system = SYSTEM_PROMPT
    messages: list[dict] = [
        {"role": "user", "content": _problem_message(instance, representation)}
    ]

    best: _Attempt | None = None
    last: _Attempt | None = None

    for round_idx in range(max_rounds):
        key = _cache_key(model, system, messages) if cache_dir else None
        routes: list[list[int]] | None = None
        if cache_dir and key:
            routes = _cache_read(cache_dir, key)
        if routes is None:
            routes = llm(system, messages, model)
            if cache_dir and key:
                _cache_write(cache_dir, key, routes)

        solution = _build_solution(instance, routes)
        errors = validate(instance, solution)
        cost = compute_cost(instance, solution) if not errors else float("inf")
        if not errors:
            solution.total_cost = cost
        attempt = _Attempt(routes=routes, solution=solution, errors=errors, cost=cost)
        last = attempt

        if not errors and (best is None or cost < best.cost):
            best = attempt

        # If this is the last round, stop — no point building another prompt.
        if round_idx == max_rounds - 1:
            break

        # Otherwise, append the assistant's response + critique and continue.
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps({"routes": routes}),
            }
        )
        messages.append({"role": "user", "content": _critique(attempt)})

    if best is None:
        issues = "; ".join(last.errors) if last else "no attempts produced"
        raise RuntimeError(
            f"single_agent failed to produce a valid solution after {max_rounds} "
            f"round(s): {issues}"
        )
    return best.solution
