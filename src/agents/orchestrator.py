"""Multi-agent orchestration: propose -> critique -> refine -> synthesise.

Each persona holds its own conversation history. The orchestrator:

1. Asks every persona to propose a solution (parallel, independent).
2. For each ordered pair (critic, target) where critic != target, asks
   ``critic`` to critique ``target``'s proposal.
3. Asks every persona to refine their own proposal given the critiques
   they received.
4. Picks the cheapest valid solution across all proposals + refinements.

Every LLM call and intermediate artefact is recorded in a
:class:`Transcript` for offline analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.agents.base import AgentLLM
from src.agents.personas import DEFAULT_PERSONAS, Persona
from src.models import Instance, Route, Solution
from src.solvers.single_agent import _problem_message
from src.validator import compute_cost, validate


@dataclass
class _Attempt:
    persona: str
    round: str  # "propose" | "refine"
    routes: list[list[int]]
    solution: Solution
    errors: list[str]
    cost: float  # inf if invalid


@dataclass
class Transcript:
    instance_name: str
    started_at: float
    personas: list[str]
    proposals: dict[str, list[list[int]]] = field(default_factory=dict)
    critiques: dict[str, dict[str, str]] = field(default_factory=dict)
    refinements: dict[str, list[list[int]]] = field(default_factory=dict)
    attempts: list[dict] = field(default_factory=list)
    winner: dict | None = None

    def to_dict(self) -> dict:
        return {
            "instance_name": self.instance_name,
            "started_at": self.started_at,
            "personas": self.personas,
            "proposals": self.proposals,
            "critiques": self.critiques,
            "refinements": self.refinements,
            "attempts": self.attempts,
            "winner": self.winner,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def _critique_user_message(target_name: str, target_routes: list[list[int]]) -> str:
    return (
        f"Here is a proposed solution from the {target_name}:\n"
        f"{json.dumps({'routes': target_routes})}\n\n"
        "Critique it from your perspective. Identify concrete weaknesses "
        "(capacity slack, crossing edges, suboptimal pairings, violated "
        "constraints) and suggest specific fixes. Be precise — refer to "
        "route indices and customer ids. Keep it under 200 words. Do not "
        "produce a new solution; only critique."
    )


def _refine_user_message(received_critiques: list[tuple[str, str]]) -> str:
    bullets = "\n\n".join(
        f"From the {critic}:\n{text}" for critic, text in received_critiques
    )
    return (
        "Other solvers have reviewed your proposal:\n\n"
        f"{bullets}\n\n"
        "Please produce a refined solution as JSON. Keep what still makes "
        "sense from your perspective; adopt critiques only where they "
        "genuinely improve the outcome."
    )


def _build_attempt(
    instance: Instance, persona: Persona, round_kind: str, routes: list[list[int]]
) -> _Attempt:
    sol = Solution(
        instance_name=instance.name,
        routes=[Route(customer_ids=list(r)) for r in routes if r],
    )
    errors = validate(instance, sol)
    cost = compute_cost(instance, sol) if not errors else float("inf")
    if not errors:
        sol.total_cost = cost
    return _Attempt(
        persona=persona.name,
        round=round_kind,
        routes=routes,
        solution=sol,
        errors=errors,
        cost=cost,
    )


def run_orchestrator(
    instance: Instance,
    llm: AgentLLM,
    *,
    personas: tuple[Persona, ...] = DEFAULT_PERSONAS,
    model: str = "claude-sonnet-4-6",
    transcript_path: Path | None = None,
) -> tuple[Solution, Transcript]:
    """Run the four-round orchestration and return the winning solution + transcript."""
    transcript = Transcript(
        instance_name=instance.name,
        started_at=time.time(),
        personas=[p.name for p in personas],
    )
    problem = _problem_message(instance, "coords")

    # ------- Round 1: propose ----------------------------------------------
    histories: dict[str, list[dict]] = {}
    attempts: list[_Attempt] = []
    for persona in personas:
        messages = [{"role": "user", "content": problem}]
        routes = llm.propose(persona.name, persona.system_prompt, messages, model)
        histories[persona.name] = messages + [
            {"role": "assistant", "content": json.dumps({"routes": routes})}
        ]
        transcript.proposals[persona.name] = routes
        attempt = _build_attempt(instance, persona, "propose", routes)
        attempts.append(attempt)
        transcript.attempts.append(
            {
                "persona": persona.name,
                "round": "propose",
                "routes": routes,
                "errors": attempt.errors,
                "cost": None if attempt.cost == float("inf") else attempt.cost,
            }
        )

    # ------- Round 2: critique ---------------------------------------------
    # critiques[critic][target] = text
    for critic in personas:
        transcript.critiques[critic.name] = {}
        for target in personas:
            if target is critic:
                continue
            target_routes = transcript.proposals[target.name]
            messages = [
                {"role": "user", "content": problem},
                {
                    "role": "user",
                    "content": _critique_user_message(target.name, target_routes),
                },
            ]
            text = llm.critique(critic.name, critic.system_prompt, messages, model)
            transcript.critiques[critic.name][target.name] = text

    # ------- Round 3: refine -----------------------------------------------
    for persona in personas:
        received = [
            (critic.name, transcript.critiques[critic.name][persona.name])
            for critic in personas
            if critic is not persona
        ]
        messages = histories[persona.name] + [
            {"role": "user", "content": _refine_user_message(received)}
        ]
        routes = llm.propose(persona.name, persona.system_prompt, messages, model)
        transcript.refinements[persona.name] = routes
        attempt = _build_attempt(instance, persona, "refine", routes)
        attempts.append(attempt)
        transcript.attempts.append(
            {
                "persona": persona.name,
                "round": "refine",
                "routes": routes,
                "errors": attempt.errors,
                "cost": None if attempt.cost == float("inf") else attempt.cost,
            }
        )

    # ------- Round 4: synthesise (best valid by cost) ----------------------
    valid = [a for a in attempts if not a.errors]
    if not valid:
        reasons = "; ".join(
            f"{a.persona}/{a.round}: " + ", ".join(a.errors[:2]) for a in attempts
        )
        if transcript_path is not None:
            transcript.write(transcript_path)
        raise RuntimeError(f"No valid solution from any persona. {reasons}")

    winner = min(valid, key=lambda a: a.cost)
    transcript.winner = {
        "persona": winner.persona,
        "round": winner.round,
        "cost": winner.cost,
    }
    if transcript_path is not None:
        transcript.write(transcript_path)
    return winner.solution, transcript
