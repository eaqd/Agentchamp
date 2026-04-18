"""Tests for src.solvers.single_agent — uses a stub LLMCaller, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models import Instance
from src.solvers.single_agent import (
    _cache_key,
    _problem_message,
    single_agent,
)
from src.validator import validate


class StubCaller:
    """Returns canned `routes` responses in order and counts invocations."""

    def __init__(self, responses: list[list[list[int]]]):
        self.responses = responses
        self.calls = 0
        self.last_system: str | None = None
        self.last_messages: list[dict] | None = None

    def __call__(self, system: str, messages: list[dict], model: str) -> list[list[int]]:
        self.last_system = system
        self.last_messages = list(messages)
        if self.calls >= len(self.responses):
            raise RuntimeError("StubCaller ran out of canned responses")
        routes = self.responses[self.calls]
        self.calls += 1
        return routes


def test_returns_valid_solution_single_round(ring_instance: Instance, tmp_path: Path) -> None:
    stub = StubCaller([[[2, 3], [4, 5]]])  # valid on first try
    solution = single_agent(
        ring_instance,
        llm=stub,
        max_rounds=1,
        cache_dir=tmp_path,
    )
    assert validate(ring_instance, solution) == []
    assert solution.total_cost is not None and solution.total_cost > 0
    assert stub.calls == 1


def test_refines_after_invalid_first_attempt(ring_instance: Instance, tmp_path: Path) -> None:
    # Round 1: capacity violation (all four on one route, load 40 > cap 30).
    # Round 2: valid split.
    stub = StubCaller([
        [[2, 3, 4, 5]],
        [[2, 3], [4, 5]],
    ])
    solution = single_agent(
        ring_instance,
        llm=stub,
        max_rounds=2,
        cache_dir=tmp_path,
    )
    assert validate(ring_instance, solution) == []
    assert stub.calls == 2

    # The second prompt must include the critique of the first attempt.
    assert stub.last_messages is not None
    assistant_turns = [m for m in stub.last_messages if m["role"] == "assistant"]
    assert len(assistant_turns) == 1
    critique_turn = stub.last_messages[-1]
    assert critique_turn["role"] == "user"
    assert "invalid" in critique_turn["content"].lower()


def test_tracks_best_valid_when_later_round_is_worse(
    ring_instance: Instance, tmp_path: Path
) -> None:
    # Both valid, but the first split is cheaper (paired opposite corners
    # would be longer than paired adjacent ones). We construct the stub so
    # that round 1 is cheap and round 2 is more expensive; solver should keep
    # round 1's solution as best.
    cheap = [[2, 3], [4, 5]]
    expensive = [[2, 5], [3, 4]]  # longer routes through the ring
    stub = StubCaller([cheap, expensive])
    solution = single_agent(
        ring_instance,
        llm=stub,
        max_rounds=2,
        cache_dir=tmp_path,
    )
    assert validate(ring_instance, solution) == []
    # best kept = cheap (solution has same ids as `cheap`).
    returned_ids = {tuple(r.customer_ids) for r in solution.routes}
    assert returned_ids == {tuple(cheap[0]), tuple(cheap[1])}


def test_raises_when_all_rounds_invalid(ring_instance: Instance, tmp_path: Path) -> None:
    stub = StubCaller([[[2, 3, 4, 5]], [[2, 3, 4, 5]]])  # always exceeds capacity
    with pytest.raises(RuntimeError, match="failed to produce a valid"):
        single_agent(
            ring_instance,
            llm=stub,
            max_rounds=2,
            cache_dir=tmp_path,
        )


def test_disk_cache_short_circuits_llm(ring_instance: Instance, tmp_path: Path) -> None:
    stub1 = StubCaller([[[2, 3], [4, 5]]])
    single_agent(ring_instance, llm=stub1, max_rounds=1, cache_dir=tmp_path)
    assert stub1.calls == 1

    # Second run with a stub that would error if called — cache should serve it.
    class ExplodingCaller:
        def __call__(self, system, messages, model):
            raise AssertionError("LLM should not be called when cache is warm")

    solution = single_agent(
        ring_instance, llm=ExplodingCaller(), max_rounds=1, cache_dir=tmp_path
    )
    assert validate(ring_instance, solution) == []


def test_cache_key_stable_across_calls() -> None:
    sys = "system prompt"
    msgs = [{"role": "user", "content": "hi"}]
    assert _cache_key("m", sys, msgs) == _cache_key("m", sys, msgs)
    assert _cache_key("m", sys, msgs) != _cache_key("n", sys, msgs)


def test_representations_produce_different_prompts(ring_instance: Instance) -> None:
    coords = _problem_message(ring_instance, "coords")
    matrix = _problem_message(ring_instance, "matrix")
    assert coords != matrix
    assert "demand=" in coords
    assert "->" in matrix  # matrix lists pairs
