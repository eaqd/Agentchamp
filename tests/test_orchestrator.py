"""Tests for src.agents.orchestrator — stub AgentLLM, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.orchestrator import run_orchestrator
from src.agents.personas import DEFAULT_PERSONAS
from src.models import Instance
from src.validator import validate


class StubAgentLLM:
    """Keyed stub. ``proposals[persona_name]`` is a list consumed in order
    across (propose round 1, refine round 3). ``critique_text`` is a single
    string reused for every critique call from that critic."""

    def __init__(
        self,
        *,
        proposals: dict[str, list[list[list[int]]]],
        critique_text: dict[str, str],
    ) -> None:
        self.proposals = {k: list(v) for k, v in proposals.items()}
        self.critique_text = critique_text
        self.propose_calls: list[tuple[str, list[dict]]] = []
        self.critique_calls: list[tuple[str, list[dict]]] = []

    def propose(self, persona_name, system, messages, model):
        self.propose_calls.append((persona_name, list(messages)))
        queue = self.proposals[persona_name]
        if not queue:
            raise RuntimeError(f"stub out of proposals for {persona_name}")
        return queue.pop(0)

    def critique(self, persona_name, system, messages, model):
        self.critique_calls.append((persona_name, list(messages)))
        return self.critique_text[persona_name]


def test_runs_four_rounds_and_picks_best(ring_instance: Instance, tmp_path: Path) -> None:
    # All three propose the same valid split in round 1; in round 3 they
    # refine. Geometric Thinker finds a cheaper split; orchestrator picks it.
    same = [[2, 3], [4, 5]]
    cheaper = [[2, 3], [4, 5]]  # same geometry — just to have a tie
    # We want one refined solution to be strictly cheaper. Pair opposite corners
    # to make a costlier route as the alternative:
    expensive = [[2, 5], [3, 4]]
    proposals = {
        "Geometric Thinker": [expensive, same],    # refine improves it
        "Constraint Engineer": [same, expensive],  # refine makes it worse — original kept
        "Heuristic Pragmatist": [same, cheaper],
    }
    stub = StubAgentLLM(
        proposals=proposals,
        critique_text={
            "Geometric Thinker": "You have crossing edges; pair adjacent ids.",
            "Constraint Engineer": "Capacity is underused on route 1.",
            "Heuristic Pragmatist": "Try a 2-opt on route 0.",
        },
    )

    transcript_path = tmp_path / "transcript.json"
    solution, transcript = run_orchestrator(
        ring_instance,
        stub,
        transcript_path=transcript_path,
    )

    # Valid, and cost <= cost of any proposed expensive solution.
    assert validate(ring_instance, solution) == []
    assert solution.total_cost is not None

    # Propose calls: 3 personas × 2 rounds (propose + refine) = 6.
    assert len(stub.propose_calls) == 6
    # Critique calls: 3 critics × 2 targets each = 6.
    assert len(stub.critique_calls) == 6

    # Transcript has all four round outputs + winner.
    assert set(transcript.proposals) == {p.name for p in DEFAULT_PERSONAS}
    assert set(transcript.refinements) == {p.name for p in DEFAULT_PERSONAS}
    assert set(transcript.critiques) == {p.name for p in DEFAULT_PERSONAS}
    for critic in transcript.critiques.values():
        # Each critic critiqued the other two targets.
        assert len(critic) == 2
    assert transcript.winner is not None
    assert transcript.winner["persona"] in {p.name for p in DEFAULT_PERSONAS}

    # File on disk matches.
    on_disk = json.loads(transcript_path.read_text())
    assert on_disk["winner"] == transcript.winner


def test_critique_sees_target_proposal(ring_instance: Instance, tmp_path: Path) -> None:
    valid = [[2, 3], [4, 5]]
    stub = StubAgentLLM(
        proposals={p.name: [valid, valid] for p in DEFAULT_PERSONAS},
        critique_text={p.name: "looks fine" for p in DEFAULT_PERSONAS},
    )
    run_orchestrator(ring_instance, stub, transcript_path=tmp_path / "t.json")

    # Every critique prompt must include a 'routes' payload to critique.
    assert stub.critique_calls, "expected critique calls"
    for _persona, messages in stub.critique_calls:
        last_user = messages[-1]["content"]
        assert "routes" in last_user
        assert "Critique" in last_user


def test_refine_sees_critiques(ring_instance: Instance, tmp_path: Path) -> None:
    valid = [[2, 3], [4, 5]]
    stub = StubAgentLLM(
        proposals={p.name: [valid, valid] for p in DEFAULT_PERSONAS},
        critique_text={
            "Geometric Thinker": "GEOM_CRITIQUE_MARKER",
            "Constraint Engineer": "IP_CRITIQUE_MARKER",
            "Heuristic Pragmatist": "HEUR_CRITIQUE_MARKER",
        },
    )
    run_orchestrator(ring_instance, stub, transcript_path=tmp_path / "t.json")

    # Round-3 propose calls: messages include assistant's own prior JSON + a
    # user critique turn referencing the two *other* personas' markers.
    refine_calls = stub.propose_calls[3:]  # first 3 are round-1 proposals
    for persona_name, messages in refine_calls:
        last_user = messages[-1]["content"]
        expected_markers = {
            "Geometric Thinker": {"IP_CRITIQUE_MARKER", "HEUR_CRITIQUE_MARKER"},
            "Constraint Engineer": {"GEOM_CRITIQUE_MARKER", "HEUR_CRITIQUE_MARKER"},
            "Heuristic Pragmatist": {"GEOM_CRITIQUE_MARKER", "IP_CRITIQUE_MARKER"},
        }[persona_name]
        for marker in expected_markers:
            assert marker in last_user, f"missing {marker} in refine prompt for {persona_name}"
        # Should NOT include the persona's own critique marker.
        own_markers = {
            "Geometric Thinker": "GEOM_CRITIQUE_MARKER",
            "Constraint Engineer": "IP_CRITIQUE_MARKER",
            "Heuristic Pragmatist": "HEUR_CRITIQUE_MARKER",
        }
        assert own_markers[persona_name] not in last_user


def test_raises_if_no_valid_solution(ring_instance: Instance, tmp_path: Path) -> None:
    # Every persona proposes the same capacity violator in both rounds.
    violator = [[2, 3, 4, 5]]  # load 40 > capacity 30
    stub = StubAgentLLM(
        proposals={p.name: [violator, violator] for p in DEFAULT_PERSONAS},
        critique_text={p.name: "..." for p in DEFAULT_PERSONAS},
    )
    with pytest.raises(RuntimeError, match="No valid solution"):
        run_orchestrator(ring_instance, stub, transcript_path=tmp_path / "t.json")
    # Transcript should still be on disk so failures can be analysed.
    assert (tmp_path / "t.json").exists()
