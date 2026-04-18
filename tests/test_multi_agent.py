"""End-to-end multi_agent solver test with a stub LLM."""

from __future__ import annotations

from pathlib import Path

from src.agents.personas import DEFAULT_PERSONAS
from src.models import Instance
from src.solvers.multi_agent import multi_agent
from src.validator import validate
from tests.test_orchestrator import StubAgentLLM


def test_multi_agent_end_to_end(ring_instance: Instance, tmp_path: Path) -> None:
    valid = [[2, 3], [4, 5]]
    stub = StubAgentLLM(
        proposals={p.name: [valid, valid] for p in DEFAULT_PERSONAS},
        critique_text={p.name: "fine" for p in DEFAULT_PERSONAS},
    )
    solution = multi_agent(
        ring_instance,
        llm=stub,
        transcript_dir=tmp_path,
    )
    assert validate(ring_instance, solution) == []
    assert solution.total_cost is not None
    # Transcript written to <tmp_path>/<instance_name>.json.
    assert (tmp_path / f"{ring_instance.name}.json").exists()


def test_multi_agent_respects_transcript_disable(
    ring_instance: Instance, tmp_path: Path
) -> None:
    valid = [[2, 3], [4, 5]]
    stub = StubAgentLLM(
        proposals={p.name: [valid, valid] for p in DEFAULT_PERSONAS},
        critique_text={p.name: "fine" for p in DEFAULT_PERSONAS},
    )
    solution = multi_agent(
        ring_instance,
        llm=stub,
        transcript_dir=None,
    )
    assert validate(ring_instance, solution) == []
    # Nothing written under tmp_path.
    assert not list(tmp_path.iterdir())
