"""Multi-agent CVRP solver — thin wrapper over the orchestrator."""

from __future__ import annotations

from pathlib import Path

from src.agents.base import AgentLLM, anthropic_agent_llm
from src.agents.orchestrator import run_orchestrator
from src.agents.personas import DEFAULT_PERSONAS, Persona
from src.models import Instance, Solution

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_TRANSCRIPT_DIR = Path(".cache") / "multi_agent_transcripts"


def multi_agent(
    instance: Instance,
    *,
    llm: AgentLLM | None = None,
    personas: tuple[Persona, ...] = DEFAULT_PERSONAS,
    model: str = DEFAULT_MODEL,
    transcript_dir: Path | None = DEFAULT_TRANSCRIPT_DIR,
) -> Solution:
    """Run the multi-agent orchestrator and return the winning solution.

    By default writes a transcript JSON to ``transcript_dir`` keyed by the
    instance name. Pass ``transcript_dir=None`` to disable logging.
    """
    llm = llm or anthropic_agent_llm()
    transcript_path = (
        transcript_dir / f"{instance.name}.json" if transcript_dir is not None else None
    )
    solution, _ = run_orchestrator(
        instance,
        llm,
        personas=personas,
        model=model,
        transcript_path=transcript_path,
    )
    return solution
