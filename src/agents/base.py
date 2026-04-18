"""Agent LLM abstraction for the multi-agent orchestrator.

Two calls are distinguished so stubs in tests can return the right shape:

- ``propose``: structured JSON ``{"routes": [[int, ...], ...]}`` — used for
  initial proposals and for refinement.
- ``critique``: free-form text — used when one persona critiques another's
  proposal.

The real implementation is thin glue over the Anthropic SDK; tests replace
it with a :class:`StubAgentLLM` that returns canned responses.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from src.solvers.single_agent import ROUTES_JSON_SCHEMA


@runtime_checkable
class AgentLLM(Protocol):
    def propose(
        self, persona_name: str, system: str, messages: list[dict], model: str
    ) -> list[list[int]]: ...

    def critique(
        self, persona_name: str, system: str, messages: list[dict], model: str
    ) -> str: ...


def anthropic_agent_llm(client=None, *, max_tokens: int = 4000) -> AgentLLM:
    """Build a real :class:`AgentLLM` backed by the Anthropic SDK.

    Imported lazily so the module is importable without ``anthropic`` installed.
    Prompt caching is applied to each persona's system prompt so a persona's
    own multi-turn calls (propose -> refine) share a cached prefix.
    """
    import anthropic  # noqa: PLC0415

    if client is None:
        client = anthropic.Anthropic()

    def _system_blocks(system: str) -> list[dict]:
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    class _Impl:
        def propose(self, persona_name, system, messages, model):
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=_system_blocks(system),
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
            return json.loads(text)["routes"]

        def critique(self, persona_name, system, messages, model):
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=_system_blocks(system),
                messages=messages,
                thinking={"type": "adaptive"},
            )
            return next(b.text for b in response.content if b.type == "text")

    return _Impl()
