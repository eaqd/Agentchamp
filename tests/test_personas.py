"""Tests for src.agents.personas — sanity that prompts are distinct and shaped right."""

from __future__ import annotations

from src.agents.personas import (
    CONSTRAINT_ENGINEER,
    DEFAULT_PERSONAS,
    GEOMETRIC_THINKER,
    HEURISTIC_PRAGMATIST,
    SHARED_RULES,
)


def test_three_default_personas() -> None:
    assert len(DEFAULT_PERSONAS) == 3
    names = {p.name for p in DEFAULT_PERSONAS}
    assert names == {"Geometric Thinker", "Constraint Engineer", "Heuristic Pragmatist"}


def test_each_persona_embeds_shared_rules() -> None:
    for persona in DEFAULT_PERSONAS:
        assert SHARED_RULES in persona.system_prompt
        # Contract lines must all appear.
        assert "depot id must NOT" in persona.system_prompt
        assert '{"routes":' in persona.system_prompt


def test_persona_voices_are_distinct() -> None:
    # Voice texts must differ — if two are identical, personas have collapsed.
    voices = {p.voice for p in DEFAULT_PERSONAS}
    assert len(voices) == 3


def test_geometric_voice_signals_geometry() -> None:
    v = GEOMETRIC_THINKER.voice.lower()
    assert "sweep" in v or "polar" in v
    assert "convex" in v or "hull" in v


def test_constraint_voice_signals_ip_thinking() -> None:
    v = CONSTRAINT_ENGINEER.voice.lower()
    assert "capacity" in v and ("pack" in v or "slack" in v)


def test_heuristic_voice_signals_local_search() -> None:
    v = HEURISTIC_PRAGMATIST.voice.lower()
    assert "2-opt" in v or "or-opt" in v
    assert "savings" in v or "nearest" in v


def test_short_ids_unique() -> None:
    short_ids = [p.short_id for p in DEFAULT_PERSONAS]
    assert len(set(short_ids)) == len(short_ids)
