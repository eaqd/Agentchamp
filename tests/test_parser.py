"""Parser smoke test against the canonical A-n32-k5 instance."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.parser import parse_vrp

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTANCE_PATH = REPO_ROOT / "data" / "raw" / "A-n32-k5.vrp"


def test_parse_a_n32_k5(capsys: pytest.CaptureFixture[str]) -> None:
    if not INSTANCE_PATH.exists():
        pytest.skip(
            f"{INSTANCE_PATH.relative_to(REPO_ROOT)} not found — "
            "run `uv run python scripts/download_data.py --instances A-n32-k5` first."
        )

    instance = parse_vrp(INSTANCE_PATH)

    assert instance.name == "A-n32-k5"
    assert instance.dimension == 32
    assert instance.capacity == 100
    assert instance.depot_id == 1
    assert instance.depot.demand == 0
    assert len(instance.customers) == 32
    assert len(instance.non_depot_customers) == 31

    total_demand = sum(c.demand for c in instance.non_depot_customers)
    print(f"name:         {instance.name}")
    print(f"dimension:    {instance.dimension}")
    print(f"capacity:     {instance.capacity}")
    print(f"total demand: {total_demand}")

    # Make the summary visible in pytest output.
    captured = capsys.readouterr()
    assert "A-n32-k5" in captured.out
