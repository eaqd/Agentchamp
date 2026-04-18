"""Run a CVRP solver across benchmark instances and print a summary table.

Example:
    uv run python scripts/run_benchmark.py --solver clarke_wright \\
        --instances A-n32-k5,A-n45-k6,A-n60-k9
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

from src.models import Instance, Solution
from src.parser import parse_vrp
from src.solvers.classical import clarke_wright
from src.solvers.single_agent import single_agent
from src.validator import compute_cost, validate

REPO_ROOT = Path(__file__).resolve().parent.parent

SOLVERS: dict[str, Callable[[Instance], Solution]] = {
    "clarke_wright": clarke_wright,
    "single_agent": single_agent,
}


def parse_sol_cost(path: Path) -> float | None:
    """Extract the ``Cost`` value from a CVRPLIB ``.sol`` file."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("cost"):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    return float(parts[-1])
                except ValueError:
                    return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="clarke_wright", choices=sorted(SOLVERS))
    parser.add_argument(
        "--instances",
        required=True,
        help="Comma-separated instance names, e.g. A-n32-k5,A-n45-k6",
    )
    args = parser.parse_args(argv)

    solver = SOLVERS[args.solver]
    names = [n.strip() for n in args.instances.split(",") if n.strip()]

    header = f"{'instance':<12} {'cost':>10} {'optimal':>10} {'gap%':>8} {'time(s)':>8} {'valid':>6}"
    print(header)
    print("-" * len(header))

    any_failed = False
    for name in names:
        vrp_path = REPO_ROOT / "data" / "raw" / f"{name}.vrp"
        sol_path = REPO_ROOT / "data" / "solutions" / f"{name}.sol"
        if not vrp_path.exists():
            print(f"{name:<12} MISSING — run scripts/download_data.py")
            any_failed = True
            continue

        instance = parse_vrp(vrp_path)
        start = time.perf_counter()
        solution = solver(instance)
        elapsed = time.perf_counter() - start

        errors = validate(instance, solution)
        cost = solution.total_cost if solution.total_cost is not None else compute_cost(instance, solution)
        optimal = parse_sol_cost(sol_path)
        gap_str = "N/A"
        if optimal is not None and optimal > 0:
            gap = (cost - optimal) / optimal * 100
            gap_str = f"{gap:+.2f}"

        opt_str = f"{optimal:.2f}" if optimal is not None else "N/A"
        valid_str = "yes" if not errors else "NO"
        print(
            f"{name:<12} {cost:>10.2f} {opt_str:>10} {gap_str:>8} {elapsed:>8.3f} {valid_str:>6}"
        )
        if errors:
            any_failed = True
            for err in errors:
                print(f"  ! {err}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
