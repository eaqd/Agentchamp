"""Download Augerat Set A CVRP benchmark instances from CVRPLIB.

Default source: CVRPLIB (http://vrp.galgos.inf.puc-rio.br/). If that host is
unreachable from your network, override ``--base-url`` or place the ``.vrp``
files into ``data/raw/`` manually — they are TSPLIB-style text files and the
parser only needs the standard sections.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

SET_A_INSTANCES: list[str] = [
    "A-n32-k5", "A-n33-k5", "A-n33-k6", "A-n34-k5", "A-n36-k5", "A-n37-k5",
    "A-n37-k6", "A-n38-k5", "A-n39-k5", "A-n39-k6", "A-n44-k6", "A-n45-k6",
    "A-n45-k7", "A-n46-k7", "A-n48-k7", "A-n53-k7", "A-n54-k7", "A-n55-k9",
    "A-n60-k9", "A-n61-k9", "A-n62-k8", "A-n63-k9", "A-n63-k10", "A-n64-k9",
    "A-n65-k9", "A-n69-k9", "A-n80-k10",
]

DEFAULT_BASE_URL = "http://vrp.galgos.inf.puc-rio.br/media/com_vrp"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _fetch(url: str, dest: Path, *, force: bool, timeout: float) -> str:
    if dest.exists() and not force:
        return "skip"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            dest.write_bytes(resp.read())
        return "ok"
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return "fail"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        default=",".join(SET_A_INSTANCES),
        help="Comma-separated instance names (default: all 27 Set A instances).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for CVRPLIB files (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--skip-solutions",
        action="store_true",
        help="Don't download the .sol companion files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist locally.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-request timeout in seconds (default: 20).",
    )
    args = parser.parse_args(argv)

    raw_dir = REPO_ROOT / "data" / "raw"
    sol_dir = REPO_ROOT / "data" / "solutions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sol_dir.mkdir(parents=True, exist_ok=True)

    names = [n.strip() for n in args.instances.split(",") if n.strip()]
    failures = 0
    for name in names:
        vrp_url = f"{args.base_url}/instances/A/{name}.vrp"
        vrp_dest = raw_dir / f"{name}.vrp"
        status = _fetch(vrp_url, vrp_dest, force=args.force, timeout=args.timeout)
        print(f"[{status:4s}] {name}.vrp")
        if status == "fail":
            failures += 1

        if not args.skip_solutions:
            sol_url = f"{args.base_url}/sol/A/{name}.sol"
            sol_dest = sol_dir / f"{name}.sol"
            sol_status = _fetch(sol_url, sol_dest, force=args.force, timeout=args.timeout)
            print(f"[{sol_status:4s}] {name}.sol")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
