"""Parser for CVRPLIB `.vrp` instance files (TSPLIB-style format)."""

from __future__ import annotations

from pathlib import Path

from src.models import Customer, Instance

_SUPPORTED_EDGE_WEIGHT_TYPE = "EUC_2D"


def parse_vrp(path: Path) -> Instance:
    """Parse a CVRPLIB `.vrp` file and return an :class:`Instance`.

    Supports the standard Augerat-style header (NAME, DIMENSION, CAPACITY,
    EDGE_WEIGHT_TYPE) followed by NODE_COORD_SECTION, DEMAND_SECTION, and
    DEPOT_SECTION (terminated by ``-1``). Raises ``ValueError`` on unsupported
    edge-weight types or missing required sections.
    """
    lines = [ln.rstrip() for ln in path.read_text().splitlines() if ln.strip()]

    name: str | None = None
    dimension: int | None = None
    capacity: int | None = None
    edge_weight_type: str | None = None

    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    depot_ids: list[int] = []

    section: str | None = None

    for raw in lines:
        line = raw.strip()
        if line == "EOF":
            break

        # Header lines use "KEY : VALUE" (or "KEY: VALUE").
        if ":" in line and section is None:
            key, _, value = line.partition(":")
            key = key.strip().upper()
            value = value.strip()
            if key == "NAME":
                name = value
            elif key == "DIMENSION":
                dimension = int(value)
            elif key == "CAPACITY":
                capacity = int(value)
            elif key == "EDGE_WEIGHT_TYPE":
                edge_weight_type = value
            # Silently ignore TYPE, COMMENT, etc.
            continue

        # Section markers switch parser state.
        upper = line.upper()
        if upper in {"NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"}:
            section = upper
            continue

        if section == "NODE_COORD_SECTION":
            parts = line.split()
            cid = int(parts[0])
            coords[cid] = (float(parts[1]), float(parts[2]))
        elif section == "DEMAND_SECTION":
            parts = line.split()
            demands[int(parts[0])] = int(parts[1])
        elif section == "DEPOT_SECTION":
            val = int(line)
            if val == -1:
                section = None
            else:
                depot_ids.append(val)

    if edge_weight_type is not None and edge_weight_type != _SUPPORTED_EDGE_WEIGHT_TYPE:
        raise ValueError(
            f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type!r} "
            f"(only {_SUPPORTED_EDGE_WEIGHT_TYPE} is supported)"
        )
    if name is None or dimension is None or capacity is None:
        raise ValueError(f"Missing NAME/DIMENSION/CAPACITY in {path}")
    if not coords:
        raise ValueError(f"Missing NODE_COORD_SECTION in {path}")
    if not demands:
        raise ValueError(f"Missing DEMAND_SECTION in {path}")
    if not depot_ids:
        raise ValueError(f"Missing DEPOT_SECTION in {path}")
    if len(depot_ids) > 1:
        raise ValueError(f"Multi-depot instance not supported: {depot_ids}")

    customers = [
        Customer(id=cid, x=coords[cid][0], y=coords[cid][1], demand=demands[cid])
        for cid in sorted(coords)
    ]

    return Instance(
        name=name,
        dimension=dimension,
        capacity=capacity,
        depot_id=depot_ids[0],
        customers=customers,
    )
