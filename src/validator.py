"""Solution validation and cost computation for CVRP instances."""

from __future__ import annotations

from src.models import Instance, Solution


def compute_cost(instance: Instance, solution: Solution) -> float:
    """Total Euclidean tour length across all routes.

    Each route is implicitly bracketed by the depot: depot -> c1 -> ... -> cn -> depot.
    Empty routes contribute zero. Unknown customer ids raise ``KeyError``.
    """
    total = 0.0
    for route in solution.routes:
        if not route.customer_ids:
            continue
        prev = instance.depot_id
        for cid in route.customer_ids:
            total += instance.distance(prev, cid)
            prev = cid
        total += instance.distance(prev, instance.depot_id)
    return total


def validate(instance: Instance, solution: Solution) -> list[str]:
    """Return a list of human-readable errors (empty list means valid).

    Checks:
    - no route contains the depot id
    - every customer id in every route exists in the instance
    - every non-depot customer is visited exactly once across all routes
    - each route's total demand does not exceed the vehicle capacity
    """
    errors: list[str] = []
    non_depot_ids = {c.id for c in instance.non_depot_customers}
    demand_by_id = {c.id: c.demand for c in instance.customers}
    visit_counts: dict[int, int] = {}

    for idx, route in enumerate(solution.routes):
        load = 0
        for cid in route.customer_ids:
            if cid == instance.depot_id:
                errors.append(f"Route {idx}: contains depot id {cid}")
                continue
            if cid not in non_depot_ids:
                errors.append(f"Route {idx}: unknown customer id {cid}")
                continue
            visit_counts[cid] = visit_counts.get(cid, 0) + 1
            load += demand_by_id[cid]
        if load > instance.capacity:
            errors.append(
                f"Route {idx}: load {load} exceeds capacity {instance.capacity}"
            )

    for cid in non_depot_ids:
        count = visit_counts.get(cid, 0)
        if count == 0:
            errors.append(f"Customer {cid} not visited")
        elif count > 1:
            errors.append(f"Customer {cid} visited {count} times")

    return errors
