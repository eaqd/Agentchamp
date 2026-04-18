"""Classical CVRP heuristics — Clarke-Wright parallel savings."""

from __future__ import annotations

from src.models import Instance, Route, Solution
from src.validator import compute_cost


def clarke_wright(instance: Instance) -> Solution:
    """Parallel Clarke-Wright savings heuristic.

    Starts with one route per customer (depot -> c -> depot), computes
    savings ``s(i, j) = d(0,i) + d(0,j) - d(i,j)`` for every pair, and
    greedily merges routes in decreasing order of savings whenever:

    - ``i`` and ``j`` belong to different routes,
    - each of ``i`` and ``j`` sits at an endpoint of its route (adjacent
      to the depot), and
    - the combined load fits in one vehicle.

    Returns a :class:`Solution` with ``total_cost`` populated.
    """
    depot = instance.depot_id
    customers = instance.non_depot_customers
    if not customers:
        return Solution(instance_name=instance.name, routes=[], total_cost=0.0)

    # route_key -> ordered list of customer ids (depot implicit at both ends).
    routes: dict[int, list[int]] = {c.id: [c.id] for c in customers}
    loads: dict[int, int] = {c.id: c.demand for c in customers}
    route_of: dict[int, int] = {c.id: c.id for c in customers}

    savings: list[tuple[float, int, int]] = []
    for idx, ci in enumerate(customers):
        for cj in customers[idx + 1 :]:
            s = (
                instance.distance(depot, ci.id)
                + instance.distance(depot, cj.id)
                - instance.distance(ci.id, cj.id)
            )
            if s > 0:
                savings.append((s, ci.id, cj.id))
    savings.sort(reverse=True)

    for _, i, j in savings:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        if loads[ri] + loads[rj] > instance.capacity:
            continue

        route_i = list(routes[ri])
        route_j = list(routes[rj])

        # i must end up as the last element of route_i.
        if route_i[-1] != i:
            if route_i[0] == i:
                route_i.reverse()
            else:
                continue  # i is internal; can't merge.

        # j must end up as the first element of route_j.
        if route_j[0] != j:
            if route_j[-1] == j:
                route_j.reverse()
            else:
                continue  # j is internal; can't merge.

        merged = route_i + route_j
        routes[ri] = merged
        loads[ri] = loads[ri] + loads[rj]
        for cid in route_j:
            route_of[cid] = ri
        del routes[rj]
        del loads[rj]

    result = Solution(
        instance_name=instance.name,
        routes=[Route(customer_ids=r) for r in routes.values()],
    )
    result.total_cost = compute_cost(instance, result)
    return result
