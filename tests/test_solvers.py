"""Tests for src.solvers — ensures classical solvers produce valid solutions."""

from __future__ import annotations

from src.models import Customer, Instance
from src.solvers.classical import clarke_wright
from src.validator import validate


def test_clarke_wright_produces_valid_solution(ring_instance: Instance) -> None:
    solution = clarke_wright(ring_instance)
    errors = validate(ring_instance, solution)
    assert errors == [], f"Invalid solution: {errors}"
    assert solution.total_cost is not None and solution.total_cost > 0


def test_clarke_wright_visits_all_customers(ring_instance: Instance) -> None:
    solution = clarke_wright(ring_instance)
    visited = {cid for route in solution.routes for cid in route.customer_ids}
    assert visited == {2, 3, 4, 5}


def test_clarke_wright_respects_capacity() -> None:
    # Tight instance: 3 customers each demanding 10, capacity 15 -> 3 routes.
    customers = [
        Customer(id=1, x=0.0, y=0.0, demand=0),
        Customer(id=2, x=1.0, y=0.0, demand=10),
        Customer(id=3, x=2.0, y=0.0, demand=10),
        Customer(id=4, x=3.0, y=0.0, demand=10),
    ]
    instance = Instance(
        name="tight-4", dimension=4, capacity=15, depot_id=1, customers=customers,
    )
    solution = clarke_wright(instance)
    assert validate(instance, solution) == []
    assert len(solution.routes) == 3


def test_clarke_wright_empty_non_depot() -> None:
    # Degenerate case: only a depot. Solver should not crash.
    instance = Instance(
        name="depot-only",
        dimension=1,
        capacity=100,
        depot_id=1,
        customers=[Customer(id=1, x=0.0, y=0.0, demand=0)],
    )
    solution = clarke_wright(instance)
    assert solution.routes == []
    assert solution.total_cost == 0.0
