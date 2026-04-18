"""Tests for src.validator — cost computation and solution correctness."""

from __future__ import annotations

import math

from src.models import Instance, Route, Solution
from src.validator import compute_cost, validate


def test_valid_solution_has_no_errors(ring_instance: Instance) -> None:
    # Two routes, each load=20, capacity=30 -> valid.
    solution = Solution(
        instance_name=ring_instance.name,
        routes=[Route(customer_ids=[2, 3]), Route(customer_ids=[4, 5])],
    )
    assert validate(ring_instance, solution) == []


def test_compute_cost_ring(ring_instance: Instance) -> None:
    # Route 0 -> 2 (10,0) -> 3 (0,10) -> 0: 10 + sqrt(200) + 10
    # Route 0 -> 4 (-10,0) -> 5 (0,-10) -> 0: 10 + sqrt(200) + 10
    solution = Solution(
        instance_name=ring_instance.name,
        routes=[Route(customer_ids=[2, 3]), Route(customer_ids=[4, 5])],
    )
    expected = 2 * (10 + math.sqrt(200) + 10)
    assert compute_cost(ring_instance, solution) == expected


def test_capacity_violation_detected(ring_instance: Instance) -> None:
    # All four customers on one route: load 40 > capacity 30.
    solution = Solution(
        instance_name=ring_instance.name,
        routes=[Route(customer_ids=[2, 3, 4, 5])],
    )
    errors = validate(ring_instance, solution)
    assert any("exceeds capacity" in e for e in errors)


def test_missing_customer_detected(ring_instance: Instance) -> None:
    solution = Solution(
        instance_name=ring_instance.name,
        routes=[Route(customer_ids=[2, 3, 4])],  # 5 missing
    )
    errors = validate(ring_instance, solution)
    assert any("Customer 5 not visited" in e for e in errors)


def test_duplicate_customer_detected(ring_instance: Instance) -> None:
    solution = Solution(
        instance_name=ring_instance.name,
        routes=[Route(customer_ids=[2, 3]), Route(customer_ids=[3, 4, 5])],
    )
    errors = validate(ring_instance, solution)
    assert any("Customer 3 visited 2 times" in e for e in errors)


def test_depot_in_route_detected(ring_instance: Instance) -> None:
    solution = Solution(
        instance_name=ring_instance.name,
        routes=[Route(customer_ids=[1, 2, 3]), Route(customer_ids=[4, 5])],
    )
    errors = validate(ring_instance, solution)
    assert any("contains depot id" in e for e in errors)


def test_unknown_customer_detected(ring_instance: Instance) -> None:
    solution = Solution(
        instance_name=ring_instance.name,
        routes=[Route(customer_ids=[2, 3, 99]), Route(customer_ids=[4, 5])],
    )
    errors = validate(ring_instance, solution)
    assert any("unknown customer id 99" in e for e in errors)
