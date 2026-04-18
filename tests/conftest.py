"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from src.models import Customer, Instance


@pytest.fixture
def ring_instance() -> Instance:
    """Small synthetic CVRP: depot at origin, 4 customers on a cross pattern."""
    customers = [
        Customer(id=1, x=0.0, y=0.0, demand=0),   # depot
        Customer(id=2, x=10.0, y=0.0, demand=10),
        Customer(id=3, x=0.0, y=10.0, demand=10),
        Customer(id=4, x=-10.0, y=0.0, demand=10),
        Customer(id=5, x=0.0, y=-10.0, demand=10),
    ]
    return Instance(
        name="ring-5",
        dimension=5,
        capacity=30,
        depot_id=1,
        customers=customers,
    )
