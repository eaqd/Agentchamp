"""Pydantic data models for CVRP instances and solutions."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field


class Customer(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    x: float
    y: float
    demand: int = Field(ge=0)


class Instance(BaseModel):
    name: str
    dimension: int = Field(gt=0)
    capacity: int = Field(gt=0)
    depot_id: int
    customers: list[Customer]

    @property
    def depot(self) -> Customer:
        return self._by_id(self.depot_id)

    @property
    def non_depot_customers(self) -> list[Customer]:
        return [c for c in self.customers if c.id != self.depot_id]

    def _by_id(self, customer_id: int) -> Customer:
        for c in self.customers:
            if c.id == customer_id:
                return c
        raise KeyError(f"Customer id {customer_id} not found in instance {self.name}")

    def distance(self, a: int, b: int) -> float:
        ca, cb = self._by_id(a), self._by_id(b)
        return math.hypot(ca.x - cb.x, ca.y - cb.y)


class Route(BaseModel):
    customer_ids: list[int]


class Solution(BaseModel):
    instance_name: str
    routes: list[Route]
    total_cost: float | None = None
