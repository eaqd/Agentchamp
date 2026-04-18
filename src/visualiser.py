"""Matplotlib visualisation of CVRP instances and solutions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.models import Instance, Solution


def plot_solution(
    instance: Instance,
    solution: Solution,
    output_path: Path | None = None,
    *,
    show: bool = False,
) -> None:
    """Render depot, customers, and route polylines.

    If ``output_path`` is provided, the figure is saved there (dpi=120).
    If ``show`` is True, calls ``plt.show()``. Otherwise the figure is closed
    silently — useful for batch plotting in scripts.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    depot = instance.depot

    non_depot = instance.non_depot_customers
    ax.scatter(
        [c.x for c in non_depot],
        [c.y for c in non_depot],
        c="#1f77b4",
        s=30,
        zorder=3,
        label="customers",
    )
    ax.scatter(
        [depot.x], [depot.y],
        c="#d62728", s=140, marker="s", zorder=4, label="depot",
    )

    cmap = plt.get_cmap("tab10")
    by_id = {c.id: c for c in instance.customers}
    for i, route in enumerate(solution.routes):
        if not route.customer_ids:
            continue
        xs = [depot.x] + [by_id[cid].x for cid in route.customer_ids] + [depot.x]
        ys = [depot.y] + [by_id[cid].y for cid in route.customer_ids] + [depot.y]
        ax.plot(xs, ys, "-", color=cmap(i % 10), linewidth=1.3, alpha=0.8, zorder=2)

    title = instance.name
    if solution.total_cost is not None:
        title += f"   cost = {solution.total_cost:.1f}"
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
