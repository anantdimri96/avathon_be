"""
Hungarian Algorithm (Kuhn-Munkres) for Batch Optimal Assignment

Constructs a cost matrix and finds the globally optimal 1-to-1 assignment
between trucks and orders using scipy's linear_sum_assignment.

Each truck is expanded into max_deliveries virtual slots. All slots use
the truck's depot location (no route chaining possible in batch mode).

Strengths: Globally optimal for its cost model, considers all options simultaneously.
Weakness: Uses static depot location per truck, computationally heavier O(n^3).
"""
import time
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from models import Truck, DeliveryOrder, Assignment, AllocationResult
from algorithms.base import (
    can_serve,
    assignment_cost,
    haversine_km,
    travel_time_hours,
    compute_metrics,
    INFEASIBLE_COST,
)


def hungarian_allocate(
    technicians: List[Truck],
    requests: List[DeliveryOrder],
) -> AllocationResult:
    trucks = technicians
    orders = requests
    start_ms = time.monotonic()

    slots: List[Tuple[Truck, int]] = []
    for truck in trucks:
        for slot_idx in range(truck.max_deliveries):
            slots.append((truck, slot_idx))

    n_slots  = len(slots)
    n_orders = len(orders)

    if n_slots == 0 or n_orders == 0:
        elapsed_ms = (time.monotonic() - start_ms) * 1000
        return AllocationResult(
            algorithm="hungarian",
            assignments=[],
            unassigned_order_ids=[o.id for o in orders],
            metrics=compute_metrics([], [o.id for o in orders], [t.id for t in trucks]),
            execution_time_ms=round(elapsed_ms, 2),
        )

    cost_matrix = np.full((n_slots, n_orders), fill_value=INFEASIBLE_COST)

    for i, (truck, _slot_idx) in enumerate(slots):
        for j, order in enumerate(orders):
            cost = assignment_cost(truck, order, truck.shift_start, truck.location)
            cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments: List[Assignment] = []
    assigned_order_ids: set = set()

    for row, col in zip(row_ind, col_ind):
        cost = cost_matrix[row, col]
        if cost >= INFEASIBLE_COST * 0.9:
            continue

        truck, _slot_idx = slots[row]
        order = orders[col]

        if order.id in assigned_order_ids:
            continue

        tt      = travel_time_hours(truck.location, order.location)
        dist    = haversine_km(truck.location, order.location)
        arrival = truck.shift_start + tt

        assignments.append(Assignment(
            truck_id=truck.id,
            order_id=order.id,
            travel_time_hours=round(tt, 3),
            distance_km=round(dist, 2),
            arrival_time=round(arrival, 3),
            score=round(cost, 3),
            explanation=(
                f"Globally optimal match: {dist:.1f} km from depot, "
                f"arrives {arrival:.2f}h (window {order.window_start}–{order.window_end}h), "
                f"cost={cost:.3f}"
            ),
        ))
        assigned_order_ids.add(order.id)

    unassigned = [o.id for o in orders if o.id not in assigned_order_ids]

    elapsed_ms = (time.monotonic() - start_ms) * 1000
    metrics    = compute_metrics(assignments, [o.id for o in orders], [t.id for t in trucks])

    return AllocationResult(
        algorithm="hungarian",
        assignments=assignments,
        unassigned_order_ids=unassigned,
        metrics=metrics,
        execution_time_ms=round(elapsed_ms, 2),
    )
