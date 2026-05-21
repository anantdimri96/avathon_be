"""
Greedy Allocation Algorithm

Processes orders sequentially in priority order. For each order, selects
the best available truck using a local cost function. Truck state
(location, time) is updated after each assignment, allowing multiple
deliveries per truck (route chaining).

Strengths: Fast, handles multiple deliveries, adapts to truck state.
Weakness: Local optimality — early choices may block better global solutions.
"""
import time
from typing import List, Dict

from models import Truck, DeliveryOrder, Assignment, AllocationResult, Location
from algorithms.base import (
    can_serve,
    assignment_cost,
    haversine_km,
    travel_time_hours,
    compute_metrics,
    INFEASIBLE_COST,
)


def greedy_allocate(
    technicians: List[Truck],      # parameter kept generic for ALGORITHMS dispatch
    requests: List[DeliveryOrder],
) -> AllocationResult:
    trucks  = technicians
    orders  = requests
    start_ms = time.monotonic()

    truck_state: Dict[str, dict] = {
        t.id: {
            "location":         t.location,
            "current_time":     t.shift_start,
            "assignment_count": 0,
        }
        for t in trucks
    }

    truck_map = {t.id: t for t in trucks}

    sorted_orders = sorted(orders, key=lambda o: (-o.priority, o.window_end))

    assignments: List[Assignment] = []
    assigned_order_ids: set = set()

    for order in sorted_orders:
        best_cost    = INFEASIBLE_COST
        best_truck_id = None

        for truck in trucks:
            state = truck_state[truck.id]
            if state["assignment_count"] >= truck.max_deliveries:
                continue
            cost = assignment_cost(truck, order, state["current_time"], state["location"])
            if cost < best_cost:
                best_cost     = cost
                best_truck_id = truck.id

        if best_truck_id is None:
            continue

        truck = truck_map[best_truck_id]
        state = truck_state[best_truck_id]
        tt    = travel_time_hours(state["location"], order.location)
        dist  = haversine_km(state["location"], order.location)
        arrival       = state["current_time"] + tt
        service_start = max(arrival, order.window_start)

        assignments.append(Assignment(
            truck_id=best_truck_id,
            order_id=order.id,
            travel_time_hours=round(tt, 3),
            distance_km=round(dist, 2),
            arrival_time=round(arrival, 3),
            score=round(best_cost, 3),
            explanation=(
                f"Best local match: {dist:.1f} km away, "
                f"arrives {arrival:.2f}h (window {order.window_start}–{order.window_end}h), "
                f"cost={best_cost:.3f}"
            ),
        ))
        assigned_order_ids.add(order.id)

        state["location"]         = order.location
        state["current_time"]     = service_start + order.duration
        state["assignment_count"] += 1

    unassigned = [o.id for o in orders if o.id not in assigned_order_ids]

    elapsed_ms = (time.monotonic() - start_ms) * 1000
    metrics    = compute_metrics(assignments, [o.id for o in orders], [t.id for t in trucks])

    return AllocationResult(
        algorithm="greedy",
        assignments=assignments,
        unassigned_order_ids=unassigned,
        metrics=metrics,
        execution_time_ms=round(elapsed_ms, 2),
    )
