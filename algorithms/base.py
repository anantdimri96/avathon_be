"""
Shared utilities for all allocation algorithms.
"""
import math
from typing import List, Tuple
from models import Location, Truck, DeliveryOrder, Assignment

INFEASIBLE_COST = 1e9
URBAN_SPEED_KMH = 30.0


def haversine_km(loc1: Location, loc2: Location) -> float:
    """Compute great-circle distance in km between two lat/lng points."""
    R = 6371.0
    lat1, lon1 = math.radians(loc1.lat), math.radians(loc1.lng)
    lat2, lon2 = math.radians(loc2.lat), math.radians(loc2.lng)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def travel_time_hours(loc1: Location, loc2: Location) -> float:
    """Estimate travel time in hours at urban speed."""
    return haversine_km(loc1, loc2) / URBAN_SPEED_KMH


def can_serve(
    truck: Truck,
    order: DeliveryOrder,
    truck_current_time: float,
    truck_current_loc: Location,
) -> Tuple[bool, str]:
    """
    Check hard constraints for a truck-order pair.
    Returns (feasible, reason).
    """
    if order.cargo_type not in truck.cargo_types:
        return False, f"cargo mismatch: truck cannot carry {order.cargo_type}"

    tt = travel_time_hours(truck_current_loc, order.location)
    arrival = truck_current_time + tt

    if arrival > order.window_end:
        return False, f"too late: arrives {arrival:.2f}h but window closes {order.window_end:.2f}h"

    finish_time = max(arrival, order.window_start) + order.duration
    if finish_time > truck.shift_end:
        return False, f"exceeds shift: finishes {finish_time:.2f}h, shift ends {truck.shift_end:.2f}h"

    return True, "ok"


def assignment_cost(
    truck: Truck,
    order: DeliveryOrder,
    truck_current_time: float,
    truck_current_loc: Location,
) -> float:
    """
    Compute a scalar cost for assigning this truck to this order.
    Lower is better.
    """
    feasible, _ = can_serve(truck, order, truck_current_time, truck_current_loc)
    if not feasible:
        return INFEASIBLE_COST

    dist = haversine_km(truck_current_loc, order.location)
    tt = travel_time_hours(truck_current_loc, order.location)
    arrival = truck_current_time + tt

    distance_cost  = dist / 10.0
    priority_bonus = (5 - order.priority) * 0.1
    slack          = order.window_end - arrival
    urgency_cost   = max(0.0, 0.2 - slack * 0.05)

    return distance_cost + priority_bonus + urgency_cost


def compute_metrics(
    assignments: List[Assignment],
    all_order_ids: List[str],
    all_truck_ids: List[str],
) -> dict:
    """Compute performance metrics for an allocation result."""
    n_orders   = len(all_order_ids)
    n_assigned = len(assignments)
    n_unassigned = n_orders - n_assigned

    fulfilled_rate = n_assigned / n_orders if n_orders > 0 else 0.0

    if assignments:
        avg_distance   = sum(a.distance_km for a in assignments) / n_assigned
        total_distance = sum(a.distance_km for a in assignments)
        avg_score      = sum(a.score for a in assignments) / n_assigned

        load: dict = {}
        for a in assignments:
            load[a.truck_id] = load.get(a.truck_id, 0) + 1

        loads    = list(load.values())
        avg_load = sum(loads) / len(loads) if loads else 0
        std_load = math.sqrt(sum((l - avg_load) ** 2 for l in loads) / len(loads)) if len(loads) > 1 else 0
        load_balance = 1.0 - min(std_load / (avg_load + 1e-9), 1.0)
    else:
        avg_distance   = 0.0
        total_distance = 0.0
        avg_score      = 0.0
        load_balance   = 1.0

    return {
        "total_orders":       n_orders,
        "total_requests":     n_orders,   # alias for frontend compatibility
        "assigned":           n_assigned,
        "unassigned":         n_unassigned,
        "fulfillment_rate":   round(fulfilled_rate * 100, 1),
        "avg_distance_km":    round(avg_distance, 2),
        "total_distance_km":  round(total_distance, 2),
        "avg_cost_score":     round(avg_score, 3),
        "load_balance_score": round(load_balance * 100, 1),
    }
