"""
Priority-Cluster Heuristic Algorithm

1. Clusters delivery orders geographically using centroid-based grouping.
2. Assigns trucks to clusters based on proximity and cargo-type overlap.
3. Within each cluster, serves orders in strict priority order with
   route chaining to capture travel efficiency.
4. Applies workload balancing across trucks as a soft constraint.

Strengths: Reduced total travel, better load distribution, priority-aware.
Weakness: Cluster quality depends on k; not globally provably optimal.
"""
import time
import math
from typing import List, Dict, Tuple, Optional

from models import Truck, DeliveryOrder, Assignment, AllocationResult, Location
from algorithms.base import (
    can_serve,
    assignment_cost,
    haversine_km,
    travel_time_hours,
    compute_metrics,
    INFEASIBLE_COST,
)


def _kmeans_clusters(
    locations: List[Location],
    k: int,
    max_iter: int = 20,
) -> List[int]:
    n = len(locations)
    if n == 0:
        return []
    k = min(k, n)

    centroids = [(locations[i].lat, locations[i].lng) for i in range(k)]
    labels    = [0] * n

    for _ in range(max_iter):
        new_labels = []
        for loc in locations:
            best_c, best_dist = 0, float("inf")
            for c_idx, (clat, clng) in enumerate(centroids):
                d = math.sqrt((loc.lat - clat) ** 2 + (loc.lng - clng) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_c    = c_idx
            new_labels.append(best_c)

        if new_labels == labels:
            break
        labels = new_labels

        for c_idx in range(k):
            members = [locations[i] for i, lbl in enumerate(labels) if lbl == c_idx]
            if members:
                centroids[c_idx] = (
                    sum(m.lat for m in members) / len(members),
                    sum(m.lng for m in members) / len(members),
                )

    return labels


def _cluster_centroid(orders: List[DeliveryOrder]) -> Tuple[float, float]:
    if not orders:
        return (0.0, 0.0)
    return (
        sum(o.location.lat for o in orders) / len(orders),
        sum(o.location.lng for o in orders) / len(orders),
    )


def heuristic_allocate(
    technicians: List[Truck],
    requests: List[DeliveryOrder],
) -> AllocationResult:
    trucks = technicians
    orders = requests
    start_ms = time.monotonic()

    if not trucks or not orders:
        elapsed_ms = (time.monotonic() - start_ms) * 1000
        return AllocationResult(
            algorithm="heuristic",
            assignments=[],
            unassigned_order_ids=[o.id for o in orders],
            metrics=compute_metrics([], [o.id for o in orders], [t.id for t in trucks]),
            execution_time_ms=round(elapsed_ms, 2),
        )

    # --- Step 1: Cluster orders ---
    n_clusters = max(1, len(trucks) // 2)
    locations  = [o.location for o in orders]
    labels     = _kmeans_clusters(locations, k=n_clusters)

    clusters: Dict[int, List[DeliveryOrder]] = {}
    for order, label in zip(orders, labels):
        clusters.setdefault(label, []).append(order)

    for label in clusters:
        clusters[label].sort(key=lambda o: (-o.priority, o.window_end))

    # --- Step 2: Assign trucks to clusters ---
    truck_state: Dict[str, dict] = {
        t.id: {
            "location":         t.location,
            "current_time":     t.shift_start,
            "assignment_count": 0,
            "cluster":          None,
        }
        for t in trucks
    }
    truck_map = {t.id: t for t in trucks}

    cluster_cargo: Dict[int, set] = {}
    for label, cluster_orders in clusters.items():
        cluster_cargo[label] = {o.cargo_type for o in cluster_orders}

    sorted_clusters = sorted(
        clusters.items(),
        key=lambda kv: -sum(o.priority for o in kv[1]),
    )

    assignments: List[Assignment] = []
    assigned_order_ids: set = set()

    # --- Step 3: Process each cluster ---
    for label, cluster_orders in sorted_clusters:
        clat, clng  = _cluster_centroid(cluster_orders)
        needed_cargo = cluster_cargo[label]

        def truck_score(truck: Truck) -> float:
            state = truck_state[truck.id]
            if state["assignment_count"] >= truck.max_deliveries:
                return float("inf")
            cargo_match    = len(needed_cargo & set(truck.cargo_types)) / max(len(needed_cargo), 1)
            dist_to_cluster = math.sqrt(
                (truck.location.lat - clat) ** 2 + (truck.location.lng - clng) ** 2
            )
            load_penalty   = state["assignment_count"] * 0.1
            return dist_to_cluster - cargo_match * 0.01 + load_penalty

        ranked_trucks = sorted(trucks, key=truck_score)

        for order in cluster_orders:
            if order.id in assigned_order_ids:
                continue

            best_cost      = INFEASIBLE_COST
            best_truck_id: Optional[str] = None

            for truck in ranked_trucks:
                state = truck_state[truck.id]
                if state["assignment_count"] >= truck.max_deliveries:
                    continue
                load_factor = state["assignment_count"] / truck.max_deliveries
                cost = assignment_cost(truck, order, state["current_time"], state["location"])
                if cost >= INFEASIBLE_COST:
                    continue
                cost += load_factor * 0.15

                if cost < best_cost:
                    best_cost      = cost
                    best_truck_id  = truck.id

            if best_truck_id is None:
                continue

            truck = truck_map[best_truck_id]
            state = truck_state[best_truck_id]
            tt    = travel_time_hours(state["location"], order.location)
            dist  = haversine_km(state["location"], order.location)
            arrival       = state["current_time"] + tt
            service_start = max(arrival, order.window_start)
            load_pct      = state["assignment_count"] / truck.max_deliveries * 100

            assignments.append(Assignment(
                truck_id=best_truck_id,
                order_id=order.id,
                travel_time_hours=round(tt, 3),
                distance_km=round(dist, 2),
                arrival_time=round(arrival, 3),
                score=round(best_cost, 3),
                explanation=(
                    f"Cluster {label} assignment: {dist:.1f} km away, "
                    f"arrives {arrival:.2f}h (window {order.window_start}–{order.window_end}h), "
                    f"truck load {load_pct:.0f}%, cost={best_cost:.3f}"
                ),
            ))
            assigned_order_ids.add(order.id)

            state["location"]         = order.location
            state["current_time"]     = service_start + order.duration
            state["assignment_count"] += 1

    # --- Step 4: Second pass for any unassigned orders ---
    unassigned_orders = [o for o in orders if o.id not in assigned_order_ids]
    unassigned_orders.sort(key=lambda o: (-o.priority, o.window_end))

    for order in unassigned_orders:
        best_cost     = INFEASIBLE_COST
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
                f"Second-pass assignment (no cluster match): {dist:.1f} km away, "
                f"arrives {arrival:.2f}h, cost={best_cost:.3f}"
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
        algorithm="heuristic",
        assignments=assignments,
        unassigned_order_ids=unassigned,
        metrics=metrics,
        execution_time_ms=round(elapsed_ms, 2),
    )
