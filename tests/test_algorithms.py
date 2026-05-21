"""
Algorithm correctness and comparison tests.
Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from models import Truck, DeliveryOrder, Location, CargoType, Priority, Assignment
from algorithms.greedy import greedy_allocate
from algorithms.hungarian import hungarian_allocate
from algorithms.heuristic import heuristic_allocate
from algorithms.ml_allocator import ml_allocate
from algorithms.base import haversine_km, travel_time_hours, can_serve, assignment_cost, compute_metrics, INFEASIBLE_COST
from data.seed_data import TRUCKS, DELIVERY_ORDERS


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_truck(id, lat, lng, cargo_types, shift_start=6.0, shift_end=17.0, max_del=5):
    return Truck(
        id=id, name=f"Driver {id}",
        location=Location(lat=lat, lng=lng),
        cargo_types=cargo_types,
        shift_start=shift_start, shift_end=shift_end, max_deliveries=max_del,
    )


def make_order(id, lat, lng, cargo, priority=3, ws=8.0, we=15.0, dur=0.5, desc=""):
    return DeliveryOrder(
        id=id,
        location=Location(lat=lat, lng=lng),
        cargo_type=cargo,
        priority=priority,
        window_start=ws, window_end=we,
        duration=dur,
        description=desc or f"Order {id}",
    )


# ─── Base utility tests ───────────────────────────────────────────────────────

class TestBaseUtils:
    def test_haversine_same_point(self):
        loc = Location(lat=37.7749, lng=-122.4194)
        assert haversine_km(loc, loc) == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        sf  = Location(lat=37.7749, lng=-122.4194)
        oak = Location(lat=37.8044, lng=-122.2712)
        dist = haversine_km(sf, oak)
        assert 11.0 < dist < 15.0

    def test_travel_time_positive(self):
        a = Location(lat=37.7749, lng=-122.4194)
        b = Location(lat=37.7599, lng=-122.4148)
        assert travel_time_hours(a, b) > 0

    def test_can_serve_cargo_mismatch(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1", 37.77, -122.41, CargoType.REFRIGERATED)
        ok, reason = can_serve(truck, order, 6.0, truck.location)
        assert not ok
        assert "cargo" in reason.lower()

    def test_can_serve_too_late(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1", 38.0, -122.41, CargoType.STANDARD, we=6.01)
        ok, reason = can_serve(truck, order, 6.0, truck.location)
        assert not ok

    def test_can_serve_happy_path(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1", 37.77, -122.41, CargoType.STANDARD)
        ok, _ = can_serve(truck, order, 6.0, truck.location)
        assert ok

    def test_can_serve_shift_end_exceeded(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD], shift_end=9.0)
        order = make_order("O1", 37.77, -122.41, CargoType.STANDARD, ws=8.5, we=12.0, dur=1.0)
        ok, reason = can_serve(truck, order, 6.0, truck.location)
        assert not ok
        assert any(word in reason.lower() for word in ("shift", "finish", "exceed"))

    def test_assignment_cost_infeasible_pair(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1", 37.77, -122.41, CargoType.REFRIGERATED)
        cost = assignment_cost(truck, order, 6.0, truck.location)
        assert cost == INFEASIBLE_COST

    def test_assignment_cost_prefers_closer_truck(self):
        order       = make_order("O1", 37.77, -122.41, CargoType.STANDARD)
        truck_close = make_truck("T_CLOSE", 37.771, -122.411, [CargoType.STANDARD])
        truck_far   = make_truck("T_FAR",   37.80,  -122.45,  [CargoType.STANDARD])
        cost_close  = assignment_cost(truck_close, order, 6.0, truck_close.location)
        cost_far    = assignment_cost(truck_far,   order, 6.0, truck_far.location)
        assert cost_close < cost_far

    def test_assignment_cost_high_priority_lower_than_low(self):
        truck      = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order_high = make_order("O_H", 37.77, -122.41, CargoType.STANDARD, priority=Priority.CRITICAL)
        order_low  = make_order("O_L", 37.77, -122.41, CargoType.STANDARD, priority=Priority.LOW)
        cost_high  = assignment_cost(truck, order_high, 6.0, truck.location)
        cost_low   = assignment_cost(truck, order_low,  6.0, truck.location)
        assert cost_high < cost_low

    def test_compute_metrics_empty_assignments(self):
        metrics = compute_metrics([], ["O1", "O2"], ["TRK1"])
        assert metrics["assigned"]          == 0
        assert metrics["unassigned"]        == 2
        assert metrics["fulfillment_rate"]  == 0.0
        assert metrics["total_distance_km"] == 0.0
        assert metrics["load_balance_score"] == 100.0

    def test_compute_metrics_counts(self):
        asg = Assignment(
            truck_id="TRK1", order_id="O1",
            travel_time_hours=0.1, distance_km=3.0,
            arrival_time=6.1, score=0.5,
            explanation="test",
        )
        metrics = compute_metrics([asg], ["O1", "O2"], ["TRK1", "TRK2"])
        assert metrics["assigned"]          == 1
        assert metrics["unassigned"]        == 1
        assert metrics["fulfillment_rate"]  == 50.0
        assert metrics["total_distance_km"] == pytest.approx(3.0)


# ─── Greedy tests ─────────────────────────────────────────────────────────────

class TestGreedy:
    def test_assigns_when_feasible(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1",   37.77, -122.41, CargoType.STANDARD)
        result = greedy_allocate([truck], [order])
        assert len(result.assignments) == 1
        assert result.assignments[0].order_id  == "O1"
        assert result.assignments[0].truck_id  == "TRK1"

    def test_cargo_constraint_respected(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1",   37.77, -122.41, CargoType.REFRIGERATED)
        result = greedy_allocate([truck], [order])
        assert len(result.assignments) == 0
        assert "O1" in result.unassigned_order_ids

    def test_priority_order(self):
        truck      = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD], max_del=1)
        order_high = make_order("O_HIGH", 37.77, -122.41, CargoType.STANDARD, priority=Priority.CRITICAL)
        order_low  = make_order("O_LOW",  37.77, -122.42, CargoType.STANDARD, priority=Priority.LOW)
        result = greedy_allocate([truck], [order_high, order_low])
        assigned_ids = [a.order_id for a in result.assignments]
        assert "O_HIGH" in assigned_ids
        assert "O_LOW"  not in assigned_ids

    def test_multiple_deliveries_per_truck(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD], max_del=3)
        orders = [
            make_order(f"O{i}", 37.77 + i * 0.001, -122.41, CargoType.STANDARD, ws=6.0 + i, we=17.0)
            for i in range(3)
        ]
        result = greedy_allocate([truck], orders)
        assert len(result.assignments) == 3

    def test_max_deliveries_cap(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD], max_del=2)
        orders = [
            make_order(f"O{i}", 37.77, -122.41, CargoType.STANDARD, ws=6.0, we=17.0, dur=0.1)
            for i in range(5)
        ]
        result = greedy_allocate([truck], orders)
        assert len(result.assignments) <= 2

    def test_empty_inputs(self):
        result = greedy_allocate([], [])
        assert result.assignments         == []
        assert result.unassigned_order_ids == []

    def test_metrics_computed(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1",   37.77, -122.41, CargoType.STANDARD)
        result = greedy_allocate([truck], [order])
        assert "assigned"          in result.metrics
        assert "total_distance_km" in result.metrics
        assert result.metrics["fulfillment_rate"] == 100.0

    def test_explanation_present(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1",   37.77, -122.41, CargoType.STANDARD)
        result = greedy_allocate([truck], [order])
        assert len(result.assignments[0].explanation) > 10


# ─── Hungarian tests ──────────────────────────────────────────────────────────

class TestHungarian:
    def test_assigns_when_feasible(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1",   37.77, -122.41, CargoType.STANDARD)
        result = hungarian_allocate([truck], [order])
        assert len(result.assignments) == 1

    def test_cargo_constraint_respected(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1",   37.77, -122.41, CargoType.REFRIGERATED)
        result = hungarian_allocate([truck], [order])
        assert len(result.assignments) == 0

    def test_globally_optimal_distance(self):
        truck_north = make_truck("TN", 37.80, -122.41, [CargoType.STANDARD])
        truck_south = make_truck("TS", 37.75, -122.41, [CargoType.STANDARD])
        order_north = make_order("ON", 37.81, -122.41, CargoType.STANDARD)
        order_south = make_order("OS", 37.74, -122.41, CargoType.STANDARD)
        result = hungarian_allocate([truck_north, truck_south], [order_north, order_south])
        asg = {a.truck_id: a.order_id for a in result.assignments}
        if len(result.assignments) == 2:
            assert asg.get("TN") == "ON" or asg.get("TS") == "OS"

    def test_empty_inputs(self):
        result = hungarian_allocate([], [])
        assert result.assignments == []


# ─── Heuristic tests ──────────────────────────────────────────────────────────

class TestHeuristic:
    def test_assigns_when_feasible(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1",   37.77, -122.41, CargoType.STANDARD)
        result = heuristic_allocate([truck], [order])
        assert len(result.assignments) == 1

    def test_cargo_constraint_respected(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1",   37.77, -122.41, CargoType.REFRIGERATED)
        result = heuristic_allocate([truck], [order])
        assert len(result.assignments) == 0

    def test_load_balancing(self):
        trucks = [
            make_truck("TRK1", 37.80, -122.41, [CargoType.STANDARD], max_del=6),
            make_truck("TRK2", 37.74, -122.41, [CargoType.STANDARD], max_del=6),
        ]
        orders = (
            [make_order(f"ON{i}", 37.80 + i * 0.003, -122.41, CargoType.STANDARD, ws=6.0, we=17.0, dur=0.3)
             for i in range(3)]
            + [make_order(f"OS{i}", 37.74 + i * 0.003, -122.41, CargoType.STANDARD, ws=6.0, we=17.0, dur=0.3)
               for i in range(3)]
        )
        result = heuristic_allocate(trucks, orders)
        from collections import Counter
        load = Counter(a.truck_id for a in result.assignments)
        assert len(load) == 2, f"Expected both trucks assigned, got: {dict(load)}"
        assert max(load.values()) <= 5

    def test_empty_inputs(self):
        result = heuristic_allocate([], [])
        assert result.assignments == []


# ─── Algorithm comparison tests ───────────────────────────────────────────────

class TestAlgorithmComparison:
    """Cross-algorithm property tests using full seed data."""

    def test_all_algorithms_produce_valid_results(self):
        for fn in [greedy_allocate, hungarian_allocate, heuristic_allocate]:
            result = fn(TRUCKS, DELIVERY_ORDERS)
            assert result.algorithm in {"greedy", "hungarian", "heuristic"}
            assert isinstance(result.assignments, list)
            assert isinstance(result.metrics, dict)
            assert result.execution_time_ms >= 0

    def test_no_duplicate_order_assignments(self):
        for fn in [greedy_allocate, hungarian_allocate, heuristic_allocate]:
            result = fn(TRUCKS, DELIVERY_ORDERS)
            order_ids = [a.order_id for a in result.assignments]
            assert len(order_ids) == len(set(order_ids)), \
                f"{result.algorithm}: duplicate order assignment found"

    def test_all_assignments_respect_cargo_constraints(self):
        truck_cargo = {t.id: set(t.cargo_types) for t in TRUCKS}
        order_cargo = {o.id: o.cargo_type for o in DELIVERY_ORDERS}
        for fn in [greedy_allocate, hungarian_allocate, heuristic_allocate]:
            result = fn(TRUCKS, DELIVERY_ORDERS)
            for asg in result.assignments:
                required = order_cargo[asg.order_id]
                assert required in truck_cargo[asg.truck_id], (
                    f"{result.algorithm}: {asg.truck_id} cannot carry {required} "
                    f"for {asg.order_id}"
                )

    def test_fulfillment_rates_reasonable(self):
        for fn in [greedy_allocate, hungarian_allocate, heuristic_allocate]:
            result = fn(TRUCKS, DELIVERY_ORDERS)
            assert result.metrics["fulfillment_rate"] >= 70.0, \
                f"{result.algorithm}: fulfillment rate too low: {result.metrics['fulfillment_rate']}%"

    def test_hungarian_competitive_total_distance(self):
        greedy_r   = greedy_allocate(TRUCKS, DELIVERY_ORDERS)
        hungarian_r = hungarian_allocate(TRUCKS, DELIVERY_ORDERS)
        heuristic_r = heuristic_allocate(TRUCKS, DELIVERY_ORDERS)
        min_dist = min(
            greedy_r.metrics["total_distance_km"],
            hungarian_r.metrics["total_distance_km"],
            heuristic_r.metrics["total_distance_km"],
        )
        hungarian_dist = hungarian_r.metrics["total_distance_km"]
        assert hungarian_dist <= min_dist * 1.3

    def test_max_deliveries_per_truck_not_exceeded(self):
        truck_max = {t.id: t.max_deliveries for t in TRUCKS}
        from collections import Counter
        for fn in [greedy_allocate, heuristic_allocate]:
            result = fn(TRUCKS, DELIVERY_ORDERS)
            load = Counter(a.truck_id for a in result.assignments)
            for tid, count in load.items():
                assert count <= truck_max[tid], \
                    f"{result.algorithm}: {tid} has {count} deliveries, max is {truck_max[tid]}"

    def test_assignments_have_positive_distance(self):
        for fn in [greedy_allocate, hungarian_allocate, heuristic_allocate]:
            result = fn(TRUCKS, DELIVERY_ORDERS)
            for asg in result.assignments:
                assert asg.distance_km       >= 0
                assert asg.travel_time_hours >= 0

    def test_execution_times_are_fast(self):
        for fn in [greedy_allocate, hungarian_allocate, heuristic_allocate]:
            result = fn(TRUCKS, DELIVERY_ORDERS)
            assert result.execution_time_ms < 500, \
                f"{result.algorithm} took {result.execution_time_ms:.1f}ms"

    def test_unassigned_plus_assigned_equals_total(self):
        for fn in [greedy_allocate, hungarian_allocate, heuristic_allocate, ml_allocate]:
            result = fn(TRUCKS, DELIVERY_ORDERS)
            total = result.metrics["total_orders"]
            assert result.metrics["assigned"] + result.metrics["unassigned"] == total, \
                f"{result.algorithm}: assigned + unassigned != total"

    def test_arrival_within_window_all_algos(self):
        order_map = {o.id: o for o in DELIVERY_ORDERS}
        for fn in [greedy_allocate, hungarian_allocate, heuristic_allocate, ml_allocate]:
            result = fn(TRUCKS, DELIVERY_ORDERS)
            for asg in result.assignments:
                order = order_map[asg.order_id]
                assert asg.arrival_time <= order.window_end + 1e-6, (
                    f"{result.algorithm}: {asg.truck_id}→{asg.order_id} "
                    f"arrives {asg.arrival_time:.3f} after window end {order.window_end}"
                )

    def test_scores_non_negative_all_algos(self):
        for fn in [greedy_allocate, hungarian_allocate, heuristic_allocate, ml_allocate]:
            result = fn(TRUCKS, DELIVERY_ORDERS)
            for asg in result.assignments:
                assert asg.score >= 0, \
                    f"{result.algorithm}: negative score on {asg.order_id}"


# ─── Greedy route-chaining tests ──────────────────────────────────────────────

class TestGreedyRouteChaining:
    def test_second_delivery_costed_from_first_stop_location(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD], max_del=3)
        o1 = make_order("O1", 37.800, -122.410, CargoType.STANDARD, ws=6.0, we=17.0, dur=0.3)
        o2 = make_order("O2", 37.802, -122.411, CargoType.STANDARD, ws=7.0, we=17.0, dur=0.3)
        result = greedy_allocate([truck], [o1, o2])
        assigned = {a.order_id for a in result.assignments}
        assert "O1" in assigned
        assert "O2" in assigned

    def test_truck_time_advances_between_deliveries(self):
        truck = make_truck("TRK1", 37.77, -122.41, [CargoType.STANDARD], max_del=2)
        o1 = make_order("O1", 37.77, -122.41, CargoType.STANDARD, ws=6.0, we=9.0,  dur=1.0)
        o2 = make_order("O2", 37.77, -122.42, CargoType.STANDARD, ws=7.5, we=17.0, dur=0.5)
        result = greedy_allocate([truck], [o1, o2])
        assert len(result.assignments) == 2
