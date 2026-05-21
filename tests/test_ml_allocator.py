"""
ML allocator and model status tests.
Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from models import Truck, DeliveryOrder, Location, CargoType, Priority
from algorithms.ml_allocator import ml_allocate, get_status
from data.seed_data import TRUCKS, DELIVERY_ORDERS


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_truck(id, lat, lng, cargo_types, shift_start=8.0, shift_end=17.0, max_del=5):
    return Truck(
        id=id, name=f"Truck {id}",
        location=Location(lat=lat, lng=lng),
        cargo_types=cargo_types,
        shift_start=shift_start, shift_end=shift_end, max_deliveries=max_del,
    )


def make_order(id, lat, lng, cargo, priority=3, ws=9.0, we=15.0, dur=1.0):
    return DeliveryOrder(
        id=id, description=f"Order {id}",
        location=Location(lat=lat, lng=lng),
        cargo_type=cargo,
        priority=priority,
        window_start=ws, window_end=we,
        duration=dur,
    )


# ─── ML Allocation tests ──────────────────────────────────────────────────────

class TestMLAllocate:
    def test_assigns_when_feasible(self):
        truck = make_truck("T1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1", 37.77, -122.41, CargoType.STANDARD)
        result = ml_allocate([truck], [order])
        assert len(result.assignments) == 1
        assert result.assignments[0].order_id == "O1"
        assert result.assignments[0].truck_id == "T1"

    def test_cargo_constraint_respected(self):
        truck = make_truck("T1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1", 37.77, -122.41, CargoType.REFRIGERATED)
        result = ml_allocate([truck], [order])
        assert len(result.assignments) == 0
        assert "O1" in result.unassigned_order_ids

    def test_no_duplicate_order_assignments(self):
        result = ml_allocate(TRUCKS, DELIVERY_ORDERS)
        ids = [a.order_id for a in result.assignments]
        assert len(ids) == len(set(ids))

    def test_max_deliveries_cap(self):
        truck = make_truck("T1", 37.77, -122.41, [CargoType.STANDARD], max_del=2)
        orders = [
            make_order(f"O{i}", 37.77, -122.41, CargoType.STANDARD, ws=8.0, we=17.0, dur=0.1)
            for i in range(5)
        ]
        result = ml_allocate([truck], orders)
        assert len(result.assignments) <= 2

    def test_explanation_contains_ml_score(self):
        truck = make_truck("T1", 37.77, -122.41, [CargoType.STANDARD])
        order = make_order("O1", 37.77, -122.41, CargoType.STANDARD)
        result = ml_allocate([truck], [order])
        assert "ML score" in result.assignments[0].explanation

    def test_algorithm_label(self):
        result = ml_allocate(TRUCKS, DELIVERY_ORDERS)
        assert result.algorithm == "ml"

    def test_all_assignments_respect_cargo_constraints(self):
        truck_cargo = {t.id: set(t.cargo_types) for t in TRUCKS}
        order_cargo = {o.id: o.cargo_type for o in DELIVERY_ORDERS}
        result = ml_allocate(TRUCKS, DELIVERY_ORDERS)
        for asg in result.assignments:
            assert order_cargo[asg.order_id] in truck_cargo[asg.truck_id], (
                f"ML: {asg.truck_id} cannot carry cargo for {asg.order_id}"
            )

    def test_arrival_within_window(self):
        order_map = {o.id: o for o in DELIVERY_ORDERS}
        result = ml_allocate(TRUCKS, DELIVERY_ORDERS)
        for asg in result.assignments:
            order = order_map[asg.order_id]
            assert asg.arrival_time <= order.window_end + 1e-6, (
                f"ML: arrived {asg.arrival_time:.3f} but window closes {order.window_end}"
            )

    def test_metrics_present(self):
        result = ml_allocate(TRUCKS, DELIVERY_ORDERS)
        for key in ["assigned", "total_orders", "fulfillment_rate", "total_distance_km"]:
            assert key in result.metrics

    def test_fulfillment_rate_reasonable(self):
        result = ml_allocate(TRUCKS, DELIVERY_ORDERS)
        assert result.metrics["fulfillment_rate"] >= 50.0

    def test_empty_inputs(self):
        result = ml_allocate([], [])
        assert result.assignments == []
        assert result.unassigned_order_ids == []

    def test_execution_time_present(self):
        result = ml_allocate(TRUCKS, DELIVERY_ORDERS)
        assert result.execution_time_ms >= 0

    def test_route_chaining_updates_location(self):
        """
        After assigning the first order, the truck's cost for the second order
        should be computed from the first order's location, not home base.
        """
        truck = make_truck("T1", 37.77, -122.41, [CargoType.STANDARD], max_del=3)
        o1 = make_order("O1", 37.771, -122.410, CargoType.STANDARD, ws=8.0, we=17.0, dur=0.5)
        o3 = make_order("O3", 37.772, -122.411, CargoType.STANDARD, ws=9.0, we=17.0, dur=0.5)
        result = ml_allocate([truck], [o1, o3])
        assigned_ids = {a.order_id for a in result.assignments}
        assert "O1" in assigned_ids
        assert "O3" in assigned_ids

    def test_priority_order_respected(self):
        """Critical order should be assigned before Low when only one slot available."""
        truck = make_truck("T1", 37.77, -122.41, [CargoType.STANDARD], max_del=1)
        order_high = make_order("O_HIGH", 37.77, -122.41, CargoType.STANDARD, priority=Priority.CRITICAL)
        order_low  = make_order("O_LOW",  37.77, -122.42, CargoType.STANDARD, priority=Priority.LOW)
        result = ml_allocate([truck], [order_high, order_low])
        assigned_ids = [a.order_id for a in result.assignments]
        assert "O_HIGH" in assigned_ids
        assert "O_LOW" not in assigned_ids


# ─── ML Model status tests ────────────────────────────────────────────────────

class TestMLStatus:
    def test_has_required_keys(self):
        status = get_status()
        for key in ["trained", "n_training_samples", "n_scenarios", "r2_score",
                    "trained_at", "feature_importances", "feature_labels"]:
            assert key in status, f"Missing key: {key}"

    def test_trained_flag(self):
        assert get_status()["trained"] is True

    def test_r2_score_valid(self):
        r2 = get_status()["r2_score"]
        assert r2 is not None
        assert 0.0 <= r2 <= 1.0

    def test_feature_importances_sum_to_one(self):
        imps = get_status()["feature_importances"]
        total = sum(imps.values())
        assert abs(total - 1.0) < 0.01, f"Importances sum to {total}, expected ~1.0"

    def test_n_training_samples_large(self):
        assert get_status()["n_training_samples"] > 10_000

    def test_feature_importances_has_all_nine_features(self):
        expected = {
            "distance_km", "travel_time_h", "cargo_match", "window_slack_h",
            "wait_time_h", "priority", "duration_h", "truck_hours_remaining", "load_ratio",
        }
        assert set(get_status()["feature_importances"].keys()) == expected

    def test_cargo_match_is_dominant_feature(self):
        """Model should learn that cargo match is the most important predictor."""
        imps = get_status()["feature_importances"]
        top_feature = max(imps, key=imps.get)
        assert top_feature == "cargo_match", (
            f"Expected cargo_match to dominate, got {top_feature} "
            f"(importances: {dict(sorted(imps.items(), key=lambda x: -x[1]))})"
        )

    def test_feature_labels_match_features(self):
        status = get_status()
        assert set(status["feature_labels"].keys()) == set(status["feature_importances"].keys())
