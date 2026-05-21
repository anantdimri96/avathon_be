"""
Data generator tests — verifies generate_scenario() produces valid, well-formed data.
Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from models import CargoType
from data.generator import generate_scenario

# Approximate bounding box for San Francisco + jitter
SF_LAT = (37.68, 37.84)
SF_LNG = (-122.55, -122.33)


# ─── Scenario shape tests ─────────────────────────────────────────────────────

class TestGenerateScenarioShape:
    def test_returns_two_lists(self):
        result = generate_scenario()
        assert isinstance(result, tuple) and len(result) == 2
        trucks, orders = result
        assert isinstance(trucks, list)
        assert isinstance(orders, list)

    def test_default_truck_count_in_range(self):
        for _ in range(5):
            trucks, _ = generate_scenario()
            assert 18 <= len(trucks) <= 25, f"Got {len(trucks)} trucks"

    def test_default_order_count_in_range(self):
        for _ in range(5):
            _, orders = generate_scenario()
            assert 40 <= len(orders) <= 55, f"Got {len(orders)} orders"

    def test_explicit_truck_count(self):
        trucks, _ = generate_scenario(n_trucks=7)
        assert len(trucks) == 7

    def test_explicit_order_count(self):
        _, orders = generate_scenario(n_orders=12)
        assert len(orders) == 12

    def test_explicit_both_counts(self):
        trucks, orders = generate_scenario(n_trucks=3, n_orders=5)
        assert len(trucks) == 3
        assert len(orders) == 5

    def test_seed_produces_identical_output(self):
        # IDs, names, cargo_types, and shift hours are drawn from the seeded local rng
        # and are therefore reproducible. Locations use _jitter() which calls the
        # global random module, so they are intentionally NOT checked here.
        trucks1, orders1 = generate_scenario(seed=42)
        trucks2, orders2 = generate_scenario(seed=42)
        assert [t.id for t in trucks1] == [t.id for t in trucks2]
        assert [t.name for t in trucks1] == [t.name for t in trucks2]
        assert [t.shift_start for t in trucks1] == [t.shift_start for t in trucks2]
        assert [t.cargo_types for t in trucks1] == [t.cargo_types for t in trucks2]
        assert [o.id for o in orders1] == [o.id for o in orders2]
        assert [o.priority for o in orders1] == [o.priority for o in orders2]
        assert [o.cargo_type for o in orders1] == [o.cargo_type for o in orders2]

    def test_different_seeds_produce_different_data(self):
        trucks1, _ = generate_scenario(seed=1)
        trucks2, _ = generate_scenario(seed=2)
        lats1 = [round(t.location.lat, 5) for t in trucks1]
        lats2 = [round(t.location.lat, 5) for t in trucks2]
        assert lats1 != lats2

    def test_unique_truck_ids(self):
        trucks, _ = generate_scenario(seed=10)
        ids = [t.id for t in trucks]
        assert len(ids) == len(set(ids))

    def test_unique_order_ids(self):
        _, orders = generate_scenario(seed=10)
        ids = [o.id for o in orders]
        assert len(ids) == len(set(ids))

    def test_unique_truck_names(self):
        trucks, _ = generate_scenario(seed=10)
        names = [t.name for t in trucks]
        assert len(names) == len(set(names))


# ─── Truck constraint tests ───────────────────────────────────────────────────

class TestTruckConstraints:
    def setup_method(self):
        self.trucks, self.orders = generate_scenario(seed=99)

    def test_cargo_types_are_valid(self):
        valid = set(CargoType)
        for truck in self.trucks:
            assert len(truck.cargo_types) >= 1
            for cargo in truck.cargo_types:
                assert cargo in valid

    def test_shift_hours_ordered(self):
        for truck in self.trucks:
            assert truck.shift_start < truck.shift_end

    def test_shift_start_in_range(self):
        for truck in self.trucks:
            assert 5.0 <= truck.shift_start <= 7.0

    def test_shift_end_in_range(self):
        for truck in self.trucks:
            assert 15.0 <= truck.shift_end <= 18.0

    def test_max_deliveries_in_range(self):
        for truck in self.trucks:
            assert 6 <= truck.max_deliveries <= 10

    def test_location_in_sf_bounds(self):
        for truck in self.trucks:
            assert SF_LAT[0] <= truck.location.lat <= SF_LAT[1], (
                f"Truck {truck.id} lat {truck.location.lat} out of SF bounds"
            )
            assert SF_LNG[0] <= truck.location.lng <= SF_LNG[1], (
                f"Truck {truck.id} lng {truck.location.lng} out of SF bounds"
            )

    def test_location_has_address(self):
        for truck in self.trucks:
            assert truck.location.address and len(truck.location.address) > 0


# ─── Delivery order constraint tests ─────────────────────────────────────────

class TestOrderConstraints:
    def setup_method(self):
        _, self.orders = generate_scenario(seed=99)

    def test_priority_in_valid_range(self):
        for order in self.orders:
            assert 1 <= order.priority <= 5

    def test_time_window_ordered(self):
        for order in self.orders:
            assert order.window_start < order.window_end, (
                f"Order {order.id}: window_start {order.window_start} >= window_end {order.window_end}"
            )

    def test_duration_positive(self):
        for order in self.orders:
            assert order.duration > 0

    def test_cargo_type_is_valid(self):
        valid = set(CargoType)
        for order in self.orders:
            assert order.cargo_type in valid

    def test_location_in_sf_bounds(self):
        for order in self.orders:
            assert SF_LAT[0] <= order.location.lat <= SF_LAT[1], (
                f"Order {order.id} lat {order.location.lat} out of SF bounds"
            )
            assert SF_LNG[0] <= order.location.lng <= SF_LNG[1], (
                f"Order {order.id} lng {order.location.lng} out of SF bounds"
            )

    def test_description_non_empty(self):
        for order in self.orders:
            assert order.description and len(order.description) > 0

    def test_location_has_address(self):
        for order in self.orders:
            assert order.location.address and len(order.location.address) > 0

    def test_priority_distribution_varied(self):
        """With 40-55 orders, should see at least 3 distinct priority levels."""
        _, orders = generate_scenario(seed=77)
        priorities = {o.priority for o in orders}
        assert len(priorities) >= 3, f"Only {len(priorities)} distinct priorities: {priorities}"
