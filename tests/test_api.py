"""
FastAPI endpoint tests.
Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestTrucksEndpoint:
    def test_get_trucks(self):
        resp = client.get("/api/trucks")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_truck_has_required_fields(self):
        resp = client.get("/api/trucks")
        truck = resp.json()[0]
        for field in ["id", "name", "location", "cargo_types", "shift_start", "shift_end"]:
            assert field in truck, f"Missing field: {field}"

    def test_truck_location_has_lat_lng(self):
        resp = client.get("/api/trucks")
        loc = resp.json()[0]["location"]
        assert "lat" in loc
        assert "lng" in loc


class TestOrdersEndpoint:
    def test_get_orders(self):
        resp = client.get("/api/orders")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_order_has_required_fields(self):
        resp = client.get("/api/orders")
        order = resp.json()[0]
        for field in ["id", "location", "cargo_type", "priority", "window_start", "window_end", "duration"]:
            assert field in order


class TestAllocateEndpoint:
    def test_greedy_allocate(self):
        resp = client.post("/api/allocate", json={"algorithm": "greedy"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["algorithm"] == "greedy"
        assert "assignments" in data
        assert "metrics" in data

    def test_hungarian_allocate(self):
        resp = client.post("/api/allocate", json={"algorithm": "hungarian"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["algorithm"] == "hungarian"

    def test_heuristic_allocate(self):
        resp = client.post("/api/allocate", json={"algorithm": "heuristic"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["algorithm"] == "heuristic"

    def test_invalid_algorithm_returns_400(self):
        resp = client.post("/api/allocate", json={"algorithm": "magic"})
        assert resp.status_code == 400

    def test_allocate_with_custom_data(self):
        custom = {
            "algorithm": "greedy",
            "trucks": [{
                "id": "TX", "name": "Test Truck",
                "location": {"lat": 37.77, "lng": -122.41, "address": "Test Depot"},
                "cargo_types": ["STANDARD"],
                "shift_start": 8.0, "shift_end": 17.0, "max_deliveries": 3,
            }],
            "orders": [{
                "id": "OX", "description": "Test delivery",
                "location": {"lat": 37.77, "lng": -122.41, "address": "Test Stop"},
                "cargo_type": "STANDARD",
                "priority": 3,
                "window_start": 9.0, "window_end": 15.0, "duration": 1.0,
            }],
        }
        resp = client.post("/api/allocate", json=custom)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["assignments"]) == 1
        assert data["assignments"][0]["truck_id"] == "TX"

    def test_metrics_in_response(self):
        resp = client.post("/api/allocate", json={"algorithm": "greedy"})
        metrics = resp.json()["metrics"]
        for key in ["assigned", "total_orders", "fulfillment_rate", "total_distance_km"]:
            assert key in metrics

    def test_execution_time_present(self):
        resp = client.post("/api/allocate", json={"algorithm": "greedy"})
        assert "execution_time_ms" in resp.json()


class TestCompareEndpoint:
    def test_compare_returns_all_algorithms(self):
        resp = client.get("/api/compare")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 4
        algorithms = {r["algorithm"] for r in data}
        assert algorithms == {"greedy", "hungarian", "heuristic", "ml"}

    def test_compare_post_with_custom_data(self):
        resp = client.post("/api/compare", json={})
        assert resp.status_code == 200
        assert len(resp.json()) == 4

    def test_compare_results_have_metrics(self):
        resp = client.get("/api/compare")
        for result in resp.json():
            assert "metrics" in result
            assert "assignments" in result
