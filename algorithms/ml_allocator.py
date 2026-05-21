"""
ML-Based Allocation — learns from synthetic delivery history.

How it works
------------
1. TRAINING  (runs once on startup, cached to disk)
   - Generates ~1650 random dispatch scenarios across SF.
   - For each scenario, enumerates every feasible (truck, order) pair
     and labels it with a quality score (0-1) that captures:
       * Distance efficiency   (35% weight)
       * Time-window utilization (20%)
       * Load balance          (20%)
       * Order priority        (25%)
   - Trains a RandomForestRegressor on 9 engineered features.

2. INFERENCE  (at allocation time)
   - Processes orders in priority order (same as greedy).
   - For each order, batch-scores all feasible trucks with the model.
   - Assigns the truck with the highest predicted quality.
   - Updates truck state (location, time, load) after each assignment.
"""
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from algorithms.base import (
    INFEASIBLE_COST,
    can_serve,
    compute_metrics,
    haversine_km,
    travel_time_hours,
)
from models import Assignment, AllocationResult, Location, Priority, DeliveryOrder, CargoType, Truck

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml_model.pkl")

FEATURE_NAMES = [
    "distance_km",
    "travel_time_h",
    "cargo_match",
    "window_slack_h",
    "wait_time_h",
    "priority",
    "duration_h",
    "truck_hours_remaining",
    "load_ratio",
]

FEATURE_LABELS = {
    "distance_km":            "Distance to delivery",
    "travel_time_h":          "Travel time",
    "cargo_match":            "Cargo type match",
    "window_slack_h":         "Time-window buffer",
    "wait_time_h":            "Early-arrival wait",
    "priority":               "Order priority",
    "duration_h":             "Unloading duration",
    "truck_hours_remaining":  "Truck hours remaining",
    "load_ratio":             "Current load ratio",
}

# Singleton model state
_state: Dict = {
    "model":               None,
    "trained":             False,
    "n_samples":           0,
    "n_scenarios":         0,
    "r2_score":            None,
    "trained_at":          None,
    "feature_importances": {},
}


# ── Feature extraction ────────────────────────────────────────────────────────

def _features(
    truck: Truck,
    order: DeliveryOrder,
    current_time: float,
    current_loc: Location,
    load_ratio: float,
) -> List[float]:
    dist      = haversine_km(current_loc, order.location)
    tt        = travel_time_hours(current_loc, order.location)
    arrival   = current_time + tt
    cargo_match   = 1.0 if order.cargo_type in truck.cargo_types else 0.0
    window_slack  = max(0.0, order.window_end - max(arrival, order.window_start))
    wait_time     = max(0.0, order.window_start - arrival)
    hours_remaining = max(0.0, truck.shift_end - max(current_time, arrival))
    return [
        dist,
        tt,
        cargo_match,
        window_slack,
        wait_time,
        float(order.priority),
        order.duration,
        hours_remaining,
        load_ratio,
    ]


def _quality_label(
    truck: Truck,
    order: DeliveryOrder,
    current_time: float,
    current_loc: Location,
    load_ratio: float,
) -> float:
    """Ground-truth quality score used as training label (0 = infeasible, 1 = perfect)."""
    if order.cargo_type not in truck.cargo_types:
        return 0.0
    dist    = haversine_km(current_loc, order.location)
    tt      = travel_time_hours(current_loc, order.location)
    arrival = current_time + tt
    if arrival > order.window_end:
        return 0.0
    if max(arrival, order.window_start) + order.duration > truck.shift_end:
        return 0.0

    dist_score     = max(0.0, 1.0 - dist / 20.0)
    slack          = order.window_end - max(arrival, order.window_start)
    window_size    = max(order.window_end - order.window_start, 1.0)
    time_score     = min(slack / window_size, 1.0)
    load_score     = max(0.0, 1.0 - load_ratio)
    priority_score = order.priority / 5.0

    return 0.35 * dist_score + 0.20 * time_score + 0.20 * load_score + 0.25 * priority_score


# ── Training data generation ─────────────────────────────────────────────────

def _generate_training_data(n_scenarios: int = 1650) -> Tuple[np.ndarray, np.ndarray]:
    """Build the training corpus (~250k labeled (truck, order) pairs)."""
    import random
    from data.generator import generate_scenario

    SF_LAT = (37.695, 37.825)
    SF_LNG = (-122.525, -122.355)
    cargo_all  = list(CargoType)
    priorities = [1, 2, 3, 4, 5]
    rng = random.Random()

    X_rows, y_rows = [], []

    live_trucks, live_orders = generate_scenario()
    base_scenarios: List[Tuple[List[Truck], List[DeliveryOrder]]] = [
        (live_trucks, live_orders)
    ]

    for _ in range(n_scenarios):
        n_t = rng.randint(5, 14)
        n_o = rng.randint(8, 24)

        trucks = [
            Truck(
                id=f"T{i}", name=f"Driver{i}",
                location=Location(lat=rng.uniform(*SF_LAT), lng=rng.uniform(*SF_LNG)),
                cargo_types=rng.sample(cargo_all, rng.randint(1, 3)),
                shift_start=rng.choice([5.0, 6.0, 7.0, 8.0]),
                shift_end=rng.choice([15.0, 16.0, 17.0, 18.0]),
                max_deliveries=rng.randint(5, 10),
            )
            for i in range(n_t)
        ]
        orders = []
        for j in range(n_o):
            ws = rng.uniform(6.0, 13.0)
            we = min(ws + rng.uniform(1.5, 6.0), 18.5)
            orders.append(DeliveryOrder(
                id=f"O{j}", description="",
                location=Location(lat=rng.uniform(*SF_LAT), lng=rng.uniform(*SF_LNG)),
                cargo_type=rng.choice(cargo_all),
                priority=rng.choice(priorities),
                window_start=ws, window_end=we,
                duration=rng.uniform(0.25, 2.0),
            ))
        base_scenarios.append((trucks, orders))

    for trucks, orders in base_scenarios:
        truck_state = {
            t.id: {"loc": t.location, "time": t.shift_start, "count": 0}
            for t in trucks
        }

        for order in sorted(orders, key=lambda o: (-o.priority, o.window_end)):
            for truck in trucks:
                st = truck_state[truck.id]
                load_ratio = st["count"] / max(truck.max_deliveries, 1)
                feats = _features(truck, order, st["time"], st["loc"], load_ratio)
                label = _quality_label(truck, order, st["time"], st["loc"], load_ratio)
                X_rows.append(feats)
                y_rows.append(label)

            best_q, best_tid = -1.0, None
            for truck in trucks:
                st = truck_state[truck.id]
                if st["count"] >= truck.max_deliveries:
                    continue
                load_ratio = st["count"] / max(truck.max_deliveries, 1)
                q = _quality_label(truck, order, st["time"], st["loc"], load_ratio)
                if q > best_q:
                    best_q, best_tid = q, truck.id

            if best_tid and best_q > 0.0:
                truck = next(t for t in trucks if t.id == best_tid)
                st = truck_state[best_tid]
                tt = travel_time_hours(st["loc"], order.location)
                arrival = st["time"] + tt
                st["loc"]   = order.location
                st["time"]  = max(arrival, order.window_start) + order.duration
                st["count"] += 1

    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=float)


# ── Model training ────────────────────────────────────────────────────────────

def train(n_scenarios: int = 1650) -> None:
    """Train (or retrain) the model on ~250k samples. Blocks until complete."""
    X, y = _generate_training_data(n_scenarios)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    r2 = float(model.score(X_test, y_test))

    importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model, "importances": importances,
            "r2": r2, "n_samples": len(X), "n_scenarios": n_scenarios + 1,
        }, f)

    _state["model"]               = model
    _state["trained"]             = True
    _state["n_samples"]           = len(X)
    _state["n_scenarios"]         = n_scenarios + 1
    _state["r2_score"]            = r2
    _state["trained_at"]          = datetime.utcnow().isoformat()
    _state["feature_importances"] = importances


def _load_or_train() -> None:
    if _state["trained"]:
        return
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            saved = pickle.load(f)
        _state["model"]               = saved["model"]
        _state["trained"]             = True
        _state["n_samples"]           = saved.get("n_samples", 0)
        _state["r2_score"]            = saved.get("r2", None)
        _state["feature_importances"] = saved.get("importances", {})
        _state["trained_at"]          = "loaded from disk"
        _state["n_scenarios"]         = saved.get("n_scenarios", 500)
    else:
        train()


def get_status() -> dict:
    _load_or_train()
    return {
        "trained":               _state["trained"],
        "n_training_samples":    _state["n_samples"],
        "n_scenarios":           _state["n_scenarios"],
        "r2_score":              round(_state["r2_score"], 4) if _state["r2_score"] else None,
        "trained_at":            _state["trained_at"],
        "feature_importances":   _state["feature_importances"],
        "feature_labels":        FEATURE_LABELS,
    }


# ── Allocation ────────────────────────────────────────────────────────────────

def ml_allocate(
    technicians: List[Truck],
    requests: List[DeliveryOrder],
) -> AllocationResult:
    trucks = technicians
    orders = requests
    _load_or_train()

    start_ms = time.monotonic()
    model    = _state["model"]

    truck_state = {
        t.id: {"loc": t.location, "time": t.shift_start, "count": 0}
        for t in trucks
    }
    truck_map = {t.id: t for t in trucks}

    sorted_orders = sorted(orders, key=lambda o: (-o.priority, o.window_end))

    assignments: List[Assignment] = []
    assigned_ids: set = set()

    for order in sorted_orders:
        candidates = []
        for truck in trucks:
            st = truck_state[truck.id]
            if st["count"] >= truck.max_deliveries:
                continue
            feasible, _ = can_serve(truck, order, st["time"], st["loc"])
            if not feasible:
                continue
            load_ratio = st["count"] / max(truck.max_deliveries, 1)
            feats = _features(truck, order, st["time"], st["loc"], load_ratio)
            candidates.append((truck.id, feats))

        if not candidates:
            continue

        X      = np.array([f for _, f in candidates])
        scores = model.predict(X)

        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score <= 0:
            continue

        best_truck_id = candidates[best_idx][0]
        truck = truck_map[best_truck_id]
        st    = truck_state[best_truck_id]

        dist          = haversine_km(st["loc"], order.location)
        tt            = travel_time_hours(st["loc"], order.location)
        arrival       = st["time"] + tt
        service_start = max(arrival, order.window_start)

        feat_vals = dict(zip(FEATURE_NAMES, candidates[best_idx][1]))
        imps  = _state["feature_importances"]
        top3  = sorted(imps.items(), key=lambda kv: -kv[1])[:3]
        factors = ", ".join(
            f"{FEATURE_LABELS.get(k, k)}={feat_vals.get(k, 0):.2f}"
            for k, _ in top3
        )
        explanation = f"ML score {best_score:.3f} — top factors: {factors}"

        assignments.append(Assignment(
            truck_id=best_truck_id,
            order_id=order.id,
            travel_time_hours=round(tt, 3),
            distance_km=round(dist, 2),
            arrival_time=round(arrival, 3),
            score=round(best_score, 3),
            explanation=explanation,
        ))
        assigned_ids.add(order.id)

        st["loc"]   = order.location
        st["time"]  = service_start + order.duration
        st["count"] += 1

    unassigned = [o.id for o in orders if o.id not in assigned_ids]
    elapsed_ms = (time.monotonic() - start_ms) * 1000
    metrics    = compute_metrics(
        assignments, [o.id for o in orders], [t.id for t in trucks]
    )

    return AllocationResult(
        algorithm="ml",
        assignments=assignments,
        unassigned_order_ids=unassigned,
        metrics=metrics,
        execution_time_ms=round(elapsed_ms, 2),
    )
