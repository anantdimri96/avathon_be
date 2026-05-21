"""
Delivery Fleet Allocation Engine — FastAPI backend.

On every startup a fresh random SF delivery scenario is generated
(different trucks, depots, and orders each time).
The ML model is loaded from disk if available, otherwise trained
on 250k synthetic samples before the first request is served.
"""
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from models import AllocationResult, AllocateRequest, CompareRequest, DeliveryOrder, RegenerateRequest, Truck
from algorithms import ALGORITHMS
from algorithms.ml_allocator import get_status as ml_get_status, train as ml_train
from data.generator import generate_scenario

app = FastAPI(title="Delivery Fleet Allocation Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Generate a fresh scenario on startup ─────────────────────────────────────
TRUCKS, DELIVERY_ORDERS = generate_scenario()


# ── Seed data endpoints ───────────────────────────────────────────────────────

@app.get("/api/trucks", response_model=List[Truck])
def get_trucks():
    return TRUCKS


@app.get("/api/orders", response_model=List[DeliveryOrder])
def get_orders():
    return DELIVERY_ORDERS


# ── Allocation ────────────────────────────────────────────────────────────────

@app.post("/api/allocate", response_model=AllocationResult)
def allocate(body: AllocateRequest):
    algorithm = body.algorithm.lower()
    if algorithm not in ALGORITHMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm '{algorithm}'. Choose from: {list(ALGORITHMS.keys())}",
        )
    trucks = body.trucks  or TRUCKS
    orders = body.orders  or DELIVERY_ORDERS
    return ALGORITHMS[algorithm](trucks, orders)


@app.post("/api/compare", response_model=List[AllocationResult])
def compare_post(body: CompareRequest):
    trucks = body.trucks or TRUCKS
    orders = body.orders or DELIVERY_ORDERS
    return [fn(trucks, orders) for fn in ALGORITHMS.values()]


@app.get("/api/compare", response_model=List[AllocationResult])
def compare_get():
    return [fn(TRUCKS, DELIVERY_ORDERS) for fn in ALGORITHMS.values()]


# ── ML management ─────────────────────────────────────────────────────────────

@app.get("/api/ml/status")
def ml_status():
    return ml_get_status()


@app.post("/api/ml/train")
def ml_retrain(background_tasks: BackgroundTasks):
    """Kick off a background retrain of the ML model on 250k samples."""
    background_tasks.add_task(ml_train)
    return {"message": "Retraining started — this takes ~60 seconds for 250k samples"}


# ── Scenario regeneration ─────────────────────────────────────────────────────

@app.post("/api/regenerate")
def regenerate_scenario(body: RegenerateRequest):
    """Generate a new random SF delivery scenario, replacing the current one."""
    global TRUCKS, DELIVERY_ORDERS
    TRUCKS, DELIVERY_ORDERS = generate_scenario(
        n_trucks=body.n_trucks,
        n_orders=body.n_orders,
    )
    return {"trucks": len(TRUCKS), "orders": len(DELIVERY_ORDERS)}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "trucks": len(TRUCKS), "orders": len(DELIVERY_ORDERS)}
