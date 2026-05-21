from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class CargoType(str, Enum):
    REFRIGERATED  = "REFRIGERATED"
    HEAVY_FREIGHT = "HEAVY_FREIGHT"
    STANDARD      = "STANDARD"
    EXPRESS       = "EXPRESS"
    FRAGILE       = "FRAGILE"


class Priority(int, Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    URGENT   = 4
    CRITICAL = 5


class Location(BaseModel):
    lat: float
    lng: float
    address: str = ""


class Truck(BaseModel):
    id: str
    name: str                    # driver name
    location: Location           # starting depot
    cargo_types: List[CargoType] # what this truck can carry
    shift_start: float = 6.0    # hours from midnight
    shift_end: float   = 18.0   # hours from midnight
    max_deliveries: int = 8


class DeliveryOrder(BaseModel):
    id: str
    description: str = ""
    location: Location           # delivery destination
    cargo_type: CargoType        # required truck capability
    priority: Priority
    window_start: float          # earliest delivery time (hours from midnight)
    window_end: float            # latest delivery time (hours from midnight)
    duration: float              # estimated unloading time in hours


class Assignment(BaseModel):
    truck_id: str
    order_id: str
    travel_time_hours: float
    distance_km: float
    arrival_time: float
    score: float
    explanation: str


class AllocationResult(BaseModel):
    algorithm: str
    assignments: List[Assignment]
    unassigned_order_ids: List[str]
    metrics: dict
    execution_time_ms: float


class CompareRequest(BaseModel):
    trucks:  Optional[List[Truck]]         = None
    orders:  Optional[List[DeliveryOrder]] = None


class AllocateRequest(BaseModel):
    algorithm: str  # "greedy" | "hungarian" | "heuristic" | "ml"
    trucks:  Optional[List[Truck]]         = None
    orders:  Optional[List[DeliveryOrder]] = None


class RegenerateRequest(BaseModel):
    n_trucks: Optional[int] = Field(None, ge=5, le=30)
    n_orders: Optional[int] = Field(None, ge=10, le=80)
