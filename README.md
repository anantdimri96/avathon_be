# Backend — Delivery Fleet Allocation Engine

FastAPI service implementing four allocation algorithms for assigning delivery trucks to orders across San Francisco.

## Working Application

https://github.com/user-attachments/assets/c4d1f7ad-5b88-48c7-9f0b-6d9f996f9ead


## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The ML model trains automatically on first startup (~60 seconds). Subsequent starts load the cached model from `data/ml_model.pkl`.

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Running Tests

```bash
pytest tests/ -v
```

105 tests across four files covering algorithms, API endpoints, ML model, and data generation.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/trucks` | List current trucks |
| GET | `/api/orders` | List current delivery orders |
| POST | `/api/allocate` | Run a single algorithm (`{"algorithm": "greedy"}`) |
| GET | `/api/compare` | Run all 4 algorithms on the current scenario |
| POST | `/api/compare` | Same, with optional custom trucks/orders in body |
| POST | `/api/regenerate` | Replace scenario with a new random one (`{"n_trucks": 20, "n_orders": 45}`) |
| GET | `/api/ml/status` | ML model stats + feature importances |
| POST | `/api/ml/train` | Retrain the ML model in the background |
| GET | `/health` | Health check |

## Structure

```
backend/
├── main.py              # FastAPI app, all routes
├── models.py            # Pydantic models: Truck, DeliveryOrder, Assignment, AllocationResult
├── requirements.txt
├── algorithms/
│   ├── base.py          # Haversine distance, can_serve(), assignment_cost(), compute_metrics()
│   ├── greedy.py        # Sequential priority-order assignment with route chaining
│   ├── hungarian.py     # Batch optimal via scipy.optimize.linear_sum_assignment
│   ├── heuristic.py     # k-means geographic clustering + load-balance soft constraint
│   └── ml_allocator.py  # RandomForest trained on 250k synthetic dispatch scenarios
├── data/
│   ├── generator.py     # Generates random SF scenarios (18–25 trucks, 40–55 orders)
│   ├── seed_data.py     # Fixed 12-truck / 20-order dataset used by tests
│   └── ml_model.pkl     # Trained model (auto-generated, not committed to git)
└── tests/
    ├── test_algorithms.py   # 53 unit + property tests for all 4 algorithms
    ├── test_api.py          # 12 endpoint integration tests
    ├── test_ml_allocator.py # 22 ML model tests
    └── test_generator.py    # 20 data generator tests
```

## Algorithms

| Name | File | Approach |
|------|------|----------|
| Greedy | `greedy.py` | Sort by priority, assign cheapest feasible truck per order, chain routes |
| Hungarian | `hungarian.py` | Build full cost matrix, solve with Kuhn-Munkres (O(n³)) |
| Heuristic | `heuristic.py` | Cluster orders geographically, route trucks through clusters, balance load |
| ML | `ml_allocator.py` | RandomForest quality predictor trained on synthetic historical assignments |

## Hard Constraints (enforced by `base.can_serve`)

- Cargo type match (truck must support the order's cargo type)
- Arrival before time window closes
- Delivery finishes before truck's shift ends
- Truck has not exceeded `max_deliveries`
