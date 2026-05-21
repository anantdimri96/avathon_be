"""
Random delivery scenario generator.

Every time the backend starts it calls generate_scenario() to produce
a fresh set of trucks and delivery orders across San Francisco.
This means the map and comparison table show different data on each run,
demonstrating that the algorithms work on arbitrary inputs.
"""
import random
from models import Truck, DeliveryOrder, Location, CargoType, Priority

# ── Real SF neighborhoods with accurate lat/lng ──────────────────────────────

NEIGHBORHOODS = [
    ("Downtown",        37.7749, -122.4194),
    ("Mission",         37.7599, -122.4148),
    ("SoMa",            37.7785, -122.3948),
    ("Castro",          37.7609, -122.4350),
    ("Richmond",        37.7800, -122.4836),
    ("Outer Sunset",    37.7464, -122.5012),
    ("Inner Sunset",    37.7566, -122.4674),
    ("Marina",          37.8030, -122.4358),
    ("Haight",          37.7693, -122.4481),
    ("Noe Valley",      37.7502, -122.4338),
    ("Potrero Hill",    37.7587, -122.4016),
    ("North Beach",     37.8060, -122.4103),
    ("Pacific Heights", 37.7925, -122.4382),
    ("Tenderloin",      37.7840, -122.4124),
    ("Bayview",         37.7298, -122.3892),
    ("Bernal Heights",  37.7419, -122.4157),
    ("Chinatown",       37.7941, -122.4078),
    ("Fillmore",        37.7840, -122.4314),
    ("Excelsior",       37.7247, -122.4262),
    ("Cow Hollow",      37.7975, -122.4352),
    ("Dogpatch",        37.7601, -122.3890),
    ("Glen Park",       37.7337, -122.4338),
    ("Russian Hill",    37.8011, -122.4175),
    ("Outer Richmond",  37.7793, -122.5040),
    ("Parkside",        37.7397, -122.4831),
]

# SF depot/warehouse names for truck starting locations
DEPOTS = [
    "SoMa Distribution Center",
    "Bayview Logistics Hub",
    "Mission Freight Depot",
    "Potrero Warehouse",
    "Dogpatch Cold Storage",
    "Hunters Point Depot",
    "Outer Sunset Depot",
    "Richmond Distribution",
    "North Beach Depot",
    "Excelsior Warehouse",
]

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry",
    "Iris", "James", "Karen", "Luis", "Maria", "Nathan", "Olivia", "Paul",
    "Quinn", "Rosa", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zach", "Andre", "Beth", "Carlos", "Diana",
]

LAST_NAMES = [
    "Chen", "Martinez", "Singh", "Kim", "Rodriguez", "Lee", "Patel",
    "Thompson", "Nakamura", "Wilson", "Davis", "Gomez", "Johnson", "Brown",
    "Taylor", "Anderson", "Jackson", "White", "Harris", "Martin",
]

# Delivery descriptions keyed by cargo type
ORDER_DESCRIPTIONS = {
    CargoType.REFRIGERATED: [
        "Fresh produce — daily restaurant restock",
        "Pharmaceutical cold chain delivery",
        "Vaccine transport — hospital order",
        "Dairy products — supermarket restock",
        "Frozen seafood — 30 boxes",
        "Cold-pressed juice delivery",
        "Organic grocery order — 20 crates",
        "Blood bank specimens — urgent",
        "Chilled meat — butcher restock",
        "Ice cream distribution — 15 units",
    ],
    CargoType.HEAVY_FREIGHT: [
        "Industrial machinery — 2 pallets",
        "Office furniture — full floor fit-out",
        "Construction steel beams",
        "HVAC units — rooftop installation",
        "Retail appliances — 8 units",
        "Commercial kitchen equipment",
        "Concrete mixer delivery",
        "Generator unit — data center",
        "Steel shelving — warehouse install",
        "Server rack delivery — 4 units",
    ],
    CargoType.STANDARD: [
        "E-commerce parcels — 80 packages",
        "Office supplies — weekly order",
        "Retail clothing stock — 15 boxes",
        "Books and media — bookstore restock",
        "Cleaning supplies — janitorial",
        "Auto parts — mechanic shop",
        "Hardware supplies — contractor",
        "Stationery — corporate order",
        "Sporting goods — gym restock",
        "General merchandise — mixed pallets",
    ],
    CargoType.EXPRESS: [
        "Same-day medical supplies",
        "Priority legal documents",
        "Emergency spare parts",
        "Critical IT components",
        "Urgent pharmaceutical refill",
        "Time-sensitive court filings",
        "Next-flight overnight parcel",
        "Bank security delivery",
        "Press-release packages — embargo",
        "Event materials — tonight's conference",
    ],
    CargoType.FRAGILE: [
        "Museum exhibit — crated sculptures",
        "Electronics retail — 30 units",
        "Luxury glassware — 12 boxes",
        "Fine art gallery shipment",
        "Medical imaging equipment",
        "High-end audio components",
        "Telescope — observatory delivery",
        "Vintage wine — restaurant order",
        "Custom ceramic tiles — 50 boxes",
        "Scientific instruments — university",
    ],
}


def _jitter(lat: float, lng: float, radius: float = 0.008) -> tuple[float, float]:
    """Add small random offset so markers in the same neighborhood don't stack."""
    return (
        lat + random.uniform(-radius, radius),
        lng + random.uniform(-radius, radius),
    )


def generate_scenario(
    n_trucks: int | None = None,
    n_orders: int | None = None,
    seed: int | None = None,
) -> tuple[list[Truck], list[DeliveryOrder]]:
    """
    Generate a fresh random delivery scenario.

    Parameters
    ----------
    n_trucks : int, optional
        How many trucks to create (default: random 18–25).
    n_orders : int, optional
        How many delivery orders to create (default: random 40–55).
    seed : int, optional
        Fix the random seed for reproducibility.
    """
    rng = random.Random(seed)

    n_t = n_trucks or rng.randint(18, 25)
    n_o = n_orders or rng.randint(40, 55)

    cargo_all = list(CargoType)

    # ── Trucks ───────────────────────────────────────────────────────────────
    used_names: set[str] = set()
    trucks: list[Truck] = []

    hoods_pool = rng.sample(NEIGHBORHOODS, min(n_t, len(NEIGHBORHOODS)))
    while len(hoods_pool) < n_t:
        hoods_pool.append(rng.choice(NEIGHBORHOODS))

    for i in range(n_t):
        while True:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            if name not in used_names:
                used_names.add(name)
                break

        hood_name, base_lat, base_lng = hoods_pool[i]
        lat, lng = _jitter(base_lat, base_lng, radius=0.005)

        # 1–3 cargo types this truck can carry
        n_types = rng.randint(1, 3)
        truck_cargo = rng.sample(cargo_all, n_types)

        shift_start = rng.choice([5.0, 5.5, 6.0, 6.5, 7.0])
        shift_end   = rng.choice([15.0, 16.0, 17.0, 18.0])
        max_del     = rng.randint(6, 10)

        depot = rng.choice(DEPOTS)

        trucks.append(Truck(
            id=f"TRK{i+1:02d}",
            name=name,
            location=Location(lat=round(lat, 6), lng=round(lng, 6), address=depot),
            cargo_types=truck_cargo,
            shift_start=shift_start,
            shift_end=shift_end,
            max_deliveries=max_del,
        ))

    # ── Delivery orders ───────────────────────────────────────────────────────
    orders: list[DeliveryOrder] = []

    for j in range(n_o):
        hood_name, base_lat, base_lng = rng.choice(NEIGHBORHOODS)
        lat, lng = _jitter(base_lat, base_lng, radius=0.008)

        cargo = rng.choice(cargo_all)
        description = rng.choice(ORDER_DESCRIPTIONS[cargo])

        priority = rng.choices(
            [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.URGENT, Priority.CRITICAL],
            weights=[10, 25, 30, 20, 15],
        )[0]

        ws = round(rng.uniform(6.0, 12.0), 1)
        we = round(min(ws + rng.uniform(2.0, 6.0), 18.0), 1)
        duration = round(rng.uniform(0.25, 2.0), 2)

        orders.append(DeliveryOrder(
            id=f"ORD{j+1:02d}",
            description=description,
            location=Location(lat=round(lat, 6), lng=round(lng, 6), address=hood_name),
            cargo_type=cargo,
            priority=int(priority),
            window_start=ws,
            window_end=we,
            duration=duration,
        ))

    return trucks, orders
