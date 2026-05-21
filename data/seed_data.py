"""
Seed data: 12 delivery trucks and 20 delivery orders across San Francisco.
Used by algorithm unit tests for deterministic, repeatable results.
"""
from models import Truck, DeliveryOrder, Location, CargoType, Priority

TRUCKS = [
    Truck(
        id="TRK01", name="Alice Chen",
        location=Location(lat=37.7749, lng=-122.4194, address="Downtown Depot"),
        cargo_types=[CargoType.STANDARD, CargoType.EXPRESS],
        shift_start=6.0, shift_end=16.0, max_deliveries=8,
    ),
    Truck(
        id="TRK02", name="Bob Martinez",
        location=Location(lat=37.7599, lng=-122.4148, address="Mission Depot"),
        cargo_types=[CargoType.REFRIGERATED, CargoType.STANDARD],
        shift_start=5.0, shift_end=15.0, max_deliveries=7,
    ),
    Truck(
        id="TRK03", name="Carol Singh",
        location=Location(lat=37.8030, lng=-122.4358, address="Marina Depot"),
        cargo_types=[CargoType.HEAVY_FREIGHT, CargoType.STANDARD],
        shift_start=6.0, shift_end=18.0, max_deliveries=6,
    ),
    Truck(
        id="TRK04", name="David Kim",
        location=Location(lat=37.7785, lng=-122.3948, address="SoMa Depot"),
        cargo_types=[CargoType.FRAGILE, CargoType.EXPRESS],
        shift_start=7.0, shift_end=17.0, max_deliveries=8,
    ),
    Truck(
        id="TRK05", name="Eva Rodriguez",
        location=Location(lat=37.7800, lng=-122.4836, address="Richmond Depot"),
        cargo_types=[CargoType.REFRIGERATED, CargoType.FRAGILE],
        shift_start=6.0, shift_end=16.0, max_deliveries=7,
    ),
    Truck(
        id="TRK06", name="Frank Lee",
        location=Location(lat=37.7566, lng=-122.4896, address="Sunset Depot"),
        cargo_types=[CargoType.HEAVY_FREIGHT, CargoType.STANDARD],
        shift_start=5.0, shift_end=15.0, max_deliveries=6,
    ),
    Truck(
        id="TRK07", name="Grace Patel",
        location=Location(lat=37.7925, lng=-122.4382, address="Pacific Heights Depot"),
        cargo_types=[CargoType.EXPRESS, CargoType.FRAGILE],
        shift_start=8.0, shift_end=18.0, max_deliveries=9,
    ),
    Truck(
        id="TRK08", name="Henry Thompson",
        location=Location(lat=37.7693, lng=-122.4481, address="Haight Depot"),
        cargo_types=[CargoType.STANDARD, CargoType.HEAVY_FREIGHT],
        shift_start=6.0, shift_end=16.0, max_deliveries=7,
    ),
    Truck(
        id="TRK09", name="Iris Nakamura",
        location=Location(lat=37.7419, lng=-122.4157, address="Bernal Heights Depot"),
        cargo_types=[CargoType.REFRIGERATED, CargoType.EXPRESS],
        shift_start=6.0, shift_end=16.0, max_deliveries=8,
    ),
    Truck(
        id="TRK10", name="James Wilson",
        location=Location(lat=37.7298, lng=-122.3892, address="Bayview Depot"),
        cargo_types=[CargoType.HEAVY_FREIGHT, CargoType.STANDARD],
        shift_start=5.0, shift_end=15.0, max_deliveries=6,
    ),
    Truck(
        id="TRK11", name="Karen Davis",
        location=Location(lat=37.8060, lng=-122.4103, address="North Beach Depot"),
        cargo_types=[CargoType.FRAGILE, CargoType.STANDARD],
        shift_start=6.0, shift_end=16.0, max_deliveries=8,
    ),
    Truck(
        id="TRK12", name="Luis Gomez",
        location=Location(lat=37.7609, lng=-122.4350, address="Castro Depot"),
        cargo_types=[CargoType.REFRIGERATED, CargoType.HEAVY_FREIGHT, CargoType.STANDARD],
        shift_start=6.0, shift_end=18.0, max_deliveries=9,
    ),
]

DELIVERY_ORDERS = [
    DeliveryOrder(
        id="ORD01", description="Fresh produce — 40 crates",
        location=Location(lat=37.7840, lng=-122.4124, address="Tenderloin"),
        cargo_type=CargoType.REFRIGERATED, priority=Priority.HIGH,
        window_start=7.0, window_end=11.0, duration=0.75,
    ),
    DeliveryOrder(
        id="ORD02", description="Emergency pharmaceutical delivery",
        location=Location(lat=37.7587, lng=-122.4016, address="Potrero Hill"),
        cargo_type=CargoType.REFRIGERATED, priority=Priority.CRITICAL,
        window_start=6.0, window_end=9.0, duration=0.5,
    ),
    DeliveryOrder(
        id="ORD03", description="Industrial HVAC units — 2 pallets",
        location=Location(lat=37.7975, lng=-122.4352, address="Cow Hollow"),
        cargo_type=CargoType.HEAVY_FREIGHT, priority=Priority.MEDIUM,
        window_start=8.0, window_end=14.0, duration=1.5,
    ),
    DeliveryOrder(
        id="ORD04", description="Office furniture — full floor fit-out",
        location=Location(lat=37.7247, lng=-122.4262, address="Excelsior"),
        cargo_type=CargoType.HEAVY_FREIGHT, priority=Priority.LOW,
        window_start=9.0, window_end=17.0, duration=2.0,
    ),
    DeliveryOrder(
        id="ORD05", description="Same-day courier — legal documents",
        location=Location(lat=37.7941, lng=-122.4078, address="Chinatown"),
        cargo_type=CargoType.EXPRESS, priority=Priority.URGENT,
        window_start=7.0, window_end=10.0, duration=0.25,
    ),
    DeliveryOrder(
        id="ORD06", description="Restaurant refrigerated goods",
        location=Location(lat=37.7502, lng=-122.4338, address="Noe Valley"),
        cargo_type=CargoType.REFRIGERATED, priority=Priority.HIGH,
        window_start=6.0, window_end=10.0, duration=0.5,
    ),
    DeliveryOrder(
        id="ORD07", description="E-commerce parcels — 80 packages",
        location=Location(lat=37.7464, lng=-122.5012, address="Outer Sunset"),
        cargo_type=CargoType.STANDARD, priority=Priority.MEDIUM,
        window_start=9.0, window_end=16.0, duration=1.0,
    ),
    DeliveryOrder(
        id="ORD08", description="Museum exhibit — crated sculptures",
        location=Location(lat=37.7840, lng=-122.4314, address="Fillmore"),
        cargo_type=CargoType.FRAGILE, priority=Priority.LOW,
        window_start=8.0, window_end=17.0, duration=2.0,
    ),
    DeliveryOrder(
        id="ORD09", description="Electronics retail — 30 units",
        location=Location(lat=37.7609, lng=-122.4350, address="Castro"),
        cargo_type=CargoType.FRAGILE, priority=Priority.MEDIUM,
        window_start=10.0, window_end=15.0, duration=0.75,
    ),
    DeliveryOrder(
        id="ORD10", description="Construction steel beams",
        location=Location(lat=37.7298, lng=-122.3892, address="Bayview"),
        cargo_type=CargoType.HEAVY_FREIGHT, priority=Priority.URGENT,
        window_start=6.0, window_end=10.0, duration=1.5,
    ),
    DeliveryOrder(
        id="ORD11", description="Grocery chain weekly restock",
        location=Location(lat=37.8030, lng=-122.4358, address="Marina"),
        cargo_type=CargoType.REFRIGERATED, priority=Priority.HIGH,
        window_start=5.0, window_end=9.0, duration=1.0,
    ),
    DeliveryOrder(
        id="ORD12", description="Luxury glassware — 12 boxes",
        location=Location(lat=37.7800, lng=-122.4836, address="Richmond"),
        cargo_type=CargoType.FRAGILE, priority=Priority.MEDIUM,
        window_start=10.0, window_end=16.0, duration=0.5,
    ),
    DeliveryOrder(
        id="ORD13", description="IT equipment — 20 servers",
        location=Location(lat=37.7785, lng=-122.3948, address="SoMa"),
        cargo_type=CargoType.FRAGILE, priority=Priority.LOW,
        window_start=9.0, window_end=17.0, duration=1.5,
    ),
    DeliveryOrder(
        id="ORD14", description="Priority office supplies — same day",
        location=Location(lat=37.7749, lng=-122.4194, address="Downtown"),
        cargo_type=CargoType.EXPRESS, priority=Priority.HIGH,
        window_start=8.0, window_end=12.0, duration=0.25,
    ),
    DeliveryOrder(
        id="ORD15", description="Cold chain — vaccine delivery",
        location=Location(lat=37.7599, lng=-122.4148, address="Mission"),
        cargo_type=CargoType.REFRIGERATED, priority=Priority.CRITICAL,
        window_start=6.0, window_end=9.0, duration=0.5,
    ),
    DeliveryOrder(
        id="ORD16", description="Retail appliances — 8 units",
        location=Location(lat=37.7925, lng=-122.4382, address="Pacific Heights"),
        cargo_type=CargoType.HEAVY_FREIGHT, priority=Priority.MEDIUM,
        window_start=10.0, window_end=16.0, duration=1.0,
    ),
    DeliveryOrder(
        id="ORD17", description="Emergency medical equipment",
        location=Location(lat=37.8060, lng=-122.4103, address="North Beach"),
        cargo_type=CargoType.EXPRESS, priority=Priority.CRITICAL,
        window_start=6.0, window_end=9.0, duration=0.5,
    ),
    DeliveryOrder(
        id="ORD18", description="Fine art gallery shipment",
        location=Location(lat=37.7693, lng=-122.4481, address="Haight"),
        cargo_type=CargoType.FRAGILE, priority=Priority.URGENT,
        window_start=9.0, window_end=14.0, duration=1.0,
    ),
    DeliveryOrder(
        id="ORD19", description="Supermarket ambient goods",
        location=Location(lat=37.7419, lng=-122.4157, address="Bernal Heights"),
        cargo_type=CargoType.STANDARD, priority=Priority.LOW,
        window_start=8.0, window_end=16.0, duration=1.0,
    ),
    DeliveryOrder(
        id="ORD20", description="Tech startup equipment install",
        location=Location(lat=37.7566, lng=-122.4896, address="Sunset"),
        cargo_type=CargoType.FRAGILE, priority=Priority.MEDIUM,
        window_start=9.0, window_end=15.0, duration=0.75,
    ),
]
