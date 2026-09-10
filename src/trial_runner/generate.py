"""Generating tasks whose correct answer is known by construction.

Hand-writing 2,500 tasks is not slow, it is unreliable: someone has to state
the right answer for each one, and a task whose stated answer is wrong is worse
than no task at all -- it fails correct agents forever and nobody looks again.

So tasks are built the other way round. Each template starts from records it
generated itself, so it already knows which hotel is cheapest and which booking
is the hotel one. The expectation is computed from that, never typed in. A
generated task cannot disagree with its own answer.

Scale comes from combinations. One template over 40 cities, several date ranges
and a few price shapes is thousands of tasks, and the parameters are what make
them different rather than reworded prose.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from trial_runner.grading import Expectation, ExpectedBooking
from trial_runner.tasks import Task
from trial_runner.world import Booking, Flight, Hotel, World

CITIES = [
    ("Denver", "DEN"),
    ("Seattle", "SEA"),
    ("Boston", "BOS"),
    ("Austin", "AUS"),
    ("Chicago", "ORD"),
    ("Miami", "MIA"),
    ("Portland", "PDX"),
    ("Nashville", "BNA"),
    ("Phoenix", "PHX"),
    ("Atlanta", "ATL"),
    ("Denver", "DEN"),
    ("San Diego", "SAN"),
    ("New Orleans", "MSY"),
    ("Minneapolis", "MSP"),
    ("Detroit", "DTW"),
    ("Raleigh", "RDU"),
    ("Salt Lake City", "SLC"),
    ("Kansas City", "MCI"),
    ("Pittsburgh", "PIT"),
    ("Tampa", "TPA"),
]
CHAINS = ["Marriott", "Hyatt", "Hilton", "Westin", "Sheraton", "Kimpton"]
AIRLINES = ["AA", "UA", "DL", "AS", "B6"]
TRAVELLERS = ["rakshita", "arjun", "mei", "sofia", "tomas", "priya", "andre", "lena"]


@dataclass(frozen=True)
class Scenario:
    """The records for one task, plus the facts the generator knows about them."""

    world: World
    city: str
    airport: str
    cheapest_hotel: str
    dearest_hotel: str
    cheapest_flight: str
    date: str
    check_out: str


def build_scenario(rng: random.Random) -> Scenario:
    """Invent a small city with hotels and flights, and remember which is which."""
    city, airport = rng.choice(CITIES)
    day = rng.randint(3, 25)
    date = f"2026-03-{day:02d}"
    check_out = f"2026-03-{min(day + rng.randint(1, 4), 28):02d}"

    # Distinct prices, so "the cheapest" is never ambiguous. A task with two
    # right answers is a task that fails correct agents at random.
    prices = rng.sample(range(110, 400), 3)
    hotels = {}
    for index, (chain, price) in enumerate(zip(rng.sample(CHAINS, 3), prices, strict=True)):
        hotel_id = f"H{index + 1}"
        hotels[hotel_id] = Hotel(hotel_id, city, f"{city} {chain}", price)

    fares = rng.sample(range(90, 600), 3)
    flights = {}
    for index, (airline, fare) in enumerate(zip(rng.sample(AIRLINES, 3), fares, strict=True)):
        flight_id = f"{airline}{rng.randint(100, 999)}"
        flights[flight_id] = Flight(flight_id, "SFO", airport, date, fare, rng.randint(2, 20))

    cheapest_hotel = min(hotels.values(), key=lambda h: h.price_per_night)
    dearest_hotel = max(hotels.values(), key=lambda h: h.price_per_night)
    cheapest_flight = min(flights.values(), key=lambda f: f.price)

    return Scenario(
        world=World(flights=flights, hotels=hotels),
        city=city,
        airport=airport,
        cheapest_hotel=cheapest_hotel.hotel_id,
        dearest_hotel=dearest_hotel.hotel_id,
        cheapest_flight=cheapest_flight.flight_id,
        date=date,
        check_out=check_out,
    )


def _task(traveller: str, request: str, world: World, expectation: Expectation) -> Task:
    return Task(
        task_id=Task.make_id(request, traveller, expectation),
        traveller=traveller,
        request=request,
        start=world,
        expectation=expectation,
    )


def book_cheapest_hotel(scenario: Scenario, traveller: str) -> Task:
    request = (
        f"I'm {traveller}. Book me the cheapest hotel in {scenario.city}, "
        f"checking in {scenario.date} and checking out {scenario.check_out}."
    )
    return _task(
        traveller,
        request,
        scenario.world,
        Expectation(
            traveller=traveller,
            required=(
                ExpectedBooking(
                    kind="hotel",
                    subject_id=scenario.cheapest_hotel,
                    check_in=scenario.date,
                    check_out=scenario.check_out,
                ),
            ),
        ),
    )


def book_cheapest_flight(scenario: Scenario, traveller: str) -> Task:
    request = (
        f"I'm {traveller}. Book me the cheapest flight from SFO to "
        f"{scenario.airport} on {scenario.date}."
    )
    return _task(
        traveller,
        request,
        scenario.world,
        Expectation(
            traveller=traveller,
            required=(ExpectedBooking(kind="flight", subject_id=scenario.cheapest_flight),),
        ),
    )


def cancel_the_hotel(scenario: Scenario, traveller: str) -> Task:
    """A cancellation, with a flight alongside that must survive.

    The interesting failure is not cancelling the wrong hotel; it is cancelling
    everything. An agent that reads "cancel my booking" as "cancel my bookings"
    costs the traveller a flight, and a task with only one booking in it cannot
    catch that.
    """
    world = scenario.world.copy()
    world.bookings["B0001"] = Booking("B0001", traveller, scenario.dearest_hotel, "hotel")
    world.bookings["B0002"] = Booking("B0002", traveller, scenario.cheapest_flight, "flight")
    world.next_booking = 3

    request = f"I'm {traveller}. Cancel my {scenario.city} hotel. Keep my flight."
    return _task(
        traveller,
        request,
        world,
        Expectation(
            traveller=traveller,
            required=(ExpectedBooking(kind="flight", subject_id=scenario.cheapest_flight),),
            cancelled=(ExpectedBooking(kind="hotel", subject_id=scenario.dearest_hotel),),
        ),
    )


def rebook_cheaper(scenario: Scenario, traveller: str) -> Task:
    """Two actions in one request: cancel the expensive one, book the cheap one."""
    world = scenario.world.copy()
    world.bookings["B0001"] = Booking(
        "B0001",
        traveller,
        scenario.dearest_hotel,
        "hotel",
        check_in=scenario.date,
        check_out=scenario.check_out,
    )
    world.next_booking = 2

    request = (
        f"I'm {traveller}. My {scenario.city} hotel is too expensive. Cancel it and "
        f"book the cheapest one instead, same dates ({scenario.date} to {scenario.check_out})."
    )
    return _task(
        traveller,
        request,
        world,
        Expectation(
            traveller=traveller,
            required=(
                ExpectedBooking(
                    kind="hotel",
                    subject_id=scenario.cheapest_hotel,
                    check_in=scenario.date,
                    check_out=scenario.check_out,
                ),
            ),
            cancelled=(ExpectedBooking(kind="hotel", subject_id=scenario.dearest_hotel),),
        ),
    )


def book_both(scenario: Scenario, traveller: str) -> Task:
    """A flight and a hotel in one go -- the shortest multi-step task."""
    request = (
        f"I'm {traveller}. Book me the cheapest flight from SFO to {scenario.airport} on "
        f"{scenario.date}, and the cheapest hotel in {scenario.city} from {scenario.date} "
        f"to {scenario.check_out}."
    )
    return _task(
        traveller,
        request,
        scenario.world,
        Expectation(
            traveller=traveller,
            required=(
                ExpectedBooking(kind="flight", subject_id=scenario.cheapest_flight),
                ExpectedBooking(
                    kind="hotel",
                    subject_id=scenario.cheapest_hotel,
                    check_in=scenario.date,
                    check_out=scenario.check_out,
                ),
            ),
        ),
    )


TEMPLATES = [
    book_cheapest_hotel,
    book_cheapest_flight,
    cancel_the_hotel,
    rebook_cheaper,
    book_both,
]


def generate(count: int, *, seed: int = 0) -> list[Task]:
    """`count` tasks, deterministically. The same seed gives the same corpus.

    Duplicates are dropped by content hash rather than counted, so a corpus of
    2,500 is 2,500 distinct tasks and not 2,500 draws that happened to collide.
    """
    rng = random.Random(seed)
    tasks: dict[str, Task] = {}
    attempts = 0
    limit = count * 50

    while len(tasks) < count and attempts < limit:
        attempts += 1
        scenario = build_scenario(rng)
        template = rng.choice(TEMPLATES)
        task = template(scenario, rng.choice(TRAVELLERS))
        tasks.setdefault(task.task_id, task)

    if len(tasks) < count:  # pragma: no cover - only with an exhausted space
        raise ValueError(
            f"only {len(tasks)} distinct tasks from {attempts} attempts; "
            "add templates or parameters before asking for more"
        )
    return list(tasks.values())


def by_template(tasks: list[Task]) -> Iterator[tuple[str, int]]:
    """How many of each kind, so a corpus is not silently 90% one template."""
    counts: dict[str, int] = {}
    for task in tasks:
        kind = task.request.split(".")[1].strip().split(" ")[0] if "." in task.request else "?"
        counts[kind] = counts.get(kind, 0) + 1
    yield from sorted(counts.items())
