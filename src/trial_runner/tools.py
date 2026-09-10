"""The only things an agent is allowed to do to the records.

Every tool returns a result rather than raising, including for bad input. An
agent that asks to cancel a booking that does not exist has made an ordinary
mistake and should be told so and given the chance to recover -- the same as a
real API would. Raising here would turn the agent's mistake into the harness's
crash, and those two must never be confused: one is a failing trial, the other
is a broken platform.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from trial_runner.world import Booking, BookingStatus, World, same_traveller


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None


def search_flights(world: World, *, origin: str, destination: str, date: str) -> ToolResult:
    matches = [
        flight
        for flight in world.flights.values()
        if flight.origin == origin and flight.destination == destination and flight.date == date
    ]
    return ToolResult(
        ok=True,
        data=[
            {
                "flight_id": f.flight_id,
                "price": f.price,
                "seats_left": f.seats_left,
                "date": f.date,
            }
            for f in sorted(matches, key=lambda f: f.flight_id)
        ],
    )


def book_flight(world: World, *, flight_id: str, traveller: str) -> ToolResult:
    flight = world.flights.get(flight_id)
    if flight is None:
        return ToolResult(ok=False, error=f"no flight {flight_id}")
    if flight.seats_left < 1:
        return ToolResult(ok=False, error=f"flight {flight_id} is full")

    booking_id = world.new_booking_id()
    world.bookings[booking_id] = Booking(
        booking_id=booking_id, traveller=traveller, subject_id=flight_id, kind="flight"
    )
    return ToolResult(ok=True, data={"booking_id": booking_id})


def search_hotels(world: World, *, city: str) -> ToolResult:
    matches = [hotel for hotel in world.hotels.values() if hotel.city == city]
    return ToolResult(
        ok=True,
        data=[
            {"hotel_id": h.hotel_id, "name": h.name, "price_per_night": h.price_per_night}
            for h in sorted(matches, key=lambda h: h.hotel_id)
        ],
    )


def book_hotel(
    world: World, *, hotel_id: str, traveller: str, check_in: str, check_out: str
) -> ToolResult:
    if hotel_id not in world.hotels:
        return ToolResult(ok=False, error=f"no hotel {hotel_id}")
    if check_out <= check_in:
        return ToolResult(ok=False, error="check_out must be after check_in")

    booking_id = world.new_booking_id()
    world.bookings[booking_id] = Booking(
        booking_id=booking_id,
        traveller=traveller,
        subject_id=hotel_id,
        kind="hotel",
        check_in=check_in,
        check_out=check_out,
    )
    return ToolResult(ok=True, data={"booking_id": booking_id})


def list_bookings(world: World, *, traveller: str) -> ToolResult:
    return ToolResult(
        ok=True,
        data=[
            {
                "booking_id": b.booking_id,
                "kind": b.kind,
                "subject_id": b.subject_id,
                "status": b.status.value,
                "check_in": b.check_in,
                "check_out": b.check_out,
            }
            for b in sorted(world.bookings.values(), key=lambda b: b.booking_id)
            if same_traveller(b.traveller, traveller)
        ],
    )


def cancel_booking(world: World, *, booking_id: str) -> ToolResult:
    booking = world.bookings.get(booking_id)
    if booking is None:
        return ToolResult(ok=False, error=f"no booking {booking_id}")
    if booking.status is BookingStatus.CANCELLED:
        return ToolResult(ok=False, error=f"booking {booking_id} is already cancelled")

    world.cancel(booking_id)
    return ToolResult(ok=True, data={"booking_id": booking_id, "status": "cancelled"})


#: The agent's entire vocabulary. Anything not here, it cannot do.
TOOLS: dict[str, Callable[..., ToolResult]] = {
    "search_flights": search_flights,
    "book_flight": book_flight,
    "search_hotels": search_hotels,
    "book_hotel": book_hotel,
    "list_bookings": list_bookings,
    "cancel_booking": cancel_booking,
}
