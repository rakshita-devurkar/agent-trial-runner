"""The records an agent can read and change: flights, hotels, and bookings.

Deliberately a plain in-memory dataclass rather than a database. Every trial
needs its own untouched copy -- if one trial's booking leaked into the next,
the second trial would fail for a reason that has nothing to do with the agent
-- and copying a dataclass is cheaper and harder to get wrong than resetting a
database between three hundred thousand runs.

Grading works by comparing this object at the end of a trial against what it
should have been. Nothing here reads the agent's own account of what it did,
which is the point: an agent that says "cancelled!" without cancelling anything
has to be caught.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from enum import StrEnum


def same_traveller(one: str | None, two: str | None) -> bool:
    """Whether two spellings name the same person.

    An agent told "I'm lena" that books for "Lena" has done nothing wrong, and
    an exact match failed it. On the first real run that was 28 of 33 failures
    -- the results were measuring the grader's case sensitivity rather than the
    agent, and would have read as a model weakness.
    """
    return (one or "").strip().casefold() == (two or "").strip().casefold()


class BookingStatus(StrEnum):
    ACTIVE = "active"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Flight:
    flight_id: str
    origin: str
    destination: str
    date: str
    price: int
    seats_left: int


@dataclass(frozen=True)
class Hotel:
    hotel_id: str
    city: str
    name: str
    price_per_night: int


@dataclass(frozen=True)
class Booking:
    booking_id: str
    traveller: str
    #: A flight_id or a hotel_id; which one is implied by `kind`.
    subject_id: str
    kind: str
    status: BookingStatus = BookingStatus.ACTIVE
    #: Only set for hotels. Flights carry their date on the flight itself.
    check_in: str | None = None
    check_out: str | None = None


@dataclass
class World:
    """One trial's private copy of the travel company's records."""

    flights: dict[str, Flight] = field(default_factory=dict)
    hotels: dict[str, Hotel] = field(default_factory=dict)
    bookings: dict[str, Booking] = field(default_factory=dict)
    #: Bookings are numbered per world so two trials cannot collide on an ID.
    next_booking: int = 1

    def copy(self) -> World:
        """A fresh, independent copy. What every trial starts from."""
        return copy.deepcopy(self)

    def new_booking_id(self) -> str:
        booking_id = f"B{self.next_booking:04d}"
        self.next_booking += 1
        return booking_id

    def cancel(self, booking_id: str) -> None:
        booking = self.bookings[booking_id]
        self.bookings[booking_id] = replace(booking, status=BookingStatus.CANCELLED)

    def active_bookings(self, traveller: str) -> list[Booking]:
        return [
            booking
            for booking in self.bookings.values()
            if same_traveller(booking.traveller, traveller)
            and booking.status is BookingStatus.ACTIVE
        ]
