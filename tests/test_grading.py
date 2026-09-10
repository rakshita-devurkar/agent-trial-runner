"""What the grader must catch, stated as the mistakes agents actually make."""

from __future__ import annotations

import pytest

from trial_runner.grading import Expectation, ExpectedBooking, grade
from trial_runner.tools import book_hotel, cancel_booking
from trial_runner.world import Booking, Hotel, World


@pytest.fixture
def world() -> World:
    return World(
        hotels={
            "H1": Hotel("H1", "Denver", "Denver Marriott", 180),
            "H2": Hotel("H2", "Denver", "Denver Hyatt", 150),
        }
    )


def wants_h1() -> Expectation:
    return Expectation(
        traveller="rakshita",
        required=(ExpectedBooking(kind="hotel", subject_id="H1"),),
    )


class TestBooking:
    def test_the_right_booking_passes(self, world: World) -> None:
        book_hotel(world, hotel_id="H1", traveller="rakshita", check_in="03-03", check_out="03-05")
        assert grade(world, wants_h1()).passed

    def test_the_wrong_hotel_fails(self, world: World) -> None:
        book_hotel(world, hotel_id="H2", traveller="rakshita", check_in="03-03", check_out="03-05")
        result = grade(world, wants_h1())
        assert not result.passed
        assert any("missing" in r for r in result.reasons)

    def test_doing_nothing_fails(self, world: World) -> None:
        assert not grade(world, wants_h1()).passed

    def test_booking_it_twice_fails(self, world: World) -> None:
        """The traveller asked for one room and would be charged for two."""
        for _ in range(2):
            book_hotel(
                world, hotel_id="H1", traveller="rakshita", check_in="03-03", check_out="03-05"
            )
        result = grade(world, wants_h1())
        assert not result.passed
        assert any("unwanted" in r for r in result.reasons)

    def test_fixing_its_own_mistake_passes(self, world: World) -> None:
        """Booked the wrong hotel, noticed, cancelled it, booked the right one."""
        wrong = book_hotel(
            world, hotel_id="H2", traveller="rakshita", check_in="03-03", check_out="03-05"
        )
        cancel_booking(world, booking_id=wrong.data["booking_id"])
        book_hotel(world, hotel_id="H1", traveller="rakshita", check_in="03-03", check_out="03-05")
        assert grade(world, wants_h1()).passed

    def test_someone_else_s_booking_is_not_ours(self, world: World) -> None:
        book_hotel(
            world, hotel_id="H1", traveller="someone-else", check_in="03-03", check_out="03-05"
        )
        assert not grade(world, wants_h1()).passed


class TestCancelling:
    @pytest.fixture
    def booked(self, world: World) -> World:
        world.bookings["B0001"] = Booking("B0001", "rakshita", "H1", "hotel")
        return world

    def wants_cancelled(self) -> Expectation:
        return Expectation(
            traveller="rakshita",
            cancelled=(ExpectedBooking(kind="hotel", subject_id="H1"),),
        )

    def test_cancelling_it_passes(self, booked: World) -> None:
        cancel_booking(booked, booking_id="B0001")
        assert grade(booked, self.wants_cancelled()).passed

    def test_saying_it_cancelled_without_cancelling_fails(self, booked: World) -> None:
        """The whole reason grading reads records and not replies."""
        result = grade(booked, self.wants_cancelled())
        assert not result.passed
        assert any("not cancelled" in r for r in result.reasons)

    def test_cancelling_then_rebooking_fails(self, booked: World) -> None:
        """Cancelled as asked, then helpfully rebooked what was cancelled."""
        cancel_booking(booked, booking_id="B0001")
        book_hotel(booked, hotel_id="H1", traveller="rakshita", check_in="03-03", check_out="03-05")
        assert not grade(booked, self.wants_cancelled()).passed


class TestIsolation:
    def test_a_copy_does_not_leak_back(self, world: World) -> None:
        """Every trial gets its own records; one trial's booking must not
        become the next trial's starting state."""
        mine = world.copy()
        book_hotel(mine, hotel_id="H1", traveller="rakshita", check_in="03-03", check_out="03-05")
        assert mine.bookings and not world.bookings


class TestTravellerNames:
    """Found by reading a trace from the first real run: an agent told "I'm
    lena" booked for "Lena", and an exact match failed it. That was 28 of 33
    failures -- the run was measuring the grader, not the agent."""

    def test_a_capitalised_name_is_the_same_traveller(self, world: World) -> None:
        book_hotel(world, hotel_id="H1", traveller="Lena", check_in="03-03", check_out="03-05")
        assert grade(
            world,
            Expectation(
                traveller="lena", required=(ExpectedBooking(kind="hotel", subject_id="H1"),)
            ),
        ).passed

    def test_surrounding_whitespace_is_the_same_traveller(self, world: World) -> None:
        book_hotel(world, hotel_id="H1", traveller=" lena ", check_in="03-03", check_out="03-05")
        assert grade(
            world,
            Expectation(
                traveller="lena", required=(ExpectedBooking(kind="hotel", subject_id="H1"),)
            ),
        ).passed

    def test_a_different_person_is_still_a_different_person(self, world: World) -> None:
        """The fix must not make every traveller the same traveller."""
        book_hotel(world, hotel_id="H1", traveller="arjun", check_in="03-03", check_out="03-05")
        assert not grade(
            world,
            Expectation(
                traveller="lena", required=(ExpectedBooking(kind="hotel", subject_id="H1"),)
            ),
        ).passed

    def test_an_unwanted_booking_is_still_caught_across_spellings(self, world: World) -> None:
        """Otherwise the fix opens a hole: book twice, spell it differently, pass."""
        book_hotel(world, hotel_id="H1", traveller="lena", check_in="03-03", check_out="03-05")
        book_hotel(world, hotel_id="H2", traveller="Lena", check_in="03-03", check_out="03-05")
        assert not grade(
            world,
            Expectation(
                traveller="lena", required=(ExpectedBooking(kind="hotel", subject_id="H1"),)
            ),
        ).passed

    def test_listing_finds_bookings_made_under_another_spelling(self, world: World) -> None:
        """An agent that cannot see a booking cannot cancel it."""
        from trial_runner.tools import list_bookings

        book_hotel(world, hotel_id="H1", traveller="Lena", check_in="03-03", check_out="03-05")
        assert list_bookings(world, traveller="lena").data
