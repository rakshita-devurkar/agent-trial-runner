"""Deciding whether a trial passed, by looking only at what changed.

Two rules shape this.

*Compare outcomes, not transcripts.* An agent that replies "cancelled!" without
cancelling anything must fail, so nothing here reads the agent's words.

*Compare what the task cares about, not the whole record set.* Requiring an
identical world would fail an agent for a harmless extra search, or for a
booking ID that came out B0003 instead of B0002. A task states the changes it
demands; anything it does not mention only has to be left alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trial_runner.world import BookingStatus, World


@dataclass(frozen=True)
class ExpectedBooking:
    """One booking a task requires to exist, described by what matters."""

    kind: str
    subject_id: str
    status: BookingStatus = BookingStatus.ACTIVE
    check_in: str | None = None
    check_out: str | None = None

    def matches(self, booking: object) -> bool:
        from trial_runner.world import Booking

        if not isinstance(booking, Booking):  # pragma: no cover - defensive
            return False
        if (booking.kind, booking.subject_id, booking.status) != (
            self.kind,
            self.subject_id,
            self.status,
        ):
            return False
        if self.check_in is not None and booking.check_in != self.check_in:
            return False
        return not (self.check_out is not None and booking.check_out != self.check_out)


@dataclass(frozen=True)
class Expectation:
    """What a task requires of the records once the agent is finished."""

    traveller: str
    #: Bookings that must exist and be active when the trial ends.
    required: tuple[ExpectedBooking, ...] = ()
    #: Bookings that must have been cancelled.
    cancelled: tuple[ExpectedBooking, ...] = ()
    #: True when the agent must not create anything beyond `required`. Set for
    #: tasks whose whole point is restraint -- booking one room, not three.
    forbid_extras: bool = True


@dataclass
class Grade:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def grade(final: World, expectation: Expectation) -> Grade:
    """Compare the records the agent left behind against what was asked."""
    reasons: list[str] = []
    mine = [b for b in final.bookings.values() if b.traveller == expectation.traveller]
    unclaimed = list(mine)

    for wanted in expectation.required:
        found = next((b for b in unclaimed if wanted.matches(b)), None)
        if found is None:
            reasons.append(f"missing {wanted.kind} booking for {wanted.subject_id}")
        else:
            unclaimed.remove(found)

    for wanted in expectation.cancelled:
        target = ExpectedBooking(
            kind=wanted.kind,
            subject_id=wanted.subject_id,
            status=BookingStatus.CANCELLED,
            check_in=wanted.check_in,
            check_out=wanted.check_out,
        )
        found = next((b for b in unclaimed if target.matches(b)), None)
        if found is None:
            reasons.append(f"{wanted.kind} booking for {wanted.subject_id} was not cancelled")
        else:
            unclaimed.remove(found)

    if expectation.forbid_extras:
        # A cancelled leftover is the agent undoing its own mistake, which is
        # allowed. An active one is a booking the traveller did not ask for and
        # will be charged for.
        extras = [b for b in unclaimed if b.status is BookingStatus.ACTIVE]
        for extra in extras:
            reasons.append(f"unwanted {extra.kind} booking for {extra.subject_id}")

    return Grade(passed=not reasons, reasons=reasons)
