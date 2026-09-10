"""Whether the generated corpus is trustworthy.

The danger with generated tasks is a task whose stated answer is wrong or
unreachable. It fails every correct agent forever, and because nobody re-reads
2,500 tasks, it looks like a model weakness rather than a corpus bug.

So the central test is an oracle: an agent that does exactly what each task
asks, using only the tools a real agent has. If the oracle cannot score 100%,
the corpus is broken, not the model.
"""

from __future__ import annotations

from trial_runner.generate import generate
from trial_runner.grading import grade
from trial_runner.tasks import Task
from trial_runner.tools import book_flight, book_hotel, cancel_booking
from trial_runner.world import BookingStatus, World


def oracle(task: Task) -> World:
    """Do exactly what the task requires, through the same tools an agent uses.

    Deliberately not a shortcut that edits the records directly: the point is to
    prove every expectation is reachable through the agent's own vocabulary.
    """
    world = task.start.copy()

    for wanted in task.expectation.cancelled:
        booking = next(
            b
            for b in world.bookings.values()
            if b.traveller == task.traveller
            and b.subject_id == wanted.subject_id
            and b.status is BookingStatus.ACTIVE
        )
        cancel_booking(world, booking_id=booking.booking_id)

    for wanted in task.expectation.required:
        already = any(
            b.traveller == task.traveller
            and b.subject_id == wanted.subject_id
            and b.status is BookingStatus.ACTIVE
            for b in world.bookings.values()
        )
        if already:
            continue
        if wanted.kind == "hotel":
            book_hotel(
                world,
                hotel_id=wanted.subject_id,
                traveller=task.traveller,
                check_in=wanted.check_in or "2026-03-03",
                check_out=wanted.check_out or "2026-03-05",
            )
        else:
            book_flight(world, flight_id=wanted.subject_id, traveller=task.traveller)
    return world


class TestTheCorpusIsSolvable:
    def test_a_perfect_agent_scores_100_percent(self) -> None:
        """The one test that says the corpus is not lying."""
        tasks = generate(500, seed=1)
        failures = [
            (task.task_id, grade(oracle(task), task.expectation).reasons)
            for task in tasks
            if not grade(oracle(task), task.expectation).passed
        ]
        assert not failures, failures[:3]

    def test_doing_nothing_scores_nearly_zero(self) -> None:
        """If idleness passes, the tasks are not asking for anything."""
        tasks = generate(200, seed=2)
        passed = [t for t in tasks if grade(t.start.copy(), t.expectation).passed]
        assert not passed, [t.request for t in passed[:3]]


class TestScale:
    def test_it_produces_two_thousand_five_hundred_distinct_tasks(self) -> None:
        tasks = generate(2500, seed=3)
        assert len(tasks) == 2500
        assert len({t.task_id for t in tasks}) == 2500

    def test_the_same_seed_gives_the_same_corpus(self) -> None:
        """A corpus has to be reproducible or results cannot be compared later."""
        first = [t.task_id for t in generate(200, seed=7)]
        second = [t.task_id for t in generate(200, seed=7)]
        assert first == second

    def test_different_seeds_give_different_corpora(self) -> None:
        first = {t.task_id for t in generate(200, seed=8)}
        second = {t.task_id for t in generate(200, seed=9)}
        assert len(first & second) < 40

    def test_every_template_is_represented(self) -> None:
        """A corpus that is 90% one template measures one behaviour."""
        tasks = generate(500, seed=4)
        shapes = {
            ("cancel" in t.request.lower(), "cheapest flight" in t.request.lower()) for t in tasks
        }
        assert len(shapes) >= 3


class TestTasksAreUnambiguous:
    def test_the_cheapest_hotel_is_never_a_tie(self) -> None:
        """Two right answers means a correct agent fails at random."""
        for task in generate(300, seed=5):
            prices = [h.price_per_night for h in task.start.hotels.values()]
            assert len(prices) == len(set(prices))

    def test_the_cheapest_flight_is_never_a_tie(self) -> None:
        for task in generate(300, seed=6):
            fares = [f.price for f in task.start.flights.values()]
            assert len(fares) == len(set(fares))

    def test_task_ids_come_from_content(self) -> None:
        """An edited task must become a new task, not silently reuse history."""
        task = generate(1, seed=11)[0]
        assert task.task_id == Task.make_id(task.request, task.traveller, task.expectation)
