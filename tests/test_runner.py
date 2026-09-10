"""What the runner must guarantee, tested without touching Bedrock.

A fake client stands in for the model so these can assert on isolation,
concurrency and error handling deterministically. The real model is exercised
separately; none of the properties below are about the model.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from trial_runner.agent import Budget, Outcome
from trial_runner.grading import Expectation, ExpectedBooking
from trial_runner.report import summarize
from trial_runner.runner import Config, Result, run_matrix
from trial_runner.tasks import Task
from trial_runner.world import Hotel, World


def reply(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "output": {"message": {"content": content}},
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }


class FakeClient:
    """Replays a fixed script of model turns, so a trial is deterministic."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = script
        self.calls = 0

    def converse(self, **_: Any) -> dict[str, Any]:
        turn = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return turn


def books_h1() -> dict[str, Any]:
    return reply(
        [
            {
                "toolUse": {
                    "toolUseId": "t1",
                    "name": "book_hotel",
                    "input": {
                        "hotel_id": "H1",
                        "traveller": "rakshita",
                        "check_in": "03-03",
                        "check_out": "03-05",
                    },
                }
            }
        ]
    )


DONE = reply([{"text": "All set."}])


@pytest.fixture
def task() -> Task:
    world = World(hotels={"H1": Hotel("H1", "Denver", "Denver Marriott", 180)})
    return Task(
        task_id="t-book",
        traveller="rakshita",
        request="Book me the Denver Marriott, 3rd to 5th.",
        start=world,
        expectation=Expectation(
            traveller="rakshita", required=(ExpectedBooking(kind="hotel", subject_id="H1"),)
        ),
    )


CONFIG = Config(config_id="nova/v1", model_id="fake", prompt_version="v1")


class TestIsolation:
    def test_trials_do_not_see_each_other_s_bookings(self, task: Task) -> None:
        """The reason every trial copies the records. Ten trials each book the
        same hotel; if they shared state, nine would fail on the leftovers."""
        results = run_matrix(
            lambda: FakeClient([books_h1(), DONE]), [task], [CONFIG], repetitions=10, workers=4
        )
        assert len(results) == 10
        assert all(r.outcome is Outcome.PASS for r in results)

    def test_the_task_s_own_records_are_never_touched(self, task: Task) -> None:
        run_matrix(
            lambda: FakeClient([books_h1(), DONE]), [task], [CONFIG], repetitions=3, workers=2
        )
        assert not task.start.bookings


class TestInfraErrors:
    class Broken:
        def converse(self, **_: Any) -> dict[str, Any]:
            raise RuntimeError("ThrottlingException: rate exceeded")

    def test_a_throttled_call_is_not_a_failing_agent(self, task: Task) -> None:
        results = run_matrix(
            lambda: TestInfraErrors.Broken(), [task], [CONFIG], repetitions=4, workers=2
        )
        assert all(r.outcome is Outcome.INFRA_ERROR for r in results)
        assert all(not r.scored for r in results)

    def test_infra_errors_are_excluded_from_the_rate(self, task: Task) -> None:
        """The whole point: an outage must not read as the agent getting worse."""
        good = run_matrix(
            lambda: FakeClient([books_h1(), DONE]), [task], [CONFIG], repetitions=2, workers=1
        )
        bad = run_matrix(
            lambda: TestInfraErrors.Broken(), [task], [CONFIG], repetitions=8, workers=1
        )
        summary = summarize(good + bad)["nova/v1"]
        assert summary.rate == 1.0
        assert summary.scored == 2
        assert summary.infra_errors == 8

    def test_the_error_is_recorded_not_swallowed(self, task: Task) -> None:
        results = run_matrix(
            lambda: TestInfraErrors.Broken(), [task], [CONFIG], repetitions=1, workers=1
        )
        assert "Throttling" in (results[0].infra_error or "")


class TestBudgets:
    def test_an_agent_that_never_stops_is_cut_off(self, task: Task) -> None:
        """Left alone this loops forever; at 300k trials that is a daily event."""
        results = run_matrix(
            lambda: FakeClient([books_h1()]),
            [task],
            [CONFIG],
            repetitions=1,
            workers=1,
            budget=Budget(max_steps=4),
        )
        assert results[0].outcome is Outcome.EXHAUSTED
        assert results[0].reasons == ["out of steps"]

    def test_being_cut_off_is_not_counted_as_wrong(self, task: Task) -> None:
        results = run_matrix(
            lambda: FakeClient([books_h1()]),
            [task],
            [CONFIG],
            repetitions=1,
            workers=1,
            budget=Budget(max_steps=3),
        )
        assert not results[0].scored


class TestConcurrency:
    def test_workers_bound_how_many_run_at_once(self, task: Task) -> None:
        """`workers` is the entire throttling story, so it has to actually hold."""
        live = 0
        peak = 0
        lock = threading.Lock()

        class Counting(FakeClient):
            def converse(self, **kw: Any) -> dict[str, Any]:
                nonlocal live, peak
                with lock:
                    live += 1
                    peak = max(peak, live)
                try:
                    return super().converse(**kw)
                finally:
                    with lock:
                        live -= 1

        run_matrix(
            lambda: Counting([books_h1(), DONE]), [task], [CONFIG], repetitions=40, workers=5
        )
        assert peak <= 5

    def test_every_cell_of_the_matrix_runs_exactly_once(self, task: Task) -> None:
        configs = [Config(f"c{i}", "fake", "v1") for i in range(3)]
        results = run_matrix(
            lambda: FakeClient([books_h1(), DONE]), [task], configs, repetitions=7, workers=4
        )
        assert len(results) == 21
        seen = {(r.config_id, r.repetition) for r in results}
        assert len(seen) == 21


class TestFlakiness:
    def test_a_sometimes_right_agent_is_reported_as_flaky(self, task: Task) -> None:
        """What repetitions are for: a single trial cannot see this at all."""
        results = [
            Result(task_id="t-book", config_id="nova/v1", repetition=i, outcome=outcome)
            for i, outcome in enumerate([Outcome.PASS] * 6 + [Outcome.FAIL] * 4)
        ]
        summary = summarize(results)["nova/v1"]
        assert summary.rate == 0.6
        assert summary.flaky_tasks == ["t-book"]
        assert summary.solid_tasks == []

    def test_an_always_right_agent_is_not_flaky(self, task: Task) -> None:
        results = [
            Result(task_id="t-book", config_id="nova/v1", repetition=i, outcome=Outcome.PASS)
            for i in range(10)
        ]
        summary = summarize(results)["nova/v1"]
        assert summary.solid_tasks == ["t-book"]
        assert summary.flaky_tasks == []
