"""What the runner must guarantee, tested without touching Bedrock.

A fake client stands in for the model so these can assert on isolation,
concurrency and error handling deterministically. The real model is exercised
separately; none of the properties below are about the model.
"""

from __future__ import annotations

import threading
from typing import Any

from tests.conftest import DONE, BrokenClient, FakeClient, books_h1, reply
from trial_runner.agent import Budget, Outcome
from trial_runner.report import summarize
from trial_runner.runner import Config, Result, run_matrix
from trial_runner.tasks import Task
from trial_runner.versions import CAREFUL

CONFIG = Config(config_id="nova/v1", model_id="fake", prompt=CAREFUL)


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
    def test_a_throttled_call_is_not_a_failing_agent(self, task: Task) -> None:
        results = run_matrix(lambda: BrokenClient(), [task], [CONFIG], repetitions=4, workers=2)
        assert all(r.outcome is Outcome.INFRA_ERROR for r in results)
        assert all(not r.scored for r in results)

    def test_infra_errors_are_excluded_from_the_rate(self, task: Task) -> None:
        """The whole point: an outage must not read as the agent getting worse."""
        good = run_matrix(
            lambda: FakeClient([books_h1(), DONE]), [task], [CONFIG], repetitions=2, workers=1
        )
        bad = run_matrix(lambda: BrokenClient(), [task], [CONFIG], repetitions=8, workers=1)
        summary = summarize(good + bad)["nova/v1"]
        assert summary.rate == 1.0
        assert summary.scored == 2
        assert summary.infra_errors == 8

    def test_the_error_is_recorded_not_swallowed(self, task: Task) -> None:
        results = run_matrix(lambda: BrokenClient(), [task], [CONFIG], repetitions=1, workers=1)
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
        configs = [Config(f"c{i}", "fake", CAREFUL) for i in range(3)]
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


class TestMalformedToolNames:
    """Found on a real run: gpt-oss emitted a tool name Bedrock then refused to
    accept back, so the agent's malformed output became our crashed trial."""

    def _calling(self, name: str) -> dict[str, Any]:
        return reply(
            [{"toolUse": {"toolUseId": "t1", "name": name, "input": {"traveller": "rakshita"}}}]
        )

    def test_a_rejected_name_does_not_kill_the_trial(self, task: Task) -> None:
        from trial_runner.agent import Outcome

        results = run_matrix(
            lambda: FakeClient([self._calling("functions.book hotel!"), DONE]),
            [task],
            [CONFIG],
            repetitions=1,
            workers=1,
        )
        # A wrong answer, because it never booked anything -- not an infra error.
        assert results[0].outcome is Outcome.FAIL

    def test_the_agent_is_told_the_tool_does_not_exist(self, task: Task) -> None:
        results = run_matrix(
            lambda: FakeClient([self._calling("bad name!"), DONE]),
            [task],
            [CONFIG],
            repetitions=1,
            workers=1,
        )
        trace = results[0].trace
        assert trace is not None
        assert any("no such tool" in (s.error or "") for s in trace.steps)

    def test_what_is_echoed_back_is_always_acceptable(self) -> None:
        """The actual failure: the *next* request was rejected, not this one."""
        import re

        from trial_runner.agent import _sanitize

        content = [{"toolUse": {"toolUseId": "t", "name": "functions.book hotel!", "input": {}}}]
        name = _sanitize(content)[0]["toolUse"]["name"]
        assert re.match(r"^[a-zA-Z0-9_-]+$", name)

    def test_a_valid_name_is_left_alone(self) -> None:
        from trial_runner.agent import _sanitize

        content = [{"toolUse": {"toolUseId": "t", "name": "book_hotel", "input": {}}}]
        assert _sanitize(content)[0]["toolUse"]["name"] == "book_hotel"

    def test_text_parts_are_untouched(self) -> None:
        from trial_runner.agent import _sanitize

        content: list[dict[str, Any]] = [
            {"text": "thinking..."},
            {"toolUse": {"toolUseId": "t", "name": "a b", "input": {}}},
        ]
        assert _sanitize(content)[0] == {"text": "thinking..."}
