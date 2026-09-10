"""What must survive an interrupted run.

A run of 300,000 trials takes hours, and hours is long enough that something
will go wrong. These tests are about the two things that make that survivable:
results are on disk the moment they exist, and a restart knows what is already
done.
"""

from __future__ import annotations

from pathlib import Path

from trial_runner.agent import Outcome, Trial
from trial_runner.runner import Result
from trial_runner.store import Store


def result(task: str = "t1", config: str = "c1", rep: int = 0, **kw: object) -> Result:
    return Result(task_id=task, config_id=config, repetition=rep, **kw)  # type: ignore[arg-type]


class TestDurability:
    def test_a_result_is_on_disk_before_the_run_ends(self, tmp_path: Path) -> None:
        """Buffered rows are exactly what disappears when a run is killed."""
        store = Store(tmp_path, "run-1")
        store.record(result(outcome=Outcome.PASS))
        assert Store(tmp_path, "run-1").results()

    def test_a_half_written_final_line_does_not_block_resuming(self, tmp_path: Path) -> None:
        """What a crash mid-write actually leaves behind."""
        store = Store(tmp_path, "run-2")
        store.record(result(rep=0, outcome=Outcome.PASS))
        store.close()
        with store.results_path.open("a") as fh:
            fh.write('{"task_id": "t1", "config_i')

        reopened = Store(tmp_path, "run-2")
        assert reopened.done() == {("t1", "c1", 0)}
        assert len(reopened.results()) == 1

    def test_every_field_survives_the_round_trip(self, tmp_path: Path) -> None:
        store = Store(tmp_path, "run-3")
        store.record(
            result(
                outcome=Outcome.FAIL,
                model_id="nova",
                prompt_version="abc123",
                reasons=["missing hotel booking"],
                input_tokens=100,
                output_tokens=20,
                steps=4,
            )
        )
        back = Store(tmp_path, "run-3").results()[0]
        assert back.outcome is Outcome.FAIL
        assert back.model_id == "nova"
        assert back.prompt_version == "abc123"
        assert back.reasons == ["missing hotel booking"]
        assert back.input_tokens == 100


class TestResuming:
    def test_finished_cells_are_reported(self, tmp_path: Path) -> None:
        store = Store(tmp_path, "run-4")
        for rep in range(3):
            store.record(result(rep=rep, outcome=Outcome.PASS))
        assert store.done() == {("t1", "c1", 0), ("t1", "c1", 1), ("t1", "c1", 2)}

    def test_a_fresh_run_has_nothing_to_skip(self, tmp_path: Path) -> None:
        assert Store(tmp_path, "run-5").done() == set()

    def test_the_runner_skips_what_is_already_done(self, tmp_path: Path) -> None:
        """The property that makes an interrupted 300,000-trial run recoverable."""
        from tests.conftest import DONE, FakeClient, booking_task, books_h1
        from trial_runner.runner import Config, run_matrix
        from trial_runner.versions import CAREFUL

        task = booking_task()
        config = Config("c1", "fake", CAREFUL)
        done = {(task.task_id, "c1", rep) for rep in range(7)}
        results = run_matrix(
            lambda: FakeClient([books_h1(), DONE]),
            [task],
            [config],
            repetitions=10,
            workers=2,
            skip=done,
        )
        assert len(results) == 3
        assert {r.repetition for r in results} == {7, 8, 9}


class TestTraces:
    def test_a_failure_keeps_its_trace(self, tmp_path: Path) -> None:
        store = Store(tmp_path, "run-6")
        store.record(result(outcome=Outcome.FAIL, trace=Trial(stopped_because="agent finished")))
        assert list(store.traces.iterdir())

    def test_a_pass_does_not(self, tmp_path: Path) -> None:
        """300,000 traces of successes nobody opens is tens of gigabytes."""
        store = Store(tmp_path, "run-7")
        store.record(result(outcome=Outcome.PASS, trace=Trial(stopped_because="agent finished")))
        assert not list(store.traces.iterdir())


class TestThroughput:
    """The dry run could not report its own sustained rate, which was the one
    number it existed to produce, because results carried no timestamp."""

    def test_a_recorded_time_survives_the_round_trip(self, tmp_path: Path) -> None:
        store = Store(tmp_path, "run-8")
        store.record(result(outcome=Outcome.PASS, finished_at=1_760_000_000.5))
        assert Store(tmp_path, "run-8").results()[0].finished_at == 1_760_000_000.5

    def test_throughput_is_measured_across_the_run(self) -> None:
        from trial_runner.report import throughput

        results = [
            Result(
                task_id="t",
                config_id="c",
                repetition=i,
                outcome=Outcome.PASS,
                finished_at=1_760_000_000 + i,
            )
            for i in range(60)
        ]
        # 60 trials spread over 59 seconds is about one a second.
        assert "1.02 trials/s" in throughput(results)

    def test_it_says_so_when_it_cannot_tell(self) -> None:
        """Better than inventing a rate from results that predate the field."""
        from trial_runner.report import throughput

        results = [Result(task_id="t", config_id="c", repetition=0, outcome=Outcome.PASS)]
        assert "not recorded" in throughput(results)
