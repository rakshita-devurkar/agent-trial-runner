"""The AWS-backed store, against fakes.

The properties worth protecting are about our logic, not Amazon's: that a
resumed run cannot double-count a cell, that traces are kept only for trials
worth reading, and that a result survives the round trip. Testing those against
real DynamoDB would make them slow and would prove nothing extra.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from trial_runner.agent import Outcome, Trial
from trial_runner.aws_store import AwsStore
from trial_runner.runner import Result


class ConditionalCheckFailedException(Exception):
    """Named to match what botocore raises when a conditional write is refused."""


class FakeTable:
    """Enough DynamoDB to exercise conditional writes and paged queries."""

    def __init__(self, page_size: int = 100) -> None:
        self.items: dict[tuple[str, str], dict[str, Any]] = {}
        self.page_size = page_size

    def put_item(self, *, Item: dict[str, Any], ConditionExpression: str = "") -> None:
        key = (Item["run_id"], Item["cell"])
        if ConditionExpression and key in self.items:
            raise ConditionalCheckFailedException("the conditional request failed")
        self.items[key] = dict(Item)

    def query(self, **kwargs: Any) -> dict[str, Any]:
        run_id = kwargs["ExpressionAttributeValues"][":r"]
        rows = [v for (r, _), v in sorted(self.items.items()) if r == run_id]
        start = 0
        token = kwargs.get("ExclusiveStartKey")
        if token:
            start = int(token["offset"])
        page = rows[start : start + self.page_size]
        result: dict[str, Any] = {"Items": page}
        if start + self.page_size < len(rows):
            result["LastEvaluatedKey"] = {"offset": start + self.page_size}
        return result


class FakeBucket:
    def __init__(self) -> None:
        self.objects_written: dict[str, bytes] = {}

    def put_object(self, *, Key: str, Body: bytes, ContentType: str = "") -> None:
        self.objects_written[Key] = Body


def store(page_size: int = 100) -> tuple[AwsStore, FakeTable, FakeBucket]:
    table, bucket = FakeTable(page_size), FakeBucket()
    return AwsStore("run-1", table=table, bucket=bucket, bucket_name="b"), table, bucket


def result(rep: int = 0, outcome: Outcome = Outcome.PASS, **kw: Any) -> Result:
    return Result(
        task_id="t1",
        config_id="nova/careful",
        repetition=rep,
        outcome=outcome,
        trace=Trial(stopped_because="agent finished"),
        **kw,
    )


class TestIdempotence:
    def test_replaying_a_finished_cell_does_not_double_count(self) -> None:
        """A resumed run may re-attempt a cell; a rate must stay a rate."""
        s, table, _ = store()
        s.record(result(0, Outcome.PASS))
        s.record(result(0, Outcome.FAIL))
        assert len(table.items) == 1
        assert len(s.results()) == 1

    def test_the_first_write_wins(self) -> None:
        s, _, _ = store()
        s.record(result(0, Outcome.PASS))
        s.record(result(0, Outcome.FAIL))
        assert s.results()[0].outcome is Outcome.PASS

    def test_different_repetitions_are_different_cells(self) -> None:
        s, table, _ = store()
        for rep in range(5):
            s.record(result(rep))
        assert len(table.items) == 5


class TestRoundTrip:
    def test_a_result_survives(self) -> None:
        s, _, _ = store()
        s.record(
            result(
                0,
                Outcome.FAIL,
                model_id="nova-micro",
                prompt_version="abc123",
                reasons=["missing hotel booking"],
                input_tokens=100,
                output_tokens=20,
                seconds=1.5,
                steps=3,
            )
        )
        back = s.results()[0]
        assert back.outcome is Outcome.FAIL
        assert back.model_id == "nova-micro"
        assert back.reasons == ["missing hotel booking"]
        assert back.seconds == 1.5

    def test_seconds_are_stored_as_decimals(self) -> None:
        """DynamoDB rejects floats; a silent failure here loses every timing."""
        s, table, _ = store()
        s.record(result(0, seconds=1.234))
        assert isinstance(next(iter(table.items.values()))["seconds"], Decimal)

    def test_a_long_infra_error_is_truncated(self) -> None:
        """DynamoDB items have a size limit; a stack trace must not break a write."""
        s, table, _ = store()
        s.record(result(0, Outcome.INFRA_ERROR, infra_error="x" * 5000))
        assert len(next(iter(table.items.values()))["infra_error"]) <= 900


class TestResuming:
    def test_done_lists_every_finished_cell(self) -> None:
        s, _, _ = store()
        for rep in range(4):
            s.record(result(rep))
        assert s.done() == {("t1", "nova/careful", rep) for rep in range(4)}

    def test_paging_is_followed_to_the_end(self) -> None:
        """A run of 300,000 will not come back in one page."""
        s, _, _ = store(page_size=10)
        for rep in range(35):
            s.record(result(rep))
        assert len(s.done()) == 35
        assert len(s.results()) == 35


class TestTraces:
    def test_a_failure_keeps_its_trace(self) -> None:
        s, _, bucket = store()
        s.record(result(0, Outcome.FAIL))
        assert bucket.objects_written

    def test_a_pass_does_not(self) -> None:
        s, _, bucket = store()
        s.record(result(0, Outcome.PASS))
        assert not bucket.objects_written

    def test_a_refused_write_writes_no_trace(self) -> None:
        """Otherwise a resumed run re-uploads traces it already has."""
        s, _, bucket = store()
        s.record(result(0, Outcome.FAIL))
        bucket.objects_written.clear()
        s.record(result(0, Outcome.FAIL))
        assert not bucket.objects_written
