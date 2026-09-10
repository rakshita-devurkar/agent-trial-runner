"""The same store, backed by S3 and DynamoDB instead of local files.

Split by what each is good at rather than by habit.

*DynamoDB holds one row per trial.* Three hundred thousand small writes that
must survive the process, be queryable by run, and never collide between
workers. A conditional write makes a resumed run idempotent: replaying a cell
that already finished is refused by the database rather than double-counted in
the results.

*S3 holds the traces.* They are large, written once and read rarely, and only
for trials that did not pass.

The interface matches the local store deliberately. A run does not know or care
which one it is writing to, which is what lets the same code run on a laptop
and on EC2.
"""

from __future__ import annotations

import json
import threading
from decimal import Decimal
from typing import Any

from trial_runner.agent import Outcome, to_json
from trial_runner.runner import Result

Cell = tuple[str, str, int]


class AwsStore:
    """Results in DynamoDB, traces in S3, keyed by run."""

    def __init__(
        self,
        run_id: str,
        *,
        table: Any,
        bucket: Any,
        bucket_name: str,
        keep_passes: bool = False,
    ) -> None:
        self.run_id = run_id
        self.table = table
        self.bucket = bucket
        self.bucket_name = bucket_name
        self.keep_passes = keep_passes
        self._lock = threading.Lock()

    @staticmethod
    def cell_key(result: Result) -> str:
        """One trial's identity. Sorting by it groups a task's repetitions."""
        return f"{result.config_id}#{result.task_id}#{result.repetition:04d}"

    def record(self, result: Result, *, keep_trace: bool = True) -> None:
        """Write one trial. Safe to call twice for the same cell."""
        item: dict[str, Any] = {
            "run_id": self.run_id,
            "cell": self.cell_key(result),
            "task_id": result.task_id,
            "config_id": result.config_id,
            "repetition": result.repetition,
            "model_id": result.model_id,
            "prompt_version": result.prompt_version,
            "outcome": result.outcome.value,
            "reasons": result.reasons,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            # DynamoDB stores exact decimals, not floats.
            "seconds": Decimal(str(round(result.seconds, 3))),
            "steps": result.steps,
            "finished_at": Decimal(str(round(result.finished_at, 3))),
        }
        if result.infra_error:
            item["infra_error"] = result.infra_error[:900]

        try:
            self.table.put_item(
                Item=item,
                # Idempotent by construction: a resumed run that re-runs a cell
                # cannot double-count it, so a rate stays a rate.
                ConditionExpression="attribute_not_exists(cell)",
            )
        except Exception as exc:
            if "ConditionalCheckFailed" not in type(exc).__name__ + str(exc):
                raise
            return

        if keep_trace and result.trace is not None:
            if result.outcome is Outcome.PASS and not self.keep_passes:
                return
            key = f"{self.run_id}/traces/{self.cell_key(result).replace('#', '/')}.json"
            self.bucket.put_object(
                Key=key, Body=to_json(result.trace).encode(), ContentType="application/json"
            )

    def done(self) -> set[Cell]:
        """Every cell already recorded for this run, so a restart can skip them."""
        finished: set[Cell] = set()
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": "run_id = :r",
            "ExpressionAttributeValues": {":r": self.run_id},
            "ProjectionExpression": "task_id, config_id, repetition",
        }
        while True:
            page = self.table.query(**kwargs)
            for row in page.get("Items", []):
                finished.add((row["task_id"], row["config_id"], int(row["repetition"])))
            token = page.get("LastEvaluatedKey")
            if not token:
                return finished
            kwargs["ExclusiveStartKey"] = token

    def results(self) -> list[Result]:
        """Read the whole run back for reporting."""
        rows: list[Result] = []
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": "run_id = :r",
            "ExpressionAttributeValues": {":r": self.run_id},
        }
        while True:
            page = self.table.query(**kwargs)
            for row in page.get("Items", []):
                rows.append(
                    Result(
                        task_id=row["task_id"],
                        config_id=row["config_id"],
                        repetition=int(row["repetition"]),
                        model_id=row.get("model_id", ""),
                        prompt_version=row.get("prompt_version", ""),
                        outcome=Outcome(row["outcome"]),
                        reasons=list(row.get("reasons", [])),
                        infra_error=row.get("infra_error"),
                        input_tokens=int(row.get("input_tokens", 0)),
                        output_tokens=int(row.get("output_tokens", 0)),
                        seconds=float(row.get("seconds", 0)),
                        steps=int(row.get("steps", 0)),
                        finished_at=float(row.get("finished_at", 0)),
                    )
                )
            token = page.get("LastEvaluatedKey")
            if not token:
                return rows
            kwargs["ExclusiveStartKey"] = token

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.bucket.put_object(
            Key=f"{self.run_id}/manifest.json",
            Body=json.dumps(manifest, indent=1, default=str).encode(),
            ContentType="application/json",
        )

    def close(self) -> None:
        """Nothing to close; both writes are already durable."""
