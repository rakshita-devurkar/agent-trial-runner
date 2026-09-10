"""Where results and traces go, and how a run survives being interrupted.

A three hundred thousand trial run takes hours. Hours is long enough that
something will go wrong -- a laptop sleeps, a token expires, a quota is hit --
and a run that loses everything at hour four is not a platform. So results are
written as they are produced, never at the end, and a restart reads back what is
already done and skips it.

Local files rather than DynamoDB while the shape is still moving. The interface
is small on purpose: `record`, `done`, `results`. Swapping the backend for S3 and
DynamoDB later changes this file and nothing else.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Self

from trial_runner.agent import Outcome, to_json
from trial_runner.runner import Result

#: A cell of the matrix: this task, under this configuration, on this repetition.
Cell = tuple[str, str, int]


class Store:
    """Append-only results, plus one trace file per trial.

    Append-only matters more than it looks. A run that crashes mid-write leaves
    one broken final line, which is recoverable; a run that rewrites a results
    file in place can lose all of it.
    """

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root / run_id
        self.traces = self.root / "traces"
        self.traces.mkdir(parents=True, exist_ok=True)
        self.results_path = self.root / "results.jsonl"
        self._lock = threading.Lock()
        self._handle = self.results_path.open("a", encoding="utf-8")

    def record(self, result: Result, *, keep_trace: bool = True) -> None:
        """Write one result now. Called from many threads, so it is serialized."""
        row = {
            "task_id": result.task_id,
            "config_id": result.config_id,
            "repetition": result.repetition,
            "model_id": result.model_id,
            "prompt_version": result.prompt_version,
            "outcome": result.outcome.value,
            "reasons": result.reasons,
            "infra_error": result.infra_error,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "seconds": round(result.seconds, 3),
            "steps": result.steps,
        }
        with self._lock:
            self._handle.write(json.dumps(row) + "\n")
            # Flushed per row rather than buffered: an unflushed buffer is
            # exactly the work that disappears when a run is interrupted.
            self._handle.flush()

        # Traces are the expensive part, and only failures are worth reading
        # back. Keeping all 300,000 would be tens of gigabytes of successes
        # nobody opens.
        if keep_trace and result.trace is not None and result.outcome is not Outcome.PASS:
            name = f"{result.config_id}--{result.task_id}--{result.repetition}.json"
            (self.traces / name.replace("/", "_")).write_text(to_json(result.trace))

    def done(self) -> set[Cell]:
        """Cells already recorded, so a restart does not redo them.

        A truncated final line from a crash is dropped rather than raising: one
        lost trial is cheaper than refusing to resume a run of 300,000.
        """
        if not self.results_path.exists():
            return set()
        finished: set[Cell] = set()
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            finished.add((row["task_id"], row["config_id"], row["repetition"]))
        return finished

    def results(self) -> list[Result]:
        """Read every recorded result back, for reporting after the fact."""
        rows: list[Result] = []
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(
                Result(
                    task_id=row["task_id"],
                    config_id=row["config_id"],
                    repetition=row["repetition"],
                    model_id=row.get("model_id", ""),
                    prompt_version=row.get("prompt_version", ""),
                    outcome=Outcome(row["outcome"]),
                    reasons=row.get("reasons", []),
                    infra_error=row.get("infra_error"),
                    input_tokens=row.get("input_tokens", 0),
                    output_tokens=row.get("output_tokens", 0),
                    seconds=row.get("seconds", 0.0),
                    steps=row.get("steps", 0),
                )
            )
        return rows

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        """What this run was: versions, corpus, budget. Written once, up front."""
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def manifest_for(
    configs: list[Any], corpus_seed: int, corpus_size: int, budget: Any
) -> dict[str, Any]:
    return {
        "corpus_seed": corpus_seed,
        "corpus_size": corpus_size,
        "budget": asdict(budget),
        "configs": [
            {
                "config_id": c.config_id,
                "model_id": c.model_id,
                "prompt_name": c.prompt.name,
                "prompt_version": c.prompt.version,
            }
            for c in configs
        ],
    }
