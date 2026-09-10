"""Running many trials at once, and turning them into a comparison.

The matrix is tasks x configurations x repetitions, and at three hundred
thousand cells the interesting problems are not about any single trial.

*Bounded concurrency.* Unlimited parallelism does not go faster; it gets
throttled, and throttling arrives as failures that look like the agent being
wrong. A fixed number of workers keeps the request rate under the account's
limit on purpose rather than by luck.

*Infrastructure errors are excluded, never counted.* A trial that could not run
is not a trial that failed. Mixing them makes a pass rate partly a measurement
of our own outages, and the direction of the error flatters whichever
configuration happened to run when the service was healthy.

*A result records its versions.* Every row carries the model, the prompt version
and the task version that produced it, because a pass rate with no provenance
cannot be compared against anything later.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from trial_runner.agent import Budget, InfraError, Outcome, Trial, run_trial
from trial_runner.grading import grade
from trial_runner.tasks import Task
from trial_runner.versions import Prompt


@dataclass(frozen=True)
class Config:
    """One version of the agent under test: a model and a prompt, both pinned."""

    config_id: str
    model_id: str
    prompt: Prompt

    @property
    def prompt_version(self) -> str:
        return self.prompt.version


@dataclass
class Result:
    """One cell of the matrix, with everything needed to compare or replay it."""

    task_id: str
    config_id: str
    repetition: int
    #: Recorded on every row: a rate with no provenance cannot be compared later.
    model_id: str = ""
    prompt_version: str = ""
    #: Defaults to the outcome that counts toward nothing. A cell whose outcome
    #: was never set means the runner itself broke, and a fail-safe default
    #: keeps that out of the rates rather than silently scoring it.
    outcome: Outcome = Outcome.INFRA_ERROR
    reasons: list[str] = field(default_factory=list)
    #: Why the platform failed, when it did. Never a statement about the agent.
    infra_error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    steps: int = 0
    trace: Trial | None = None

    @property
    def scored(self) -> bool:
        """Only real answers count toward a rate."""
        return self.outcome in {Outcome.PASS, Outcome.FAIL}


def run_one(
    make_client: Callable[[], Any], task: Task, config: Config, repetition: int, budget: Budget
) -> Result:
    """Run a single cell. Never raises: a broken platform is a recorded outcome."""
    result = Result(
        task_id=task.task_id,
        config_id=config.config_id,
        repetition=repetition,
        model_id=config.model_id,
        prompt_version=config.prompt_version,
    )
    # This trial's own copy of the records. Nothing it does can reach another.
    world = task.start.copy()
    try:
        trial = run_trial(
            make_client(), config.model_id, world, task.request, budget, config.prompt.text
        )
    except InfraError as exc:
        result.outcome = Outcome.INFRA_ERROR
        result.infra_error = str(exc)
        return result

    result.trace = trial
    result.input_tokens = trial.input_tokens
    result.output_tokens = trial.output_tokens
    result.seconds = trial.seconds
    result.steps = len(trial.steps)

    if trial.stopped_because != "agent finished":
        # Out of steps, tokens or time. The agent's own doing, but it never gave
        # an answer, so calling it wrong would overstate what we know.
        result.outcome = Outcome.EXHAUSTED
        result.reasons = [trial.stopped_because]
        return result

    verdict = grade(world, task.expectation)
    result.outcome = Outcome.PASS if verdict.passed else Outcome.FAIL
    result.reasons = verdict.reasons
    return result


def run_matrix(
    make_client: Callable[[], Any],
    tasks: Iterable[Task],
    configs: Iterable[Config],
    *,
    repetitions: int = 10,
    workers: int = 8,
    budget: Budget | None = None,
    on_result: Callable[[Result], None] | None = None,
    skip: set[tuple[str, str, int]] | None = None,
) -> list[Result]:
    """Run every task against every config, `repetitions` times each.

    Threads rather than processes: a trial is almost entirely waiting on a
    network call, so the work is IO-bound and threads keep the client and the
    records cheap to share. `workers` is the whole throttling story -- it is the
    largest number of requests that can ever be in flight.
    """
    budget = budget or Budget()
    # Cells already recorded by an earlier attempt at this run. Resuming is not
    # an optimisation at this size: redoing 200,000 finished trials because the
    # last 100 failed would cost hours and money for nothing.
    already = skip or set()
    cells: queue.Queue[tuple[Task, Config, int]] = queue.Queue()
    configs = list(configs)
    for task in tasks:
        for config in configs:
            for repetition in range(repetitions):
                if (task.task_id, config.config_id, repetition) not in already:
                    cells.put((task, config, repetition))

    results: list[Result] = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                task, config, repetition = cells.get_nowait()
            except queue.Empty:
                return
            result = run_one(make_client, task, config, repetition, budget)
            with lock:
                results.append(result)
                if on_result is not None:
                    on_result(result)
            cells.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results
