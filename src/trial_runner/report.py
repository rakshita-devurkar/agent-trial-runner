"""Turning trial results into rates, and rates into an honest comparison.

Repetitions exist so that a per-task result is a probability rather than a
verdict, and this is where that pays off. A configuration passing a task 6 times
in 10 is not "passing"; it is a coin the traveller has to flip. So the report
distinguishes three things a single run cannot: reliably right, reliably wrong,
and unreliable.

Rates are computed over scored trials only. Infrastructure errors are reported
beside the rate, never inside it, because an outage is not evidence about an
agent.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from trial_runner.agent import Outcome
from trial_runner.runner import Result


@dataclass
class TaskSummary:
    """How one configuration did on one task, across its repetitions."""

    task_id: str
    config_id: str
    passes: int = 0
    scored: int = 0
    exhausted: int = 0
    infra_errors: int = 0

    @property
    def rate(self) -> float | None:
        """None when nothing was scored -- a missing number, not a zero."""
        return self.passes / self.scored if self.scored else None

    @property
    def flaky(self) -> bool:
        """Sometimes right, sometimes wrong, on identical input.

        The failure mode a single trial cannot see and users feel every day.
        """
        return self.scored > 1 and 0 < self.passes < self.scored


@dataclass
class ConfigSummary:
    config_id: str
    tasks: dict[str, TaskSummary] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def scored(self) -> int:
        return sum(t.scored for t in self.tasks.values())

    @property
    def passes(self) -> int:
        return sum(t.passes for t in self.tasks.values())

    @property
    def rate(self) -> float | None:
        return self.passes / self.scored if self.scored else None

    @property
    def infra_errors(self) -> int:
        return sum(t.infra_errors for t in self.tasks.values())

    @property
    def exhausted(self) -> int:
        return sum(t.exhausted for t in self.tasks.values())

    @property
    def flaky_tasks(self) -> list[str]:
        return sorted(task_id for task_id, t in self.tasks.items() if t.flaky)

    @property
    def solid_tasks(self) -> list[str]:
        """Passed every time it was scored. The only tasks you can rely on."""
        return sorted(
            task_id for task_id, t in self.tasks.items() if t.scored and t.passes == t.scored
        )


def summarize(results: list[Result]) -> dict[str, ConfigSummary]:
    """Group raw trial results by configuration and task."""
    configs: dict[str, ConfigSummary] = {}
    by_cell: dict[tuple[str, str], TaskSummary] = defaultdict(
        lambda: TaskSummary(task_id="", config_id="")
    )

    for result in results:
        config = configs.setdefault(result.config_id, ConfigSummary(config_id=result.config_id))
        config.input_tokens += result.input_tokens
        config.output_tokens += result.output_tokens
        config.seconds += result.seconds

        key = (result.config_id, result.task_id)
        task = by_cell[key]
        task.task_id, task.config_id = result.task_id, result.config_id
        if result.scored:
            task.scored += 1
            task.passes += result.outcome is Outcome.PASS
        elif result.outcome is Outcome.EXHAUSTED:
            task.exhausted += 1
        else:
            task.infra_errors += 1
        config.tasks[result.task_id] = task

    return configs


def render(configs: dict[str, ConfigSummary]) -> str:
    """A comparison table. Every number states what it is computed over."""
    header = (
        f"{'config':22} {'pass rate':>10} {'scored':>7} {'flaky':>6} "
        f"{'exhausted':>10} {'infra':>6} {'cost':>9}"
    )
    lines = [header]
    for config_id in sorted(configs):
        c = configs[config_id]
        rate = "n/a" if c.rate is None else f"{c.rate:.1%}"
        cost = _cost(c.input_tokens, c.output_tokens)
        lines.append(
            f"{config_id:22} {rate:>10} {c.scored:>7} {len(c.flaky_tasks):>6} "
            f"{c.exhausted:>10} {c.infra_errors:>6} {cost:>9}"
        )

    total_infra = sum(c.infra_errors for c in configs.values())
    if total_infra:
        lines.append("")
        lines.append(f"{total_infra} trial(s) never ran and are excluded from every rate above.")
    return "\n".join(lines)


#: Amazon Nova Micro, USD per million tokens. Kept here so a cost is always
#: reported next to a rate: a configuration that is two points better and four
#: times dearer is a decision, not an improvement.
INPUT_PER_MILLION = 0.035
OUTPUT_PER_MILLION = 0.14


def throughput(results: list[Result]) -> str:
    """Sustained trials per second, measured rather than assumed.

    Wall clock between the first and last trial to finish, which is what a run
    actually costs in time -- not the sum of trial durations, which counts the
    workers' waiting many times over.
    """
    times = sorted(r.finished_at for r in results if r.finished_at)
    if len(times) < 2:
        return "throughput: not recorded"
    elapsed = times[-1] - times[0]
    if elapsed <= 0:
        return "throughput: not recorded"
    rate = len(times) / elapsed
    return (
        f"{rate:.2f} trials/s over {elapsed / 60:.1f}m  |  "
        f"300,000 would take {300_000 / rate / 3600:.1f}h"
    )


def _cost(input_tokens: int, output_tokens: int) -> str:
    dollars = (input_tokens * INPUT_PER_MILLION + output_tokens * OUTPUT_PER_MILLION) / 1_000_000
    return f"${dollars:.4f}"
