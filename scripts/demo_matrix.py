"""A small real run: a few tasks, two configurations, several repetitions each.

Deliberately tiny -- this exists to prove the matrix works end to end against a
live model before anything is generated at scale.
"""

from __future__ import annotations

import boto3
from botocore.config import Config as BotoConfig

from trial_runner.agent import Budget
from trial_runner.grading import Expectation, ExpectedBooking
from trial_runner.report import render, summarize
from trial_runner.runner import Config, run_matrix
from trial_runner.tasks import Task
from trial_runner.world import Booking, Flight, Hotel, World

REGION = "us-east-1"


def base_world() -> World:
    return World(
        flights={
            "AA100": Flight("AA100", "SFO", "DEN", "2026-03-03", 220, 12),
            "UA455": Flight("UA455", "SFO", "DEN", "2026-03-03", 310, 4),
        },
        hotels={
            "H1": Hotel("H1", "Denver", "Denver Marriott", 180),
            "H2": Hotel("H2", "Denver", "Denver Hyatt", 150),
        },
    )


def with_booking(**kwargs: object) -> World:
    world = base_world()
    world.bookings["B0001"] = Booking(**kwargs)  # type: ignore[arg-type]
    world.next_booking = 2
    return world


def tasks() -> list[Task]:
    return [
        Task(
            task_id="cancel-hotel",
            traveller="rakshita",
            request="I'm rakshita. Cancel my Denver Marriott hotel booking.",
            start=with_booking(
                booking_id="B0001", traveller="rakshita", subject_id="H1", kind="hotel"
            ),
            expectation=Expectation(
                traveller="rakshita",
                cancelled=(ExpectedBooking(kind="hotel", subject_id="H1"),),
            ),
        ),
        Task(
            task_id="book-cheapest-hotel",
            traveller="rakshita",
            request=(
                "I'm rakshita. Book me the cheapest hotel in Denver, "
                "checking in 2026-03-03 and out 2026-03-05."
            ),
            start=base_world(),
            expectation=Expectation(
                traveller="rakshita",
                required=(
                    ExpectedBooking(
                        kind="hotel", subject_id="H2", check_in="2026-03-03", check_out="2026-03-05"
                    ),
                ),
            ),
        ),
        Task(
            task_id="book-cheapest-flight",
            traveller="rakshita",
            request="I'm rakshita. Book me the cheapest flight from SFO to DEN on 2026-03-03.",
            start=base_world(),
            expectation=Expectation(
                traveller="rakshita",
                required=(ExpectedBooking(kind="flight", subject_id="AA100"),),
            ),
        ),
        Task(
            task_id="cancel-only-the-hotel",
            traveller="rakshita",
            # The trap: a careless agent cancels both.
            request="I'm rakshita. Cancel just my hotel, please. Keep my flight.",
            start=_flight_and_hotel(),
            expectation=Expectation(
                traveller="rakshita",
                required=(ExpectedBooking(kind="flight", subject_id="AA100"),),
                cancelled=(ExpectedBooking(kind="hotel", subject_id="H1"),),
            ),
        ),
    ]


def _flight_and_hotel() -> World:
    world = base_world()
    world.bookings["B0001"] = Booking("B0001", "rakshita", "H1", "hotel")
    world.bookings["B0002"] = Booking("B0002", "rakshita", "AA100", "flight")
    world.next_booking = 3
    return world


def make_client() -> object:
    # Retries are the platform's job, not the agent's. Anything still failing
    # after these is recorded as an infrastructure error, never as a wrong answer.
    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=BotoConfig(retries={"max_attempts": 4, "mode": "adaptive"}),
    )


CONFIGS = [
    Config(config_id="nova-micro/v1", model_id="us.amazon.nova-micro-v1:0", prompt_version="v1"),
    Config(config_id="nova-lite/v1", model_id="us.amazon.nova-lite-v1:0", prompt_version="v1"),
]


def main() -> None:
    done = 0

    def tick(_: object) -> None:
        nonlocal done
        done += 1
        print(f"\r  {done} trials", end="", flush=True)

    results = run_matrix(
        make_client,
        tasks(),
        CONFIGS,
        repetitions=5,
        workers=8,
        budget=Budget(max_steps=8, max_seconds=45),
        on_result=tick,
    )
    print("\n")
    summaries = summarize(results)
    print(render(summaries))

    for config_id in sorted(summaries):
        summary = summaries[config_id]
        print(f"\n{config_id}")
        for task_id in sorted(summary.tasks):
            task = summary.tasks[task_id]
            rate = "n/a" if task.rate is None else f"{task.rate:.0%}"
            flag = "  <- flaky" if task.flaky else ""
            print(f"  {task_id:24} {rate:>5}  ({task.passes}/{task.scored}){flag}")


if __name__ == "__main__":
    main()
