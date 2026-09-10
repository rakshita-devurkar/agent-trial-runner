"""Shared fakes, so tests about the runner never depend on a live model.

Isolation, worker bounds, budgets, error handling and resuming are properties of
the platform, not of any model. Testing them against Bedrock would make them slow,
costly and non-deterministic for no gain.
"""

from __future__ import annotations

from typing import Any

import pytest

from trial_runner.grading import Expectation, ExpectedBooking
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


class BrokenClient:
    """Every call fails the way a throttled account fails."""

    def converse(self, **_: Any) -> dict[str, Any]:
        raise RuntimeError("ThrottlingException: rate exceeded")


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


def booking_task() -> Task:
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


@pytest.fixture
def task() -> Task:
    return booking_task()
