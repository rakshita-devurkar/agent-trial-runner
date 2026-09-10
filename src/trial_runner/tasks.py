"""A task: the records to start from, what to ask, and what must be true after.

A task is data, not code, so that the same 2,500 of them can be generated,
versioned, frozen and replayed. `task_id` is derived from the content, so an
edited task becomes a different task rather than quietly changing the meaning
of results already recorded against it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from trial_runner.grading import Expectation
from trial_runner.world import World


@dataclass(frozen=True)
class Task:
    task_id: str
    traveller: str
    #: What the traveller says. The only thing the agent is told.
    request: str
    start: World
    expectation: Expectation

    @staticmethod
    def make_id(request: str, traveller: str, expectation: Expectation) -> str:
        """Content-addressed, so editing a task cannot silently reuse its history."""
        payload = json.dumps(
            {
                "request": request,
                "traveller": traveller,
                "required": [tuple(vars(b).values()) for b in expectation.required],
                "cancelled": [tuple(vars(b).values()) for b in expectation.cancelled],
                "forbid_extras": expectation.forbid_extras,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.blake2b(payload.encode(), digest_size=6).hexdigest()
