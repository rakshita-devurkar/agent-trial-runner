"""The travel agent under test, and the loop that runs one trial of it.

Three things here are platform concerns rather than agent concerns, and they are
the reason this file exists instead of a bare API call.

*Budgets.* An agent that loops forever is not a rare event at three hundred
thousand trials; it is a daily one. Every trial is capped on steps, tokens and
wall-clock, and hitting a cap is its own recorded outcome rather than a silent
failure.

*The trace.* Every step is recorded as it happens, so a failure can be read back
afterwards instead of reproduced by rerunning a non-deterministic agent and
hoping it goes wrong the same way.

*Infrastructure errors.* A throttled request is not the agent being wrong. If
the two are mixed, a pass rate partly measures our own outages, so they are
separated here and never counted together.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from trial_runner.tools import TOOLS
from trial_runner.world import World

SYSTEM = (
    "You are a travel booking assistant. Use the tools to carry out the traveller's "
    "request against their account. Do only what was asked: extra bookings cost the "
    "traveller money. When the request is complete, reply with a short confirmation "
    "and no further tool calls."
)


class Outcome(StrEnum):
    """How a trial ended. Only PASS and FAIL are scored."""

    PASS = "pass"
    FAIL = "fail"
    #: The agent ran out of budget before finishing. Its own doing, but not a
    #: wrong answer, so it is reported apart from FAIL.
    EXHAUSTED = "exhausted"
    #: Ours, not the agent's: throttling, timeouts, a malformed response.
    INFRA_ERROR = "infra_error"


class InfraError(Exception):
    """Raised when the platform failed, so it is never scored as a wrong answer."""


@dataclass(frozen=True)
class Budget:
    max_steps: int = 10
    max_tokens: int = 20_000
    max_seconds: float = 60.0


@dataclass
class Step:
    """One turn: what the agent said, what it called, what came back."""

    index: int
    text: str = ""
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None


@dataclass
class Trial:
    """Everything one trial produced, enough to replay it without rerunning it."""

    steps: list[Step] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    stopped_because: str = ""


def tool_config() -> dict[str, Any]:
    """The agent's vocabulary, in the shape Bedrock's Converse API expects."""
    schemas: dict[str, dict[str, Any]] = {
        "search_flights": {
            "origin": "string",
            "destination": "string",
            "date": "string",
        },
        "book_flight": {"flight_id": "string", "traveller": "string"},
        "search_hotels": {"city": "string"},
        "book_hotel": {
            "hotel_id": "string",
            "traveller": "string",
            "check_in": "string",
            "check_out": "string",
        },
        "list_bookings": {"traveller": "string"},
        "cancel_booking": {"booking_id": "string"},
    }
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": name,
                    "description": f"Call {name}.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {arg: {"type": kind} for arg, kind in properties.items()},
                            "required": list(properties),
                        }
                    },
                }
            }
            for name, properties in schemas.items()
        ]
    }


def run_trial(client: Any, model_id: str, world: World, request: str, budget: Budget) -> Trial:
    """Run one trial to completion, a budget, or an infrastructure failure.

    `world` is mutated in place and is expected to be this trial's private copy.
    """
    trial = Trial()
    started = time.monotonic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": request}]}]

    for index in range(budget.max_steps):
        trial.seconds = time.monotonic() - started
        if trial.seconds > budget.max_seconds:
            trial.stopped_because = "out of time"
            return trial
        if trial.input_tokens + trial.output_tokens > budget.max_tokens:
            trial.stopped_because = "out of tokens"
            return trial

        response = _converse(client, model_id, messages)
        usage = response.get("usage", {})
        trial.input_tokens += int(usage.get("inputTokens", 0))
        trial.output_tokens += int(usage.get("outputTokens", 0))

        content = response["output"]["message"].get("content", [])
        step = Step(index=index)
        step.text = " ".join(part["text"] for part in content if "text" in part).strip()
        calls = [part["toolUse"] for part in content if "toolUse" in part]

        if not calls:
            trial.steps.append(step)
            trial.stopped_because = "agent finished"
            trial.seconds = time.monotonic() - started
            return trial

        messages.append({"role": "assistant", "content": content})
        results = []
        for call in calls:
            outcome = _run_tool(world, call)
            results.append(
                {
                    "toolResult": {
                        "toolUseId": call["toolUseId"],
                        "content": [{"json": {"result": outcome}}],
                    }
                }
            )
            trial.steps.append(
                Step(
                    index=index,
                    text=step.text,
                    tool=call["name"],
                    arguments=call.get("input") or {},
                    result=outcome.get("data"),
                    error=outcome.get("error"),
                )
            )
        messages.append({"role": "user", "content": results})

    trial.stopped_because = "out of steps"
    trial.seconds = time.monotonic() - started
    return trial


def _run_tool(world: World, call: dict[str, Any]) -> dict[str, Any]:
    """Execute one tool call. An unknown tool is the agent's mistake, not ours."""
    function = TOOLS.get(call["name"])
    if function is None:
        return {"ok": False, "error": f"no such tool {call['name']}"}
    arguments = call.get("input") or {}
    try:
        result = function(world, **arguments)
    except TypeError as exc:
        # Wrong or missing arguments: the agent misused a real tool.
        return {"ok": False, "error": f"bad arguments: {exc}"}
    return {"ok": result.ok, "data": result.data, "error": result.error}


def _converse(client: Any, model_id: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """One model call, with every failure mode named as ours rather than the agent's."""
    try:
        response: dict[str, Any] = client.converse(
            modelId=model_id,
            messages=messages,
            system=[{"text": SYSTEM}],
            toolConfig=tool_config(),
            inferenceConfig={"maxTokens": 1024, "temperature": 1.0},
        )
        return response
    except Exception as exc:
        # Anything the SDK raises -- throttling, a timeout, a malformed
        # response, a model that is not enabled -- is the platform failing, and
        # must not be recorded as the agent answering incorrectly.
        raise InfraError(f"{type(exc).__name__}: {exc}") from exc


def to_json(trial: Trial) -> str:
    """The trace, for replay. Written per trial and never edited afterwards."""
    return json.dumps(
        {
            "stopped_because": trial.stopped_because,
            "input_tokens": trial.input_tokens,
            "output_tokens": trial.output_tokens,
            "seconds": round(trial.seconds, 3),
            "steps": [
                {
                    "index": s.index,
                    "text": s.text,
                    "tool": s.tool,
                    "arguments": s.arguments,
                    "result": s.result,
                    "error": s.error,
                }
                for s in trial.steps
            ],
        },
        indent=1,
        default=str,
    )
