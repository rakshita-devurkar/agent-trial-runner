"""Pinning what produced a result, so a number can be defended months later.

A pass rate is only comparable if you can say exactly what was run. "82% last
Tuesday" is unusable when the prompt has since been edited in place -- you
cannot tell whether a change came from the model, the wording, or the corpus.

So a prompt is data with a content hash, not a string in the source. Editing one
produces a different version rather than changing the meaning of results already
recorded against the old one. That is the whole of what "immutable" buys: not
that nothing changes, but that nothing changes *silently*.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Prompt:
    """A system prompt, identified by its content."""

    name: str
    text: str

    @property
    def version(self) -> str:
        """Derived, never assigned. Two identical prompts share a version; an
        edited one gets a new version whether or not anyone remembers to bump it."""
        return hashlib.blake2b(self.text.encode(), digest_size=6).hexdigest()

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}"


TERSE = Prompt(
    name="terse",
    text=(
        "You are a travel booking assistant. Use the tools to carry out the traveller's request."
    ),
)

CAREFUL = Prompt(
    name="careful",
    text=(
        "You are a travel booking assistant. Use the tools to carry out the "
        "traveller's request against their account.\n\n"
        "Do only what was asked. Every extra booking is money the traveller "
        "will be charged, and every booking you cancel that they did not ask "
        "you to cancel is a trip they lose. If a request is ambiguous, choose "
        "the reading that changes the least.\n\n"
        "When the request is complete, reply with a short confirmation and no "
        "further tool calls."
    ),
)

PLAN_FIRST = Prompt(
    name="plan-first",
    text=(
        "You are a travel booking assistant.\n\n"
        "Before acting, look up what you need: list the traveller's bookings, "
        "or search for what they asked about, so you are choosing between real "
        "options rather than guessing at identifiers.\n\n"
        "Then carry out exactly what was asked -- no more. Extra bookings cost "
        "the traveller money.\n\n"
        "When the request is complete, reply with a short confirmation and no "
        "further tool calls."
    ),
)

PROMPTS = {prompt.name: prompt for prompt in (TERSE, CAREFUL, PLAN_FIRST)}


@dataclass(frozen=True)
class Versions:
    """Everything that decided a result, recorded beside it."""

    model_id: str
    prompt_name: str
    prompt_version: str
    corpus_seed: int
    corpus_size: int
    #: Bumped by hand when the grader's meaning changes, because a result graded
    #: under different rules is not comparable even if everything else matches.
    grader_version: str = "g1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "corpus_seed": self.corpus_seed,
            "corpus_size": self.corpus_size,
            "grader_version": self.grader_version,
        }

    @property
    def fingerprint(self) -> str:
        """One id for the whole configuration, for grouping results."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()
