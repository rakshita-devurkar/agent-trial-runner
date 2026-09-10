"""Running as fast as Bedrock allows, and no faster.

Bedrock's rate limit belongs to the account, not the machine, so throughput has
a ceiling that adding workers cannot raise. Past that ceiling extra workers do
not go faster; they convert work into ThrottlingExceptions, and a throttled
trial that is not retried is a trial thrown away. At three hundred thousand
trials a few percent of that is hours.

So two things happen here.

*A retry with backoff and jitter.* Backoff because retrying immediately at the
same rate reproduces the problem. Jitter because without it every throttled
worker waits the same interval and they all return together, which is the same
stampede one beat later.

*A shared rate limit the workers cooperate on.* Retries alone leave every worker
discovering the ceiling separately and repeatedly. One token bucket across all
of them keeps the request rate under the limit deliberately rather than by
trial and error, and it narrows automatically when the account pushes back.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")

#: What Bedrock calls its refusals. Matched on name because botocore raises
#: these as generated classes rather than one importable type.
THROTTLE_NAMES = (
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceQuotaExceededException",
    "ModelNotReadyException",
)


def is_throttle(error: BaseException) -> bool:
    name = type(error).__name__
    if name in THROTTLE_NAMES:
        return True
    # botocore surfaces the real name inside ClientError rather than as a type.
    code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
    return code in THROTTLE_NAMES or "Throttl" in f"{name}{code}{error}"


@dataclass
class Limiter:
    """A token bucket shared by every worker, that narrows when refused.

    Additive increase, multiplicative decrease -- the same shape as TCP
    congestion control, and for the same reason: the limit is not published, so
    it has to be found by probing upward and retreating fast when it is hit.
    """

    #: Requests per second to aim for, adjusted as the account responds.
    rate: float = 20.0
    min_rate: float = 1.0
    max_rate: float = 200.0
    #: How sharply to retreat on a refusal, and how gently to recover.
    backoff_factor: float = 0.6
    recovery_per_second: float = 0.5

    _allowance: float = 0.0
    _last: float = 0.0
    # default_factory, not a default: a Lock() evaluated once in the class body
    # would be shared by every instance.
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._last = time.monotonic()
        self._allowance = self.rate

    def acquire(self) -> None:
        """Block until this worker may send a request."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._allowance = min(self.rate, self._allowance + elapsed * self.rate)
                # Recover slowly towards the ceiling while nothing is refusing.
                self.rate = min(self.max_rate, self.rate + elapsed * self.recovery_per_second)
                if self._allowance >= 1.0:
                    self._allowance -= 1.0
                    return
                wait = (1.0 - self._allowance) / self.rate
            time.sleep(min(wait, 0.5))

    def refused(self) -> None:
        """Called on a throttle. Retreat now; recovery is gradual by design."""
        with self._lock:
            self.rate = max(self.min_rate, self.rate * self.backoff_factor)


@dataclass
class RetryPolicy:
    attempts: int = 6
    base_seconds: float = 0.5
    max_seconds: float = 20.0


@dataclass
class Throttling:
    """Counts for the report: how much of the run was spent being refused."""

    refusals: int = 0
    retried: int = 0
    gave_up: int = 0
    waited_seconds: float = 0.0
    # default_factory, not a default: a Lock() evaluated once in the class body
    # would be shared by every instance.
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def note(self, *, refused: bool = False, waited: float = 0.0, gave_up: bool = False) -> None:
        with self._lock:
            self.refusals += refused
            self.retried += refused and not gave_up
            self.gave_up += gave_up
            self.waited_seconds += waited


def with_retry(
    call: Callable[[], T],
    *,
    limiter: Limiter | None = None,
    policy: RetryPolicy | None = None,
    stats: Throttling | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `call`, retrying only what is worth retrying.

    A throttle is the account being busy and is always worth another attempt. A
    validation error is a bug in the request and will fail identically forever,
    so it is raised immediately rather than retried five times.
    """
    policy = policy or RetryPolicy()
    last: BaseException | None = None

    for attempt in range(policy.attempts):
        if limiter is not None:
            limiter.acquire()
        try:
            return call()
        except BaseException as exc:
            if not is_throttle(exc):
                raise
            last = exc
            if limiter is not None:
                limiter.refused()
            if attempt == policy.attempts - 1:
                break
            delay = min(policy.max_seconds, policy.base_seconds * (2**attempt))
            # Full jitter: without it, every worker refused at the same moment
            # waits the same interval and returns together, which is the same
            # stampede one beat later.
            delay = random.uniform(0, delay)
            if stats is not None:
                stats.note(refused=True, waited=delay)
            sleep(delay)

    if stats is not None:
        stats.note(gave_up=True)
    assert last is not None  # only reachable after a throttle
    raise last


def wrap_client(client: Any, **kwargs: Any) -> Any:
    """A Bedrock client whose `converse` retries throttles. Same interface."""

    class Retrying:
        def converse(self, **call_kwargs: Any) -> Any:
            return with_retry(lambda: client.converse(**call_kwargs), **kwargs)

    return Retrying()
