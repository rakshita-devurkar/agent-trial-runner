"""What must happen when the account says slow down.

A throttled trial that is not retried is a trial thrown away, and at three
hundred thousand trials a few percent of that is hours of work and money. None
of this can be tested against Bedrock -- provoking real throttling is slow,
expensive and not reproducible -- so a fake refuses on demand instead.
"""

from __future__ import annotations

import threading

import pytest

from trial_runner.throttle import Limiter, RetryPolicy, Throttling, is_throttle, with_retry


class Throttled(Exception):
    """Shaped like what botocore raises for a rate limit."""

    def __init__(self) -> None:
        super().__init__("Rate exceeded")
        self.response = {"Error": {"Code": "ThrottlingException"}}


class Broken(Exception):
    """A request that is simply wrong, and will be wrong every time."""

    def __init__(self) -> None:
        super().__init__("bad input")
        self.response = {"Error": {"Code": "ValidationException"}}


def never_sleep(_: float) -> None:
    return None


class TestRecognisingRefusals:
    def test_a_throttle_is_recognised(self) -> None:
        assert is_throttle(Throttled())

    def test_a_validation_error_is_not(self) -> None:
        """Retrying a malformed request just fails five more times."""
        assert not is_throttle(Broken())


class TestRetrying:
    def test_it_succeeds_once_the_account_recovers(self) -> None:
        calls = 0

        def flaky() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise Throttled()
            return "ok"

        assert with_retry(flaky, sleep=never_sleep) == "ok"
        assert calls == 3

    def test_a_real_error_is_raised_at_once(self) -> None:
        calls = 0

        def broken() -> str:
            nonlocal calls
            calls += 1
            raise Broken()

        with pytest.raises(Broken):
            with_retry(broken, sleep=never_sleep)
        assert calls == 1, "a bad request must not be retried"

    def test_it_gives_up_eventually(self) -> None:
        """Retrying forever turns a bad afternoon into a stuck run."""
        with pytest.raises(Throttled):
            with_retry(
                lambda: (_ for _ in ()).throw(Throttled()),
                policy=RetryPolicy(attempts=3),
                sleep=never_sleep,
            )

    def test_waits_grow_and_are_jittered(self) -> None:
        waits: list[float] = []

        def failing() -> str:
            raise Throttled()

        with pytest.raises(Throttled):
            with_retry(
                failing,
                policy=RetryPolicy(attempts=6, base_seconds=1.0),
                sleep=waits.append,
            )
        assert len(waits) == 5
        # Jitter means no fixed sequence, so assert the envelope instead: each
        # wait is drawn from [0, 2^n) and the ceiling doubles.
        assert all(0 <= w <= 1.0 * 2**i for i, w in enumerate(waits))
        assert len(set(waits)) > 1, "identical waits mean every worker returns together"


class TestCounting:
    def test_refusals_are_reported(self) -> None:
        stats = Throttling()
        calls = 0

        def flaky() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise Throttled()
            return "ok"

        with_retry(flaky, stats=stats, sleep=never_sleep)
        assert stats.refusals == 2
        assert stats.gave_up == 0

    def test_giving_up_is_reported_separately(self) -> None:
        stats = Throttling()
        with pytest.raises(Throttled):
            with_retry(
                lambda: (_ for _ in ()).throw(Throttled()),
                policy=RetryPolicy(attempts=2),
                stats=stats,
                sleep=never_sleep,
            )
        assert stats.gave_up == 1


class TestLimiter:
    def test_it_narrows_when_refused(self) -> None:
        limiter = Limiter(rate=100.0)
        limiter.refused()
        assert limiter.rate < 100.0

    def test_it_never_narrows_to_a_stop(self) -> None:
        """A rate of zero is a hung run, not a slow one."""
        limiter = Limiter(rate=100.0, min_rate=2.0)
        for _ in range(200):
            limiter.refused()
        assert limiter.rate >= 2.0

    def test_it_holds_the_rate_across_workers(self) -> None:
        """The point of sharing one limiter: the ceiling is per account."""
        limiter = Limiter(rate=50.0, recovery_per_second=0.0)
        sent = 0
        lock = threading.Lock()

        def worker() -> None:
            nonlocal sent
            for _ in range(20):
                limiter.acquire()
                with lock:
                    sent += 1

        threads = [threading.Thread(target=worker) for _ in range(5)]
        import time

        started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.monotonic() - started

        assert sent == 100
        # 100 requests at 50/s cannot finish faster than the bucket allows.
        assert elapsed >= (100 - 50) / 50.0 * 0.8

    def test_each_limiter_has_its_own_lock(self) -> None:
        """A Lock() as a dataclass default is created once and shared by all."""
        assert Limiter()._lock is not Limiter()._lock
