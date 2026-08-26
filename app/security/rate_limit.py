from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import monotonic


class RateLimitExceeded(RuntimeError):
    def __init__(self, *, retry_after: float):
        super().__init__("rate limit exceeded")
        self.retry_after = max(retry_after, 0.0)


@dataclass(frozen=True, slots=True)
class RateLimitSnapshot:
    limit: int
    remaining: int


class RateLimitLease:
    def __init__(
        self,
        *,
        limiter: "RateLimiter",
        key: str,
        snapshot: RateLimitSnapshot,
    ):
        self.snapshot = snapshot
        self._limiter = limiter
        self._key = key
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._limiter.release(self._key)


class RateLimiter:
    def __init__(
        self,
        *,
        enabled: bool,
        requests: int,
        window_seconds: float,
        max_concurrent: int,
        clock: Callable[[], float] = monotonic,
    ):
        if requests < 1 or window_seconds <= 0 or max_concurrent < 1:
            raise ValueError("invalid rate limiter configuration")
        self._enabled = enabled
        self._requests = requests
        self._window_seconds = window_seconds
        self._max_concurrent = max_concurrent
        self._clock = clock
        self._requests_by_key: dict[str, deque[float]] = {}
        self._active_by_key: dict[str, int] = {}
        self._lock = Lock()

    def acquire(self, key: str) -> RateLimitLease:
        if not self._enabled:
            return RateLimitLease(
                limiter=self,
                key=key,
                snapshot=RateLimitSnapshot(
                    limit=self._requests,
                    remaining=self._requests,
                ),
            )

        now = self._clock()
        with self._lock:
            timestamps = self._requests_by_key.setdefault(key, deque())
            cutoff = now - self._window_seconds
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            active = self._active_by_key.get(key, 0)
            if active >= self._max_concurrent:
                raise RateLimitExceeded(retry_after=1.0)
            if len(timestamps) >= self._requests:
                raise RateLimitExceeded(
                    retry_after=(timestamps[0] + self._window_seconds) - now
                )

            timestamps.append(now)
            self._active_by_key[key] = active + 1
            remaining = self._requests - len(timestamps)
        return RateLimitLease(
            limiter=self,
            key=key,
            snapshot=RateLimitSnapshot(
                limit=self._requests,
                remaining=remaining,
            ),
        )

    def release(self, key: str) -> None:
        if not self._enabled:
            return
        with self._lock:
            active = self._active_by_key.get(key, 0)
            if active <= 1:
                self._active_by_key.pop(key, None)
            else:
                self._active_by_key[key] = active - 1
