import logging
from collections.abc import Callable
from threading import Lock
from time import monotonic, sleep

import httpx

from app.websearch.base import WebSearchProvider
from app.websearch.models import WebSearchResult


logger = logging.getLogger(__name__)


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int,
        reset_seconds: float,
        clock: Callable[[], float] = monotonic,
    ):
        if failure_threshold < 1 or reset_seconds <= 0:
            raise ValueError("invalid circuit breaker configuration")
        self._failure_threshold = failure_threshold
        self._reset_seconds = reset_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False
        self._lock = Lock()

    def before_call(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            if self._clock() - self._opened_at < self._reset_seconds:
                raise CircuitOpenError("external provider circuit is open")
            if self._probe_in_flight:
                raise CircuitOpenError("external provider circuit probe is active")
            self._probe_in_flight = True

    def success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._probe_in_flight or self._failures >= self._failure_threshold:
                self._opened_at = self._clock()
            self._probe_in_flight = False


class ResilientWebSearchProvider:
    def __init__(
        self,
        provider: WebSearchProvider,
        *,
        max_attempts: int,
        backoff_seconds: float,
        circuit_failure_threshold: int,
        circuit_reset_seconds: float,
        sleeper: Callable[[float], None] = sleep,
        clock: Callable[[], float] = monotonic,
    ):
        if max_attempts < 1 or backoff_seconds < 0:
            raise ValueError("invalid retry configuration")
        self._provider = provider
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._sleeper = sleeper
        self._circuit = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            reset_seconds=circuit_reset_seconds,
            clock=clock,
        )

    def search(self, query: str) -> list[WebSearchResult]:
        for attempt in range(1, self._max_attempts + 1):
            self._circuit.before_call()
            try:
                results = self._provider.search(query)
            except Exception as error:
                if not _is_transient(error):
                    raise
                self._circuit.failure()
                if attempt == self._max_attempts:
                    raise
                delay = self._backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Transient web provider failure; retrying",
                    extra={"attempt": attempt, "delay_seconds": delay},
                )
                self._sleeper(delay)
            else:
                self._circuit.success()
                return results
        raise RuntimeError("web retry loop ended unexpectedly")


def _is_transient(error: Exception) -> bool:
    if isinstance(
        error,
        (
            TimeoutError,
            ConnectionError,
            httpx.TimeoutException,
            httpx.NetworkError,
        ),
    ):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return False
