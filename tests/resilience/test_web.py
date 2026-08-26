import httpx
import pytest

from app.resilience.web import CircuitOpenError, ResilientWebSearchProvider


class FlakyProvider:
    def __init__(self, failures: list[Exception]):
        self.failures = failures
        self.calls = 0

    def search(self, query: str):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return []


def test_transient_failure_is_retried_with_backoff() -> None:
    provider = FlakyProvider([httpx.ConnectError("offline")])
    delays: list[float] = []
    resilient = ResilientWebSearchProvider(
        provider,
        max_attempts=3,
        backoff_seconds=0.5,
        circuit_failure_threshold=5,
        circuit_reset_seconds=30,
        sleeper=delays.append,
    )

    assert resilient.search("query") == []
    assert provider.calls == 2
    assert delays == [0.5]


def test_non_transient_failure_is_not_retried() -> None:
    provider = FlakyProvider([ValueError("invalid payload")])
    resilient = ResilientWebSearchProvider(
        provider,
        max_attempts=3,
        backoff_seconds=0,
        circuit_failure_threshold=2,
        circuit_reset_seconds=30,
    )

    with pytest.raises(ValueError, match="invalid payload"):
        resilient.search("query")
    assert provider.calls == 1


def test_circuit_opens_and_allows_probe_after_reset() -> None:
    now = [0.0]
    provider = FlakyProvider(
        [httpx.ConnectError("one"), httpx.ConnectError("two")]
    )
    resilient = ResilientWebSearchProvider(
        provider,
        max_attempts=1,
        backoff_seconds=0,
        circuit_failure_threshold=2,
        circuit_reset_seconds=10,
        clock=lambda: now[0],
    )

    with pytest.raises(httpx.ConnectError):
        resilient.search("query")
    with pytest.raises(httpx.ConnectError):
        resilient.search("query")
    with pytest.raises(CircuitOpenError):
        resilient.search("query")
    assert provider.calls == 2

    now[0] = 10.1
    assert resilient.search("query") == []
    assert provider.calls == 3
