import pytest

from app.security.rate_limit import RateLimitExceeded, RateLimiter


def test_rate_limiter_enforces_window_and_concurrency() -> None:
    now = [0.0]
    limiter = RateLimiter(
        enabled=True,
        requests=2,
        window_seconds=10,
        max_concurrent=1,
        clock=lambda: now[0],
    )

    first = limiter.acquire("principal")
    with pytest.raises(RateLimitExceeded, match="rate limit"):
        limiter.acquire("principal")
    first.release()

    second = limiter.acquire("principal")
    second.release()
    with pytest.raises(RateLimitExceeded) as error:
        limiter.acquire("principal")
    assert error.value.retry_after == 10.0

    now[0] = 10.1
    renewed = limiter.acquire("principal")
    assert renewed.snapshot.remaining == 1
    renewed.release()
