from math import ceil

from fastapi import HTTPException, Request

from app.security.access import (
    AccessContext,
    AuthenticationError,
    TenantAccessError,
)
from app.security.rate_limit import RateLimitExceeded, RateLimitLease


def authorize_request(
    request: Request,
    *,
    api_key: str | None,
    tenant_id: str | None,
) -> AccessContext:
    """Apply the API key policy and translate its errors into HTTP status."""
    metrics = request.app.state.metrics
    try:
        return request.app.state.access_policy.authorize(
            api_key=api_key,
            tenant_id=tenant_id,
        )
    except AuthenticationError as error:
        metrics.auth_failed()
        raise HTTPException(
            status_code=401,
            detail="invalid API credential",
            headers={"WWW-Authenticate": "ApiKey"},
        ) from error
    except TenantAccessError as error:
        metrics.auth_failed()
        raise HTTPException(
            status_code=403,
            detail="tenant access denied",
        ) from error


def acquire_rate_limit(request: Request, access: AccessContext) -> RateLimitLease:
    try:
        return request.app.state.rate_limiter.acquire(access.quota_key)
    except RateLimitExceeded as error:
        request.app.state.metrics.rate_limit_rejected()
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(max(ceil(error.retry_after), 1))},
        ) from error
