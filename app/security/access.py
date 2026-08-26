import json
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest

from app.settings import Settings


class AuthenticationError(PermissionError):
    pass


class TenantAccessError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class AccessContext:
    principal: str
    tenant_id: str | None

    @property
    def quota_key(self) -> str:
        return f"{self.principal}:{self.tenant_id or '-'}"

    def thread_id(self, conversation_id: str) -> str:
        identity = (
            f"{self.principal}\0{self.tenant_id or '-'}\0{conversation_id.strip()}"
        )
        return sha256(identity.encode("utf-8")).hexdigest()

    @property
    def retrieval_filters(self) -> dict[str, str] | None:
        if self.tenant_id is None:
            return None
        return {"tenant_id": self.tenant_id}


@dataclass(frozen=True, slots=True)
class _KeyPolicy:
    key_digest: str
    principal: str
    tenants: frozenset[str]


class AccessPolicy:
    def __init__(self, *, enabled: bool, policies: list[_KeyPolicy]):
        if enabled and not policies:
            raise ValueError("enabled access policy requires at least one API key")
        self._enabled = enabled
        self._policies = policies

    @classmethod
    def from_settings(cls, config: Settings) -> "AccessPolicy":
        if not config.auth_enabled:
            return cls(enabled=False, policies=[])
        raw = config.api_key_policies.get_secret_value()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("API_KEY_POLICIES must contain valid JSON") from error
        if not isinstance(payload, list):
            raise ValueError("API_KEY_POLICIES must be a JSON list")

        policies: list[_KeyPolicy] = []
        digests: set[str] = set()
        for item in payload:
            policy = _parse_policy(item)
            if policy.key_digest in digests:
                raise ValueError("API keys must be unique")
            digests.add(policy.key_digest)
            policies.append(policy)
        return cls(enabled=True, policies=policies)

    def authorize(
        self,
        *,
        api_key: str | None,
        tenant_id: str | None,
    ) -> AccessContext:
        normalized_tenant = _optional_text(tenant_id)
        if not self._enabled:
            return AccessContext(
                principal="anonymous",
                tenant_id=normalized_tenant,
            )
        if api_key is None or not api_key.strip():
            raise AuthenticationError("API key is required")

        candidate_digest = _key_digest(api_key.strip())
        policy = next(
            (
                item
                for item in self._policies
                if compare_digest(item.key_digest, candidate_digest)
            ),
            None,
        )
        if policy is None:
            raise AuthenticationError("API key is invalid")
        if normalized_tenant is None or (
            "*" not in policy.tenants and normalized_tenant not in policy.tenants
        ):
            raise TenantAccessError("tenant access denied")
        return AccessContext(
            principal=policy.principal,
            tenant_id=normalized_tenant,
        )


def _parse_policy(value: object) -> _KeyPolicy:
    if not isinstance(value, dict) or set(value) != {"key", "principal", "tenants"}:
        raise ValueError(
            "each API key policy needs exactly key, principal and tenants"
        )
    key = value["key"]
    principal = value["principal"]
    tenants = value["tenants"]
    if not isinstance(key, str) or len(key.strip()) < 16:
        raise ValueError("API keys must contain at least 16 characters")
    if not isinstance(principal, str) or not principal.strip():
        raise ValueError("API key principal must not be blank")
    if (
        not isinstance(tenants, list)
        or not tenants
        or any(not isinstance(tenant, str) or not tenant.strip() for tenant in tenants)
    ):
        raise ValueError("API key tenants must be a non-empty text list")
    normalized_tenants = [tenant.strip() for tenant in tenants]
    if len(normalized_tenants) != len(set(normalized_tenants)):
        raise ValueError("API key tenants must be unique")
    return _KeyPolicy(
        key_digest=_key_digest(key.strip()),
        principal=principal.strip(),
        tenants=frozenset(normalized_tenants),
    )


def _key_digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
