import json

import pytest

from app.security.access import (
    AccessPolicy,
    AuthenticationError,
    TenantAccessError,
)
from app.settings import Settings


def make_policy() -> AccessPolicy:
    return AccessPolicy.from_settings(
        Settings(
            auth_enabled=True,
            api_key_policies=json.dumps(
                [
                    {
                        "key": "a-secure-key-with-32-characters-01",
                        "principal": "frontend",
                        "tenants": ["tenant-a", "tenant-b"],
                    }
                ]
            ),
            _env_file=None,
        )
    )


def test_access_policy_isolates_thread_and_filters_by_tenant() -> None:
    access = make_policy().authorize(
        api_key="a-secure-key-with-32-characters-01",
        tenant_id="tenant-a",
    )

    assert access.retrieval_filters == {"tenant_id": "tenant-a"}
    assert len(access.thread_id("conversation")) == 64
    assert access.thread_id("conversation") != AccessPolicy.from_settings(
        Settings(auth_enabled=False, _env_file=None)
    ).authorize(api_key=None, tenant_id="tenant-a").thread_id("conversation")


def test_access_policy_rejects_bad_key_and_tenant() -> None:
    policy = make_policy()

    with pytest.raises(AuthenticationError):
        policy.authorize(api_key="wrong-key-with-32-characters-000000", tenant_id="tenant-a")
    with pytest.raises(TenantAccessError):
        policy.authorize(
            api_key="a-secure-key-with-32-characters-01",
            tenant_id="tenant-c",
        )


def test_plaintext_api_key_is_not_retained_in_policy_representation() -> None:
    policy = make_policy()

    assert "a-secure-key" not in repr(policy.__dict__)


def test_key_rotation_can_preserve_the_same_principal() -> None:
    config = Settings(
        auth_enabled=True,
        api_key_policies=json.dumps(
            [
                {
                    "key": "old-key-with-32-characters-0000001",
                    "principal": "frontend",
                    "tenants": ["tenant-a"],
                },
                {
                    "key": "new-key-with-32-characters-0000002",
                    "principal": "frontend",
                    "tenants": ["tenant-a"],
                },
            ]
        ),
        _env_file=None,
    )
    policy = AccessPolicy.from_settings(config)

    old_access = policy.authorize(
        api_key="old-key-with-32-characters-0000001",
        tenant_id="tenant-a",
    )
    new_access = policy.authorize(
        api_key="new-key-with-32-characters-0000002",
        tenant_id="tenant-a",
    )

    assert old_access.thread_id("conversation") == new_access.thread_id(
        "conversation"
    )
