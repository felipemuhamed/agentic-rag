import pytest
from pydantic import ValidationError

from app.settings import Settings


def test_max_rag_attempts_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        Settings(max_rag_attempts=0, _env_file=None)


@pytest.mark.parametrize("max_results", [0, 21])
def test_web_search_result_limit_is_validated(max_results: int) -> None:
    with pytest.raises(ValidationError):
        Settings(web_search_max_results=max_results, _env_file=None)


def test_web_search_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(web_search_timeout_seconds=0, _env_file=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("multimodal_max_images", 0),
        ("multimodal_max_images", 6),
        ("multimodal_max_image_bytes", 1023),
        ("multimodal_max_image_bytes", 20_000_001),
    ],
)
def test_multimodal_limits_are_validated(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value}, _env_file=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("visual_max_results", 0),
        ("visual_max_results", 21),
        ("visual_max_distance", -0.01),
        ("visual_max_distance", 4.01),
    ],
)
def test_visual_catalog_limits_are_validated(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value}, _env_file=None)


def test_web_search_requires_tavily_key_only_when_enabled() -> None:
    disabled = Settings(web_search_enabled=False, _env_file=None)

    assert disabled.web_search_enabled is False

    with pytest.raises(ValidationError, match="TAVILY_API_KEY"):
        Settings(web_search_enabled=True, _env_file=None)

    enabled = Settings(
        web_search_enabled=True,
        tavily_api_key="test-tavily-key",
        _env_file=None,
    )
    assert enabled.web_search_enabled is True


def test_production_requires_durable_secured_observable_configuration() -> None:
    with pytest.raises(ValidationError, match="durable checkpointer"):
        Settings(app_environment="production", _env_file=None)

    with pytest.raises(ValidationError, match="authentication"):
        Settings(
            app_environment="production",
            checkpointer_mode="sqlite",
            _env_file=None,
        )


def test_complete_production_configuration_is_accepted() -> None:
    config = Settings(
        app_environment="production",
        checkpointer_mode="sqlite",
        auth_enabled=True,
        api_key_policies=(
            '[{"key":"a-secure-production-key-0001",'
            '"principal":"frontend","tenants":["public"]}]'
        ),
        rate_limit_enabled=True,
        json_logs=True,
        metrics_api_key="metrics-key",
        _env_file=None,
    )

    assert config.checkpointer_mode == "sqlite"
    assert config.api_key_policies.get_secret_value().startswith("[")


def test_credentials_are_hidden_from_settings_representation() -> None:
    config = Settings(
        tavily_api_key="tavily-private-value",
        api_key_policies="api-policy-private-value",
        metrics_api_key="metrics-private-value",
        _env_file=None,
    )

    representation = repr(config)

    assert "tavily-private-value" not in representation
    assert "api-policy-private-value" not in representation
    assert "metrics-private-value" not in representation
