import pytest

from app.graph.factory import create_rag_graph
from app.llm.models import create_chat_model
from app.settings import Settings


def test_model_name_must_be_configured_before_provider_initialization() -> None:
    with pytest.raises(ValueError, match="LLM_MODEL must be configured"):
        create_chat_model("   ")


def test_openrouter_converts_timeout_seconds_to_milliseconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    expected_model = object()

    def fake_init_chat_model(model_name: str, **parameters: object):
        calls.append((model_name, parameters))
        return expected_model

    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model",
        fake_init_chat_model,
    )

    model = create_chat_model(
        "openrouter:anthropic/claude-sonnet-4.6",
        timeout=30,
        max_retries=0,
    )

    assert model is expected_model
    assert calls == [
        (
            "openrouter:anthropic/claude-sonnet-4.6",
            {"timeout": 30_000, "max_retries": 0},
        )
    ]


def test_timeout_for_other_providers_remains_in_provider_native_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters: dict[str, object] = {}

    def fake_init_chat_model(model_name: str, **values: object):
        parameters.update(values)
        return object()

    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model",
        fake_init_chat_model,
    )

    create_chat_model("openai:gpt-4.1-mini", timeout=30.5)

    assert parameters == {"timeout": 30.5}


def test_rag_runtime_disables_provider_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class ModelInitializationObserved(Exception):
        pass

    def fake_create_chat_model(
        model_name: str,
        **parameters: object,
    ) -> object:
        calls.append({"model_name": model_name, **parameters})
        raise ModelInitializationObserved

    monkeypatch.setattr(
        "app.graph.factory.create_chat_model",
        fake_create_chat_model,
    )
    config = Settings(
        llm_model="openrouter:anthropic/claude-sonnet-4.6",
        llm_timeout_seconds=45,
        llm_max_attempts=3,
        _env_file=None,
    )

    with pytest.raises(ModelInitializationObserved):
        create_rag_graph(config)

    assert calls == [
        {
            "model_name": "openrouter:anthropic/claude-sonnet-4.6",
            "timeout": 45.0,
            "max_retries": 0,
        }
    ]
