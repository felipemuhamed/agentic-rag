import pytest

from app.multimodal.factory import create_image_analyzer
from app.settings import Settings


class ModelInitializationObserved(Exception):
    pass


@pytest.mark.parametrize(
    ("vision_model", "expected"),
    [
        ("", "openrouter:anthropic/claude-sonnet-4.6"),
        ("openrouter:google/gemini-2.5-flash", "openrouter:google/gemini-2.5-flash"),
    ],
)
def test_image_analyzer_uses_dedicated_model_or_llm_fallback(
    monkeypatch: pytest.MonkeyPatch,
    vision_model: str,
    expected: str,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_create_chat_model(model_name: str, **parameters: object):
        calls.append({"model_name": model_name, **parameters})
        raise ModelInitializationObserved

    monkeypatch.setattr(
        "app.multimodal.factory.create_chat_model",
        fake_create_chat_model,
    )
    config = Settings(
        llm_model="openrouter:anthropic/claude-sonnet-4.6",
        vision_model=vision_model,
        llm_timeout_seconds=45,
        _env_file=None,
    )

    with pytest.raises(ModelInitializationObserved):
        create_image_analyzer(config)

    assert calls == [
        {
            "model_name": expected,
            "timeout": 45.0,
            "max_retries": 0,
        }
    ]
