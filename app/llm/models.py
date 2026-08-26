from typing import Any


def create_chat_model(
    model_name: str,
    *,
    temperature: float | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> Any:
    normalized_name = model_name.strip()
    if not normalized_name:
        raise ValueError(
            "LLM_MODEL must be configured as 'provider:model', "
            "for example 'openrouter:anthropic/claude-sonnet-4.6'"
        )

    from langchain.chat_models import init_chat_model

    parameters: dict[str, object] = {}
    if temperature is not None:
        parameters["temperature"] = temperature
    if timeout is not None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        parameters["timeout"] = (
            max(1, round(timeout * 1000))
            if _model_provider(normalized_name) == "openrouter"
            else timeout
        )
    if max_retries is not None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        parameters["max_retries"] = max_retries
    return init_chat_model(normalized_name, **parameters)


def _model_provider(model_name: str) -> str | None:
    provider, separator, _model = model_name.partition(":")
    if not separator:
        return None
    return provider.replace("-", "_").lower()
