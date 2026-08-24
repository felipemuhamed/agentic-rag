from typing import Any


def create_chat_model(
    model_name: str,
    *,
    temperature: float | None = None,
) -> Any:
    normalized_name = model_name.strip()
    if not normalized_name:
        raise ValueError(
            "LLM_MODEL must be configured as 'provider:model', "
            "for example 'openai:gpt-4.1-mini'"
        )

    from langchain.chat_models import init_chat_model

    if temperature is None:
        return init_chat_model(normalized_name)
    return init_chat_model(normalized_name, temperature=temperature)
