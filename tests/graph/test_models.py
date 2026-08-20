import pytest

from app.llm.models import create_chat_model


def test_model_name_must_be_configured_before_provider_initialization() -> None:
    with pytest.raises(ValueError, match="LLM_MODEL must be configured"):
        create_chat_model("   ")
