from collections.abc import Sequence
from typing import Any

from app.llm.models import create_chat_model
from app.multimodal.analyzer import ImageAnalyzer, LangChainImageAnalyzer
from app.settings import Settings


def create_image_analyzer(
    config: Settings,
    *,
    callbacks: Sequence[Any] | None = None,
) -> ImageAnalyzer:
    model_name = config.vision_model.strip() or config.llm_model
    model = create_chat_model(
        model_name,
        timeout=config.llm_timeout_seconds,
        max_retries=0,
    )
    return LangChainImageAnalyzer(
        model,
        max_attempts=config.llm_max_attempts,
        callbacks=callbacks,
    )
