from functools import lru_cache

from langchain.chat_models import init_chat_model

from app.settings import settings


@lru_cache(maxsize=1)
def get_internal_model():

    return init_chat_model(
        settings.llm_model,
    ).with_config({
        "tags": ["nostream"],
    })


@lru_cache(maxsize=1)
def get_answer_model():

    return init_chat_model(
        settings.llm_model,
    )