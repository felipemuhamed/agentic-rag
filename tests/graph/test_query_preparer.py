from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.llm.query_preparer import (
    PREPARE_QUERY_INSTRUCTIONS,
    LangChainQueryPreparer,
)


@dataclass
class FakeResponse:
    content: object


class RecordingChatModel:
    def __init__(self, response: object):
        self.response = response
        self.inputs: list[object] = []

    def invoke(self, input: object) -> FakeResponse:
        self.inputs.append(input)
        return FakeResponse(self.response)


def test_first_turn_is_preserved_without_calling_model() -> None:
    model = RecordingChatModel("must not be used")
    preparer = LangChainQueryPreparer(model)

    query = preparer.prepare("  Como configurar OAuth?  ", [])

    assert query == "Como configurar OAuth?"
    assert model.inputs == []


def test_follow_up_is_prepared_from_conversation_history() -> None:
    model = RecordingChatModel("  Como configurar OAuth no Keycloak?  ")
    preparer = LangChainQueryPreparer(model)
    history = [
        HumanMessage(content="Como configurar OAuth?"),
        AIMessage(content="Configure o provedor e o cliente OAuth."),
    ]

    query = preparer.prepare("E no Keycloak?", history)

    assert query == "Como configurar OAuth no Keycloak?"
    assert len(model.inputs) == 1
    messages = model.inputs[0]
    assert isinstance(messages, list)
    assert messages[0] == ("system", PREPARE_QUERY_INSTRUCTIONS)
    prompt = messages[1][1]
    assert "Como configurar OAuth?" in prompt
    assert "Configure o provedor" in prompt
    assert "E no Keycloak?" in prompt


def test_preparer_rejects_empty_model_response() -> None:
    preparer = LangChainQueryPreparer(RecordingChatModel("   "))

    with pytest.raises(RuntimeError, match="empty or unsupported response"):
        preparer.prepare(
            "E no Keycloak?",
            [HumanMessage(content="Como configurar OAuth?")],
        )
