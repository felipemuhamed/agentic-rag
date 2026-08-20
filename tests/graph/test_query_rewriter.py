from dataclasses import dataclass

import pytest

from app.llm.query_rewriter import (
    LangChainQueryRewriter,
    REWRITE_INSTRUCTIONS,
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


def test_rewriter_preserves_inputs_in_restricted_prompt() -> None:
    model = RecordingChatModel(
        "  renovação de access token usando refresh token  "
    )
    rewriter = LangChainQueryRewriter(model)

    query = rewriter.rewrite(
        "Como renovar o token?",
        "renovar token",
        "A consulta não especifica qual tipo de token.",
    )

    assert query == "renovação de access token usando refresh token"
    assert len(model.inputs) == 1
    messages = model.inputs[0]
    assert isinstance(messages, list)
    assert messages[0] == ("system", REWRITE_INSTRUCTIONS)
    assert "Como renovar o token?" in messages[1][1]
    assert "renovar token" in messages[1][1]
    assert "não especifica" in messages[1][1]


def test_rewriter_rejects_empty_model_response() -> None:
    rewriter = LangChainQueryRewriter(RecordingChatModel("   "))

    with pytest.raises(RuntimeError, match="empty or unsupported response"):
        rewriter.rewrite("Pergunta", "consulta", "contexto insuficiente")
