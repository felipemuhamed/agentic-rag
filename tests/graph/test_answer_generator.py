from dataclasses import dataclass

from app.knowledge.retrieval.models import RetrievedDocument
from app.llm.answer_generator import (
    GROUNDING_INSTRUCTIONS,
    LangChainAnswerGenerator,
    NO_CONTEXT_ANSWER,
)


@dataclass
class FakeResponse:
    content: object


class RecordingChatModel:
    def __init__(self, response: object = "Resposta baseada no contexto."):
        self.response = response
        self.inputs: list[object] = []

    def invoke(self, input: object) -> FakeResponse:
        self.inputs.append(input)
        return FakeResponse(self.response)


def make_document() -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id="oauth-1",
        content="O access token é renovado com um refresh token.",
        metadata={"source": "oauth.md", "section": "Renovação"},
        retrieval_score=0.8,
        rerank_score=0.9,
    )


def test_answer_prompt_is_restricted_to_retrieved_context() -> None:
    model = RecordingChatModel()
    generator = LangChainAnswerGenerator(model)

    answer = generator.generate("Como renovar o token?", [make_document()])

    assert answer == "Resposta baseada no contexto."
    assert len(model.inputs) == 1
    messages = model.inputs[0]
    assert isinstance(messages, list)
    assert messages[0] == ("system", GROUNDING_INSTRUCTIONS)
    assert "exclusivamente" in messages[0][1]
    assert "Como renovar o token?" in messages[1][1]
    assert "refresh token" in messages[1][1]
    assert '"chunk_id": "oauth-1"' in messages[1][1]


def test_no_documents_returns_safe_answer_without_calling_model() -> None:
    model = RecordingChatModel()
    generator = LangChainAnswerGenerator(model)

    answer = generator.generate("Qual é a política?", [])

    assert answer == NO_CONTEXT_ANSWER
    assert model.inputs == []


def test_text_content_blocks_are_supported() -> None:
    model = RecordingChatModel([{"type": "text", "text": "Resposta em bloco."}])
    generator = LangChainAnswerGenerator(model)

    assert generator.generate("Pergunta", [make_document()]) == "Resposta em bloco."
