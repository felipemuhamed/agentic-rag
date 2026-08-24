from dataclasses import dataclass

from app.knowledge.retrieval.models import RetrievedDocument
from app.llm.answer_generator import (
    GROUNDING_INSTRUCTIONS,
    LangChainAnswerGenerator,
    NO_CONTEXT_ANSWER,
)
from app.sources.normalizer import normalize_rag_sources, normalize_web_sources
from app.websearch.models import WebSearchResult


@dataclass
class FakeResponse:
    content: object
    usage_metadata: dict[str, int] | None = None


class RecordingChatModel:
    def __init__(self, response: object = "Resposta baseada no contexto [1]."):
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

    document = make_document()
    answer = generator.generate(
        "Como renovar o token?",
        [document],
        normalize_rag_sources([document]),
    )

    assert answer == "Resposta baseada no contexto [1]."
    assert len(model.inputs) == 1
    messages = model.inputs[0]
    assert isinstance(messages, list)
    assert messages[0] == ("system", GROUNDING_INSTRUCTIONS)
    assert "exclusivamente" in messages[0][1]
    assert "Como renovar o token?" in messages[1][1]
    assert "refresh token" in messages[1][1]
    assert '"chunk_id": "oauth-1"' in messages[1][1]
    assert '"citation_id": 1' in messages[1][1]
    assert '"citation": "[1]"' in messages[1][1]


def test_no_documents_returns_safe_answer_without_calling_model() -> None:
    model = RecordingChatModel()
    generator = LangChainAnswerGenerator(model)

    answer = generator.generate("Qual é a política?", [], [])

    assert answer == NO_CONTEXT_ANSWER
    assert model.inputs == []


def test_text_content_blocks_are_supported() -> None:
    model = RecordingChatModel(
        [{"type": "text", "text": "Resposta em bloco [1]."}]
    )
    generator = LangChainAnswerGenerator(model)
    document = make_document()

    assert generator.generate(
        "Pergunta",
        [document],
        normalize_rag_sources([document]),
    ) == "Resposta em bloco [1]."


def test_generation_metadata_preserves_provider_token_usage() -> None:
    model = RecordingChatModel()
    model.response = "Resposta medida [1]."
    document = make_document()
    generator = LangChainAnswerGenerator(model)
    model.invoke = lambda input: FakeResponse(
        content="Resposta medida [1].",
        usage_metadata={
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
        },
    )

    generation = generator.generate_with_metadata(
        "Pergunta",
        [document],
        normalize_rag_sources([document]),
    )

    assert generation.answer == "Resposta medida [1]."
    assert generation.input_tokens == 20
    assert generation.output_tokens == 5
    assert generation.total_tokens == 25


def test_web_result_url_is_preserved_in_grounding_context() -> None:
    model = RecordingChatModel()
    generator = LangChainAnswerGenerator(model)
    result = WebSearchResult(
        title="Keycloak documentation",
        url="https://www.keycloak.org/docs/oauth",
        content="Keycloak supports OAuth 2.0.",
        score=0.9,
        provider="tavily",
    )

    generator.generate(
        "Como configurar OAuth?",
        [result],
        normalize_web_sources([result]),
    )

    messages = model.inputs[0]
    assert isinstance(messages, list)
    assert '"url": "https://www.keycloak.org/docs/oauth"' in messages[1][1]
    assert '"provider": "tavily"' in messages[1][1]
