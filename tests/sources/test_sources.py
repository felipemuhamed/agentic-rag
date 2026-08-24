import pytest

from app.knowledge.retrieval.models import RetrievedDocument
from app.sources.citations import (
    CitationValidationError,
    select_cited_sources,
)
from app.sources.models import Source
from app.sources.normalizer import normalize_rag_sources, normalize_web_sources
from app.websearch.models import WebSearchResult


def test_rag_source_preserves_chunk_page_section_and_url() -> None:
    document = RetrievedDocument(
        chunk_id="chunk-oauth-7",
        content="Procedimento OAuth.",
        metadata={
            "title": "Manual de autenticação",
            "source": "manual-oauth.pdf",
            "url": "https://docs.example.com/manual-oauth.pdf",
            "page": 7,
            "section": "Configuração OAuth",
        },
        retrieval_score=0.8,
        rerank_score=0.9,
    )

    sources = normalize_rag_sources([document])

    assert sources[0].to_public() == {
        "id": 1,
        "citation": "[1]",
        "type": "rag",
        "title": "Manual de autenticação",
        "source": "manual-oauth.pdf",
        "chunkId": "chunk-oauth-7",
        "url": "https://docs.example.com/manual-oauth.pdf",
        "page": 7,
        "section": "Configuração OAuth",
    }
    assert Source.from_public(sources[0].to_public()) == sources[0]


def test_web_source_maps_citation_directly_to_url() -> None:
    result = WebSearchResult(
        title="Keycloak OAuth documentation",
        url="https://www.keycloak.org/docs/oauth",
        content="Documentação OAuth.",
        score=0.95,
        provider="tavily",
    )

    assert normalize_web_sources([result])[0].to_public() == {
        "id": 1,
        "citation": "[1]",
        "type": "web",
        "title": "Keycloak OAuth documentation",
        "source": "https://www.keycloak.org/docs/oauth",
        "chunkId": None,
        "url": "https://www.keycloak.org/docs/oauth",
        "page": None,
        "section": None,
    }


def test_only_sources_cited_by_answer_are_exposed() -> None:
    sources = [
        Source(1, "rag", "Guia A", "a.md", chunk_id="a-1"),
        Source(2, "rag", "Guia B", "b.md", chunk_id="b-1"),
    ]

    selected = select_cited_sources("A configuração está no guia B [2].", sources)

    assert selected == [sources[1]]


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        ("Resposta sem referência.", "must contain a citation"),
        ("Resposta com fonte inventada [9].", "unknown citations: 9"),
        ("Resposta com identificador [01].", "not canonical"),
    ],
)
def test_grounded_answer_must_reference_an_existing_source(
    answer: str,
    error: str,
) -> None:
    sources = [Source(1, "rag", "Guia", "guia.md", chunk_id="chunk-1")]

    with pytest.raises(CitationValidationError, match=error):
        select_cited_sources(answer, sources)
