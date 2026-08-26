from app.graph.nodes import RagNodes
from app.knowledge.retrieval.models import (
    RetrievalDiagnostics,
    RetrievalResult,
    RetrievedDocument,
)
from app.llm.answer_generator import NO_CONTEXT_ANSWER
from app.llm.rag_grader import RagGrade
from app.llm.web_grader import WebGrade
from app.sources.models import Source
from app.websearch.models import WebSearchResult


def make_document() -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id="chunk-1",
        content="Conteúdo recuperado.",
        metadata={"source": "guia.md"},
        retrieval_score=0.7,
        rerank_score=0.9,
    )


def make_web_result() -> WebSearchResult:
    return WebSearchResult(
        title="Documentação Keycloak",
        url="https://www.keycloak.org/docs/latest/",
        content="O Keycloak suporta configuração OAuth.",
        score=0.9,
        provider="test",
    )


class StubRetriever:
    def __init__(self, documents: list[RetrievedDocument]):
        self.queries: list[str] = []
        self.result = RetrievalResult(
            documents=documents,
            diagnostics=RetrievalDiagnostics(
                fetched_count=len(documents),
                mmr_selected_count=len(documents),
                reranked_count=len(documents),
                returned_count=len(documents),
            ),
        )

    def retrieve(self, query: str) -> RetrievalResult:
        self.queries.append(query)
        return self.result


class FilteringStubRetriever(StubRetriever):
    def __init__(self, documents: list[RetrievedDocument]):
        super().__init__(documents)
        self.filters: list[object] = []

    def retrieve(self, query: str, filters=None) -> RetrievalResult:
        self.queries.append(query)
        self.filters.append(filters)
        return self.result


class StubAnswerGenerator:
    def __init__(self):
        self.calls: list[
            tuple[str, list[RetrievedDocument], list[Source]]
        ] = []

    def generate(
        self,
        question: str,
        documents: list[RetrievedDocument],
        sources: list[Source],
    ) -> str:
        self.calls.append((question, documents, sources))
        return "Resposta grounded [1]."


class StubGrader:
    def __init__(self, relevant: bool = True):
        self.calls: list[tuple[str, list[RetrievedDocument]]] = []
        self.relevant = relevant

    def grade(
        self,
        question: str,
        documents: list[RetrievedDocument],
    ) -> RagGrade:
        self.calls.append((question, documents))
        return RagGrade(
            relevant=self.relevant,
            confidence=0.9,
            reason=(
                "Contexto suficiente."
                if self.relevant
                else "Contexto insuficiente."
            ),
        )


class StubQueryRewriter:
    def __init__(self, rewritten_query: str = "consulta reescrita"):
        self.rewritten_query = rewritten_query
        self.calls: list[tuple[str, str, str]] = []

    def rewrite(
        self,
        question: str,
        search_query: str,
        failure_reason: str,
    ) -> str:
        self.calls.append((question, search_query, failure_reason))
        return self.rewritten_query


class StubQueryPreparer:
    def __init__(self, prepared_query: str | None = None):
        self.prepared_query = prepared_query
        self.calls: list[tuple[str, list[object]]] = []

    def prepare(self, question: str, history: list[object]) -> str:
        self.calls.append((question, history))
        return self.prepared_query or question


class StubWebSearch:
    def __init__(self, results: list[WebSearchResult] | None = None):
        self.results = results or []
        self.queries: list[str] = []

    def search(self, query: str) -> list[WebSearchResult]:
        self.queries.append(query)
        return self.results


class StubWebGrader:
    def __init__(self, relevant: bool = False):
        self.relevant = relevant
        self.calls: list[tuple[str, list[WebSearchResult]]] = []

    def grade(
        self,
        question: str,
        results: list[WebSearchResult],
    ) -> WebGrade:
        self.calls.append((question, results))
        return WebGrade(
            relevant=self.relevant,
            confidence=0.9,
            reason="Evidência web suficiente." if self.relevant else "Insuficiente.",
        )


def test_nodes_are_independently_callable() -> None:
    document = make_document()
    retriever = StubRetriever([document])
    grader = StubGrader()
    preparer = StubQueryPreparer()
    rewriter = StubQueryRewriter()
    generator = StubAnswerGenerator()
    nodes = RagNodes(
        retriever=retriever,
        grader=grader,
        query_preparer=preparer,
        query_rewriter=rewriter,
        web_search=StubWebSearch(),
        web_grader=StubWebGrader(),
        answer_generator=generator,
        max_rag_attempts=2,
    )

    prepared = nodes.prepare_query({"question": "  Como funciona?  "})
    retrieved = nodes.retrieve_rag(
        {
            "question": prepared["question"],
            "prepared_query": prepared["prepared_query"],
            "search_query": prepared["search_query"],
            "retrieval_attempts": prepared["retrieval_attempts"],
        }
    )
    graded = nodes.grade_rag(
        {
            "question": prepared["question"],
            "prepared_query": prepared["prepared_query"],
            "documents": retrieved["documents"],
        }
    )
    answered = nodes.generate_answer(
        {
            "question": prepared["question"],
            "prepared_query": prepared["prepared_query"],
            "documents": retrieved["documents"],
        }
    )

    assert prepared == {
        "question": "Como funciona?",
        "prepared_query": "Como funciona?",
        "search_query": "Como funciona?",
        "retrieval_attempts": 0,
        "web_results": [],
        "web_relevant": False,
        "sources": [],
    }
    assert preparer.calls == [("Como funciona?", [])]
    assert retriever.queries == ["Como funciona?"]
    assert retrieved["documents"] == [document]
    assert retrieved["retrieval_attempts"] == 1
    assert grader.calls == [("Como funciona?", [document])]
    assert graded == {
        "rag_relevant": True,
        "rag_confidence": 0.9,
        "rag_reason": "Contexto suficiente.",
    }
    assert nodes.route_after_grade(
        {**graded, "retrieval_attempts": 1}
    ) == "answer"
    assert len(generator.calls) == 1
    generated_question, generated_documents, generated_sources = generator.calls[0]
    assert generated_question == "Como funciona?"
    assert generated_documents == [document]
    assert [source.citation for source in generated_sources] == ["[1]"]
    assert answered["answer"] == "Resposta grounded [1]."
    assert answered["source"] == "rag"
    assert answered["sources"] == [
        {
            "id": 1,
            "citation": "[1]",
            "type": "rag",
            "title": "guia",
            "source": "guia.md",
            "chunkId": "chunk-1",
            "url": None,
            "page": None,
            "section": None,
        }
    ]
    assert [message.content for message in answered["messages"]] == [
        "Como funciona?",
        "Resposta grounded [1].",
    ]


def test_prepare_query_rejects_blank_question() -> None:
    nodes = RagNodes(
        retriever=StubRetriever([]),
        grader=StubGrader(),
        query_preparer=StubQueryPreparer(),
        query_rewriter=StubQueryRewriter(),
        web_search=StubWebSearch(),
        web_grader=StubWebGrader(),
        answer_generator=StubAnswerGenerator(),
        max_rag_attempts=2,
    )

    try:
        nodes.prepare_query({"question": "   "})
    except ValueError as error:
        assert str(error) == "question must not be empty"
    else:
        raise AssertionError("blank question should fail")


def test_visual_context_drives_retrieval_and_becomes_citable_evidence() -> None:
    retriever = StubRetriever([make_document()])
    preparer = StubQueryPreparer()
    nodes = RagNodes(
        retriever=retriever,
        grader=StubGrader(),
        query_preparer=preparer,
        query_rewriter=StubQueryRewriter(),
        web_search=StubWebSearch(),
        web_grader=StubWebGrader(),
        answer_generator=StubAnswerGenerator(),
        max_rag_attempts=1,
    )
    image_document = RetrievedDocument(
        chunk_id="user-image:abc",
        content="Imagem mostra uma ONT com o LED LOS vermelho.",
        metadata={
            "source": "user-image:abc",
            "title": "Imagem enviada: ont.png",
        },
        retrieval_score=1.0,
        rerank_score=1.0,
    )

    prepared = nodes.prepare_query(
        {
            "question": "O que devo fazer?",
            "image_context": '{"visible_text":["LOS"]}',
        }
    )
    retrieved = nodes.retrieve_rag(
        {
            **prepared,
            "question": "O que devo fazer?",
            "image_document": image_document,
        }
    )

    assert "O que devo fazer?" in str(prepared["prepared_query"])
    assert "visible_text" in str(prepared["prepared_query"])
    assert preparer.calls[0][0] == prepared["prepared_query"]
    assert retriever.queries == [prepared["search_query"]]
    assert retrieved["documents"] == [image_document, make_document()]
    assert retrieved["retrieval_diagnostics"].returned_count == 2


def test_visual_matches_filter_manuals_by_equipment_and_tenant() -> None:
    retriever = FilteringStubRetriever([make_document()])
    nodes = RagNodes(
        retriever=retriever,
        grader=StubGrader(),
        query_preparer=StubQueryPreparer(),
        query_rewriter=StubQueryRewriter(),
        web_search=StubWebSearch(),
        web_grader=StubWebGrader(),
        answer_generator=StubAnswerGenerator(),
        max_rag_attempts=1,
    )
    reference = RetrievedDocument(
        chunk_id="visual-ref:one",
        content="Candidato visual proprietário.",
        metadata={
            "source": "/references/device.png",
            "equipment_id": "OLT-9000",
        },
        retrieval_score=-0.1,
        rerank_score=0.9,
    )

    result = nodes.retrieve_rag(
        {
            "question": "Qual equipamento?",
            "prepared_query": "Qual equipamento?",
            "search_query": "equipamento da imagem",
            "retrieval_attempts": 0,
            "retrieval_filters": {"tenant_id": "public"},
            "image_reference_documents": [reference],
        }
    )

    assert retriever.filters == [
        {
            "$and": [
                {"tenant_id": "public"},
                {"equipment_id": {"$in": ["OLT-9000"]}},
            ]
        }
    ]
    assert result["documents"] == [reference, make_document()]
    assert result["retrieval_diagnostics"].returned_count == 2


def test_irrelevant_rag_routes_through_web_before_no_answer() -> None:
    rewriter = StubQueryRewriter()
    web_result = make_web_result()
    web_search = StubWebSearch([web_result])
    web_grader = StubWebGrader(relevant=False)
    nodes = RagNodes(
        retriever=StubRetriever([make_document()]),
        grader=StubGrader(relevant=False),
        query_preparer=StubQueryPreparer(),
        query_rewriter=rewriter,
        web_search=web_search,
        web_grader=web_grader,
        answer_generator=StubAnswerGenerator(),
        max_rag_attempts=2,
    )

    retry_state = {
        "question": "Pergunta original",
        "prepared_query": "Pergunta original autocontida",
        "search_query": "consulta original",
        "retrieval_attempts": 1,
        "rag_relevant": False,
        "rag_reason": "Faltam termos específicos.",
    }
    route = nodes.route_after_grade(retry_state)
    rewritten = nodes.rewrite_query(retry_state)
    terminal_state = {**retry_state, "retrieval_attempts": 2}
    terminal_route = nodes.route_after_grade(terminal_state)
    web_update = nodes.search_web(terminal_state)
    web_state = {**terminal_state, **web_update}
    web_grade = nodes.grade_web(web_state)
    web_route = nodes.route_after_web_grade({**web_state, **web_grade})
    result = nodes.no_answer({**web_state, **web_grade})

    assert route == "rewrite"
    assert rewritten == {"search_query": "consulta reescrita"}
    assert rewriter.calls == [
        (
            "Pergunta original autocontida",
            "consulta original",
            "Faltam termos específicos.",
        )
    ]
    assert terminal_route == "web"
    assert web_search.queries == ["consulta original"]
    assert web_update == {"web_results": [web_result]}
    assert web_grader.calls == [
        ("Pergunta original autocontida", [web_result])
    ]
    assert web_grade == {
        "web_relevant": False,
        "web_confidence": 0.9,
        "web_reason": "Insuficiente.",
    }
    assert web_route == "no_answer"
    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["source"] == "none"
    assert result["sources"] == []
    assert [message.content for message in result["messages"]] == [
        "Pergunta original",
        NO_CONTEXT_ANSWER,
    ]


def test_max_rag_attempts_must_be_positive() -> None:
    try:
        RagNodes(
            retriever=StubRetriever([]),
            grader=StubGrader(),
            query_preparer=StubQueryPreparer(),
            query_rewriter=StubQueryRewriter(),
            web_search=StubWebSearch(),
            web_grader=StubWebGrader(),
            answer_generator=StubAnswerGenerator(),
            max_rag_attempts=0,
        )
    except ValueError as error:
        assert str(error) == "max_rag_attempts must be at least 1"
    else:
        raise AssertionError("zero attempts should fail")
