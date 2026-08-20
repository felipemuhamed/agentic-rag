from typing import Any

import pytest
from pydantic import ValidationError

from app.knowledge.retrieval.models import RetrievedDocument
from app.llm.rag_grader import (
    GRADE_INSTRUCTIONS,
    LangChainRagGrader,
    NO_DOCUMENTS_REASON,
    RagGrade,
)


class FakeStructuredRunnable:
    def __init__(self, response: object):
        self.response = response
        self.inputs: list[object] = []
        self.configs: list[dict[str, Any]] = []

    def with_config(self, config: dict[str, Any]) -> "FakeStructuredRunnable":
        self.configs.append(config)
        return self

    def invoke(self, input: object) -> object:
        self.inputs.append(input)
        return self.response


class FakeChatModel:
    def __init__(self, response: object):
        self.schema: type[object] | None = None
        self.structured = FakeStructuredRunnable(response)

    def with_structured_output(
        self,
        schema: type[object],
    ) -> FakeStructuredRunnable:
        self.schema = schema
        return self.structured


def make_document() -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id="policy-1",
        content="Os registros são mantidos por 30 dias.",
        metadata={"source": "politica.md"},
        retrieval_score=0.8,
        rerank_score=0.9,
    )


def test_grader_uses_validated_structured_output() -> None:
    model = FakeChatModel(
        {
            "relevant": True,
            "confidence": 0.92,
            "reason": "  O trecho informa diretamente o prazo.  ",
        }
    )
    grader = LangChainRagGrader(model)

    grade = grader.grade(
        "Por quanto tempo os registros são mantidos?",
        [make_document()],
    )

    assert grade == RagGrade(
        relevant=True,
        confidence=0.92,
        reason="O trecho informa diretamente o prazo.",
    )
    assert model.schema is RagGrade
    assert model.structured.configs == [{"tags": ["rag-grader", "nostream"]}]
    messages = model.structured.inputs[0]
    assert isinstance(messages, list)
    assert messages[0] == ("system", GRADE_INSTRUCTIONS)
    assert "Por quanto tempo" in messages[1][1]
    assert "30 dias" in messages[1][1]
    assert '"chunk_id": "policy-1"' in messages[1][1]


def test_grader_rejects_invalid_structured_output() -> None:
    model = FakeChatModel(
        {
            "relevant": True,
            "confidence": 1.5,
            "reason": "Confiança fora do intervalo.",
        }
    )
    grader = LangChainRagGrader(model)

    with pytest.raises(ValidationError):
        grader.grade("Pergunta", [make_document()])


def test_empty_retrieval_is_insufficient_without_model_call() -> None:
    model = FakeChatModel({"invalid": "must not be used"})
    grader = LangChainRagGrader(model)

    grade = grader.grade("Pergunta sem documentos", [])

    assert grade == RagGrade(
        relevant=False,
        confidence=1.0,
        reason=NO_DOCUMENTS_REASON,
    )
    assert model.structured.inputs == []
