from typing import Any

import pytest
from pydantic import ValidationError

from app.llm.web_grader import (
    NO_WEB_RESULTS_REASON,
    WEB_GRADE_INSTRUCTIONS,
    LangChainWebGrader,
    WebGrade,
)
from app.websearch.models import WebSearchResult


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


def make_result() -> WebSearchResult:
    return WebSearchResult(
        title="Keycloak documentation",
        url="https://www.keycloak.org/docs/oauth",
        content="Keycloak implements OAuth 2.0 endpoints.",
        score=0.94,
        provider="tavily",
    )


def test_web_grader_uses_validated_structured_output() -> None:
    model = FakeChatModel(
        {
            "relevant": True,
            "confidence": 0.93,
            "reason": "  A documentação responde diretamente.  ",
        }
    )
    grader = LangChainWebGrader(model)

    grade = grader.grade("Como configurar OAuth no Keycloak?", [make_result()])

    assert grade == WebGrade(
        relevant=True,
        confidence=0.93,
        reason="A documentação responde diretamente.",
    )
    assert model.schema is WebGrade
    assert model.structured.configs == [{"tags": ["web-grader", "nostream"]}]
    messages = model.structured.inputs[0]
    assert isinstance(messages, list)
    assert messages[0] == ("system", WEB_GRADE_INSTRUCTIONS)
    assert "Como configurar OAuth" in messages[1][1]
    assert "https://www.keycloak.org/docs/oauth" in messages[1][1]
    assert "Keycloak implements OAuth" in messages[1][1]


def test_web_grader_rejects_invalid_structured_output() -> None:
    grader = LangChainWebGrader(
        FakeChatModel(
            {
                "relevant": True,
                "confidence": 2.0,
                "reason": "Valor inválido.",
            }
        )
    )

    with pytest.raises(ValidationError):
        grader.grade("Pergunta", [make_result()])


def test_empty_web_results_are_insufficient_without_model_call() -> None:
    model = FakeChatModel({"invalid": "must not be used"})
    grader = LangChainWebGrader(model)

    grade = grader.grade("Pergunta", [])

    assert grade == WebGrade(
        relevant=False,
        confidence=1.0,
        reason=NO_WEB_RESULTS_REASON,
    )
    assert model.structured.inputs == []
