from app.evaluation.providers import GroundedAnswerProvider
from app.knowledge.retrieval.models import RetrievedDocument


class InvalidCitationGenerator:
    def generate(self, question, documents, sources) -> str:
        return "Answer with an unknown citation [2]."


def test_evaluation_keeps_invalid_answer_available_for_scoring() -> None:
    document = RetrievedDocument(
        chunk_id="chunk-1",
        content="Supported answer.",
        metadata={"source": "docs/source.md"},
        retrieval_score=0.8,
        rerank_score=0.9,
    )

    prediction = GroundedAnswerProvider(InvalidCitationGenerator()).answer(
        "Question?",
        [document],
    )

    assert prediction.answer.endswith("[2].")
    assert [source["citation"] for source in prediction.sources] == ["[1]"]
