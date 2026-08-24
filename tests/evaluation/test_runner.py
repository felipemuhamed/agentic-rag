import json
from pathlib import Path

import pytest

from app.evaluation.dataset import load_dataset
from app.evaluation.models import (
    EvaluationConfiguration,
    GenerationUsage,
    Pricing,
)
from app.evaluation.providers import AnswerPrediction
from app.evaluation.runner import EvaluationRunner, compare_reports
from app.knowledge.retrieval.models import (
    RetrievalDiagnostics,
    RetrievalResult,
    RetrievedDocument,
)
from app.sources.normalizer import normalize_rag_sources


class FixedRetriever:
    def __init__(self, document: RetrievedDocument):
        self.document = document

    def retrieve(
        self,
        query: str,
        filters: dict[str, object] | None = None,
    ) -> RetrievalResult:
        assert query == "How does it work?"
        assert filters is None
        return RetrievalResult(
            documents=[self.document],
            diagnostics=RetrievalDiagnostics(1, 1, 1, 1),
        )


class FixedAnswerProvider:
    def __init__(self, document: RetrievedDocument):
        self.document = document

    def answer(
        self,
        question: str,
        documents: list[RetrievedDocument],
    ) -> AnswerPrediction:
        assert documents == [self.document]
        source = normalize_rag_sources(documents)[0].to_public()
        return AnswerPrediction(
            answer="The documented answer works [1].",
            sources=[source],
            usage=GenerationUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        )


def make_dataset(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "source.md"
    source.write_text("The documented answer works.", encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "runner",
                "version": "1",
                "description": "Runner test",
                "corpus": ["docs/source.md"],
                "cases": [
                    {
                        "id": "answer",
                        "question": "How does it work?",
                        "expected_documents": [{"source": "docs/source.md"}],
                        "reference_answer": "The documented answer works.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return load_dataset(dataset_path), source


def make_configuration(name: str, lambda_mult: float = 0.65):
    return EvaluationConfiguration(
        name=name,
        mode="offline",
        app_version="0.1.0",
        chunk_size=700,
        chunk_overlap=100,
        embedding_model="embedding-v1",
        reranker_model="reranker-v1",
        fetch_k=5,
        candidate_k=3,
        final_k=1,
        lambda_mult=lambda_mult,
        llm_model="answer-v1",
        llm_temperature=0.0,
        pricing=Pricing(input_per_million=1.0, output_per_million=2.0),
    )


def test_runner_builds_auditable_report_with_cost_and_latency(tmp_path: Path) -> None:
    loaded, source = make_dataset(tmp_path)
    document = RetrievedDocument(
        chunk_id="chunk-1",
        content="The documented answer works.",
        metadata={
            "document_id": "unstable-absolute-id",
            "source": str(source),
        },
        retrieval_score=0.8,
        rerank_score=0.9,
    )
    clock = iter([0.0, 0.010, 0.020, 0.025])
    report = EvaluationRunner(
        retriever=FixedRetriever(document),
        configuration=make_configuration("baseline"),
        k_values=[1],
        answer_provider=FixedAnswerProvider(document),
        clock=lambda: next(clock),
    ).run(loaded, corpus_root=str(tmp_path))

    assert report.summary.retrieval.recall_at_k == {"1": 1.0}
    assert report.summary.retrieval.mrr == 1.0
    assert report.summary.retrieval.latency.mean_ms == 10.0
    assert report.summary.answers is not None
    assert report.summary.answers.grounding == 1.0
    assert report.summary.answers.usage.estimated_cost == 0.0002
    assert report.cases[0].retrieved[0].source == "docs/source.md"
    assert report.cases[0].retrieved[0].document_id is None


def test_report_comparison_requires_same_inputs_and_returns_deltas(
    tmp_path: Path,
) -> None:
    loaded, source = make_dataset(tmp_path)
    document = RetrievedDocument(
        chunk_id="chunk-1",
        content="The documented answer works.",
        metadata={"source": str(source)},
        retrieval_score=0.8,
        rerank_score=0.9,
    )

    def run(name: str, lambda_mult: float, times: list[float]):
        clock = iter(times)
        return EvaluationRunner(
            retriever=FixedRetriever(document),
            configuration=make_configuration(name, lambda_mult),
            k_values=[1],
            answer_provider=FixedAnswerProvider(document),
            clock=lambda: next(clock),
        ).run(loaded, corpus_root=str(tmp_path))

    baseline = run("baseline", 0.65, [0.0, 0.010, 0.020, 0.025])
    candidate = run("candidate", 0.75, [0.0, 0.020, 0.030, 0.040])
    comparison = compare_reports(baseline, candidate)

    retrieval = comparison["metric_deltas"]["retrieval"]
    assert retrieval["mrr"] == 0.0
    assert retrieval["mean_latency_ms"] == 10.0
    assert comparison["configuration_changes"]["lambda_mult"] == {
        "baseline": 0.65,
        "candidate": 0.75,
    }

    other_dataset = candidate.model_copy(
        update={
            "dataset": candidate.dataset.model_copy(
                update={"sha256": "different"}
            )
        }
    )
    with pytest.raises(ValueError, match="different dataset revisions"):
        compare_reports(baseline, other_dataset)

    other_k = candidate.model_copy(
        update={
            "summary": candidate.summary.model_copy(
                update={
                    "retrieval": candidate.summary.retrieval.model_copy(
                        update={"recall_at_k": {"1": 1.0, "2": 1.0}}
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="different Recall@K"):
        compare_reports(baseline, other_k)
