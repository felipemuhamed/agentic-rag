from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from statistics import mean
from time import perf_counter
from typing import Any, Protocol

from app.evaluation.dataset import LoadedDataset
from app.evaluation.metrics import evaluate_answer, recall_at_k, reciprocal_rank
from app.evaluation.models import (
    AnswerCaseMetrics,
    AnswerSummary,
    DatasetIdentity,
    EvaluationCaseReport,
    EvaluationConfiguration,
    EvaluationReport,
    EvaluationSummary,
    LatencySummary,
    RetrievedHit,
    RetrievalCaseMetrics,
    RetrievalSummary,
    UsageSummary,
)
from app.evaluation.providers import EvaluationAnswerProvider
from app.knowledge.retrieval.models import RetrievedDocument, RetrievalResult


class EvaluationRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult: ...


class EvaluationRunner:
    def __init__(
        self,
        *,
        retriever: EvaluationRetriever,
        configuration: EvaluationConfiguration,
        k_values: Sequence[int] = (1, 3, 5),
        answer_provider: EvaluationAnswerProvider | None = None,
        clock: Callable[[], float] = perf_counter,
    ):
        if any(
            isinstance(k, bool) or not isinstance(k, int) or k < 1
            for k in k_values
        ):
            raise ValueError("k_values must contain positive integers")
        normalized_k = tuple(sorted(set(k_values)))
        if not normalized_k:
            raise ValueError("k_values must contain positive integers")
        self._retriever = retriever
        self._configuration = configuration
        self._k_values = normalized_k
        self._answer_provider = answer_provider
        self._clock = clock

    def run(
        self,
        loaded_dataset: LoadedDataset,
        *,
        corpus_root: str,
    ) -> EvaluationReport:
        case_reports: list[EvaluationCaseReport] = []
        for case in loaded_dataset.dataset.cases:
            retrieval_start = self._clock()
            result = self._retriever.retrieve(case.question, filters=case.filters)
            retrieval_latency = _milliseconds(self._clock() - retrieval_start)
            recall = {
                str(k): _score(
                    recall_at_k(result.documents, case.expected_documents, k)
                )
                for k in self._k_values
            }
            retrieval_metrics = RetrievalCaseMetrics(
                recall_at_k=recall,
                reciprocal_rank=_score(
                    reciprocal_rank(result.documents, case.expected_documents)
                ),
                latency_ms=retrieval_latency,
            )

            answer = None
            answer_metrics = None
            if self._answer_provider is not None:
                answer_start = self._clock()
                prediction = self._answer_provider.answer(
                    case.question,
                    result.documents,
                )
                answer_latency = _milliseconds(self._clock() - answer_start)
                answer = prediction.answer
                answer_metrics = evaluate_answer(
                    answer=prediction.answer,
                    reference_answer=case.reference_answer,
                    documents=result.documents,
                    public_sources=prediction.sources,
                    expected_documents=case.expected_documents,
                    latency_ms=answer_latency,
                    usage=prediction.usage,
                )

            case_reports.append(
                EvaluationCaseReport(
                    id=case.id,
                    question=case.question,
                    retrieved=[
                        _retrieved_hit(
                            rank,
                            document,
                            loaded_dataset.dataset.corpus,
                        )
                        for rank, document in enumerate(result.documents, start=1)
                    ],
                    retrieval=retrieval_metrics,
                    answer=answer,
                    answer_metrics=answer_metrics,
                )
            )

        retrieval_summary = RetrievalSummary(
            recall_at_k={
                str(k): _score(
                    mean(
                        report.retrieval.recall_at_k[str(k)]
                        for report in case_reports
                    )
                )
                for k in self._k_values
            },
            mrr=_score(
                mean(report.retrieval.reciprocal_rank for report in case_reports)
            ),
            latency=_latency_summary(
                [report.retrieval.latency_ms for report in case_reports]
            ),
        )
        answer_reports = [
            report.answer_metrics
            for report in case_reports
            if report.answer_metrics is not None
        ]
        answers = (
            _answer_summary(answer_reports, self._configuration)
            if answer_reports
            else None
        )
        dataset = loaded_dataset.dataset
        return EvaluationReport(
            generated_at=datetime.now(UTC).isoformat(),
            dataset=DatasetIdentity(
                name=dataset.name,
                version=dataset.version,
                sha256=loaded_dataset.sha256,
                corpus_sha256=loaded_dataset.corpus_sha256(corpus_root),
                cases=len(dataset.cases),
            ),
            configuration=self._configuration,
            summary=EvaluationSummary(
                retrieval=retrieval_summary,
                answers=answers,
            ),
            cases=case_reports,
        )


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
) -> dict[str, object]:
    if baseline.dataset.sha256 != candidate.dataset.sha256:
        raise ValueError("reports use different dataset revisions")
    if baseline.dataset.corpus_sha256 != candidate.dataset.corpus_sha256:
        raise ValueError("reports use different corpus revisions")

    baseline_retrieval = baseline.summary.retrieval
    candidate_retrieval = candidate.summary.retrieval
    if set(baseline_retrieval.recall_at_k) != set(
        candidate_retrieval.recall_at_k
    ):
        raise ValueError("reports use different Recall@K values")

    answer_deltas: dict[str, object] | None = None
    if (
        baseline.summary.answers is not None
        and candidate.summary.answers is not None
    ):
        answer_deltas = {
            field: _delta(
                getattr(baseline.summary.answers, field),
                getattr(candidate.summary.answers, field),
            )
            for field in (
                "relevance",
                "grounding",
                "citation_validity",
                "expected_source_recall",
            )
        }
        answer_deltas["mean_latency_ms"] = _delta(
            baseline.summary.answers.latency.mean_ms,
            candidate.summary.answers.latency.mean_ms,
        )
        answer_deltas["usage"] = _usage_deltas(
            baseline.summary.answers.usage,
            candidate.summary.answers.usage,
        )

    baseline_config = baseline.configuration.model_dump(mode="json")
    candidate_config = candidate.configuration.model_dump(mode="json")
    changed_configuration = {
        field: {
            "baseline": baseline_config[field],
            "candidate": candidate_config[field],
        }
        for field in baseline_config
        if baseline_config[field] != candidate_config[field]
    }
    return {
        "schema_version": "1.0",
        "dataset": baseline.dataset.model_dump(mode="json"),
        "baseline": baseline.configuration.name,
        "candidate": candidate.configuration.name,
        "metric_deltas": {
            "retrieval": {
                "recall_at_k": {
                    k: _delta(
                        baseline_retrieval.recall_at_k[k],
                        candidate_retrieval.recall_at_k[k],
                    )
                    for k in baseline_retrieval.recall_at_k
                },
                "mrr": _delta(baseline_retrieval.mrr, candidate_retrieval.mrr),
                "mean_latency_ms": _delta(
                    baseline_retrieval.latency.mean_ms,
                    candidate_retrieval.latency.mean_ms,
                ),
            },
            "answers": answer_deltas,
        },
        "configuration_changes": changed_configuration,
    }


def _retrieved_hit(
    rank: int,
    document: RetrievedDocument,
    corpus_sources: Sequence[str],
) -> RetrievedHit:
    metadata = document.metadata
    document_id = metadata.get("document_id")
    source = metadata.get("source")
    chunk_index = metadata.get("chunk_index")
    portable_source = (
        _portable_source(source, corpus_sources) if isinstance(source, str) else None
    )
    portable_chunk_id = (
        f"{portable_source}#chunk:{chunk_index}"
        if portable_source is not None
        and isinstance(chunk_index, int)
        and not isinstance(chunk_index, bool)
        else None
    )
    return RetrievedHit(
        rank=rank,
        chunk_id=portable_chunk_id or document.chunk_id,
        document_id=(
            document_id
            if isinstance(document_id, str) and portable_source is None
            else None
        ),
        source=portable_source or (source if isinstance(source, str) else None),
        retrieval_score=float(document.retrieval_score),
        rerank_score=float(document.rerank_score),
    )


def _answer_summary(
    reports: list[AnswerCaseMetrics],
    configuration: EvaluationConfiguration,
) -> AnswerSummary:
    usages = [report.usage for report in reports]
    usage_complete = all(
        usage.input_tokens is not None and usage.output_tokens is not None
        for usage in usages
    )
    input_tokens = (
        sum(usage.input_tokens for usage in usages if usage.input_tokens is not None)
        if usage_complete
        else None
    )
    output_tokens = (
        sum(usage.output_tokens for usage in usages if usage.output_tokens is not None)
        if usage_complete
        else None
    )
    total_tokens = (
        sum(
            usage.total_tokens
            if usage.total_tokens is not None
            else (usage.input_tokens or 0) + (usage.output_tokens or 0)
            for usage in usages
        )
        if usage_complete
        else None
    )
    pricing = configuration.pricing
    estimated_cost = None
    if pricing is not None and input_tokens is not None and output_tokens is not None:
        estimated_cost = _score(
            (input_tokens * pricing.input_per_million / 1_000_000)
            + (output_tokens * pricing.output_per_million / 1_000_000)
        )
    return AnswerSummary(
        evaluated_cases=len(reports),
        relevance=_score(mean(report.relevance for report in reports)),
        grounding=_score(mean(report.grounding for report in reports)),
        citation_validity=_score(
            mean(report.citation_validity for report in reports)
        ),
        expected_source_recall=_score(
            mean(report.expected_source_recall for report in reports)
        ),
        latency=_latency_summary(
            [report.latency_ms for report in reports]
        ),
        usage=UsageSummary(
            calls=len(reports),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            complete=usage_complete,
            estimated_cost=estimated_cost,
            currency=pricing.currency if estimated_cost is not None else None,
        ),
    )


def _latency_summary(values: list[float]) -> LatencySummary:
    return LatencySummary(
        mean_ms=_score(mean(values), 3),
        p50_ms=_score(_percentile(values, 0.50), 3),
        p95_ms=_score(_percentile(values, 0.95), 3),
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _milliseconds(seconds: float) -> float:
    return _score(max(seconds, 0.0) * 1000, 3)


def _score(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _delta(baseline: float, candidate: float) -> float:
    return _score(candidate - baseline)


def _portable_source(
    source: str,
    corpus_sources: Sequence[str],
) -> str | None:
    normalized = source.replace("\\", "/")
    for corpus_source in corpus_sources:
        if normalized == corpus_source or normalized.endswith(f"/{corpus_source}"):
            return corpus_source
    return None


def _usage_deltas(
    baseline: UsageSummary,
    candidate: UsageSummary,
) -> dict[str, object]:
    deltas: dict[str, object] = {
        "calls": candidate.calls - baseline.calls,
        "input_tokens": _optional_int_delta(
            baseline.input_tokens,
            candidate.input_tokens,
        ),
        "output_tokens": _optional_int_delta(
            baseline.output_tokens,
            candidate.output_tokens,
        ),
        "total_tokens": _optional_int_delta(
            baseline.total_tokens,
            candidate.total_tokens,
        ),
        "estimated_cost": None,
    }
    if (
        baseline.estimated_cost is not None
        and candidate.estimated_cost is not None
        and baseline.currency == candidate.currency
    ):
        deltas["estimated_cost"] = _delta(
            baseline.estimated_cost,
            candidate.estimated_cost,
        )
        deltas["currency"] = baseline.currency
    return deltas


def _optional_int_delta(baseline: int | None, candidate: int | None) -> int | None:
    if baseline is None or candidate is None:
        return None
    return candidate - baseline
