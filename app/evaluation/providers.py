import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.evaluation.metrics import lexical_tokens
from app.evaluation.models import GenerationUsage
from app.knowledge.retrieval.models import RetrievedDocument
from app.llm.answer_generator import AnswerGeneration, AnswerGenerator
from app.sources.citations import select_cited_sources
from app.sources.models import PublicSource
from app.sources.normalizer import normalize_rag_sources


@dataclass(frozen=True, slots=True)
class AnswerPrediction:
    answer: str
    sources: list[PublicSource]
    usage: GenerationUsage


class EvaluationAnswerProvider(Protocol):
    def answer(
        self,
        question: str,
        documents: Sequence[RetrievedDocument],
    ) -> AnswerPrediction: ...


class GroundedAnswerProvider:
    def __init__(self, generator: AnswerGenerator):
        self._generator = generator

    def answer(
        self,
        question: str,
        documents: Sequence[RetrievedDocument],
    ) -> AnswerPrediction:
        retrieved = list(documents)
        sources = normalize_rag_sources(retrieved)
        generate_with_metadata = getattr(
            self._generator,
            "generate_with_metadata",
            None,
        )
        if callable(generate_with_metadata):
            generation = generate_with_metadata(question, retrieved, sources)
            if not isinstance(generation, AnswerGeneration):
                raise TypeError("generate_with_metadata returned an invalid result")
        else:
            generation = AnswerGeneration(
                answer=self._generator.generate(question, retrieved, sources)
            )

        try:
            cited_sources = select_cited_sources(generation.answer, sources)
        except ValueError:
            # Evaluation scores invalid output instead of aborting the dataset.
            # The production graph and SSE boundary still fail closed.
            cited_sources = sources
        return AnswerPrediction(
            answer=generation.answer,
            sources=[source.to_public() for source in cited_sources],
            usage=GenerationUsage(
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                total_tokens=generation.total_tokens,
            ),
        )


class ExtractiveAnswerProvider:
    """Dependency-free answer baseline; never used by the production graph."""

    _sentence_boundary = re.compile(r"(?<=[.!?])\s+|\n+")
    _existing_citation = re.compile(r"\[\d+\]")

    def answer(
        self,
        question: str,
        documents: Sequence[RetrievedDocument],
    ) -> AnswerPrediction:
        if not documents:
            return AnswerPrediction(
                answer="Não encontrei evidências suficientes para responder com segurança.",
                sources=[],
                usage=GenerationUsage(
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                ),
            )

        question_tokens = set(lexical_tokens(question))
        candidates: list[tuple[float, int, int, str]] = []
        for document_index, document in enumerate(documents):
            for sentence_index, raw_sentence in enumerate(
                self._sentence_boundary.split(document.content)
            ):
                sentence = _clean_sentence(raw_sentence)
                tokens = set(lexical_tokens(sentence))
                if len(tokens) < 3:
                    continue
                overlap = len(question_tokens & tokens)
                score = overlap / max(len(question_tokens), 1)
                candidates.append(
                    (score, -document_index, -sentence_index, sentence)
                )

        if candidates:
            _, negative_document_index, _, sentence = max(candidates)
            document_index = -negative_document_index
        else:
            document_index = 0
            sentence = _clean_sentence(documents[0].content)

        sentence = self._existing_citation.sub("", sentence).strip()
        citation_id = document_index + 1
        sources = normalize_rag_sources(list(documents))
        cited_source = sources[document_index]
        return AnswerPrediction(
            answer=f"{sentence} [{citation_id}]",
            sources=[cited_source.to_public()],
            usage=GenerationUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            ),
        )


def _clean_sentence(value: str) -> str:
    return value.strip().lstrip("#-* ").replace("`", "")
