import argparse
import json
from pathlib import Path
from typing import Sequence

from app.evaluation.dataset import load_dataset, load_report
from app.evaluation.models import EvaluationConfiguration, Pricing
from app.evaluation.offline import (
    OFFLINE_EMBEDDING_MODEL,
    OFFLINE_RERANKER_MODEL,
    create_offline_retriever,
)
from app.evaluation.providers import (
    ExtractiveAnswerProvider,
    GroundedAnswerProvider,
)
from app.evaluation.runner import EvaluationRunner, compare_reports
from app.knowledge.retrieval.factory import create_knowledge_retriever
from app.llm.answer_generator import LangChainAnswerGenerator
from app.llm.models import create_chat_model
from app.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a versioned offline evaluation of retrieval and answers."
    )
    parser.add_argument("dataset", help="Path to the versioned evaluation dataset")
    parser.add_argument("--corpus-root", default=".")
    parser.add_argument("--name", default="candidate")
    parser.add_argument(
        "--output",
        help="Write the complete JSON report to this path",
    )
    parser.add_argument("--baseline", help="Compare with an existing JSON report")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pricing = _pricing(args)
    config = Settings()
    loaded_dataset = load_dataset(args.dataset)
    corpus_paths = loaded_dataset.corpus_paths(args.corpus_root)

    if args.offline:
        retriever = create_offline_retriever(
            corpus_paths=corpus_paths,
            chunk_size=config.ingestion_chunk_size,
            chunk_overlap=config.ingestion_chunk_overlap,
            fetch_k=config.rag_fetch_k,
            candidate_k=config.rag_candidate_k,
            final_k=config.rag_final_k,
            lambda_mult=config.rag_lambda_mult,
        )
        answer_provider = (
            None if args.retrieval_only else ExtractiveAnswerProvider()
        )
        configuration = _configuration(
            args=args,
            config=config,
            mode="offline",
            embedding_model=OFFLINE_EMBEDDING_MODEL,
            reranker_model=OFFLINE_RERANKER_MODEL,
            llm_model=None
            if args.retrieval_only
            else "evaluation:extractive-v1",
            pricing=pricing,
        )
    else:
        answer_provider = None
        llm_model = None
        if not args.retrieval_only:
            model = create_chat_model(
                config.llm_model,
                temperature=args.temperature,
            )
            answer_provider = GroundedAnswerProvider(
                LangChainAnswerGenerator(model)
            )
            llm_model = config.llm_model
        retriever = create_knowledge_retriever(config)
        configuration = _configuration(
            args=args,
            config=config,
            mode="live",
            embedding_model=config.embedding_model,
            reranker_model=config.reranker_model,
            llm_model=llm_model,
            pricing=pricing,
        )

    report = EvaluationRunner(
        retriever=retriever,
        configuration=configuration,
        k_values=args.k,
        answer_provider=answer_provider,
    ).run(loaded_dataset, corpus_root=args.corpus_root)
    report_payload = report.model_dump(mode="json")

    comparison = None
    if args.baseline:
        comparison = compare_reports(load_report(args.baseline), report)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stdout_payload: dict[str, object] = {
            "report": str(output_path),
            "summary": report.summary.model_dump(mode="json"),
        }
    else:
        stdout_payload = {"report": report_payload}
    if comparison is not None:
        stdout_payload["comparison"] = comparison
    print(json.dumps(stdout_payload, ensure_ascii=False, indent=2))
    return 0


def _pricing(args: argparse.Namespace) -> Pricing | None:
    input_price = args.input_cost_per_million
    output_price = args.output_cost_per_million
    if (input_price is None) != (output_price is None):
        raise ValueError("both input and output token prices must be provided")
    if input_price is None:
        return None
    return Pricing(
        input_per_million=input_price,
        output_per_million=output_price,
    )


def _configuration(
    *,
    args: argparse.Namespace,
    config: Settings,
    mode: str,
    embedding_model: str,
    reranker_model: str,
    llm_model: str | None,
    pricing: Pricing | None,
) -> EvaluationConfiguration:
    return EvaluationConfiguration(
        name=args.name,
        mode=mode,
        app_version=config.app_version,
        chunk_size=config.ingestion_chunk_size,
        chunk_overlap=config.ingestion_chunk_overlap,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        fetch_k=config.rag_fetch_k,
        candidate_k=config.rag_candidate_k,
        final_k=config.rag_final_k,
        lambda_mult=config.rag_lambda_mult,
        collection_name=config.chroma_collection if mode == "live" else None,
        llm_model=llm_model,
        llm_temperature=args.temperature if llm_model is not None else None,
        pricing=pricing,
    )


if __name__ == "__main__":
    raise SystemExit(main())
