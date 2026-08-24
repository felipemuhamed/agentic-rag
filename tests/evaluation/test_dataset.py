import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import load_dataset, load_report


def test_versioned_project_dataset_and_corpus_are_loadable() -> None:
    loaded = load_dataset("evaluation/datasets/project-docs-v1.json")
    baseline = load_report(
        "evaluation/baselines/offline-project-docs-v1.json"
    )

    assert loaded.dataset.name == "agentic-rag-project-docs"
    assert len(loaded.dataset.cases) == 6
    assert len(loaded.sha256) == 64
    corpus_sha256 = loaded.corpus_sha256(".")
    assert len(corpus_sha256) == 64
    assert all(path.is_file() for path in loaded.corpus_paths("."))
    assert baseline.dataset.sha256 == loaded.sha256
    assert baseline.dataset.corpus_sha256 == corpus_sha256


def test_dataset_rejects_expected_source_outside_corpus(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "invalid",
                "version": "1",
                "description": "Invalid source mapping",
                "corpus": ["docs/a.md"],
                "cases": [
                    {
                        "id": "case",
                        "question": "Question?",
                        "expected_documents": [{"source": "docs/b.md"}],
                        "reference_answer": "Answer.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="absent from corpus"):
        load_dataset(dataset_path)


def test_corpus_fingerprint_changes_with_document_content(tmp_path: Path) -> None:
    document = tmp_path / "source.md"
    document.write_text("First version", encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "fingerprint",
                "version": "1",
                "description": "Fingerprint test",
                "corpus": ["source.md"],
                "cases": [
                    {
                        "id": "case",
                        "question": "Question?",
                        "expected_documents": [{"source": "source.md"}],
                        "reference_answer": "Answer.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_dataset(dataset_path)
    first = loaded.corpus_sha256(tmp_path)

    document.write_text("Second version", encoding="utf-8")

    assert loaded.corpus_sha256(tmp_path) != first
