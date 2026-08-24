import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from app.evaluation.models import EvaluationDataset, EvaluationReport


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    dataset: EvaluationDataset
    path: Path
    sha256: str

    def corpus_paths(self, root: str | Path) -> list[Path]:
        corpus_root = Path(root).expanduser().resolve()
        paths: list[Path] = []
        for relative_path in self.dataset.corpus:
            path = (corpus_root / relative_path).resolve()
            if not path.is_relative_to(corpus_root):
                raise ValueError("corpus path escapes its configured root")
            if not path.is_file():
                raise FileNotFoundError(f"corpus document not found: {path}")
            paths.append(path)
        return paths

    def corpus_sha256(self, root: str | Path) -> str:
        digest = sha256()
        for relative_path, path in zip(
            self.dataset.corpus,
            self.corpus_paths(root),
            strict=True,
        ):
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()


def load_dataset(path: str | Path) -> LoadedDataset:
    dataset_path = Path(path).expanduser().resolve()
    content = dataset_path.read_bytes()
    raw = json.loads(content.decode("utf-8"))
    return LoadedDataset(
        dataset=EvaluationDataset.model_validate(raw),
        path=dataset_path,
        sha256=sha256(content).hexdigest(),
    )


def load_report(path: str | Path) -> EvaluationReport:
    report_path = Path(path).expanduser().resolve()
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    return EvaluationReport.model_validate(raw)
