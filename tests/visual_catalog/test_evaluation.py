import base64
import json
from pathlib import Path

from app.knowledge.retrieval.models import RetrievedDocument
from app.visual_catalog.evaluation import (
    VisualEvaluationRunner,
    load_visual_dataset,
)
from app.visual_catalog.resolver import EquipmentResolver


# A 1x1 PNG: the runner validates the binary signature, so the bytes have to be
# a real image even though nothing ever decodes them here.
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKm"
    "MIQAAAABJRU5ErkJggg=="
)


class StubVisualRetriever:
    """Return fixed candidates per image name, keyed like the real retriever."""

    def __init__(self, candidates: dict[str, list[tuple[str, float]]]):
        self.candidates = candidates

    def retrieve(self, images, *, filters=None):
        name = images[0]["name"]
        return [
            RetrievedDocument(
                chunk_id=f"visual-ref:{equipment_id}",
                content="Referência visual.",
                metadata={
                    "equipment_id": equipment_id,
                    "equipment_name": equipment_id,
                    "model": _MODELS.get(equipment_id, ""),
                    "visual_distance": distance,
                },
                retrieval_score=-distance,
                rerank_score=max(0.0, 1.0 - distance),
            )
            for equipment_id, distance in self.candidates[name]
        ]


_MODELS = {"EQ-MOTO": "g86", "EQ-IPHONE": "iphone air"}


def make_runner(retriever: StubVisualRetriever) -> VisualEvaluationRunner:
    return VisualEvaluationRunner(
        retriever=retriever,
        resolver_factory=lambda: EquipmentResolver(
            identifier_weight=0.65,
            min_confidence=0.55,
            ambiguity_margin=0.08,
        ),
        identifier_weight=0.65,
        min_confidence=0.55,
        ambiguity_margin=0.08,
    )


def write_case_files(root: Path, names: list[str]) -> None:
    for name in names:
        (root / name).write_bytes(ONE_PIXEL_PNG)


def write_dataset(root: Path, cases: list[dict[str, object]]) -> Path:
    path = root / "dataset.json"
    path.write_text(
        json.dumps({"version": "test-v1", "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_sweep_recommends_the_threshold_that_separates_lookalikes(
    tmp_path: Path,
) -> None:
    write_case_files(tmp_path, ["moto.png", "iphone.png", "desconhecido.png"])
    dataset = write_dataset(
        tmp_path,
        [
            {"image": "moto.png", "expected_equipment_id": "EQ-MOTO"},
            {"image": "iphone.png", "expected_equipment_id": "EQ-IPHONE"},
            {"image": "desconhecido.png", "expected_equipment_id": None},
        ],
    )
    runner = make_runner(
        StubVisualRetriever(
            {
                "moto.png": [("EQ-MOTO", 0.40), ("EQ-IPHONE", 0.50)],
                "iphone.png": [("EQ-IPHONE", 0.40), ("EQ-MOTO", 0.50)],
                "desconhecido.png": [("EQ-MOTO", 1.30)],
            }
        )
    )

    report = runner.run(
        load_visual_dataset(dataset),
        images_root=tmp_path,
        thresholds=[0.45, 0.80],
    )

    # At 0.80 both lookalikes clear the gate and the resolver refuses to choose;
    # at 0.45 only the right one survives.
    assert report.recommended_max_distance == 0.45
    tight, loose = report.sweep
    assert (tight.correct, tight.wrong, tight.missed) == (2, 0, 0)
    assert tight.correct_abstentions == 1
    assert tight.accuracy == 1.0
    assert (loose.correct, loose.missed) == (0, 2)
    assert loose.score < tight.score


def test_read_identifiers_resolve_a_case_visual_distance_cannot(
    tmp_path: Path,
) -> None:
    write_case_files(tmp_path, ["moto.png"])
    dataset = write_dataset(
        tmp_path,
        [
            {
                "image": "moto.png",
                "expected_equipment_id": "EQ-MOTO",
                "identifiers": ["G86"],
            }
        ],
    )
    runner = make_runner(
        StubVisualRetriever(
            # The lookalike is the nearer of the two.
            {"moto.png": [("EQ-IPHONE", 0.40), ("EQ-MOTO", 0.50)]}
        )
    )

    report = runner.run(
        load_visual_dataset(dataset),
        images_root=tmp_path,
        thresholds=[0.80],
    )

    assert report.cases[0].resolved_equipment_id == "EQ-MOTO"
    assert report.cases[0].outcome == "correct"
    assert report.sweep[0].accuracy == 1.0


def test_identifying_an_uncatalogued_photo_counts_as_wrong(
    tmp_path: Path,
) -> None:
    write_case_files(tmp_path, ["estranho.png"])
    dataset = write_dataset(
        tmp_path,
        [{"image": "estranho.png", "expected_equipment_id": None}],
    )
    runner = make_runner(
        StubVisualRetriever({"estranho.png": [("EQ-MOTO", 0.10)]})
    )

    report = runner.run(
        load_visual_dataset(dataset),
        images_root=tmp_path,
        thresholds=[0.80],
    )

    assert report.cases[0].outcome == "wrong"
    assert report.sweep[0].wrong == 1
    assert report.sweep[0].false_positive_rate == 1.0
    assert report.negative_cases == 1


def test_report_records_the_dataset_digest_and_resolver_settings(
    tmp_path: Path,
) -> None:
    write_case_files(tmp_path, ["moto.png"])
    dataset = write_dataset(
        tmp_path,
        [{"image": "moto.png", "expected_equipment_id": "EQ-MOTO"}],
    )
    loaded = load_visual_dataset(dataset)
    runner = make_runner(
        StubVisualRetriever({"moto.png": [("EQ-MOTO", 0.10)]})
    )

    report = runner.run(loaded, images_root=tmp_path, thresholds=[0.80])

    # Comparing two runs is only meaningful when the dataset is identical.
    assert report.dataset_sha256 == loaded.sha256
    assert report.dataset_version == "test-v1"
    assert report.identifier_weight == 0.65
    assert report.min_confidence == 0.55
    assert report.ambiguity_margin == 0.08


def test_image_paths_cannot_escape_the_configured_root(tmp_path: Path) -> None:
    write_case_files(tmp_path, ["moto.png"])
    nested = tmp_path / "casos"
    nested.mkdir()
    dataset = write_dataset(
        nested,
        [{"image": "../moto.png", "expected_equipment_id": "EQ-MOTO"}],
    )
    runner = make_runner(
        StubVisualRetriever({"moto.png": [("EQ-MOTO", 0.10)]})
    )

    try:
        runner.run(
            load_visual_dataset(dataset),
            images_root=nested,
            thresholds=[0.80],
        )
    except ValueError as error:
        assert "escapes its configured root" in str(error)
    else:
        raise AssertionError("expected the traversal to be rejected")
