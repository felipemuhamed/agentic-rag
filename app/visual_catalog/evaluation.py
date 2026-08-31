"""Measure whether a photo identifies the right equipment.

The textual pipeline has had a versioned evaluation since the retrieval stage
was accepted; visual identification shipped without one, which left the distance
gate to be chosen by hand against a couple of sample photos. This module turns
that judgement call into a reproducible number.

Two properties matter more than raw accuracy. A **wrong** identification is worse
than no identification, because it restricts the manual search to the wrong
equipment and hides the correct document entirely; and an **abstention** is a
successful outcome for a photo of something the catalogue does not contain. The
metrics below keep those cases apart instead of collapsing them into one score.

Each image is embedded and queried once, with the gate wide open; the sweep then
re-applies every candidate threshold to those stored distances in memory. So
adding thresholds costs no extra model work.
"""

import base64
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.knowledge.retrieval.models import RetrievedDocument
from app.multimodal.models import ChatImage, ImageInput, validate_image_inputs
from app.visual_catalog.ingestion import SUPPORTED_REFERENCE_IMAGES
from app.visual_catalog.resolver import EquipmentResolver


CaseOutcome = Literal[
    "correct",
    "wrong",
    "missed",
    "correct_abstention",
]

DEFAULT_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VisualCase(StrictModel):
    image: str = Field(min_length=1)
    # None marks a negative case: equipment absent from the catalogue, where
    # abstaining is the correct behaviour.
    expected_equipment_id: str | None = None
    # Recorded OCR readings. Keeping them in the dataset lets the gate run
    # without credentials for the vision model, the same way the textual
    # evaluation has an offline baseline.
    identifiers: list[str] = Field(default_factory=list)
    note: str = ""


class VisualDataset(StrictModel):
    version: str = Field(min_length=1)
    cases: list[VisualCase] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class LoadedVisualDataset:
    dataset: VisualDataset
    path: Path
    sha256: str


class CaseCandidate(StrictModel):
    equipment_id: str
    visual_distance: float


class CaseReport(StrictModel):
    image: str
    expected_equipment_id: str | None
    resolved_equipment_id: str | None
    confidence: float
    ambiguous: bool
    reason: str
    outcome: CaseOutcome
    candidates: list[CaseCandidate]


class ThresholdMetrics(StrictModel):
    max_distance: float
    correct: int
    wrong: int
    missed: int
    correct_abstentions: int
    accuracy: float
    false_positive_rate: float
    abstention_rate: float
    score: float


class VisualEvaluationReport(StrictModel):
    dataset_version: str
    dataset_sha256: str
    case_count: int
    positive_cases: int
    negative_cases: int
    identifier_weight: float
    min_confidence: float
    ambiguity_margin: float
    recommended_max_distance: float
    sweep: list[ThresholdMetrics]
    cases: list[CaseReport]


class SupportsVisualRetrieval(Protocol):
    def retrieve(
        self,
        images: Sequence[ImageInput],
        *,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedDocument]: ...


class VisualEvaluationRunner:
    def __init__(
        self,
        *,
        retriever: SupportsVisualRetrieval,
        resolver_factory: Callable[[], EquipmentResolver],
        identifier_weight: float,
        min_confidence: float,
        ambiguity_margin: float,
    ):
        self._retriever = retriever
        self._resolver_factory = resolver_factory
        self._identifier_weight = identifier_weight
        self._min_confidence = min_confidence
        self._ambiguity_margin = ambiguity_margin

    def run(
        self,
        loaded: LoadedVisualDataset,
        *,
        images_root: str | Path,
        thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    ) -> VisualEvaluationReport:
        if not thresholds:
            raise ValueError("at least one threshold is required")

        resolver = self._resolver_factory()
        observations = [
            (case, self._candidates(case, images_root))
            for case in loaded.dataset.cases
        ]

        sweep = [
            _metrics(
                max_distance,
                [
                    _judge(
                        case,
                        resolver,
                        _within(candidates, max_distance),
                    )
                    for case, candidates in observations
                ],
            )
            for max_distance in sorted(set(thresholds))
        ]
        best = max(
            sweep,
            # Ties go to the tighter gate: it admits fewer lookalikes on photos
            # the dataset does not cover yet.
            key=lambda metrics: (metrics.score, -metrics.max_distance),
        )
        cases = [
            _judge(case, resolver, _within(candidates, best.max_distance))
            for case, candidates in observations
        ]
        positives = sum(
            1
            for case in loaded.dataset.cases
            if case.expected_equipment_id is not None
        )
        return VisualEvaluationReport(
            dataset_version=loaded.dataset.version,
            dataset_sha256=loaded.sha256,
            case_count=len(loaded.dataset.cases),
            positive_cases=positives,
            negative_cases=len(loaded.dataset.cases) - positives,
            identifier_weight=self._identifier_weight,
            min_confidence=self._min_confidence,
            ambiguity_margin=self._ambiguity_margin,
            recommended_max_distance=best.max_distance,
            sweep=sweep,
            cases=cases,
        )

    def _candidates(
        self,
        case: VisualCase,
        images_root: str | Path,
    ) -> list[RetrievedDocument]:
        return self._retriever.retrieve([_load_image(case.image, images_root)])


def load_visual_dataset(path: str | Path) -> LoadedVisualDataset:
    dataset_path = Path(path).expanduser().resolve()
    content = dataset_path.read_bytes()
    return LoadedVisualDataset(
        dataset=VisualDataset.model_validate(json.loads(content.decode("utf-8"))),
        path=dataset_path,
        sha256=sha256(content).hexdigest(),
    )


def _load_image(relative_path: str, images_root: str | Path) -> ImageInput:
    root = Path(images_root).expanduser().resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("image path escapes its configured root")
    try:
        mime_type = SUPPORTED_REFERENCE_IMAGES[path.suffix.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(SUPPORTED_REFERENCE_IMAGES))
        raise ValueError(
            f"unsupported evaluation image; expected one of: {supported}"
        ) from error
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    validated = validate_image_inputs(
        [ChatImage(name=path.name, mimeType=mime_type, data=data)],
        max_images=1,
        max_image_bytes=50_000_000,
    )
    return validated[0]


def _within(
    candidates: Sequence[RetrievedDocument],
    max_distance: float,
) -> list[RetrievedDocument]:
    return [
        candidate
        for candidate in candidates
        if _distance(candidate) <= max_distance
    ]


def _distance(candidate: RetrievedDocument) -> float:
    value = candidate.metadata.get("visual_distance")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float("inf")


def _judge(
    case: VisualCase,
    resolver: EquipmentResolver,
    candidates: Sequence[RetrievedDocument],
) -> CaseReport:
    resolution = resolver.resolve(case.identifiers, candidates)
    resolved = (
        None if resolution.resolved is None else resolution.resolved.equipment_id
    )
    if case.expected_equipment_id is None:
        outcome: CaseOutcome = "correct_abstention" if resolved is None else "wrong"
    elif resolved is None:
        outcome = "missed"
    elif resolved == case.expected_equipment_id:
        outcome = "correct"
    else:
        outcome = "wrong"
    return CaseReport(
        image=case.image,
        expected_equipment_id=case.expected_equipment_id,
        resolved_equipment_id=resolved,
        confidence=(
            0.0 if resolution.resolved is None else resolution.resolved.confidence
        ),
        ambiguous=resolution.ambiguous,
        reason=resolution.reason,
        outcome=outcome,
        candidates=[
            CaseCandidate(
                equipment_id=candidate.equipment_id,
                visual_distance=_distance(candidate.document),
            )
            for candidate in resolution.candidates
        ],
    )


def _metrics(
    max_distance: float,
    reports: Sequence[CaseReport],
) -> ThresholdMetrics:
    total = len(reports)
    counted = {
        outcome: sum(1 for report in reports if report.outcome == outcome)
        for outcome in ("correct", "wrong", "missed", "correct_abstention")
    }
    positives = sum(
        1 for report in reports if report.expected_equipment_id is not None
    )
    accuracy = counted["correct"] / positives if positives else 0.0
    false_positive_rate = counted["wrong"] / total if total else 0.0
    abstentions = counted["missed"] + counted["correct_abstention"]
    return ThresholdMetrics(
        max_distance=max_distance,
        correct=counted["correct"],
        wrong=counted["wrong"],
        missed=counted["missed"],
        correct_abstentions=counted["correct_abstention"],
        accuracy=round(accuracy, 6),
        false_positive_rate=round(false_positive_rate, 6),
        abstention_rate=round(abstentions / total, 6) if total else 0.0,
        # A wrong identification costs exactly what a right one earns: it sends
        # the manual search to another equipment, so it is not a neutral miss.
        score=round(accuracy - false_positive_rate, 6),
    )
