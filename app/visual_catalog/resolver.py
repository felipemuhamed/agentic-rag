"""Decide which catalogued equipment a photo shows.

Visual similarity alone is a weak identifier: a generic CLIP model separates
"phone" from "router" but not one black chassis from another, so a lookalike can
sit inside the distance gate and silently redirect the manual search. The text a
camera can read — a model plate, a label, a serial — is far stronger evidence
when it is present. This module fuses both, and abstains when neither is
conclusive, so the caller can widen the search or ask instead of guessing.

Nothing here imports the vision model or LangChain: the resolver takes plain
strings, which keeps it usable from the knowledge-only image and from the
evaluation CLI.
"""

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from app.knowledge.retrieval.models import RetrievedDocument


CATALOG_IDENTITY_FIELDS = (
    "model",
    "equipment_id",
    "equipment_name",
    "manufacturer",
)

# Below this length a catalog field matches too much by accident: a model named
# "G8" would hit any text containing those characters.
MINIMUM_SUBSTRING_LENGTH = 3


@dataclass(frozen=True, slots=True)
class EquipmentCandidate:
    equipment_id: str
    equipment_name: str
    visual_score: float
    identifier_score: float
    confidence: float
    matched_identifiers: tuple[str, ...]
    document: RetrievedDocument


@dataclass(frozen=True, slots=True)
class EquipmentResolution:
    candidates: tuple[EquipmentCandidate, ...]
    resolved: EquipmentCandidate | None
    ambiguous: bool
    reason: str

    @property
    def equipment_ids(self) -> list[str]:
        """Return the ids allowed to restrict the manual search.

        Empty on abstention, which leaves the textual retrieval unfiltered
        instead of pointing it at the wrong equipment.
        """
        return [self.resolved.equipment_id] if self.resolved is not None else []

    def context_text(self) -> str:
        return json.dumps(
            {
                "equipamento_resolvido": (
                    None
                    if self.resolved is None
                    else {
                        "equipment_id": self.resolved.equipment_id,
                        "equipment_name": self.resolved.equipment_name,
                        "confianca": round(self.resolved.confidence, 4),
                        "identificadores_lidos": list(
                            self.resolved.matched_identifiers
                        ),
                    }
                ),
                "ambiguo": self.ambiguous,
                "motivo": self.reason,
                "candidatos": [
                    {
                        "equipment_id": candidate.equipment_id,
                        "equipment_name": candidate.equipment_name,
                        "confianca": round(candidate.confidence, 4),
                        "similaridade_visual": round(candidate.visual_score, 4),
                        "evidencia_textual": round(
                            candidate.identifier_score, 4
                        ),
                    }
                    for candidate in self.candidates
                ],
            },
            ensure_ascii=False,
            indent=2,
        )


class EquipmentResolver:
    def __init__(
        self,
        *,
        identifier_weight: float,
        min_confidence: float,
        ambiguity_margin: float,
    ):
        if not 0.0 <= identifier_weight <= 1.0:
            raise ValueError("identifier_weight must be between zero and one")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between zero and one")
        if not 0.0 <= ambiguity_margin <= 1.0:
            raise ValueError("ambiguity_margin must be between zero and one")
        self._identifier_weight = identifier_weight
        self._min_confidence = min_confidence
        self._ambiguity_margin = ambiguity_margin

    def resolve(
        self,
        identifiers: Sequence[str],
        references: Sequence[RetrievedDocument],
    ) -> EquipmentResolution:
        scored = [
            self._score(reference, identifiers)
            for reference in references
            if _equipment_id(reference)
        ]
        if not scored:
            return EquipmentResolution(
                candidates=(),
                resolved=None,
                ambiguous=False,
                reason="nenhuma referência visual dentro do limiar",
            )

        # The blend only applies when the photo actually yielded readable text
        # that matches something in the catalog. Applying it otherwise would cap
        # every confidence at the visual weight and make abstention unavoidable
        # for equipment with no legible plate.
        textual_evidence = any(
            candidate.identifier_score > 0.0 for candidate in scored
        )
        candidates = tuple(
            sorted(
                (
                    _with_confidence(
                        candidate,
                        self._identifier_weight if textual_evidence else 0.0,
                    )
                    for candidate in scored
                ),
                key=lambda candidate: candidate.confidence,
                reverse=True,
            )
        )

        best = candidates[0]
        if best.confidence < self._min_confidence:
            return EquipmentResolution(
                candidates=candidates,
                resolved=None,
                ambiguous=False,
                reason=(
                    f"confiança {best.confidence:.3f} abaixo do mínimo "
                    f"{self._min_confidence:.3f}"
                ),
            )

        if len(candidates) > 1:
            runner_up = candidates[1]
            distance = best.confidence - runner_up.confidence
            if distance < self._ambiguity_margin:
                return EquipmentResolution(
                    candidates=candidates,
                    resolved=None,
                    ambiguous=True,
                    reason=(
                        f"{best.equipment_id} e {runner_up.equipment_id} "
                        f"separados por apenas {distance:.3f}"
                    ),
                )

        return EquipmentResolution(
            candidates=candidates,
            resolved=best,
            ambiguous=False,
            reason=(
                "identificador lido na imagem"
                if best.identifier_score > 0.0
                else "similaridade visual sem concorrente próximo"
            ),
        )

    def _score(
        self,
        reference: RetrievedDocument,
        identifiers: Sequence[str],
    ) -> EquipmentCandidate:
        metadata = reference.metadata
        identifier_score, matched = identifier_evidence(metadata, identifiers)
        return EquipmentCandidate(
            equipment_id=_equipment_id(reference),
            equipment_name=_text(metadata.get("equipment_name")),
            visual_score=_visual_score(metadata.get("visual_distance")),
            identifier_score=identifier_score,
            confidence=0.0,
            matched_identifiers=matched,
            document=reference,
        )


def _with_confidence(
    candidate: EquipmentCandidate,
    identifier_weight: float,
) -> EquipmentCandidate:
    confidence = (
        identifier_weight * candidate.identifier_score
        + (1.0 - identifier_weight) * candidate.visual_score
    )
    return EquipmentCandidate(
        equipment_id=candidate.equipment_id,
        equipment_name=candidate.equipment_name,
        visual_score=candidate.visual_score,
        identifier_score=candidate.identifier_score,
        confidence=confidence,
        matched_identifiers=candidate.matched_identifiers,
        document=candidate.document,
    )


def _visual_score(distance: object) -> float:
    """Convert a Chroma distance into a similarity in the unit interval.

    The visual embeddings are L2-normalised and the collection uses the default
    squared euclidean space, so ``d = 2 - 2*cos`` and the cosine comes back with
    ``1 - d/2``. Expressing the score as a cosine keeps the number comparable
    across catalogues instead of tied to whatever the current gate happens to be.
    """
    if not isinstance(distance, (int, float)) or isinstance(distance, bool):
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))


def identifier_evidence(
    metadata: dict[str, object],
    identifiers: Sequence[str],
) -> tuple[float, tuple[str, ...]]:
    """Score how strongly text read from an image matches a catalog entry.

    Returns the best score in the unit interval and the readings that hit,
    so a caller can show the operator why an equipment was proposed.
    """
    readings = [
        (raw, normalized)
        for raw in identifiers
        if (normalized := _normalize(raw))
    ]
    if not readings:
        return 0.0, ()

    catalog = [
        normalized
        for field in CATALOG_IDENTITY_FIELDS
        if (normalized := _normalize(_text(metadata.get(field))))
    ]
    if not catalog:
        return 0.0, ()

    best = 0.0
    matched: list[str] = []
    for raw, reading in readings:
        score = max(_pair_score(reading, entry) for entry in catalog)
        if score > 0.0:
            matched.append(raw)
        best = max(best, score)
    return best, tuple(matched)


def _pair_score(reading: str, catalog_entry: str) -> float:
    if reading == catalog_entry:
        return 1.0
    if (
        len(catalog_entry) >= MINIMUM_SUBSTRING_LENGTH
        and catalog_entry in reading
    ) or (
        len(reading) >= MINIMUM_SUBSTRING_LENGTH and reading in catalog_entry
    ):
        return 0.85

    reading_tokens = set(reading.split())
    catalog_tokens = set(catalog_entry.split())
    shared = reading_tokens & catalog_tokens
    if not shared:
        return 0.0
    # Jaccard keeps a long OCR dump from scoring highly just because it happens
    # to contain the manufacturer name.
    return 0.7 * len(shared) / len(reading_tokens | catalog_tokens)


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    collapsed = re.sub(r"[^a-z0-9]+", " ", without_accents.lower())
    return collapsed.strip()


def _equipment_id(reference: RetrievedDocument) -> str:
    return _text(reference.metadata.get("equipment_id"))


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
