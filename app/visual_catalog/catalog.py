"""Read the equipment catalog as a list, and match free text against it.

The visual catalog is stored one document per reference photo, but callers
outside the image pipeline think in equipment: a monitoring system knows a host
name and an inventory model string, not a photograph. This module folds the
photos back into equipment entries and scores arbitrary text against them, so a
`{HOST.NAME}` or a trigger description can be mapped to the `equipment_id` that
filters the right manuals.

Matching reuses the identifier scoring from the visual resolver, so text read by
a camera and text read from a monitoring API are judged by the same rules.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.knowledge.models import IndexedDocument
from app.visual_catalog.resolver import identifier_evidence


class SupportsCatalogListing(Protocol):
    def list_documents(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[IndexedDocument]: ...

    def get_document_chunks(self, document_id: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    equipment_id: str
    equipment_name: str
    manufacturer: str
    model: str
    description: str
    reference_count: int

    def as_metadata(self) -> dict[str, Any]:
        return {
            "equipment_id": self.equipment_id,
            "equipment_name": self.equipment_name,
            "manufacturer": self.manufacturer,
            "model": self.model,
        }


@dataclass(frozen=True, slots=True)
class CatalogMatch:
    entry: CatalogEntry
    score: float
    matched_terms: tuple[str, ...]


def list_equipment(
    store: SupportsCatalogListing,
    *,
    tenant_id: str | None = None,
) -> list[CatalogEntry]:
    """Fold the reference photos into one entry per equipment."""
    filters = {"tenant_id": tenant_id} if tenant_id else None
    grouped: dict[str, dict[str, Any]] = {}
    for document in store.list_documents(filters):
        equipment_id = document.equipment_id.strip()
        if not equipment_id:
            continue
        entry = grouped.get(equipment_id)
        if entry is None:
            grouped[equipment_id] = {
                "metadata": _first_chunk_metadata(store, document),
                "references": 1,
            }
            continue
        entry["references"] += 1

    catalog = [
        CatalogEntry(
            equipment_id=equipment_id,
            equipment_name=_text(entry["metadata"].get("equipment_name")),
            manufacturer=_text(entry["metadata"].get("manufacturer")),
            model=_text(entry["metadata"].get("model")),
            description=_text(entry["metadata"].get("description")),
            reference_count=entry["references"],
        )
        for equipment_id, entry in grouped.items()
    ]
    return sorted(catalog, key=lambda item: item.equipment_id)


def rank_equipment_by_text(
    text_values: Sequence[str],
    catalog: Sequence[CatalogEntry],
) -> list[CatalogMatch]:
    """Rank catalog entries against free text, best first.

    Deliberately returns every scoring candidate instead of committing to one.
    The caller here is an assistant holding the whole incident context, which is
    better placed to choose than a fixed threshold -- and a wrong commitment
    would point the manual search at the wrong equipment.
    """
    readings = [value for value in text_values if value and value.strip()]
    if not readings:
        return []

    matches = []
    for entry in catalog:
        score, matched = identifier_evidence(entry.as_metadata(), readings)
        if score > 0.0:
            matches.append(
                CatalogMatch(entry=entry, score=score, matched_terms=matched)
            )
    return sorted(matches, key=lambda match: match.score, reverse=True)


def _first_chunk_metadata(
    store: SupportsCatalogListing,
    document: IndexedDocument,
) -> dict[str, Any]:
    # Manufacturer, model and description live on the chunk, not on the
    # aggregated document row.
    chunks = store.get_document_chunks(document.document_id)
    for chunk in chunks:
        metadata = chunk.get("metadata")
        if isinstance(metadata, dict) and metadata:
            return metadata
    return {"equipment_name": document.title}


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
