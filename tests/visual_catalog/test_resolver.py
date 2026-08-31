from app.knowledge.retrieval.models import RetrievedDocument
from app.visual_catalog.resolver import EquipmentResolver


def make_resolver(
    *,
    identifier_weight: float = 0.65,
    min_confidence: float = 0.55,
    ambiguity_margin: float = 0.08,
) -> EquipmentResolver:
    return EquipmentResolver(
        identifier_weight=identifier_weight,
        min_confidence=min_confidence,
        ambiguity_margin=ambiguity_margin,
    )


def make_reference(
    equipment_id: str,
    *,
    distance: float,
    equipment_name: str = "",
    model: str = "",
    manufacturer: str = "",
) -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id=f"visual-ref:{equipment_id}",
        content="Referência visual.",
        metadata={
            "equipment_id": equipment_id,
            "equipment_name": equipment_name or equipment_id,
            "model": model,
            "manufacturer": manufacturer,
            "visual_distance": distance,
        },
        retrieval_score=-distance,
        rerank_score=max(0.0, 1.0 - distance),
    )


def test_identifier_overrides_a_closer_lookalike() -> None:
    # The measured failure: two phones sit within the visual gate, and the
    # lookalike is the nearer of the two. Only the plate read from the photo
    # separates them.
    lookalike = make_reference(
        "OLT-9000",
        distance=0.20,
        equipment_name="iPhone Air",
        model="iphone air",
        manufacturer="Apple",
    )
    correct = make_reference(
        "OLT-9001",
        distance=0.60,
        equipment_name="Moto G",
        model="g86",
        manufacturer="Motorola",
    )

    resolution = make_resolver().resolve(["G86"], [lookalike, correct])

    assert resolution.resolved is not None
    assert resolution.resolved.equipment_id == "OLT-9001"
    assert resolution.equipment_ids == ["OLT-9001"]
    assert resolution.resolved.matched_identifiers == ("G86",)
    assert resolution.reason == "identificador lido na imagem"


def test_identifier_matching_ignores_case_accents_and_punctuation() -> None:
    reference = make_reference(
        "OLT-9001",
        distance=0.60,
        equipment_name="Módulo Óptico",
        model="MX-200",
    )

    resolution = make_resolver().resolve(["mx 200"], [reference])

    assert resolution.resolved is not None
    assert resolution.resolved.identifier_score == 1.0


def test_visual_similarity_alone_resolves_when_no_text_is_readable() -> None:
    # Without OCR evidence the blend would cap every confidence at the visual
    # weight, so equipment with no legible plate could never be resolved.
    reference = make_reference("OLT-9001", distance=0.10)

    resolution = make_resolver().resolve([], [reference])

    assert resolution.resolved is not None
    assert resolution.resolved.equipment_id == "OLT-9001"
    assert resolution.reason == "similaridade visual sem concorrente próximo"


def test_distant_match_is_abstained_instead_of_filtering() -> None:
    reference = make_reference("OLT-9000", distance=1.30)

    resolution = make_resolver().resolve([], [reference])

    assert resolution.resolved is None
    assert resolution.ambiguous is False
    assert resolution.equipment_ids == []
    assert "abaixo do mínimo" in resolution.reason


def test_two_close_candidates_are_reported_as_ambiguous() -> None:
    resolution = make_resolver().resolve(
        [],
        [
            make_reference("OLT-9000", distance=0.40),
            make_reference("OLT-9001", distance=0.42),
        ],
    )

    assert resolution.resolved is None
    assert resolution.ambiguous is True
    assert resolution.equipment_ids == []
    assert "OLT-9000" in resolution.reason
    assert "OLT-9001" in resolution.reason


def test_candidates_survive_abstention_as_evidence() -> None:
    resolution = make_resolver().resolve(
        [],
        [
            make_reference("OLT-9000", distance=0.40),
            make_reference("OLT-9001", distance=0.42),
        ],
    )

    assert [candidate.equipment_id for candidate in resolution.candidates] == [
        "OLT-9000",
        "OLT-9001",
    ]
    assert "OLT-9000" in resolution.context_text()


def test_no_references_abstains_without_claiming_ambiguity() -> None:
    resolution = make_resolver().resolve(["G86"], [])

    assert resolution.resolved is None
    assert resolution.ambiguous is False
    assert resolution.candidates == ()
    assert resolution.equipment_ids == []


def test_references_without_equipment_id_are_ignored() -> None:
    orphan = RetrievedDocument(
        chunk_id="visual-ref:orphan",
        content="Referência sem equipamento.",
        metadata={"visual_distance": 0.05},
        retrieval_score=-0.05,
        rerank_score=0.95,
    )

    resolution = make_resolver().resolve([], [orphan])

    assert resolution.candidates == ()
    assert resolution.resolved is None


def test_short_catalog_values_do_not_match_by_substring() -> None:
    # A model named "G8" would otherwise match any reading containing it.
    reference = make_reference("OLT-9001", distance=1.30, model="G8")

    resolution = make_resolver().resolve(["SN G8A44710X"], [reference])

    assert resolution.candidates[0].identifier_score == 0.0
