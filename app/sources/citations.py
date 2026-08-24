import re
from collections.abc import Sequence

from app.sources.models import Source


CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class CitationValidationError(ValueError):
    pass


def citation_ids(answer: str) -> list[int]:
    identifiers: list[int] = []
    for match in CITATION_PATTERN.finditer(answer):
        identifier = int(match.group(1))
        if match.group(0) != f"[{identifier}]":
            raise CitationValidationError("citation label is not canonical")
        identifiers.append(identifier)
    return identifiers


def select_cited_sources(
    answer: str,
    sources: Sequence[Source],
) -> list[Source]:
    normalized_answer = answer.strip()
    if not normalized_answer:
        raise CitationValidationError("answer must not be empty")

    referenced_ids = citation_ids(normalized_answer)
    if not sources:
        if referenced_ids:
            raise CitationValidationError("answer cites a source that does not exist")
        return []
    if not referenced_ids:
        raise CitationValidationError("grounded answer must contain a citation")

    sources_by_id = {source.citation_id: source for source in sources}
    unknown_ids = set(referenced_ids) - set(sources_by_id)
    if unknown_ids:
        unknown = ", ".join(str(value) for value in sorted(unknown_ids))
        raise CitationValidationError(f"answer contains unknown citations: {unknown}")

    used_ids = set(referenced_ids)
    return [
        source for source in sources if source.citation_id in used_ids
    ]
