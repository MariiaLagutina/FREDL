"""Validate exact retrieved-source paths and character ranges."""

from collections.abc import Mapping

from src.models import MinimalSource
from src.retrieval.validation.models import (
    SourceValidationIssue,
    SourceValidationIssueKind,
)


MAX_SOURCE_LENGTH = 2000


def validate_source(
    source: MinimalSource,
    source_texts: Mapping[str, str],
    *,
    result_index: int,
    source_index: int,
    question_id: str,
    max_source_length: int = MAX_SOURCE_LENGTH,
) -> tuple[SourceValidationIssue, ...]:
    """Return every failed invariant for one retrieved source."""
    if max_source_length <= 0:
        raise ValueError("Maximum source length must be positive")

    issues: list[SourceValidationIssue] = []
    source_text = source_texts.get(source.file_path)
    if source_text is None:
        issues.append(
            _issue(
                SourceValidationIssueKind.UNKNOWN_PATH,
                source,
                result_index,
                source_index,
                question_id,
                "Source path is absent from the discovered corpus",
            )
        )

    start = source.first_character_index
    end = source.last_character_index
    range_is_valid = start >= 0 and start < end
    if source_text is not None:
        range_is_valid = range_is_valid and end <= len(source_text)
    if not range_is_valid:
        source_length = len(source_text) if source_text is not None else None
        issues.append(
            _issue(
                SourceValidationIssueKind.INVALID_RANGE,
                source,
                result_index,
                source_index,
                question_id,
                f"Range [{start}, {end}) is invalid for source length "
                f"{source_length}",
            )
        )

    range_length = end - start
    if range_length > max_source_length:
        issues.append(
            _issue(
                SourceValidationIssueKind.OVERSIZED,
                source,
                result_index,
                source_index,
                question_id,
                f"Source length {range_length} exceeds maximum "
                f"{max_source_length}",
            )
        )
    return tuple(issues)


def _issue(
    kind: SourceValidationIssueKind,
    source: MinimalSource,
    result_index: int,
    source_index: int,
    question_id: str,
    detail: str,
) -> SourceValidationIssue:
    """Build one issue with shared retrieval-result context."""
    return SourceValidationIssue(
        kind=kind,
        result_index=result_index,
        source_index=source_index,
        question_id=question_id,
        file_path=source.file_path,
        detail=detail,
    )
