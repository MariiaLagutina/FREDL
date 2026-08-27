"""Tests for exact retrieved-source invariants."""

import pytest

from src.models import MinimalSource
from src.retrieval.validation import (
    SourceValidationIssueKind,
    validate_source,
)


FILE_PATH = "data/raw/vllm/example.py"
SOURCE_TEXTS = {FILE_PATH: "cache = build_cache()\n"}


def source(start: int, end: int, path: str = FILE_PATH) -> MinimalSource:
    """Create one source location for invariant tests."""
    return MinimalSource(
        file_path=path,
        first_character_index=start,
        last_character_index=end,
    )


def validate(
    candidate: MinimalSource,
) -> tuple[SourceValidationIssueKind, ...]:
    """Return only issue kinds for one representative result context."""
    return tuple(
        issue.kind
        for issue in validate_source(
            candidate,
            SOURCE_TEXTS,
            result_index=2,
            source_index=3,
            question_id="q-1",
        )
    )


def test_valid_source_passes_every_invariant() -> None:
    """An exact known non-empty bounded range produces no issues."""
    assert validate(source(0, 5)) == ()


def test_unknown_path_is_reported_with_result_context() -> None:
    """A path outside the discovered corpus is an actionable failure."""
    issues = validate_source(
        source(0, 5, "data/raw/vllm/missing.py"),
        SOURCE_TEXTS,
        result_index=2,
        source_index=3,
        question_id="q-1",
    )

    assert [issue.kind for issue in issues] == [
        SourceValidationIssueKind.UNKNOWN_PATH
    ]
    assert issues[0].result_index == 2
    assert issues[0].source_index == 3
    assert issues[0].question_id == "q-1"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-1, 5),
        (5, 5),
        (6, 5),
        (0, len(SOURCE_TEXTS[FILE_PATH]) + 1),
    ],
)
def test_invalid_half_open_range_is_reported(start: int, end: int) -> None:
    """Negative, empty, reversed, and out-of-bounds ranges fail."""
    assert validate(source(start, end)) == (
        SourceValidationIssueKind.INVALID_RANGE,
    )


def test_oversized_source_is_reported_separately() -> None:
    """A valid range may still exceed the evaluator context limit."""
    texts = {FILE_PATH: "x" * 2001}

    issues = validate_source(
        source(0, 2001),
        texts,
        result_index=0,
        source_index=0,
        question_id="q-1",
    )

    assert [issue.kind for issue in issues] == [
        SourceValidationIssueKind.OVERSIZED
    ]


def test_unknown_path_can_also_have_an_invalid_range() -> None:
    """Independent failures on one source are accumulated together."""
    assert validate(source(-1, 0, "missing.py")) == (
        SourceValidationIssueKind.UNKNOWN_PATH,
        SourceValidationIssueKind.INVALID_RANGE,
    )


def test_non_positive_maximum_is_rejected() -> None:
    """Validation configuration must express a usable source limit."""
    with pytest.raises(ValueError, match="must be positive"):
        validate_source(
            source(0, 5),
            SOURCE_TEXTS,
            result_index=0,
            source_index=0,
            question_id="q-1",
            max_source_length=0,
        )
