"""Tests for structural grounded-answer validation."""

import pytest

from src.generation import (
    INSUFFICIENT_CONTEXT_RESPONSE,
    GroundedAnswerValidationError,
    validate_grounded_answer,
)
from src.models import MinimalSource


def _sources(count: int = 2) -> tuple[MinimalSource, ...]:
    return tuple(
        MinimalSource(
            file_path=f"docs/source-{number}.md",
            first_character_index=0,
            last_character_index=10,
        )
        for number in range(1, count + 1)
    )


@pytest.mark.parametrize(
    "answer",
    [
        INSUFFICIENT_CONTEXT_RESPONSE,
        "The cache uses LRU. [Source 1]",
        (
            "The sources conflict: timeout is 30 seconds [Source 1], "
            "but another source says 60 seconds [Source 2]."
        ),
    ],
)
def test_validator_accepts_supported_contract_shapes(answer: str) -> None:
    """Fallback, grounded facts, and evidenced conflicts are valid."""
    validate_grounded_answer(answer, _sources())


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (
            "The cache uses LRU.",
            "does not cite any retrieved source",
        ),
        (
            "The cache uses LRU. [Source 3]",
            "outside the prompt context",
        ),
        (
            "The sources conflict: details are unclear. [Source 1]",
            "must cite at least two sources",
        ),
    ],
)
def test_validator_rejects_untraceable_answers(
    answer: str,
    message: str,
) -> None:
    """Malformed grounding cannot cross the generation boundary."""
    with pytest.raises(GroundedAnswerValidationError, match=message):
        validate_grounded_answer(answer, _sources())


def test_exact_fallback_is_required_when_no_sources_exist() -> None:
    """Uncited prose cannot replace the versioned fallback response."""
    with pytest.raises(
        GroundedAnswerValidationError,
        match="does not cite any retrieved source",
    ):
        validate_grounded_answer("I do not know.", ())
