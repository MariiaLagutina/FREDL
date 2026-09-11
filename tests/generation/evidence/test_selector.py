"""Tests for model-independent binary evidence selection."""

from collections.abc import Mapping
import math

import pytest

from src.generation.evidence import (
    BinaryEvidenceSelector,
    EvidenceCandidate,
    EvidenceDecision,
    EvidenceSelectionError,
)
from src.models import MinimalSource


class FixedScorer:
    """Return configured binary scores by exact source number."""

    def __init__(
        self,
        scores: Mapping[int, Mapping[EvidenceDecision, float]],
    ) -> None:
        self._scores = scores
        self.calls: list[tuple[str, EvidenceCandidate]] = []

    def score_evidence(
        self,
        question: str,
        candidate: EvidenceCandidate,
    ) -> Mapping[EvidenceDecision, float]:
        self.calls.append((question, candidate))
        return self._scores[candidate.source_number]


def _candidate(number: int, text: str = "evidence") -> EvidenceCandidate:
    return EvidenceCandidate(
        source_number=number,
        source=MinimalSource(
            file_path=f"data/raw/source-{number}.txt",
            first_character_index=0,
            last_character_index=len(text),
        ),
        text=text,
    )


def _scores(
    answers: float,
    does_not_answer: float,
) -> Mapping[EvidenceDecision, float]:
    return {
        EvidenceDecision.ANSWERS: answers,
        EvidenceDecision.DOES_NOT_ANSWER: does_not_answer,
    }


def test_selector_preserves_order_and_admits_only_confident_answers() -> None:
    """Uncertain or negative decisions never become answer evidence."""
    scorer = FixedScorer(
        {
            1: _scores(-0.1, -1.1),
            2: _scores(-0.4, -0.2),
            3: _scores(-0.3, -0.35),
        }
    )
    candidates = (_candidate(1), _candidate(2), _candidate(3))

    result = BinaryEvidenceSelector(scorer, 0.5).select(
        "  What is configured?  ",
        candidates,
    )

    assert tuple(
        assessment.candidate for assessment in result.assessments
    ) == candidates
    assert result.supporting_candidates == (candidates[0],)
    assert result.assessments[0].accepted is True
    assert result.assessments[1].accepted is False
    assert result.assessments[2].accepted is False
    assert [question for question, _ in scorer.calls] == [
        "What is configured?",
    ] * 3


def test_selector_rejects_tie_even_with_zero_boundary() -> None:
    """A tie cannot be treated as confident binary evidence."""
    result = BinaryEvidenceSelector(
        FixedScorer({1: _scores(-0.4, -0.4)}),
        0,
    ).select("Question?", (_candidate(1),))

    assert result.assessments[0].accepted is False
    assert result.supporting_candidates == ()


@pytest.mark.parametrize(
    "scores",
    [
        {EvidenceDecision.ANSWERS: -0.1},
        _scores(math.nan, -0.2),
        _scores(-0.1, math.inf),
    ],
)
def test_selector_rejects_invalid_scorer_output(
    scores: Mapping[EvidenceDecision, float],
) -> None:
    """Incomplete or non-finite evidence cannot cross the boundary."""
    selector = BinaryEvidenceSelector(FixedScorer({1: scores}), 0)

    with pytest.raises(EvidenceSelectionError):
        selector.select("Question?", (_candidate(1),))


@pytest.mark.parametrize(
    "candidates",
    [
        (_candidate(2),),
        (_candidate(1, " "),),
        (_candidate(1), _candidate(1)),
    ],
)
def test_selector_rejects_invalid_candidates(
    candidates: tuple[EvidenceCandidate, ...],
) -> None:
    """Source identity and text remain trustworthy before scoring."""
    selector = BinaryEvidenceSelector(FixedScorer({}), 0)

    with pytest.raises(EvidenceSelectionError):
        selector.select("Question?", candidates)


@pytest.mark.parametrize("gap", [-0.1, math.nan, math.inf])
def test_selector_rejects_invalid_confidence_gap(gap: float) -> None:
    """The acceptance boundary must be finite and non-negative."""
    with pytest.raises(ValueError):
        BinaryEvidenceSelector(FixedScorer({}), gap)
