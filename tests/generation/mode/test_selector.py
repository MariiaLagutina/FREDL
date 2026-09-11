"""Tests for model-independent answer-mode selection policy."""

from collections.abc import Mapping, Sequence

import pytest

from src.generation import (
    AnswerMode,
    AnswerModeSelector,
    AnswerModeSelectionError,
    ContextBuildResult,
)
from src.models import MinimalSource


class RecordingScorer:
    """Return configured scores while preserving requested candidates."""

    def __init__(self, scores: Mapping[AnswerMode, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[AnswerMode, ...]] = []

    def score_modes(
        self,
        question: str,
        context: ContextBuildResult,
        candidates: Sequence[AnswerMode],
    ) -> Mapping[AnswerMode, float]:
        """Record only the candidate boundary relevant to these tests."""
        del question, context
        self.calls.append(tuple(candidates))
        return self.scores


def _context(source_count: int) -> ContextBuildResult:
    sources = tuple(
        MinimalSource(
            file_path=f"data/raw/source-{number}.md",
            first_character_index=0,
            last_character_index=10,
        )
        for number in range(1, source_count + 1)
    )
    return ContextBuildResult("evidence", sources, 1, 0)


def test_no_sources_selects_insufficient_without_model_scoring() -> None:
    """An empty context has only one deterministic valid behavior."""
    scorer = RecordingScorer({})
    selector = AnswerModeSelector(scorer, min_confidence_gap=0.1)

    selection = selector.select("What is configured?", _context(0))

    assert selection.mode is AnswerMode.INSUFFICIENT
    assert selection.confidence_gap is None
    assert selection.ranked_scores == ()
    assert scorer.calls == []


def test_one_source_never_offers_conflict_mode() -> None:
    """A single source cannot support a multi-source conflict answer."""
    scorer = RecordingScorer(
        {
            AnswerMode.SUPPORTED: -0.1,
            AnswerMode.INSUFFICIENT: -0.8,
        }
    )
    selector = AnswerModeSelector(scorer, min_confidence_gap=0.2)

    selection = selector.select("What is configured?", _context(1))

    assert scorer.calls == [
        (AnswerMode.SUPPORTED, AnswerMode.INSUFFICIENT)
    ]
    assert selection.mode is AnswerMode.SUPPORTED
    assert selection.confidence_gap == pytest.approx(0.7)


def test_multiple_sources_rank_all_modes() -> None:
    """Two sources admit conflict while preserving complete score evidence."""
    scorer = RecordingScorer(
        {
            AnswerMode.SUPPORTED: -0.7,
            AnswerMode.INSUFFICIENT: -1.2,
            AnswerMode.CONFLICT: -0.2,
        }
    )
    selector = AnswerModeSelector(scorer, min_confidence_gap=0.3)

    selection = selector.select("What is the timeout?", _context(2))

    assert scorer.calls == [tuple(AnswerMode)]
    assert selection.mode is AnswerMode.CONFLICT
    assert [score.mode for score in selection.ranked_scores] == [
        AnswerMode.CONFLICT,
        AnswerMode.SUPPORTED,
        AnswerMode.INSUFFICIENT,
    ]


def test_close_scores_are_rejected_instead_of_guessed() -> None:
    """An uncertain top mode cannot silently control answer generation."""
    scorer = RecordingScorer(
        {
            AnswerMode.SUPPORTED: -0.10,
            AnswerMode.INSUFFICIENT: -0.25,
        }
    )
    selector = AnswerModeSelector(scorer, min_confidence_gap=0.2)

    with pytest.raises(
        AnswerModeSelectionError,
        match="confidence is insufficient",
    ):
        selector.select("What is configured?", _context(1))


def test_tied_scores_are_rejected_with_zero_minimum_gap() -> None:
    """A zero policy threshold cannot turn an exact tie into a guess."""
    scorer = RecordingScorer(
        {
            AnswerMode.SUPPORTED: -0.5,
            AnswerMode.INSUFFICIENT: -0.5,
        }
    )
    selector = AnswerModeSelector(scorer, min_confidence_gap=0.0)

    with pytest.raises(
        AnswerModeSelectionError,
        match="confidence is insufficient",
    ):
        selector.select("What is configured?", _context(1))


@pytest.mark.parametrize(
    "scores",
    [
        {AnswerMode.SUPPORTED: 1.0},
        {
            AnswerMode.SUPPORTED: 1.0,
            AnswerMode.INSUFFICIENT: 0.0,
            AnswerMode.CONFLICT: -1.0,
        },
    ],
)
def test_scorer_must_cover_exact_available_candidates(
    scores: Mapping[AnswerMode, float],
) -> None:
    """Missing or impossible candidates make score evidence ambiguous."""
    selector = AnswerModeSelector(
        RecordingScorer(scores),
        min_confidence_gap=0.1,
    )

    with pytest.raises(
        AnswerModeSelectionError,
        match="score every available candidate once",
    ):
        selector.select("What is configured?", _context(1))


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_non_finite_scores_are_rejected(score: float) -> None:
    """NaN or infinity cannot create a meaningful confidence boundary."""
    selector = AnswerModeSelector(
        RecordingScorer(
            {
                AnswerMode.SUPPORTED: score,
                AnswerMode.INSUFFICIENT: 0.0,
            }
        ),
        min_confidence_gap=0.1,
    )

    with pytest.raises(
        AnswerModeSelectionError,
        match="scores must be finite numbers",
    ):
        selector.select("What is configured?", _context(1))


@pytest.mark.parametrize("gap", [-0.1, float("nan"), float("inf")])
def test_selector_rejects_invalid_confidence_boundary(gap: float) -> None:
    """The caller must choose a finite non-negative confidence policy."""
    with pytest.raises(ValueError, match="confidence gap"):
        AnswerModeSelector(RecordingScorer({}), min_confidence_gap=gap)
