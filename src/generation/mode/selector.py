"""Select one grounded-answer mode from a constrained candidate set."""

from collections.abc import Mapping
import math

from src.generation.context.models import ContextBuildResult
from src.generation.mode.models import (
    AnswerMode,
    AnswerModeScore,
    AnswerModeScorer,
    AnswerModeSelection,
    AnswerModeSelectionError,
)


def available_answer_modes(source_count: int) -> tuple[AnswerMode, ...]:
    """Return only modes possible for the admitted number of sources."""
    if source_count < 0:
        raise ValueError("Source count must not be negative")
    if source_count == 0:
        return (AnswerMode.INSUFFICIENT,)
    if source_count == 1:
        return (AnswerMode.SUPPORTED, AnswerMode.INSUFFICIENT)
    return tuple(AnswerMode)


class AnswerModeSelector:
    """Choose a mode or reject an uncertain model-scored decision."""

    def __init__(
        self,
        scorer: AnswerModeScorer,
        *,
        min_confidence_gap: float,
    ) -> None:
        """Store a scorer and validate its explicit confidence boundary."""
        if not math.isfinite(min_confidence_gap):
            raise ValueError("Mode confidence gap must be finite")
        if min_confidence_gap < 0:
            raise ValueError("Mode confidence gap must not be negative")
        self._scorer = scorer
        self._min_confidence_gap = min_confidence_gap

    def select(
        self,
        question: str,
        context: ContextBuildResult,
    ) -> AnswerModeSelection:
        """Select one available mode without guessing across close scores."""
        if not question.strip():
            raise ValueError("Answer-mode question must not be empty")

        candidates = available_answer_modes(len(context.sources))
        if candidates == (AnswerMode.INSUFFICIENT,):
            return AnswerModeSelection(
                mode=AnswerMode.INSUFFICIENT,
                confidence_gap=None,
                ranked_scores=(),
            )

        scores = self._scorer.score_modes(question, context, candidates)
        _validate_scores(scores, candidates)
        ranked_scores = tuple(
            sorted(
                (
                    AnswerModeScore(mode, scores[mode])
                    for mode in candidates
                ),
                key=lambda result: result.score,
                reverse=True,
            )
        )
        confidence_gap = ranked_scores[0].score - ranked_scores[1].score
        if confidence_gap <= 0 or confidence_gap < self._min_confidence_gap:
            raise AnswerModeSelectionError(
                "Answer mode selection confidence is insufficient"
            )
        return AnswerModeSelection(
            mode=ranked_scores[0].mode,
            confidence_gap=confidence_gap,
            ranked_scores=ranked_scores,
        )


def _validate_scores(
    scores: object,
    candidates: tuple[AnswerMode, ...],
) -> None:
    """Require exact candidate coverage and finite numeric score values."""
    if not isinstance(scores, Mapping):
        raise AnswerModeSelectionError(
            "Answer mode scorer must return a mapping"
        )
    if any(not isinstance(mode, AnswerMode) for mode in scores):
        raise AnswerModeSelectionError(
            "Answer mode scorer returned an invalid candidate"
        )
    if set(scores) != set(candidates):
        raise AnswerModeSelectionError(
            "Answer mode scorer must score every available candidate once"
        )
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        for score in scores.values()
    ):
        raise AnswerModeSelectionError(
            "Answer mode scores must be finite numbers"
        )
