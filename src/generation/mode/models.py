"""Contracts for constrained grounded-answer mode selection."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from src.generation.context.models import ContextBuildResult


class AnswerMode(str, Enum):
    """Describe one mutually exclusive grounded-answer behavior."""

    SUPPORTED = "SUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True)
class AnswerModeScore:
    """Keep one candidate mode and its model-provided score."""

    mode: AnswerMode
    score: float


@dataclass(frozen=True, slots=True)
class AnswerModeSelection:
    """Expose a selected mode and evidence for the selection boundary."""

    mode: AnswerMode
    confidence_gap: float | None
    ranked_scores: tuple[AnswerModeScore, ...]


class AnswerModeScorer(Protocol):
    """Score only the answer modes admitted for one context."""

    def score_modes(
        self,
        question: str,
        context: ContextBuildResult,
        candidates: Sequence[AnswerMode],
    ) -> Mapping[AnswerMode, float]:
        """Return one finite score for every requested candidate."""


class AnswerModeSelectionError(RuntimeError):
    """Report an invalid or insufficiently confident mode decision."""
