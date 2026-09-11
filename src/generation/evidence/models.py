"""Contracts for source-by-source binary evidence decisions."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from collections.abc import Mapping

from src.models import MinimalSource


class EvidenceDecision(str, Enum):
    """Describe whether one source directly answers the question."""

    ANSWERS = "ANSWERS"
    DOES_NOT_ANSWER = "DOES_NOT_ANSWER"


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """Pair one exact retrieved source with its admitted text."""

    source_number: int
    source: MinimalSource
    text: str


@dataclass(frozen=True, slots=True)
class EvidenceDecisionScore:
    """Preserve one raw score used for a binary evidence decision."""

    decision: EvidenceDecision
    score: float


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Report a prediction and whether it crossed the confidence boundary."""

    candidate: EvidenceCandidate
    predicted_decision: EvidenceDecision
    scores: tuple[EvidenceDecisionScore, ...]
    confidence_gap: float
    accepted: bool

    @property
    def supports_answer(self) -> bool:
        """Return whether this source is accepted as direct evidence."""
        return (
            self.accepted
            and self.predicted_decision is EvidenceDecision.ANSWERS
        )


@dataclass(frozen=True, slots=True)
class EvidenceSelection:
    """Preserve ordered assessments and expose admitted answer evidence."""

    assessments: tuple[EvidenceAssessment, ...]

    @property
    def supporting_candidates(self) -> tuple[EvidenceCandidate, ...]:
        """Return only confidently accepted answer-bearing candidates."""
        return tuple(
            assessment.candidate
            for assessment in self.assessments
            if assessment.supports_answer
        )


class BinaryEvidenceScorer(Protocol):
    """Score both evidence decisions for one question-source pair."""

    def score_evidence(
        self,
        question: str,
        candidate: EvidenceCandidate,
    ) -> Mapping[EvidenceDecision, float]:
        """Return one finite score for each binary decision."""


class EvidenceSelectionError(ValueError):
    """Report invalid evidence candidates or scorer output."""
