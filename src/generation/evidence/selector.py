"""Select direct answer evidence through small binary decisions."""

from collections.abc import Sequence
import math

from src.generation.evidence.models import (
    BinaryEvidenceScorer,
    EvidenceAssessment,
    EvidenceCandidate,
    EvidenceDecision,
    EvidenceDecisionScore,
    EvidenceSelection,
    EvidenceSelectionError,
)


class BinaryEvidenceSelector:
    """Assess each source independently and reject uncertain evidence."""

    def __init__(
        self,
        scorer: BinaryEvidenceScorer,
        minimum_confidence_gap: float,
    ) -> None:
        if (
            not math.isfinite(minimum_confidence_gap)
            or minimum_confidence_gap < 0
        ):
            raise ValueError(
                "Evidence confidence gap must be finite and non-negative"
            )
        self._scorer = scorer
        self._minimum_confidence_gap = minimum_confidence_gap

    def select(
        self,
        question: str,
        candidates: Sequence[EvidenceCandidate],
    ) -> EvidenceSelection:
        """Return ordered binary assessments for exact source candidates."""
        normalized_question = question.strip()
        if not normalized_question:
            raise EvidenceSelectionError(
                "Evidence question must not be empty"
            )
        candidate_tuple = tuple(candidates)
        _validate_candidates(candidate_tuple)
        return EvidenceSelection(
            tuple(
                self._assess(normalized_question, candidate)
                for candidate in candidate_tuple
            )
        )

    def _assess(
        self,
        question: str,
        candidate: EvidenceCandidate,
    ) -> EvidenceAssessment:
        raw_scores = self._scorer.score_evidence(question, candidate)
        expected_decisions = set(EvidenceDecision)
        if set(raw_scores) != expected_decisions:
            raise EvidenceSelectionError(
                "Evidence scorer must score both decisions exactly once"
            )
        if any(not math.isfinite(score) for score in raw_scores.values()):
            raise EvidenceSelectionError(
                "Evidence scorer returned a non-finite score"
            )

        ranked = sorted(
            raw_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        confidence_gap = ranked[0][1] - ranked[1][1]
        accepted = (
            confidence_gap > 0
            and confidence_gap >= self._minimum_confidence_gap
        )
        return EvidenceAssessment(
            candidate=candidate,
            predicted_decision=ranked[0][0],
            scores=tuple(
                EvidenceDecisionScore(decision, raw_scores[decision])
                for decision in EvidenceDecision
            ),
            confidence_gap=confidence_gap,
            accepted=accepted,
        )


def _validate_candidates(
    candidates: tuple[EvidenceCandidate, ...],
) -> None:
    """Require canonical numbering, unique identities, and usable text."""
    expected_numbers = tuple(range(1, len(candidates) + 1))
    actual_numbers = tuple(
        candidate.source_number for candidate in candidates
    )
    if actual_numbers != expected_numbers:
        raise EvidenceSelectionError(
            "Evidence source numbers must be contiguous from one"
        )

    identities: set[tuple[str, int, int]] = set()
    for candidate in candidates:
        if not candidate.text.strip():
            raise EvidenceSelectionError(
                "Evidence candidate text must not be empty"
            )
        identity = (
            candidate.source.file_path,
            candidate.source.first_character_index,
            candidate.source.last_character_index,
        )
        if identity in identities:
            raise EvidenceSelectionError(
                "Evidence candidates must contain unique sources"
            )
        identities.add(identity)
