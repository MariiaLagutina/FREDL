"""Collect objective evidence for retrieval error analysis."""

from collections.abc import Sequence

from src.evaluation.retrieval.error_models import RetrievalMissEvidence
from src.evaluation.retrieval.metrics import sources_match
from src.evaluation.retrieval.models import RetrievalEvaluationCase


TOP_FIVE = 5


def collect_top_five_misses(
    cases: Sequence[RetrievalEvaluationCase],
) -> tuple[RetrievalMissEvidence, ...]:
    """Return top-five misses in ground-truth question order."""
    return tuple(
        _miss_evidence(case)
        for case in cases
        if _first_relevant_rank(case) not in range(1, TOP_FIVE + 1)
    )


def _miss_evidence(case: RetrievalEvaluationCase) -> RetrievalMissEvidence:
    """Convert one missed case without assigning a causal category."""
    return RetrievalMissEvidence(
        question_id=case.question_id,
        question=case.question,
        references=case.references,
        retrieved=case.retrieved,
        relevant_rank=_first_relevant_rank(case),
    )


def _first_relevant_rank(case: RetrievalEvaluationCase) -> int | None:
    """Find the first retrieved source matching any labelled reference."""
    for rank, retrieved in enumerate(case.retrieved, start=1):
        if any(
            sources_match(retrieved, reference)
            for reference in case.references
        ):
            return rank
    return None
