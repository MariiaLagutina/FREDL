"""Collect objective evidence for retrieval error analysis."""

from collections.abc import Sequence
from itertools import combinations

from src.evaluation.retrieval.error_models import (
    RetrievalErrorCategory,
    RetrievalMissAnalysis,
    RetrievalMissEvidence,
)
from src.evaluation.retrieval.metrics import sources_match
from src.evaluation.retrieval.models import RetrievalEvaluationCase
from src.models import MinimalSource


TOP_FIVE = 5
MAX_BOUNDARY_GAP = 2


def classify_structural_miss(
    evidence: RetrievalMissEvidence,
) -> RetrievalMissAnalysis | None:
    """Classify a miss from rank, file diversity, or source proximity."""
    ranked = classify_ranked_miss(evidence)
    if ranked is not None:
        return ranked
    if _has_overlapping_results(evidence):
        return _structural_analysis(
            evidence,
            RetrievalErrorCategory.DUPLICATE_RESULTS,
            (
                "At least two top-five chunks from one file overlap, reducing "
                "result diversity while the labelled source remains absent."
            ),
            "Add file-level diversification after BM25 ranking.",
            (
                "Re-rank this miss with at most two chunks per file and check "
                "whether the labelled source enters the top five."
            ),
        )
    if _has_adjacent_reference_chunk(evidence):
        return _structural_analysis(
            evidence,
            RetrievalErrorCategory.CHUNK_BOUNDARY,
            (
                "A top-five chunk is in the labelled file and touches or "
                "nearly touches the reference range, but misses the required "
                "source overlap."
            ),
            "Adjust chunk boundaries or overlap around adjacent content.",
            (
                "Re-chunk the same file with a boundary-aware overlap and "
                "re-evaluate this question."
            ),
        )
    return None


def classify_ranked_miss(
    evidence: RetrievalMissEvidence,
) -> RetrievalMissAnalysis | None:
    """Classify a miss when its relevant source is present below rank five."""
    if evidence.relevant_rank is None:
        return None
    return RetrievalMissAnalysis(
        question_id=evidence.question_id,
        question=evidence.question,
        category=RetrievalErrorCategory.RELEVANT_BELOW_TOP_5,
        relevant_rank=evidence.relevant_rank,
        hypothesis=(
            "The lexical retriever found the labelled source, but ranked it "
            "below the evaluation cutoff."
        ),
        proposed_fix=(
            "Improve ranking precision without changing source matching or "
            "retrieval depth."
        ),
        next_test=(
            "Measure whether the next single-factor ranking change moves "
            "this source into the top five."
        ),
    )


def _structural_analysis(
    evidence: RetrievalMissEvidence,
    category: RetrievalErrorCategory,
    hypothesis: str,
    proposed_fix: str,
    next_test: str,
) -> RetrievalMissAnalysis:
    """Build one validated analysis from structural ranking evidence."""
    return RetrievalMissAnalysis(
        question_id=evidence.question_id,
        question=evidence.question,
        category=category,
        relevant_rank=evidence.relevant_rank,
        hypothesis=hypothesis,
        proposed_fix=proposed_fix,
        next_test=next_test,
    )


def _has_adjacent_reference_chunk(evidence: RetrievalMissEvidence) -> bool:
    """Detect a retrieved range at the boundary of a labelled source."""
    return any(
        retrieved.file_path == reference.file_path
        and _range_gap(retrieved, reference) <= MAX_BOUNDARY_GAP
        for retrieved in evidence.retrieved[:TOP_FIVE]
        for reference in evidence.references
    )


def _has_overlapping_results(evidence: RetrievalMissEvidence) -> bool:
    """Detect two top-five chunks that repeat part of the same file."""
    return any(
        left.file_path == right.file_path
        and min(left.last_character_index, right.last_character_index)
        > max(left.first_character_index, right.first_character_index)
        for left, right in combinations(evidence.retrieved[:TOP_FIVE], 2)
    )


def _range_gap(left: MinimalSource, right: MinimalSource) -> int:
    """Return zero for overlap or the number of characters between ranges."""
    return max(
        right.first_character_index - left.last_character_index,
        left.first_character_index - right.last_character_index,
        0,
    )


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
