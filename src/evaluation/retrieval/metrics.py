"""Calculate source-aware retrieval metrics using Moulinette overlap."""

from collections.abc import Sequence
from typing import TypeAlias

from src.evaluation.retrieval.models import (
    RetrievalMetrics,
    RetrievalQueryMetrics,
)
from src.ingestion import Chunk
from src.models import MinimalSource


MOULINETTE_IOU_THRESHOLD = 0.05
RetrievedSource: TypeAlias = Chunk | MinimalSource


def source_iou(
    retrieved: RetrievedSource,
    reference: MinimalSource,
) -> float:
    """Return intersection over union for two half-open source ranges."""
    retrieved_path, retrieved_start, retrieved_end = _retrieved_range(
        retrieved
    )
    reference_length = (
        reference.last_character_index - reference.first_character_index
    )
    if reference.first_character_index < 0 or reference_length <= 0:
        raise ValueError("Reference source range must be positive")
    if retrieved_path != reference.file_path:
        return 0.0
    intersection = max(
        0,
        min(retrieved_end, reference.last_character_index)
        - max(retrieved_start, reference.first_character_index),
    )
    retrieved_length = retrieved_end - retrieved_start
    union = retrieved_length + reference_length - intersection
    return intersection / union


def sources_match(
    retrieved: RetrievedSource,
    reference: MinimalSource,
) -> bool:
    """Apply exact path equality and the Moulinette IoU threshold."""
    return source_iou(retrieved, reference) >= MOULINETTE_IOU_THRESHOLD


def evaluate_query(
    retrieved: Sequence[RetrievedSource],
    references: Sequence[MinimalSource],
) -> RetrievalQueryMetrics:
    """Evaluate ranked chunks against every relevant reference source."""
    if not references:
        raise ValueError("Retrieval evaluation requires reference sources")
    return RetrievalQueryMetrics(
        recall_at_1=_recall_at_k(retrieved, references, 1),
        recall_at_3=_recall_at_k(retrieved, references, 3),
        recall_at_5=_recall_at_k(retrieved, references, 5),
        recall_at_10=_recall_at_k(retrieved, references, 10),
        reciprocal_rank=_reciprocal_rank(retrieved, references),
    )


def aggregate_query_metrics(
    query_metrics: Sequence[RetrievalQueryMetrics],
) -> RetrievalMetrics:
    """Average one already separated documentation or code query group."""
    if not query_metrics:
        raise ValueError("Retrieval metrics require at least one query")
    query_count = len(query_metrics)
    return RetrievalMetrics(
        query_count=query_count,
        recall_at_1=sum(item.recall_at_1 for item in query_metrics)
        / query_count,
        recall_at_3=sum(item.recall_at_3 for item in query_metrics)
        / query_count,
        recall_at_5=sum(item.recall_at_5 for item in query_metrics)
        / query_count,
        recall_at_10=sum(item.recall_at_10 for item in query_metrics)
        / query_count,
        mean_reciprocal_rank=sum(
            item.reciprocal_rank for item in query_metrics
        )
        / query_count,
    )


def _recall_at_k(
    retrieved: Sequence[RetrievedSource],
    references: Sequence[MinimalSource],
    k: int,
) -> float:
    """Return the fraction of reference spans covered by the first k hits."""
    top_k = retrieved[:k]
    matched_count = sum(
        any(sources_match(hit, reference) for hit in top_k)
        for reference in references
    )
    return matched_count / len(references)


def _reciprocal_rank(
    retrieved: Sequence[RetrievedSource],
    references: Sequence[MinimalSource],
) -> float:
    """Return the inverse rank of the first source-overlapping result."""
    for rank, hit in enumerate(retrieved, start=1):
        if any(sources_match(hit, reference) for reference in references):
            return 1.0 / rank
    return 0.0


def _retrieved_range(
    source: RetrievedSource,
) -> tuple[str, int, int]:
    """Normalize internal chunks and persisted sources to one range."""
    if isinstance(source, Chunk):
        return source.file_path, source.start, source.end
    return (
        source.file_path,
        source.first_character_index,
        source.last_character_index,
    )
