"""Retrieval evaluation with exact source-aware relevance rules."""

from src.evaluation.retrieval.metrics import (
    aggregate_query_metrics,
    evaluate_query,
    source_iou,
    sources_match,
)
from src.evaluation.retrieval.models import (
    RetrievalMetrics,
    RetrievalQueryMetrics,
)

__all__ = [
    "RetrievalMetrics",
    "RetrievalQueryMetrics",
    "aggregate_query_metrics",
    "evaluate_query",
    "source_iou",
    "sources_match",
]
