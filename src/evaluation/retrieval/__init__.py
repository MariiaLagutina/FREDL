"""Retrieval evaluation with exact source-aware relevance rules."""

from src.evaluation.retrieval.evaluator import evaluate_cases
from src.evaluation.retrieval.metrics import (
    aggregate_query_metrics,
    evaluate_query,
    source_iou,
    sources_match,
)
from src.evaluation.retrieval.models import (
    RetrievalDatasetKind,
    RetrievalEvaluationCase,
    RetrievalEvaluationReport,
    RetrievalMetrics,
    RetrievalQueryMetrics,
)
from src.evaluation.retrieval.workflow import load_evaluation_cases

__all__ = [
    "RetrievalDatasetKind",
    "RetrievalEvaluationReport",
    "RetrievalMetrics",
    "RetrievalEvaluationCase",
    "RetrievalQueryMetrics",
    "aggregate_query_metrics",
    "evaluate_cases",
    "evaluate_query",
    "load_evaluation_cases",
    "source_iou",
    "sources_match",
]
