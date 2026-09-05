"""Retrieval evaluation with exact source-aware relevance rules."""

from src.evaluation.retrieval.error_analysis import (
    assess_reference_matchability,
    classify_ranked_miss,
    classify_structural_miss,
    collect_top_five_misses,
)
from src.evaluation.retrieval.error_models import (
    RetrievalErrorAnalysisReport,
    RetrievalErrorCategory,
    RetrievalMissAnalysis,
    RetrievalMissEvidence,
    ReferenceMatchability,
)
from src.evaluation.retrieval.error_report import (
    render_error_analysis_markdown,
    write_error_analysis_markdown,
)
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
    "RetrievalErrorAnalysisReport",
    "RetrievalErrorCategory",
    "RetrievalEvaluationReport",
    "RetrievalMetrics",
    "RetrievalMissAnalysis",
    "RetrievalMissEvidence",
    "ReferenceMatchability",
    "RetrievalEvaluationCase",
    "RetrievalQueryMetrics",
    "aggregate_query_metrics",
    "assess_reference_matchability",
    "classify_ranked_miss",
    "classify_structural_miss",
    "collect_top_five_misses",
    "evaluate_cases",
    "evaluate_query",
    "load_evaluation_cases",
    "render_error_analysis_markdown",
    "source_iou",
    "sources_match",
    "write_error_analysis_markdown",
]
