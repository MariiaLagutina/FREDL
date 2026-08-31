"""Evaluate aligned persisted retrieval results as one labelled dataset."""

from collections.abc import Sequence

from src.evaluation.retrieval.metrics import (
    aggregate_query_metrics,
    evaluate_query,
)
from src.evaluation.retrieval.models import (
    RetrievalDatasetKind,
    RetrievalEvaluationCase,
    RetrievalEvaluationReport,
)


def evaluate_cases(
    dataset: RetrievalDatasetKind,
    cases: Sequence[RetrievalEvaluationCase],
) -> RetrievalEvaluationReport:
    """Evaluate and aggregate one required Docs or Code dataset."""
    query_metrics = [
        evaluate_query(case.retrieved, case.references)
        for case in cases
    ]
    return RetrievalEvaluationReport(
        dataset=dataset,
        metrics=aggregate_query_metrics(query_metrics),
    )
