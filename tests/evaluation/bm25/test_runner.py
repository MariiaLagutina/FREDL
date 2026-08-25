"""Tests for the fixed mini-suite and BM25 experiment runner."""

from pathlib import Path

from src.evaluation.bm25.models import EvaluationSuite, QueryRunResult
from src.evaluation.bm25.runner import run_experiment
from src.evaluation.bm25.suite import build_suite_documents, load_suite
from src.retrieval.bm25 import BM25Document, BM25Parameters


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUITE_ROOT = PROJECT_ROOT / "evals" / "bm25" / "mini"


def _prepared_suite() -> tuple[EvaluationSuite, list[BM25Document]]:
    """Load the versioned mini-suite through normal corpus ingestion."""
    suite = load_suite(SUITE_ROOT)
    documents = build_suite_documents(PROJECT_ROOT, SUITE_ROOT, suite)
    return suite, documents


def _query_result(
    results: tuple[QueryRunResult, ...],
    query_id: str,
) -> QueryRunResult:
    """Return one named query result from a complete experiment."""
    return next(
        result for result in results if result.query.query_id == query_id
    )


def test_mini_suite_indexes_fixed_docs_and_code_corpus() -> None:
    """The inspectable suite contains both query groups and eight files."""
    suite, documents = _prepared_suite()

    assert len(suite.queries) == 4
    assert len({document.chunk.file_path for document in documents}) == 8
    assert len(documents) == 9


def test_runner_reports_documentation_and_code_separately() -> None:
    """One run never hides either required dataset behind a combined mean."""
    suite, documents = _prepared_suite()

    result = run_experiment(
        suite,
        documents,
        "M0",
        BM25Parameters(metadata_weight=1.0),
        latency_repetitions=1,
    )

    assert result.documentation_metrics.query_count == 2
    assert result.code_metrics.query_count == 2
    assert result.documentation_metrics.recall_at_1 == 1.0
    assert result.code_metrics.recall_at_1 == 0.5
    assert result.source_file_count == 8
    assert result.document_count == 9
    assert result.index_size_bytes > 0
    assert result.peak_build_memory_bytes > 0


def test_mini_suite_exposes_metadata_improvement_and_regression() -> None:
    """A strong metadata weight helps one query and harms another."""
    suite, documents = _prepared_suite()
    baseline = run_experiment(
        suite,
        documents,
        "M0",
        BM25Parameters(metadata_weight=1.0),
        latency_repetitions=1,
    )
    strong_weight = run_experiment(
        suite,
        documents,
        "M3",
        BM25Parameters(metadata_weight=3.0),
        latency_repetitions=1,
    )

    baseline_cache = _query_result(
        baseline.query_results,
        "code-cache-store",
    )
    weighted_cache = _query_result(
        strong_weight.query_results,
        "code-cache-store",
    )
    baseline_retry = _query_result(
        baseline.query_results,
        "docs-request-retry-delay",
    )
    weighted_retry = _query_result(
        strong_weight.query_results,
        "docs-request-retry-delay",
    )

    assert baseline_cache.metrics.reciprocal_rank == 0.5
    assert weighted_cache.metrics.reciprocal_rank == 1.0
    assert baseline_retry.metrics.reciprocal_rank == 1.0
    assert weighted_retry.metrics.reciprocal_rank == 0.5
