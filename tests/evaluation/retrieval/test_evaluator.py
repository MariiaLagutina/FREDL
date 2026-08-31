"""Tests for labelled persisted retrieval dataset evaluation."""

from src.evaluation.retrieval import (
    RetrievalDatasetKind,
    RetrievalEvaluationCase,
    evaluate_cases,
)
from src.models import MinimalSource


def _source(file_path: str, start: int, end: int) -> MinimalSource:
    """Create one reference or retrieved source range."""
    return MinimalSource(
        file_path=file_path,
        first_character_index=start,
        last_character_index=end,
    )


def test_evaluate_cases_matches_manual_two_query_calculation() -> None:
    """Aggregate persisted metrics match a transparent manual example."""
    report = evaluate_cases(
        RetrievalDatasetKind.DOCS,
        [
            RetrievalEvaluationCase(
                question_id="q-1",
                question="Where are cache and store documented?",
                references=(
                    _source("docs/cache.md", 0, 20),
                    _source("docs/store.md", 40, 60),
                ),
                retrieved=(
                    _source("docs/cache.md", 0, 20),
                    _source("docs/noise.md", 0, 20),
                    _source("docs/store.md", 40, 60),
                ),
            ),
            RetrievalEvaluationCase(
                question_id="q-2",
                question="Where is scheduling documented?",
                references=(_source("docs/scheduler.md", 0, 20),),
                retrieved=(
                    _source("docs/noise.md", 0, 20),
                    _source("docs/scheduler.md", 0, 20),
                ),
            ),
        ],
    )

    assert report.dataset is RetrievalDatasetKind.DOCS
    assert report.metrics.query_count == 2
    assert report.metrics.recall_at_1 == 0.25
    assert report.metrics.recall_at_3 == 1.0
    assert report.metrics.recall_at_5 == 1.0
    assert report.metrics.recall_at_10 == 1.0
    assert report.metrics.mean_reciprocal_rank == 0.75
