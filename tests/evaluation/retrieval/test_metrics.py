"""Tests for Moulinette-compatible retrieval metrics."""

from math import isclose

import pytest

from src.evaluation.retrieval import (
    RetrievalQueryMetrics,
    aggregate_query_metrics,
    evaluate_query,
    source_iou,
    sources_match,
)
from src.ingestion import Chunk
from src.models import MinimalSource


def _chunk(file_path: str, start: int, end: int) -> Chunk:
    """Create one synthetic retrieved source range."""
    return Chunk(file_path, start, end, "x" * (end - start))


def _source(file_path: str, start: int, end: int) -> MinimalSource:
    """Create one synthetic reference source range."""
    return MinimalSource(
        file_path=file_path,
        first_character_index=start,
        last_character_index=end,
    )


def test_source_match_requires_exact_file_path() -> None:
    """Equal coordinates in different files are never relevant."""
    retrieved = _chunk("src/cache.py", 0, 100)
    reference = _source("src/store.py", 0, 100)

    assert source_iou(retrieved, reference) == 0
    assert not sources_match(retrieved, reference)


def test_source_match_accepts_exact_iou_threshold() -> None:
    """An IoU of exactly 0.05 satisfies the Moulinette rule."""
    retrieved = _chunk("src/cache.py", 0, 105)
    reference = _source("src/cache.py", 95, 200)

    assert isclose(source_iou(retrieved, reference), 0.05)
    assert sources_match(retrieved, reference)


def test_source_match_rejects_overlap_below_threshold() -> None:
    """Any intersection alone is insufficient for relevance."""
    retrieved = _chunk("src/cache.py", 0, 105)
    reference = _source("src/cache.py", 96, 201)

    assert source_iou(retrieved, reference) < 0.05
    assert not sources_match(retrieved, reference)


def test_source_iou_rejects_invalid_reference_range() -> None:
    """Broken labels fail instead of silently corrupting retrieval metrics."""
    retrieved = _chunk("src/cache.py", 0, 20)
    reference = _source("src/cache.py", 10, 10)

    with pytest.raises(ValueError, match="range must be positive"):
        source_iou(retrieved, reference)


def test_query_metrics_cover_distinct_references_by_rank() -> None:
    """Recall measures covered references while MRR uses the first match."""
    references = [
        _source("src/cache.py", 0, 20),
        _source("src/store.py", 40, 60),
    ]
    retrieved = [
        _chunk("src/cache.py", 0, 20),
        _chunk("src/noise.py", 0, 20),
        _chunk("src/store.py", 40, 60),
    ]

    metrics = evaluate_query(retrieved, references)

    assert metrics.recall_at_1 == 0.5
    assert metrics.recall_at_3 == 1.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.recall_at_10 == 1.0
    assert metrics.reciprocal_rank == 1.0


def test_query_metrics_report_later_first_relevant_rank() -> None:
    """MRR distinguishes a relevant second hit from a top-ranked hit."""
    metrics = evaluate_query(
        [
            _chunk("src/noise.py", 0, 20),
            _chunk("src/cache.py", 0, 20),
        ],
        [_source("src/cache.py", 0, 20)],
    )

    assert metrics.recall_at_1 == 0
    assert metrics.recall_at_3 == 1
    assert metrics.reciprocal_rank == 0.5


def test_aggregate_metrics_average_one_fixed_query_group() -> None:
    """Documentation or code query metrics remain independently reportable."""
    metrics = aggregate_query_metrics(
        [
            RetrievalQueryMetrics(1, 1, 1, 1, 1),
            RetrievalQueryMetrics(0, 1, 1, 1, 0.5),
        ]
    )

    assert metrics.query_count == 2
    assert metrics.recall_at_1 == 0.5
    assert metrics.recall_at_3 == 1.0
    assert metrics.mean_reciprocal_rank == 0.75


def test_query_metrics_require_reference_sources() -> None:
    """A query without relevance labels cannot produce recall."""
    with pytest.raises(ValueError, match="reference sources"):
        evaluate_query([], [])
