"""Tests for the public retrieval result boundary."""

import pytest

from src.ingestion import Chunk
from src.retrieval import select_sources
from src.retrieval.bm25 import BM25Document, BM25Hit


def _hit(file_path: str, start: int, text: str, score: float) -> BM25Hit:
    """Create one ranked hit with exact synthetic source evidence."""
    document = BM25Document(
        chunk=Chunk(file_path, start, start + len(text), text),
        content_terms=("term",),
    )
    return BM25Hit(document, score, score, 0.0)


def test_select_sources_preserves_ranking_coordinates_and_limit() -> None:
    """Public sources keep exact evidence from the highest-ranked hits."""
    hits = [
        _hit("data/raw/cache.py", 10, "cache", 3.0),
        _hit("data/raw/store.py", 20, "store", 2.0),
        _hit("data/raw/other.py", 30, "other", 1.0),
    ]

    sources = select_sources(hits, k=2)

    assert [source.file_path for source in sources] == [
        "data/raw/cache.py",
        "data/raw/store.py",
    ]
    assert sources[0].first_character_index == 10
    assert sources[0].last_character_index == 15


def test_select_sources_skips_exact_duplicates_before_applying_limit() -> None:
    """A duplicate hit does not consume one of the requested result slots."""
    first = _hit("data/raw/cache.py", 10, "cache", 3.0)
    duplicate = BM25Hit(first.document, 2.5, 2.5, 0.0)
    next_hit = _hit("data/raw/store.py", 20, "store", 2.0)

    sources = select_sources([first, duplicate, next_hit], k=2)

    assert [source.file_path for source in sources] == [
        "data/raw/cache.py",
        "data/raw/store.py",
    ]


def test_select_sources_rejects_non_positive_k() -> None:
    """The result boundary rejects a request for no result slots."""
    with pytest.raises(ValueError, match="k must be greater than zero"):
        select_sources([], k=0)
