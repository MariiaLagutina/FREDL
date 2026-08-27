"""Tests for the public retrieval result boundary."""

import pytest

from src.ingestion import Chunk
from src.retrieval import search_sources, select_sources
from src.retrieval.bm25 import BM25Document, BM25Hit, BM25Index


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


def test_search_sources_returns_public_sources_in_ranking_order() -> None:
    """Raw-query search hides internal BM25 scores and documents."""
    index = BM25Index(
        [
            _hit("docs/cache.md", 0, "term term", 2.0).document,
            _hit("src/cache.py", 10, "term", 1.0).document,
        ]
    )

    sources = search_sources(index, "term", k=2)

    assert [source.file_path for source in sources] == [
        "docs/cache.md",
        "src/cache.py",
    ]
    assert sources[1].first_character_index == 10
    assert sources[1].last_character_index == 14


def test_search_sources_returns_empty_list_for_empty_query() -> None:
    """An empty raw query has no fabricated source matches."""
    index = BM25Index([_hit("src/cache.py", 0, "term", 1.0).document])

    assert search_sources(index, "   ", k=1) == []


def test_search_sources_rejects_non_positive_k() -> None:
    """The public search boundary reports its own invalid-limit error."""
    index = BM25Index([_hit("src/cache.py", 0, "term", 1.0).document])

    with pytest.raises(ValueError, match="Search k"):
        search_sources(index, "term", k=0)
