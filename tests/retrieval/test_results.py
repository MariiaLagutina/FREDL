"""Tests for the public retrieval result boundary."""

import pytest

from src.ingestion import Chunk
from src.models import RagDataset, UnansweredQuestion
from src.retrieval import (
    search_dataset,
    search_question,
    search_sources,
    select_sources,
)
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


def test_search_sources_rejects_blank_query() -> None:
    """The public boundary rejects a whitespace-only user query."""
    index = BM25Index([_hit("src/cache.py", 0, "term", 1.0).document])

    with pytest.raises(ValueError, match="non-whitespace"):
        search_sources(index, "   ", k=1)


def test_search_sources_allows_nonblank_query_without_terms() -> None:
    """Punctuation-only input is valid but has no lexical matches."""
    index = BM25Index([_hit("src/cache.py", 0, "term", 1.0).document])

    assert search_sources(index, "!!!", k=1) == []


def test_search_sources_rejects_non_positive_k() -> None:
    """The public search boundary reports its own invalid-limit error."""
    index = BM25Index([_hit("src/cache.py", 0, "term", 1.0).document])

    with pytest.raises(ValueError, match="Search k"):
        search_sources(index, "term", k=0)


def test_search_question_preserves_identity_and_source_coordinates() -> None:
    """One dataset question becomes one traceable public search result."""
    index = BM25Index(
        [_hit("src/cache.py", 10, "term", 1.0).document]
    )
    question = UnansweredQuestion(
        question_id="question-7",
        question="Where is the term?",
    )

    result = search_question(index, question, k=1)

    assert result.question_id == "question-7"
    assert result.question == "Where is the term?"
    assert result.retrieved_sources[0].model_dump() == {
        "file_path": "src/cache.py",
        "first_character_index": 10,
        "last_character_index": 14,
    }


def test_search_dataset_preserves_question_order_and_reports_k() -> None:
    """Batch search reuses one index and produces the submission model."""
    index = BM25Index([_hit("src/cache.py", 0, "term", 1.0).document])
    dataset = RagDataset(
        rag_questions=[
            UnansweredQuestion(question_id="q-1", question="First term?"),
            UnansweredQuestion(question_id="q-2", question="Second term?"),
        ]
    )

    result = search_dataset(index, dataset, k=1)

    assert result.k == 1
    assert [item.question_id for item in result.search_results] == [
        "q-1",
        "q-2",
    ]
    assert all(
        len(item.retrieved_sources) == 1
        for item in result.search_results
    )


def test_search_dataset_rejects_invalid_k_for_empty_dataset() -> None:
    """An empty batch cannot bypass validation of the requested limit."""
    index = BM25Index([])
    dataset = RagDataset(rag_questions=[])

    with pytest.raises(ValueError, match="Search k"):
        search_dataset(index, dataset, k=0)
