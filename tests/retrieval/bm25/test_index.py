"""Tests for two-field BM25 corpus statistics and ranking."""

from math import isclose

import pytest

from src.ingestion import Chunk
from src.retrieval.bm25 import BM25Document, BM25Index, BM25Parameters


def _document(
    file_path: str,
    content_terms: tuple[str, ...],
    metadata_terms: tuple[str, ...] = (),
) -> BM25Document:
    """Create one exact synthetic document for scoring tests."""
    text = " ".join(content_terms)
    return BM25Document(
        chunk=Chunk(
            file_path=file_path,
            start=0,
            end=len(text),
            text=text,
        ),
        content_terms=content_terms,
        metadata_terms=metadata_terms,
    )


def test_index_stores_separate_field_statistics() -> None:
    """Content and metadata use independent lengths and frequencies."""
    index = BM25Index(
        [
            _document(
                "cache.py",
                ("cache", "cache", "lookup"),
                ("src", "cache"),
            ),
            _document("database.py", ("database",), ("src",)),
        ]
    )

    assert index.statistics.document_count == 2
    assert index.statistics.average_content_length == 2.0
    assert index.statistics.average_metadata_length == 1.5
    assert dict(index.statistics.content_document_frequencies)["cache"] == 1
    assert dict(index.statistics.metadata_document_frequencies)["src"] == 2


def test_search_exposes_content_and_metadata_contributions() -> None:
    """The final score remains explainable as two field scores."""
    index = BM25Index(
        [
            _document("cache.py", ("cache", "lookup"), ("cache",)),
            _document("other.py", ("other",), ("docs",)),
        ],
        BM25Parameters(metadata_weight=1.5),
    )

    hit = index.search(("cache",), top_k=1)[0]

    assert hit.document.chunk.file_path == "cache.py"
    assert hit.content_score > 0
    assert hit.metadata_score > 0
    assert isclose(
        hit.score,
        hit.content_score + 1.5 * hit.metadata_score,
    )


def test_fractional_metadata_weight_scales_score_exactly() -> None:
    """A fractional weight multiplies a separate score without token copies."""
    document = _document("cache.py", ("unrelated",), ("cache",))
    baseline = BM25Index(
        [document],
        BM25Parameters(metadata_weight=1.0),
    ).search(("cache",))[0]
    weighted = BM25Index(
        [document],
        BM25Parameters(metadata_weight=1.5),
    ).search(("cache",))[0]

    assert baseline.content_score == 0
    assert isclose(weighted.score, baseline.score * 1.5)
    assert weighted.metadata_score == baseline.metadata_score


def test_repeated_query_terms_do_not_change_ranking_score() -> None:
    """Accidental query duplication is not an undeclared query weight."""
    index = BM25Index([_document("cache.py", ("cache", "cache"))])

    single = index.search(("cache",))[0]
    repeated = index.search(("cache", "cache"))[0]

    assert repeated.score == single.score


def test_document_term_frequency_increases_with_saturation() -> None:
    """Repeated evidence helps while BM25 prevents linear score growth."""
    index = BM25Index(
        [
            _document("once.py", ("cache",)),
            _document("twice.py", ("cache", "cache")),
        ],
        BM25Parameters(b=0),
    )

    hits = index.search(("cache",))

    assert hits[0].document.chunk.file_path == "twice.py"
    assert hits[0].score > hits[1].score
    assert hits[0].score < 2 * hits[1].score


def test_equal_scores_use_source_key_for_stable_order() -> None:
    """Ties return the same order regardless of insertion order."""
    index = BM25Index(
        [
            _document("z.py", ("cache",)),
            _document("a.py", ("cache",)),
        ]
    )

    assert [
        hit.document.chunk.file_path for hit in index.search(("cache",))
    ] == ["a.py", "z.py"]


def test_search_omits_documents_without_query_matches() -> None:
    """Top-k never fills unused places with zero-score evidence."""
    index = BM25Index([_document("cache.py", ("cache",))])

    assert index.search(("database",)) == []


def test_index_rejects_duplicate_source_spans() -> None:
    """One evidence span cannot silently occur twice in the index."""
    document = _document("cache.py", ("cache",))

    with pytest.raises(ValueError, match="keys must be unique"):
        BM25Index([document, document])


def test_search_rejects_invalid_top_k() -> None:
    """A non-positive result limit is rejected explicitly."""
    index = BM25Index([_document("cache.py", ("cache",))])

    with pytest.raises(ValueError, match="top_k"):
        index.search(("cache",), top_k=0)


def test_search_rejects_raw_string_instead_of_token_sequence() -> None:
    """A raw query cannot be silently interpreted as character terms."""
    index = BM25Index([_document("cache.py", ("cache",))])

    with pytest.raises(TypeError, match="sequence of terms"):
        index.search("cache")
