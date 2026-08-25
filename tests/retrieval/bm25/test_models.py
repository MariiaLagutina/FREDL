"""Tests for immutable BM25 inputs and parameters."""

from typing import cast

import pytest

from src.ingestion.documents import Chunk
from src.retrieval.bm25 import BM25Document, BM25Parameters


def _chunk() -> Chunk:
    """Return one exact source chunk for model tests."""
    return Chunk(
        file_path="data/raw/corpus/src/cache.py",
        start=10,
        end=15,
        text="cache",
    )


def test_parameters_expose_explicit_baseline_defaults() -> None:
    """The first lexical baseline has stable, inspectable parameters."""
    assert BM25Parameters() == BM25Parameters(
        k1=1.5,
        b=0.75,
        metadata_weight=1.0,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"k1": 0.0}, "k1"),
        ({"b": -0.1}, "b"),
        ({"b": 1.1}, "b"),
        ({"metadata_weight": -1.0}, "metadata weight"),
    ],
)
def test_parameters_reject_invalid_values(
    overrides: dict[str, float],
    message: str,
) -> None:
    """Invalid scoring domains fail before an index is built."""
    with pytest.raises(ValueError, match=message):
        BM25Parameters(**overrides)


def test_document_keeps_source_and_search_fields_separate() -> None:
    """Metadata enrichment does not modify exact source evidence."""
    document = BM25Document(
        chunk=_chunk(),
        content_terms=("cache", "lookup"),
        metadata_terms=("src", "cache", "py"),
    )

    assert document.chunk.text == "cache"
    assert document.content_terms == ("cache", "lookup")
    assert document.metadata_terms == ("src", "cache", "py")
    assert document.key == (
        "data/raw/corpus/src/cache.py",
        10,
        15,
    )


def test_document_rejects_mutable_term_lists() -> None:
    """Index inputs cannot change after corpus statistics are calculated."""
    with pytest.raises(TypeError, match="content_terms must be a tuple"):
        BM25Document(
            chunk=_chunk(),
            content_terms=cast(tuple[str, ...], ["cache"]),
        )


def test_document_rejects_empty_terms() -> None:
    """Malformed tokens fail instead of contaminating document frequency."""
    with pytest.raises(ValueError, match="metadata_terms"):
        BM25Document(
            chunk=_chunk(),
            content_terms=("cache",),
            metadata_terms=("",),
        )
