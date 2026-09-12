"""Tests for raw-query BM25 retrieval."""

import pytest

from src.ingestion import Chunk
from src.retrieval.bm25 import (
    BM25Document,
    BM25Index,
    BM25Parameters,
    BM25Retriever,
)


def _document(
    file_path: str,
    content_terms: tuple[str, ...],
    metadata_terms: tuple[str, ...],
) -> BM25Document:
    """Create one synthetic exact chunk for retrieval tests."""
    text = " ".join(content_terms)
    return BM25Document(
        chunk=Chunk(file_path, 0, len(text), text),
        content_terms=content_terms,
        metadata_terms=metadata_terms,
    )


def test_raw_symbol_query_ranks_qualified_metadata_first() -> None:
    """Exact structural evidence outranks a partial content match."""
    retriever = BM25Retriever(
        BM25Index(
            [
                _document(
                    "cache_store.py",
                    ("return", "value"),
                    ("cachestore.get_item", "get_item"),
                ),
                _document(
                    "cache.md",
                    ("cache", "configuration"),
                    ("docs",),
                ),
            ],
            BM25Parameters(metadata_weight=1.5),
        )
    )

    hits = retriever.search("CacheStore.get_item")

    assert hits[0].document.chunk.file_path == "cache_store.py"
    assert hits[0].metadata_score > 0


def test_blank_raw_query_is_rejected() -> None:
    """Whitespace-only input is invalid rather than a successful search."""
    retriever = BM25Retriever(
        BM25Index([_document("cache.py", ("cache",), ())])
    )

    with pytest.raises(ValueError, match="must not be empty"):
        retriever.search("   ")


def test_nonblank_query_without_lexical_terms_is_rejected() -> None:
    """Punctuation-only input is not a silent successful search."""
    retriever = BM25Retriever(
        BM25Index([_document("cache.py", ("cache",), ())])
    )

    with pytest.raises(ValueError, match="contain searchable text"):
        retriever.search("!!!")
