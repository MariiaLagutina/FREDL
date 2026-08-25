"""Tests for preparing chunks as two-field BM25 documents."""

import pytest

from src.ingestion import Chunk, FileKind, SourceDocument
from src.retrieval.bm25 import build_bm25_documents


def test_text_document_uses_body_path_and_heading_terms() -> None:
    """Documentation content and structural metadata remain separate."""
    source = "# Cache Setup\nUse cache.\n"
    document = SourceDocument(
        file_path="data/raw/corpus/docs/cache.md",
        kind=FileKind.TEXT,
        text=source,
    )
    chunk = Chunk(
        file_path=document.file_path,
        start=0,
        end=len(source),
        text=source,
        section_path=("Cache Setup",),
    )

    result = build_bm25_documents(document, [chunk])[0]

    assert result.content_terms == ("cache", "setup", "use", "cache")
    assert result.metadata_terms == (
        "data",
        "raw",
        "corpus",
        "docs",
        "cache.md",
        "cache",
        "md",
        "setup",
    )


def test_python_body_chunk_inherits_qualified_symbol_terms() -> None:
    """A split method body remains searchable by its containing symbols."""
    source = (
        "class CacheStore:\n"
        "    def get_item(self):\n"
        "        return self.value\n"
    )
    document = SourceDocument(
        file_path="data/raw/corpus/cache_store.py",
        kind=FileKind.PYTHON,
        text=source,
    )
    start = source.index("return")
    end = source.index("\n", start)
    chunk = Chunk(
        file_path=document.file_path,
        start=start,
        end=end,
        text=source[start:end],
    )

    result = build_bm25_documents(document, [chunk])[0]

    assert result.content_terms == (
        "return",
        "self.value",
        "self",
        "value",
    )
    assert "cachestore" in result.metadata_terms
    assert "cachestore.get_item" in result.metadata_terms
    assert "get_item" in result.metadata_terms


def test_builder_skips_chunks_without_content_terms() -> None:
    """Path metadata cannot make punctuation-only evidence searchable."""
    document = SourceDocument(
        file_path="data/raw/corpus/fragment.py",
        kind=FileKind.PYTHON,
        text=")",
    )
    chunk = Chunk(
        file_path=document.file_path,
        start=0,
        end=1,
        text=")",
    )

    assert build_bm25_documents(document, [chunk]) == []


def test_builder_rejects_chunk_from_another_document() -> None:
    """Retrieval preparation cannot detach evidence from its source file."""
    document = SourceDocument(
        file_path="data/raw/corpus/one.py",
        kind=FileKind.PYTHON,
        text="value = 1\n",
    )
    chunk = Chunk(
        file_path="data/raw/corpus/two.py",
        start=0,
        end=5,
        text="value",
    )

    with pytest.raises(ValueError, match="paths must match"):
        build_bm25_documents(document, [chunk])
