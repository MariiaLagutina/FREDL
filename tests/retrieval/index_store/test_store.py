"""Tests for BM25 index persistence and compatibility checks."""

import json
from pathlib import Path

import pytest

from src.ingestion import Chunk
from src.retrieval.bm25 import BM25Document, BM25Index, BM25Parameters
from src.retrieval.index_store import IncompatibleIndexError, IndexStore

FINGERPRINT = "a" * 64


def _index() -> BM25Index:
    """Build a small index with scores affected by both lexical fields."""
    documents = [
        BM25Document(
            chunk=Chunk("docs/cache.md", 0, 5, "cache", ("Cache",)),
            content_terms=("cache",),
            metadata_terms=("docs", "cache"),
        ),
        BM25Document(
            chunk=Chunk("src/cache.py", 0, 6, "lookup"),
            content_terms=("lookup",),
            metadata_terms=("src", "cache"),
        ),
    ]
    return BM25Index(
        documents,
        BM25Parameters(k1=1.2, b=0.5, metadata_weight=1.5),
    )


def _ranking(index: BM25Index) -> list[tuple[str, float]]:
    """Return stable ranking evidence for a fixed normalized query."""
    return [
        (hit.document.chunk.file_path, hit.score)
        for hit in index.search(("cache",), top_k=2)
    ]


def test_save_load_preserves_parameters_documents_and_ranking(
    tmp_path: Path,
) -> None:
    """A fresh runtime index reproduces the exact stored top-k scores."""
    original = _index()
    store = IndexStore(tmp_path / "bm25-index.json")

    store.save(original, FINGERPRINT)
    loaded = store.load(FINGERPRINT)

    assert loaded is not original
    assert loaded.parameters == original.parameters
    assert loaded.documents == original.documents
    assert _ranking(loaded) == _ranking(original)


def test_load_requires_reindex_for_incompatible_schema(
    tmp_path: Path,
) -> None:
    """An unknown schema version cannot be interpreted silently."""
    path = tmp_path / "bm25-index.json"
    store = IndexStore(path)
    store.save(_index(), FINGERPRINT)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IncompatibleIndexError, match="reindex required"):
        store.load(FINGERPRINT)


def test_load_requires_reindex_for_changed_corpus(tmp_path: Path) -> None:
    """A valid index cannot be reused for different source content."""
    store = IndexStore(tmp_path / "bm25-index.json")
    store.save(_index(), FINGERPRINT)

    with pytest.raises(IncompatibleIndexError, match="corpus differs"):
        store.load("b" * 64)
