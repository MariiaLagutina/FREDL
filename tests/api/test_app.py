"""Tests for the local HTTP API application."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.api import create_app
from src.ingestion import Chunk, discover_files
from src.retrieval.bm25 import BM25Document, BM25Index
from src.retrieval.index_store import (
    SCHEMA_VERSION,
    IndexStore,
    PipelineConfig,
    fingerprint_corpus,
    fingerprint_pipeline,
)


def _build_app(tmp_path: Path) -> FastAPI:
    """Persist one compatible index and build a real app around it."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "cache.md").write_text(
        "cache implementation notes", encoding="utf-8"
    )

    pipeline_config = PipelineConfig()
    manifest = discover_files(tmp_path, corpus_root)
    corpus_fingerprint = fingerprint_corpus(tmp_path, manifest)
    pipeline_fingerprint = fingerprint_pipeline(
        pipeline_config,
        index_schema_version=SCHEMA_VERSION,
    )
    IndexStore(tmp_path / "bm25-index.json").save(
        BM25Index(
            [
                BM25Document(
                    chunk=Chunk("corpus/cache.md", 0, 5, "cache"),
                    content_terms=("cache",),
                )
            ]
        ),
        corpus_fingerprint,
        pipeline_fingerprint,
    )
    return create_app(
        project_root=tmp_path,
        corpus_root=Path("corpus"),
        index_path=Path("bm25-index.json"),
        pipeline_config=pipeline_config,
    )


def test_health_reports_ok(tmp_path: Path) -> None:
    client = TestClient(_build_app(tmp_path))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_sources_from_the_preloaded_index(
    tmp_path: Path,
) -> None:
    client = TestClient(_build_app(tmp_path))

    response = client.post(
        "/search", json={"question": "Where is cache?", "k": 1}
    )

    assert response.status_code == 200
    assert response.json() == {
        "sources": [
            {
                "file_path": "corpus/cache.md",
                "first_character_index": 0,
                "last_character_index": 5,
            }
        ]
    }


def test_search_rejects_empty_question(tmp_path: Path) -> None:
    client = TestClient(_build_app(tmp_path))

    response = client.post("/search", json={"question": "   ", "k": 1})

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_search_rejects_non_positive_k(tmp_path: Path) -> None:
    client = TestClient(_build_app(tmp_path))

    response = client.post("/search", json={"question": "cache", "k": 0})

    assert response.status_code == 400
    assert "greater than zero" in response.json()["detail"]


def test_create_app_fails_fast_without_a_compatible_index(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        create_app(
            project_root=tmp_path,
            corpus_root=Path("missing-corpus"),
            index_path=Path("missing-index.json"),
        )
