"""Tests for reproducible production BM25 index construction."""

from pathlib import Path

from src.retrieval.bm25 import BM25Parameters
from src.retrieval.index_store import PipelineConfig, build_index


def test_build_index_runs_ingestion_with_declared_configuration(
    tmp_path: Path,
) -> None:
    """The production builder preserves sources and scoring parameters."""
    corpus_root = tmp_path / "data" / "raw"
    corpus_root.mkdir(parents=True)
    source_path = corpus_root / "guide.md"
    source_path.write_text(
        "# Cache\n\nThe cache stores chunks.\n",
        encoding="utf-8",
    )
    parameters = BM25Parameters(k1=1.2, b=0.5, metadata_weight=1.5)

    build = build_index(
        tmp_path,
        corpus_root,
        PipelineConfig(max_chunk_size=20, parameters=parameters),
        index_schema_version=2,
    )

    assert build.index.parameters == parameters
    assert build.index.documents
    paths = {
        document.chunk.file_path for document in build.index.documents
    }
    assert paths == {"data/raw/guide.md"}
    assert len(build.corpus_fingerprint) == 64
    assert len(build.pipeline_fingerprint) == 64


def test_build_index_is_deterministic(tmp_path: Path) -> None:
    """Unchanged corpus and configuration produce the same complete build."""
    corpus_root = tmp_path / "data" / "raw"
    corpus_root.mkdir(parents=True)
    (corpus_root / "notes.txt").write_text("stable text", encoding="utf-8")
    config = PipelineConfig()

    first = build_index(
        tmp_path,
        corpus_root,
        config,
        index_schema_version=2,
    )
    second = build_index(
        tmp_path,
        corpus_root,
        config,
        index_schema_version=2,
    )

    assert first.index.documents == second.index.documents
    assert first.index.parameters == second.index.parameters
    assert first.corpus_fingerprint == second.corpus_fingerprint
    assert first.pipeline_fingerprint == second.pipeline_fingerprint
