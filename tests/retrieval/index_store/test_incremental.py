"""Tests for reusing unchanged files across BM25 index rebuilds."""

from pathlib import Path

from src.retrieval.index_store import (
    IncrementalIndexBuild,
    IndexStore,
    PipelineConfig,
    build_index,
    build_index_incremental,
)

SCHEMA_VERSION = 7


def _write_corpus(corpus_root: Path) -> None:
    """Create a small, deterministic three-file corpus."""
    corpus_root.mkdir(parents=True, exist_ok=True)
    (corpus_root / "cache.md").write_text(
        "# Cache\n\nThe cache stores chunks.\n", encoding="utf-8"
    )
    (corpus_root / "index.md").write_text(
        "# Index\n\nThe index ranks chunks.\n", encoding="utf-8"
    )
    (corpus_root / "search.py").write_text(
        "def search():\n    return []\n", encoding="utf-8"
    )


def _reindex(
    project_root: Path,
    corpus_root: Path,
    index_path: Path,
    config: PipelineConfig,
) -> IncrementalIndexBuild:
    """Run one incremental build and persist its result, as the CLI does."""
    incremental = build_index_incremental(
        project_root,
        corpus_root,
        config,
        index_schema_version=SCHEMA_VERSION,
        previous_index_path=index_path,
    )
    IndexStore(index_path).save(
        incremental.build.index,
        incremental.build.corpus_fingerprint,
        incremental.build.pipeline_fingerprint,
        incremental.file_fingerprints,
    )
    return incremental


def test_first_build_has_no_prior_index_to_reuse(tmp_path: Path) -> None:
    """Bootstrapping without a prior snapshot rebuilds every file."""
    corpus_root = tmp_path / "data" / "raw"
    _write_corpus(corpus_root)
    index_path = tmp_path / "bm25-index.json"

    incremental = _reindex(
        tmp_path, corpus_root, index_path, PipelineConfig()
    )

    assert incremental.total_file_count == 3
    assert incremental.reused_file_count == 0
    assert incremental.rebuilt_file_count == 3


def test_incremental_rebuild_matches_full_rebuild_when_one_file_changes(
    tmp_path: Path,
) -> None:
    """Reusing untouched files never changes the resulting index."""
    corpus_root = tmp_path / "data" / "raw"
    _write_corpus(corpus_root)
    index_path = tmp_path / "bm25-index.json"
    config = PipelineConfig()
    _reindex(tmp_path, corpus_root, index_path, config)

    (corpus_root / "cache.md").write_text(
        "# Cache\n\nThe cache now stores validated chunks.\n",
        encoding="utf-8",
    )
    incremental = _reindex(tmp_path, corpus_root, index_path, config)

    assert incremental.total_file_count == 3
    assert incremental.reused_file_count == 2
    assert incremental.rebuilt_file_count == 1

    full = build_index(
        tmp_path, corpus_root, config, index_schema_version=SCHEMA_VERSION
    )
    assert incremental.build.index.documents == full.index.documents
    assert incremental.build.corpus_fingerprint == full.corpus_fingerprint
    assert (
        incremental.build.pipeline_fingerprint == full.pipeline_fingerprint
    )


def test_incremental_rebuild_matches_full_rebuild_when_unchanged(
    tmp_path: Path,
) -> None:
    """Reindexing an untouched corpus reuses every file and stays exact."""
    corpus_root = tmp_path / "data" / "raw"
    _write_corpus(corpus_root)
    index_path = tmp_path / "bm25-index.json"
    config = PipelineConfig()
    _reindex(tmp_path, corpus_root, index_path, config)

    incremental = _reindex(tmp_path, corpus_root, index_path, config)

    assert incremental.reused_file_count == 3
    assert incremental.rebuilt_file_count == 0
    full = build_index(
        tmp_path, corpus_root, config, index_schema_version=SCHEMA_VERSION
    )
    assert incremental.build.index.documents == full.index.documents


def test_incremental_rebuild_drops_removed_files(tmp_path: Path) -> None:
    """A file deleted from the corpus disappears from the rebuilt index."""
    corpus_root = tmp_path / "data" / "raw"
    _write_corpus(corpus_root)
    index_path = tmp_path / "bm25-index.json"
    config = PipelineConfig()
    _reindex(tmp_path, corpus_root, index_path, config)

    (corpus_root / "search.py").unlink()
    incremental = _reindex(tmp_path, corpus_root, index_path, config)

    assert incremental.total_file_count == 2
    assert incremental.reused_file_count == 2
    paths = {
        document.chunk.file_path
        for document in incremental.build.index.documents
    }
    assert all(not path.endswith("search.py") for path in paths)

    full = build_index(
        tmp_path, corpus_root, config, index_schema_version=SCHEMA_VERSION
    )
    assert incremental.build.index.documents == full.index.documents


def test_incremental_rebuild_adds_new_files(tmp_path: Path) -> None:
    """A newly added file is chunked and included, not skipped."""
    corpus_root = tmp_path / "data" / "raw"
    _write_corpus(corpus_root)
    index_path = tmp_path / "bm25-index.json"
    config = PipelineConfig()
    _reindex(tmp_path, corpus_root, index_path, config)

    (corpus_root / "new.md").write_text(
        "# New\n\nA freshly added document.\n", encoding="utf-8"
    )
    incremental = _reindex(tmp_path, corpus_root, index_path, config)

    assert incremental.total_file_count == 4
    assert incremental.reused_file_count == 3
    assert incremental.rebuilt_file_count == 1
    full = build_index(
        tmp_path, corpus_root, config, index_schema_version=SCHEMA_VERSION
    )
    assert incremental.build.index.documents == full.index.documents


def test_incremental_rebuild_falls_back_when_pipeline_changes(
    tmp_path: Path,
) -> None:
    """A different chunking configuration cannot reuse prior chunks."""
    corpus_root = tmp_path / "data" / "raw"
    _write_corpus(corpus_root)
    index_path = tmp_path / "bm25-index.json"
    _reindex(tmp_path, corpus_root, index_path, PipelineConfig())

    changed_config = PipelineConfig(max_chunk_size=10)
    incremental = _reindex(
        tmp_path, corpus_root, index_path, changed_config
    )

    assert incremental.reused_file_count == 0
    assert incremental.rebuilt_file_count == 3
    full = build_index(
        tmp_path,
        corpus_root,
        changed_config,
        index_schema_version=SCHEMA_VERSION,
    )
    assert incremental.build.index.documents == full.index.documents


def test_incremental_rebuild_falls_back_when_stored_index_is_corrupt(
    tmp_path: Path,
) -> None:
    """A corrupt prior snapshot degrades to a full rebuild, not an error."""
    corpus_root = tmp_path / "data" / "raw"
    _write_corpus(corpus_root)
    index_path = tmp_path / "bm25-index.json"
    index_path.write_text("not json", encoding="utf-8")
    config = PipelineConfig()

    incremental = build_index_incremental(
        tmp_path,
        corpus_root,
        config,
        index_schema_version=SCHEMA_VERSION,
        previous_index_path=index_path,
    )

    assert incremental.reused_file_count == 0
    assert incremental.rebuilt_file_count == 3
    full = build_index(
        tmp_path, corpus_root, config, index_schema_version=SCHEMA_VERSION
    )
    assert incremental.build.index.documents == full.index.documents
