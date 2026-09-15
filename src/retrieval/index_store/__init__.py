"""Versioned persistence for lexical retrieval indexes."""

from src.retrieval.index_store.builder import IndexBuild, build_index
from src.retrieval.index_store.fingerprint import (
    fingerprint_corpus,
    fingerprint_file,
)
from src.retrieval.index_store.incremental import (
    IncrementalIndexBuild,
    build_index_incremental,
)
from src.retrieval.index_store.pipeline import (
    PipelineConfig,
    fingerprint_pipeline,
)
from src.retrieval.index_store.store import (
    IncompatibleIndexError,
    IndexStore,
    ReusableIndexSnapshot,
    SCHEMA_VERSION,
)

__all__ = [
    "build_index",
    "build_index_incremental",
    "fingerprint_corpus",
    "fingerprint_file",
    "fingerprint_pipeline",
    "IncompatibleIndexError",
    "IncrementalIndexBuild",
    "IndexBuild",
    "IndexStore",
    "PipelineConfig",
    "ReusableIndexSnapshot",
    "SCHEMA_VERSION",
]
