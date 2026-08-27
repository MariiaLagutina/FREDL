"""Versioned persistence for lexical retrieval indexes."""

from src.retrieval.index_store.fingerprint import fingerprint_corpus
from src.retrieval.index_store.pipeline import (
    PipelineConfig,
    fingerprint_pipeline,
)
from src.retrieval.index_store.store import (
    IncompatibleIndexError,
    IndexStore,
)

__all__ = [
    "fingerprint_corpus",
    "fingerprint_pipeline",
    "IncompatibleIndexError",
    "IndexStore",
    "PipelineConfig",
]
