"""Versioned persistence for lexical retrieval indexes."""

from src.retrieval.index_store.store import (
    IncompatibleIndexError,
    IndexStore,
)

__all__ = ["IncompatibleIndexError", "IndexStore"]
