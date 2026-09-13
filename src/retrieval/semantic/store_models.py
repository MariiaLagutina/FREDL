"""Validated metadata schema for a persisted semantic index."""

from pydantic import BaseModel, ConfigDict


class StoredSemanticChunk(BaseModel):
    """Persist the exact source span associated with one vector row."""

    model_config = ConfigDict(frozen=True, strict=True)

    file_path: str
    start: int
    end: int
    text: str
    section_path: tuple[str, ...]


class StoredSemanticIndex(BaseModel):
    """Describe one versioned semantic vector snapshot."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_version: int
    corpus_fingerprint: str
    pipeline_fingerprint: str
    model_name: str
    model_revision: str
    dimension: int
    embeddings_sha256: str
    documents: tuple[StoredSemanticChunk, ...]
