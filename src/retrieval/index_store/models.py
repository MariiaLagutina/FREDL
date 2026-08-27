"""Validated JSON schema for persisted BM25 indexes."""

from pydantic import BaseModel, ConfigDict


class StoredChunk(BaseModel):
    """Persist exact source evidence and structural context."""

    model_config = ConfigDict(frozen=True, strict=True)

    file_path: str
    start: int
    end: int
    text: str
    section_path: tuple[str, ...]


class StoredDocument(BaseModel):
    """Persist one chunk and its already-tokenized lexical fields."""

    model_config = ConfigDict(frozen=True, strict=True)

    chunk: StoredChunk
    content_terms: tuple[str, ...]
    metadata_terms: tuple[str, ...]


class StoredParameters(BaseModel):
    """Persist all values that affect BM25 scores."""

    model_config = ConfigDict(frozen=True, strict=True)

    k1: float
    b: float
    metadata_weight: float


class StoredBM25Index(BaseModel):
    """Define the complete versioned BM25 snapshot envelope."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_version: int
    corpus_fingerprint: str
    pipeline_fingerprint: str
    parameters: StoredParameters
    documents: tuple[StoredDocument, ...]
