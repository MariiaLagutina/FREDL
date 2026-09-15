"""Validated schemas for the local search-result cache."""

from pydantic import BaseModel, ConfigDict

from src.models import MinimalSource


class SearchCacheKey(BaseModel):
    """Identify every input that can change one search result."""

    model_config = ConfigDict(frozen=True, strict=True)

    question: str
    corpus_fingerprint: str
    pipeline_fingerprint: str
    k: int
    identifier_match_weight: float
    identifier_candidate_depth: int
    auxiliary_path_penalty: float
    path_candidate_depth: int


class StoredSearchResult(BaseModel):
    """Persist one cached search result behind its exact key."""

    model_config = ConfigDict(frozen=True, strict=True)

    key: SearchCacheKey
    retrieved_sources: tuple[MinimalSource, ...]


class StoredSearchCache(BaseModel):
    """Define the complete versioned cache envelope."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_version: int
    entries: tuple[StoredSearchResult, ...]
