"""Request and response schemas for the local HTTP API."""

from pydantic import BaseModel, ConfigDict, Field

from src.models import MinimalSource
from src.retrieval import (
    DEFAULT_AUXILIARY_PATH_PENALTY,
    DEFAULT_PATH_CANDIDATE_DEPTH,
)

DEFAULT_CONTEXT_TOKEN_BUDGET = 4096


class SearchRequest(BaseModel):
    """Describe one raw search query submitted over HTTP."""

    model_config = ConfigDict(strict=True)

    question: str
    k: int = 5
    identifier_match_weight: float = 0.0
    identifier_candidate_depth: int = 0
    auxiliary_path_penalty: float = DEFAULT_AUXILIARY_PATH_PENALTY
    path_candidate_depth: int = DEFAULT_PATH_CANDIDATE_DEPTH


class SearchResponse(BaseModel):
    """Return the ranked sources for one search query."""

    model_config = ConfigDict(strict=True)

    sources: list[MinimalSource]


class AnswerRequest(BaseModel):
    """Describe one raw question submitted for grounded generation."""

    model_config = ConfigDict(strict=True)

    question: str
    k: int = 5
    context_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET
    auxiliary_path_penalty: float = DEFAULT_AUXILIARY_PATH_PENALTY
    path_candidate_depth: int = DEFAULT_PATH_CANDIDATE_DEPTH


class AnswerResponse(BaseModel):
    """Return one grounded answer with its retrieval evidence."""

    model_config = ConfigDict(strict=True)

    answer: str
    sources: list[MinimalSource]
    retrieved_sources: list[MinimalSource]
    used_context_tokens: int
    skipped_source_count: int
    prompt_version: str
    cache_hit: bool


class HealthResponse(BaseModel):
    """Report that the server process is reachable."""

    model_config = ConfigDict(strict=True)

    status: str = Field(default="ok")
