"""Validated schemas for the local answer cache."""

from pydantic import BaseModel, ConfigDict

from src.models import MinimalSource


class AnswerCacheKey(BaseModel):
    """Identify every input that can change one generated answer."""

    model_config = ConfigDict(frozen=True, strict=True)

    question: str
    corpus_fingerprint: str
    pipeline_fingerprint: str
    k: int
    context_token_budget: int
    auxiliary_path_penalty: float
    path_candidate_depth: int
    prompt_version: str
    model_name: str
    device: str
    max_new_tokens: int
    local_files_only: bool
    enable_thinking: bool
    do_sample: bool


class StoredValidatedAnswer(BaseModel):
    """Persist one answer that already crossed the grounding boundary."""

    model_config = ConfigDict(frozen=True, strict=True)

    key: AnswerCacheKey
    answer: str
    retrieved_sources: tuple[MinimalSource, ...]
    context_sources: tuple[MinimalSource, ...]
    used_context_tokens: int
    skipped_source_count: int
    generation_device: str


class StoredAnswerCache(BaseModel):
    """Define the complete versioned cache envelope."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_version: int
    entries: tuple[StoredValidatedAnswer, ...]
