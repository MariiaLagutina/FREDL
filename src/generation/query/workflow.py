"""Orchestrate one traceable query from stored BM25 index to answer."""

from pathlib import Path

from src.generation.backend import (
    GenerationConfig,
    LoadedGenerationBackend,
    load_generation_backend,
)
from src.generation.cache import AnswerCacheKey, ValidatedAnswerCache
from src.generation.cache.store import normalize_question
from src.generation.context import HuggingFaceTokenCounter, build_context
from src.generation.prompt import (
    GROUNDING_PROMPT_VERSION,
    generate_grounded_answer,
)
from src.generation.query.models import QueryAnswerResult
from src.ingestion import discover_files
from src.retrieval import (
    DEFAULT_AUXILIARY_PATH_PENALTY,
    DEFAULT_PATH_CANDIDATE_DEPTH,
    run_stored_search,
)
from src.retrieval.index_store import (
    PipelineConfig,
    SCHEMA_VERSION,
    fingerprint_corpus,
    fingerprint_pipeline,
)


def answer_query(
    question: str,
    project_root: Path,
    corpus_root: Path,
    index_path: Path,
    backend: LoadedGenerationBackend | None,
    generation_config: GenerationConfig,
    pipeline_config: PipelineConfig,
    *,
    k: int = 5,
    context_token_budget: int,
    auxiliary_path_penalty: float = DEFAULT_AUXILIARY_PATH_PENALTY,
    path_candidate_depth: int = DEFAULT_PATH_CANDIDATE_DEPTH,
    answer_cache: ValidatedAnswerCache | None = None,
) -> QueryAnswerResult:
    """Retrieve, bound context, and generate one grounded answer."""
    if not question.strip():
        raise ValueError("Question must not be empty")
    if k <= 0:
        raise ValueError("Search k must be greater than zero")
    if context_token_budget <= 0:
        raise ValueError("Context token budget must be greater than zero")

    resolved_corpus_root = _below_root(project_root, corpus_root)
    manifest = discover_files(project_root, resolved_corpus_root)
    corpus_fingerprint = fingerprint_corpus(project_root, manifest)
    pipeline_fingerprint = fingerprint_pipeline(
        pipeline_config,
        index_schema_version=SCHEMA_VERSION,
    )
    cache_key = _answer_cache_key(
        question,
        corpus_fingerprint,
        pipeline_fingerprint,
        generation_config,
        k,
        context_token_budget,
        auxiliary_path_penalty,
        path_candidate_depth,
    )
    if answer_cache is not None:
        cached_result = answer_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

    retrieved_sources = run_stored_search(
        _below_root(project_root, index_path),
        corpus_fingerprint,
        pipeline_fingerprint,
        question,
        k,
        auxiliary_path_penalty=auxiliary_path_penalty,
        path_candidate_depth=path_candidate_depth,
    )
    if backend is None:
        backend = load_generation_backend(generation_config)
    context = build_context(
        project_root,
        resolved_corpus_root,
        retrieved_sources,
        HuggingFaceTokenCounter(backend.tokenizer),
        context_token_budget,
    )
    grounded_answer = generate_grounded_answer(
        question,
        context,
        backend,
        generation_config,
    )
    result = QueryAnswerResult(
        answer=grounded_answer.answer,
        retrieved_sources=tuple(retrieved_sources),
        context_sources=grounded_answer.sources,
        used_context_tokens=context.used_tokens,
        skipped_source_count=context.skipped_source_count,
        prompt_version=grounded_answer.prompt_version,
        generation_device=backend.device,
    )
    if answer_cache is not None:
        answer_cache.put(cache_key, result)
    return result


def _answer_cache_key(
    question: str,
    corpus_fingerprint: str,
    pipeline_fingerprint: str,
    generation_config: GenerationConfig,
    k: int,
    context_token_budget: int,
    auxiliary_path_penalty: float,
    path_candidate_depth: int,
) -> AnswerCacheKey:
    """Capture every answer-changing input in one exact cache identity."""
    return AnswerCacheKey(
        question=normalize_question(question),
        corpus_fingerprint=corpus_fingerprint,
        pipeline_fingerprint=pipeline_fingerprint,
        k=k,
        context_token_budget=context_token_budget,
        auxiliary_path_penalty=auxiliary_path_penalty,
        path_candidate_depth=path_candidate_depth,
        prompt_version=GROUNDING_PROMPT_VERSION,
        model_name=generation_config.model_name,
        device=generation_config.device.value,
        max_new_tokens=generation_config.max_new_tokens,
        local_files_only=generation_config.local_files_only,
        enable_thinking=generation_config.enable_thinking,
        do_sample=generation_config.do_sample,
    )


def _below_root(project_root: Path, path: Path) -> Path:
    """Resolve a relative workflow path from the project root."""
    if path.is_absolute():
        return path
    return project_root / path
