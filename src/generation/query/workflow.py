"""Orchestrate one traceable query from stored BM25 index to answer."""

from pathlib import Path

from src.generation.backend import GenerationConfig, LoadedGenerationBackend
from src.generation.context import HuggingFaceTokenCounter, build_context
from src.generation.prompt import generate_grounded_answer
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
    backend: LoadedGenerationBackend,
    generation_config: GenerationConfig,
    pipeline_config: PipelineConfig,
    *,
    k: int = 5,
    context_token_budget: int,
    auxiliary_path_penalty: float = DEFAULT_AUXILIARY_PATH_PENALTY,
    path_candidate_depth: int = DEFAULT_PATH_CANDIDATE_DEPTH,
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
    retrieved_sources = run_stored_search(
        _below_root(project_root, index_path),
        fingerprint_corpus(project_root, manifest),
        fingerprint_pipeline(
            pipeline_config,
            index_schema_version=SCHEMA_VERSION,
        ),
        question,
        k,
        auxiliary_path_penalty=auxiliary_path_penalty,
        path_candidate_depth=path_candidate_depth,
    )
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
    return QueryAnswerResult(
        answer=grounded_answer.answer,
        retrieved_sources=tuple(retrieved_sources),
        context_sources=grounded_answer.sources,
        used_context_tokens=context.used_tokens,
        skipped_source_count=context.skipped_source_count,
        prompt_version=grounded_answer.prompt_version,
        generation_attempts=grounded_answer.generation_attempts,
    )


def _below_root(project_root: Path, path: Path) -> Path:
    """Resolve a relative workflow path from the project root."""
    if path.is_absolute():
        return path
    return project_root / path
