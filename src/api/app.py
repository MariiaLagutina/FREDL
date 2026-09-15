"""Build the local HTTP API application."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.api.errors import error_detail
from src.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from src.api.state import AnswerState, load_search_index_state
from src.generation import (
    GenerationConfig,
    HuggingFaceTokenCounter,
    build_context,
    generate_grounded_answer,
)
from src.retrieval import search_sources
from src.retrieval.bm25 import BM25Parameters
from src.retrieval.index_store import PipelineConfig
from src.retrieval.tokenization import require_searchable_query

DEFAULT_INDEX_PATH = Path("data/processed/bm25-index.json")
DEFAULT_CORPUS_ROOT = Path("data/raw")


def create_app(
    *,
    project_root: Path = Path("."),
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
    index_path: Path = DEFAULT_INDEX_PATH,
    pipeline_config: PipelineConfig | None = None,
    generation_config: GenerationConfig | None = None,
) -> FastAPI:
    """Construct the app and load the BM25 index once for its lifetime.

    The generation backend is not loaded here: it is loaded once on the
    first `/answer` request and cached for the rest of the process, so a
    server that never answers a question never pays the model-load cost.
    """
    active_pipeline_config = pipeline_config or PipelineConfig(
        parameters=BM25Parameters(),
    )
    resolved_corpus_root = _below_root(project_root, corpus_root)
    search_state = load_search_index_state(
        project_root,
        resolved_corpus_root,
        _below_root(project_root, index_path),
        active_pipeline_config,
    )
    answer_state = AnswerState(generation_config or GenerationConfig())

    app = FastAPI(title="RAG against the machine")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report that the server process is reachable."""
        return HealthResponse()

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        """Search the already-loaded index without reacquiring it."""
        try:
            require_searchable_query(request.question)
            if request.k <= 0:
                raise ValueError("Search k must be greater than zero")
            sources = search_sources(
                search_state.index,
                request.question,
                request.k,
                identifier_match_weight=request.identifier_match_weight,
                identifier_candidate_depth=(
                    request.identifier_candidate_depth
                ),
                auxiliary_path_penalty=request.auxiliary_path_penalty,
                path_candidate_depth=request.path_candidate_depth,
            )
        except (UnicodeError, ValueError) as error:
            raise HTTPException(
                status_code=400,
                detail=error_detail(error),
            ) from None
        return SearchResponse(sources=sources)

    @app.post("/answer", response_model=AnswerResponse)
    def answer(request: AnswerRequest) -> AnswerResponse:
        """Answer one question using the already-loaded index and model."""
        try:
            require_searchable_query(request.question)
            if request.k <= 0:
                raise ValueError("Search k must be greater than zero")
            if request.context_token_budget <= 0:
                raise ValueError(
                    "Context token budget must be greater than zero"
                )
            retrieved_sources = search_sources(
                search_state.index,
                request.question,
                request.k,
                auxiliary_path_penalty=request.auxiliary_path_penalty,
                path_candidate_depth=request.path_candidate_depth,
            )
            backend = answer_state.backend()
            context = build_context(
                project_root,
                resolved_corpus_root,
                retrieved_sources,
                HuggingFaceTokenCounter(backend.tokenizer),
                request.context_token_budget,
            )
            grounded_answer = generate_grounded_answer(
                request.question,
                context,
                backend,
                answer_state.generation_config,
            )
        except (OSError, UnicodeError, ValueError, RuntimeError) as error:
            raise HTTPException(
                status_code=400,
                detail=error_detail(error),
            ) from None
        return AnswerResponse(
            answer=grounded_answer.answer,
            sources=list(context.sources),
            retrieved_sources=list(retrieved_sources),
            used_context_tokens=context.used_tokens,
            skipped_source_count=context.skipped_source_count,
            prompt_version=grounded_answer.prompt_version,
            cache_hit=False,
        )

    return app


def _below_root(project_root: Path, path: Path) -> Path:
    """Resolve a relative API path from the configured project root."""
    if path.is_absolute():
        return path
    return project_root / path
