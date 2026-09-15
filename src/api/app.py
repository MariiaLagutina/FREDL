"""Build the local HTTP API application."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.api.errors import error_detail
from src.api.schemas import HealthResponse, SearchRequest, SearchResponse
from src.api.state import load_search_index_state
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
) -> FastAPI:
    """Construct the app and load the BM25 index once for its lifetime."""
    active_pipeline_config = pipeline_config or PipelineConfig(
        parameters=BM25Parameters(),
    )
    search_state = load_search_index_state(
        project_root,
        _below_root(project_root, corpus_root),
        _below_root(project_root, index_path),
        active_pipeline_config,
    )

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

    return app


def _below_root(project_root: Path, path: Path) -> Path:
    """Resolve a relative API path from the configured project root."""
    if path.is_absolute():
        return path
    return project_root / path
