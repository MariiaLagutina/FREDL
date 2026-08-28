"""Orchestrate validated batch retrieval across filesystem boundaries."""

from pathlib import Path

from src.models import MinimalSource, RetrievalResults
from src.retrieval.bm25 import BM25Index
from src.retrieval.input import load_rag_dataset
from src.retrieval.index_store import IndexStore
from src.retrieval.output import save_search_results
from src.retrieval.results import (
    QuestionProgress,
    search_dataset,
    search_sources,
)


def run_stored_search(
    index_path: Path,
    corpus_fingerprint: str,
    pipeline_fingerprint: str,
    query: str,
    k: int = 5,
) -> list[MinimalSource]:
    """Load one compatible index and search one raw query."""
    index = IndexStore(index_path).load(
        corpus_fingerprint,
        pipeline_fingerprint,
    )
    return search_sources(index, query, k)


def run_stored_retrieval(
    index_path: Path,
    corpus_fingerprint: str,
    pipeline_fingerprint: str,
    input_path: Path,
    output_path: Path,
    k: int = 5,
    progress: QuestionProgress | None = None,
) -> RetrievalResults:
    """Load one compatible index and run retrieval for a question file."""
    index = IndexStore(index_path).load(
        corpus_fingerprint,
        pipeline_fingerprint,
    )
    return run_retrieval(
        index,
        input_path,
        output_path,
        k,
        progress,
    )


def run_retrieval(
    index: BM25Index,
    input_path: Path,
    output_path: Path,
    k: int = 5,
    progress: QuestionProgress | None = None,
) -> RetrievalResults:
    """Load questions, search one index, and save validated results."""
    dataset = load_rag_dataset(input_path)
    results = search_dataset(index, dataset, k, progress)
    save_search_results(results, output_path)
    return results
