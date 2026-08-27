"""Orchestrate validated batch retrieval across filesystem boundaries."""

from pathlib import Path

from src.models import RetrievalResults
from src.retrieval.bm25 import BM25Index
from src.retrieval.input import load_rag_dataset
from src.retrieval.output import save_search_results
from src.retrieval.results import search_dataset


def run_retrieval(
    index: BM25Index,
    input_path: Path,
    output_path: Path,
    k: int = 5,
) -> RetrievalResults:
    """Load questions, search one index, and save validated results."""
    dataset = load_rag_dataset(input_path)
    results = search_dataset(index, dataset, k)
    save_search_results(results, output_path)
    return results
