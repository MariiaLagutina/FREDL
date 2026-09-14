"""Orchestrate and measure optional semantic index construction."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.models import (
    MinimalSource,
    QuerySearchResult,
    RetrievalResults,
)
from src.retrieval.bm25 import BM25Index
from src.retrieval.semantic.builder import build_semantic_index
from src.retrieval.semantic.config import SemanticEncoderConfig
from src.retrieval.semantic.encoder import MiniLMEncoder
from src.retrieval.semantic.models import SemanticHit
from src.retrieval.semantic.runtime import load_semantic_backend
from src.retrieval.semantic.store import SemanticIndexStore
from src.retrieval.input import load_rag_dataset
from src.retrieval.output import save_search_results
from src.retrieval.tokenization import require_searchable_query


@dataclass(frozen=True, slots=True)
class SemanticIndexBuildReport:
    """Expose separate semantic build costs and resulting index shape."""

    document_count: int
    dimension: int
    model_load_seconds: float
    encoding_seconds: float
    save_seconds: float


@dataclass(frozen=True, slots=True)
class SemanticSearchReport:
    """Expose semantic sources and separate cold-query costs."""

    sources: tuple[MinimalSource, ...]
    index_load_seconds: float
    model_load_seconds: float
    query_encoding_seconds: float
    search_seconds: float


@dataclass(frozen=True, slots=True)
class SemanticBatchSearchReport:
    """Expose compatible batch results and shared-resource timings."""

    results: RetrievalResults
    index_load_seconds: float
    model_load_seconds: float
    query_encoding_seconds: float
    search_seconds: float

    @property
    def query_count(self) -> int:
        """Return the number of evaluated questions."""
        return len(self.results.search_results)

    @property
    def average_query_seconds(self) -> float:
        """Average warm encoding and search cost across all questions."""
        if self.query_count == 0:
            return 0.0
        return (
            self.query_encoding_seconds + self.search_seconds
        ) / self.query_count


def build_and_store_semantic_index(
    lexical_index: BM25Index,
    output_directory: Path,
    config: SemanticEncoderConfig,
    *,
    corpus_fingerprint: str,
    pipeline_fingerprint: str,
    clock: Callable[[], float] = perf_counter,
) -> SemanticIndexBuildReport:
    """Load MiniLM, encode BM25 documents, and publish one snapshot."""
    load_started = clock()
    backend = load_semantic_backend(config)
    load_finished = clock()

    encoding_started = clock()
    semantic_index = build_semantic_index(
        lexical_index,
        MiniLMEncoder(config, backend),
    )
    encoding_finished = clock()

    save_started = clock()
    SemanticIndexStore(output_directory).save(
        semantic_index,
        corpus_fingerprint=corpus_fingerprint,
        pipeline_fingerprint=pipeline_fingerprint,
        model_name=config.model_name,
        model_revision=config.model_revision,
    )
    save_finished = clock()
    return SemanticIndexBuildReport(
        document_count=len(semantic_index.documents),
        dimension=semantic_index.dimension,
        model_load_seconds=load_finished - load_started,
        encoding_seconds=encoding_finished - encoding_started,
        save_seconds=save_finished - save_started,
    )


def run_stored_semantic_search(
    index_directory: Path,
    query: str,
    k: int,
    config: SemanticEncoderConfig,
    *,
    corpus_fingerprint: str,
    pipeline_fingerprint: str,
    clock: Callable[[], float] = perf_counter,
) -> SemanticSearchReport:
    """Validate, load, encode, and search one optional semantic query."""
    require_searchable_query(query)
    if k <= 0:
        raise ValueError("Semantic search k must be greater than zero")

    index_load_started = clock()
    index = SemanticIndexStore(index_directory).load(
        expected_corpus_fingerprint=corpus_fingerprint,
        expected_pipeline_fingerprint=pipeline_fingerprint,
        expected_model_name=config.model_name,
        expected_model_revision=config.model_revision,
    )
    index_load_finished = clock()

    model_load_started = clock()
    encoder = MiniLMEncoder(config)
    model_load_finished = clock()

    encoding_started = clock()
    query_embedding = encoder.encode((query,))[0]
    encoding_finished = clock()

    search_started = clock()
    hits = index.search(query_embedding, top_k=k)
    search_finished = clock()
    return SemanticSearchReport(
        sources=_sources_from_hits(hits),
        index_load_seconds=index_load_finished - index_load_started,
        model_load_seconds=model_load_finished - model_load_started,
        query_encoding_seconds=encoding_finished - encoding_started,
        search_seconds=search_finished - search_started,
    )


def run_stored_semantic_retrieval(
    index_directory: Path,
    input_path: Path,
    output_path: Path,
    k: int,
    config: SemanticEncoderConfig,
    *,
    corpus_fingerprint: str,
    pipeline_fingerprint: str,
    clock: Callable[[], float] = perf_counter,
) -> SemanticBatchSearchReport:
    """Search a validated dataset with one shared index and encoder load."""
    if k <= 0:
        raise ValueError("Semantic search k must be greater than zero")
    dataset = load_rag_dataset(input_path)
    for question in dataset.rag_questions:
        require_searchable_query(question.question)
    if not dataset.rag_questions:
        results = RetrievalResults(search_results=[], k=k)
        save_search_results(results, output_path)
        return SemanticBatchSearchReport(results, 0.0, 0.0, 0.0, 0.0)

    index_load_started = clock()
    index = SemanticIndexStore(index_directory).load(
        expected_corpus_fingerprint=corpus_fingerprint,
        expected_pipeline_fingerprint=pipeline_fingerprint,
        expected_model_name=config.model_name,
        expected_model_revision=config.model_revision,
    )
    index_load_finished = clock()

    model_load_started = clock()
    encoder = MiniLMEncoder(config)
    model_load_finished = clock()

    encoding_started = clock()
    query_embeddings = encoder.encode(
        tuple(question.question for question in dataset.rag_questions)
    )
    encoding_finished = clock()

    search_started = clock()
    results = RetrievalResults(
        search_results=[
            QuerySearchResult(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=list(
                    _sources_from_hits(
                        index.search(query_embeddings[row], top_k=k)
                    )
                ),
            )
            for row, question in enumerate(dataset.rag_questions)
        ],
        k=k,
    )
    search_finished = clock()
    save_search_results(results, output_path)
    return SemanticBatchSearchReport(
        results=results,
        index_load_seconds=index_load_finished - index_load_started,
        model_load_seconds=model_load_finished - model_load_started,
        query_encoding_seconds=encoding_finished - encoding_started,
        search_seconds=search_finished - search_started,
    )


def _sources_from_hits(
    hits: Sequence[SemanticHit],
) -> tuple[MinimalSource, ...]:
    """Convert semantic hits to the assignment source contract."""
    return tuple(
        MinimalSource(
            file_path=hit.document.chunk.file_path,
            first_character_index=hit.document.chunk.start,
            last_character_index=hit.document.chunk.end,
        )
        for hit in hits
    )
