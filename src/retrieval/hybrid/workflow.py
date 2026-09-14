"""Load compatible resources and measure one stored hybrid search."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.models import MinimalSource, QuerySearchResult, RetrievalResults
from src.retrieval.hybrid.models import HybridHit, RRFParameters
from src.retrieval.hybrid.retriever import (
    HybridRetriever,
    ProductionBM25Retriever,
)
from src.retrieval.index_store import IndexStore
from src.retrieval.input import load_rag_dataset
from src.retrieval.output import save_search_results
from src.retrieval.semantic import (
    MiniLMEncoder,
    SemanticEncoderConfig,
    SemanticIndexStore,
)
from src.retrieval.tokenization import require_searchable_query


@dataclass(frozen=True, slots=True)
class HybridSearchReport:
    """Expose fused sources and separate stored-search costs."""

    sources: tuple[MinimalSource, ...]
    lexical_index_load_seconds: float
    semantic_index_load_seconds: float
    model_load_seconds: float
    query_encoding_seconds: float
    search_seconds: float


@dataclass(frozen=True, slots=True)
class HybridBatchSearchReport:
    """Expose compatible batch output and shared hybrid resource costs."""

    results: RetrievalResults
    lexical_index_load_seconds: float
    semantic_index_load_seconds: float
    model_load_seconds: float
    query_encoding_seconds: float
    search_seconds: float

    @property
    def query_count(self) -> int:
        """Return the number of searched questions."""
        return len(self.results.search_results)

    @property
    def average_query_seconds(self) -> float:
        """Average warm encoding and hybrid search time per question."""
        if self.query_count == 0:
            return 0.0
        return (
            self.query_encoding_seconds + self.search_seconds
        ) / self.query_count


def run_stored_hybrid_search(
    lexical_index_path: Path,
    semantic_index_directory: Path,
    query: str,
    k: int,
    candidate_k: int,
    config: SemanticEncoderConfig,
    *,
    corpus_fingerprint: str,
    pipeline_fingerprint: str,
    rrf_parameters: RRFParameters | None = None,
    clock: Callable[[], float] = perf_counter,
) -> HybridSearchReport:
    """Validate, load both indexes once, and run one measured query."""
    require_searchable_query(query)
    if k <= 0:
        raise ValueError("Hybrid search k must be greater than zero")
    if candidate_k < k:
        raise ValueError(
            "Hybrid candidate_k must be greater than or equal to k"
        )

    lexical_load_started = clock()
    lexical_index = IndexStore(lexical_index_path).load(
        corpus_fingerprint,
        pipeline_fingerprint,
    )
    lexical_load_finished = clock()

    semantic_load_started = clock()
    semantic_index = SemanticIndexStore(semantic_index_directory).load(
        expected_corpus_fingerprint=corpus_fingerprint,
        expected_pipeline_fingerprint=pipeline_fingerprint,
        expected_model_name=config.model_name,
        expected_model_revision=config.model_revision,
    )
    semantic_load_finished = clock()

    model_load_started = clock()
    encoder = MiniLMEncoder(config)
    model_load_finished = clock()

    retriever = HybridRetriever(
        ProductionBM25Retriever(lexical_index),
        semantic_index,
        encoder,
        rrf_parameters,
    )
    encoding_started = clock()
    query_embedding = encoder.encode((query,))[0]
    encoding_finished = clock()
    search_started = clock()
    hits = retriever.search_with_embedding(
        query,
        query_embedding,
        top_k=k,
        candidate_k=candidate_k,
    )
    search_finished = clock()
    return HybridSearchReport(
        sources=_sources_from_hits(hits),
        lexical_index_load_seconds=(
            lexical_load_finished - lexical_load_started
        ),
        semantic_index_load_seconds=(
            semantic_load_finished - semantic_load_started
        ),
        model_load_seconds=model_load_finished - model_load_started,
        query_encoding_seconds=encoding_finished - encoding_started,
        search_seconds=search_finished - search_started,
    )


def run_stored_hybrid_retrieval(
    lexical_index_path: Path,
    semantic_index_directory: Path,
    input_path: Path,
    output_path: Path,
    k: int,
    candidate_k: int,
    config: SemanticEncoderConfig,
    *,
    corpus_fingerprint: str,
    pipeline_fingerprint: str,
    rrf_parameters: RRFParameters | None = None,
    clock: Callable[[], float] = perf_counter,
) -> HybridBatchSearchReport:
    """Encode all validated questions once and save fused batch results."""
    if k <= 0:
        raise ValueError("Hybrid search k must be greater than zero")
    if candidate_k < k:
        raise ValueError(
            "Hybrid candidate_k must be greater than or equal to k"
        )
    dataset = load_rag_dataset(input_path)
    for question in dataset.rag_questions:
        require_searchable_query(question.question)
    if not dataset.rag_questions:
        results = RetrievalResults(search_results=[], k=k)
        save_search_results(results, output_path)
        return HybridBatchSearchReport(results, 0.0, 0.0, 0.0, 0.0, 0.0)

    lexical_load_started = clock()
    lexical_index = IndexStore(lexical_index_path).load(
        corpus_fingerprint,
        pipeline_fingerprint,
    )
    lexical_load_finished = clock()
    semantic_load_started = clock()
    semantic_index = SemanticIndexStore(semantic_index_directory).load(
        expected_corpus_fingerprint=corpus_fingerprint,
        expected_pipeline_fingerprint=pipeline_fingerprint,
        expected_model_name=config.model_name,
        expected_model_revision=config.model_revision,
    )
    semantic_load_finished = clock()
    model_load_started = clock()
    encoder = MiniLMEncoder(config)
    model_load_finished = clock()
    retriever = HybridRetriever(
        ProductionBM25Retriever(lexical_index),
        semantic_index,
        encoder,
        rrf_parameters,
    )

    encoding_started = clock()
    embeddings = encoder.encode(
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
                        retriever.search_with_embedding(
                            question.question,
                            embeddings[row],
                            top_k=k,
                            candidate_k=candidate_k,
                        )
                    )
                ),
            )
            for row, question in enumerate(dataset.rag_questions)
        ],
        k=k,
    )
    search_finished = clock()
    save_search_results(results, output_path)
    return HybridBatchSearchReport(
        results=results,
        lexical_index_load_seconds=(
            lexical_load_finished - lexical_load_started
        ),
        semantic_index_load_seconds=(
            semantic_load_finished - semantic_load_started
        ),
        model_load_seconds=model_load_finished - model_load_started,
        query_encoding_seconds=encoding_finished - encoding_started,
        search_seconds=search_finished - search_started,
    )


def _sources_from_hits(
    hits: Sequence[HybridHit],
) -> tuple[MinimalSource, ...]:
    """Convert fused hits to the unchanged assignment source contract."""
    return tuple(
        MinimalSource(
            file_path=hit.chunk.file_path,
            first_character_index=hit.chunk.start,
            last_character_index=hit.chunk.end,
        )
        for hit in hits
    )
