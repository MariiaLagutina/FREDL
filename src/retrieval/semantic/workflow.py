"""Orchestrate and measure optional semantic index construction."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.retrieval.bm25 import BM25Index
from src.retrieval.semantic.builder import build_semantic_index
from src.retrieval.semantic.config import SemanticEncoderConfig
from src.retrieval.semantic.encoder import MiniLMEncoder
from src.retrieval.semantic.runtime import load_semantic_backend
from src.retrieval.semantic.store import SemanticIndexStore


@dataclass(frozen=True, slots=True)
class SemanticIndexBuildReport:
    """Expose separate semantic build costs and resulting index shape."""

    document_count: int
    dimension: int
    model_load_seconds: float
    encoding_seconds: float
    save_seconds: float


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
