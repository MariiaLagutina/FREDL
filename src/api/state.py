"""Load retrieval state once for the long-running HTTP API process."""

from dataclasses import dataclass
from pathlib import Path

from src.ingestion import discover_files
from src.retrieval.bm25 import BM25Index
from src.retrieval.index_store import (
    IndexStore,
    PipelineConfig,
    SCHEMA_VERSION,
    fingerprint_corpus,
    fingerprint_pipeline,
)


@dataclass(frozen=True, slots=True)
class SearchIndexState:
    """Keep one loaded BM25 index available across requests."""

    index: BM25Index
    corpus_fingerprint: str
    pipeline_fingerprint: str


def load_search_index_state(
    project_root: Path,
    corpus_root: Path,
    index_path: Path,
    pipeline_config: PipelineConfig,
) -> SearchIndexState:
    """Load one compatible BM25 index for the lifetime of the server."""
    manifest = discover_files(project_root, corpus_root)
    corpus_fingerprint = fingerprint_corpus(project_root, manifest)
    pipeline_fingerprint = fingerprint_pipeline(
        pipeline_config,
        index_schema_version=SCHEMA_VERSION,
    )
    index = IndexStore(index_path).load(
        corpus_fingerprint,
        pipeline_fingerprint,
    )
    return SearchIndexState(index, corpus_fingerprint, pipeline_fingerprint)
