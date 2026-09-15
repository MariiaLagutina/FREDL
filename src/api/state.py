"""Load retrieval and generation state for the long-running API process."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from src.generation import (
    GenerationConfig,
    LoadedGenerationBackend,
    load_generation_backend,
)
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


class AnswerState:
    """Cache the generation backend across requests, loaded on first use."""

    def __init__(self, generation_config: GenerationConfig) -> None:
        """Keep the requested configuration without loading any model."""
        self._generation_config = generation_config
        self._backend: LoadedGenerationBackend | None = None
        self._lock = Lock()

    @property
    def generation_config(self) -> GenerationConfig:
        """Expose the configuration used to load the cached backend."""
        return self._generation_config

    def backend(self) -> LoadedGenerationBackend:
        """Return the cached backend, loading it once on first use."""
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    self._backend = load_generation_backend(
                        self._generation_config
                    )
        return self._backend
