"""Build one reproducible BM25 index from a discovered source corpus."""

from dataclasses import dataclass
from pathlib import Path

from src.ingestion import chunk_document, discover_files, read_document
from src.retrieval.bm25 import (
    BM25Document,
    BM25Index,
    build_bm25_documents,
)
from src.retrieval.index_store.fingerprint import fingerprint_corpus
from src.retrieval.index_store.pipeline import (
    PipelineConfig,
    fingerprint_pipeline,
)


@dataclass(frozen=True, slots=True)
class IndexBuild:
    """Return the runtime index and both compatibility identities."""

    index: BM25Index
    corpus_fingerprint: str
    pipeline_fingerprint: str


def build_index(
    project_root: Path,
    corpus_root: Path,
    config: PipelineConfig,
    *,
    index_schema_version: int,
) -> IndexBuild:
    """Run the declared ingestion and lexical-index pipeline once."""
    manifest = discover_files(project_root, corpus_root)
    documents: list[BM25Document] = []
    for corpus_file in manifest:
        source = read_document(project_root, corpus_file)
        chunks = chunk_document(source, config.max_chunk_size)
        documents.extend(build_bm25_documents(source, chunks))

    return IndexBuild(
        index=BM25Index(documents, config.parameters),
        corpus_fingerprint=fingerprint_corpus(project_root, manifest),
        pipeline_fingerprint=fingerprint_pipeline(
            config,
            index_schema_version=index_schema_version,
        ),
    )
