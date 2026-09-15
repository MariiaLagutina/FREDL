"""Rebuild only the corpus files that changed since a prior BM25 index."""

from dataclasses import dataclass
from pathlib import Path

from src.ingestion import chunk_document, discover_files, read_document
from src.retrieval.bm25 import BM25Document, BM25Index, build_bm25_documents
from src.retrieval.index_store.builder import IndexBuild
from src.retrieval.index_store.fingerprint import (
    fingerprint_corpus,
    fingerprint_file,
)
from src.retrieval.index_store.pipeline import (
    PipelineConfig,
    fingerprint_pipeline,
)
from src.retrieval.index_store.store import IndexStore


@dataclass(frozen=True, slots=True)
class IncrementalIndexBuild:
    """Extend a full build result with per-file reuse accounting."""

    build: IndexBuild
    file_fingerprints: dict[str, str]
    reused_file_count: int
    rebuilt_file_count: int
    total_file_count: int


def build_index_incremental(
    project_root: Path,
    corpus_root: Path,
    config: PipelineConfig,
    *,
    index_schema_version: int,
    previous_index_path: Path,
) -> IncrementalIndexBuild:
    """Reuse unchanged files' stored documents instead of re-chunking them.

    A file is only reused when its exact bytes are unchanged (verified by
    content hash) and the declared pipeline (chunker, tokenizer, BM25
    parameters, and schema) matches the one that produced the prior index.
    When no reusable prior index exists at `previous_index_path`, every file
    is rebuilt and the result is identical to a full `build_index` call.
    """
    manifest = discover_files(project_root, corpus_root)
    pipeline_fingerprint = fingerprint_pipeline(
        config,
        index_schema_version=index_schema_version,
    )

    previous = IndexStore(previous_index_path).load_for_reuse(
        pipeline_fingerprint
    )
    previous_documents_by_file = (
        previous.documents_by_file if previous is not None else {}
    )
    previous_file_fingerprints = (
        previous.file_fingerprints if previous is not None else {}
    )

    documents: list[BM25Document] = []
    file_fingerprints: dict[str, str] = {}
    reused_file_count = 0
    for corpus_file in manifest:
        content_fingerprint = fingerprint_file(project_root, corpus_file)
        file_fingerprints[corpus_file.file_path] = content_fingerprint

        if previous_file_fingerprints.get(
            corpus_file.file_path
        ) == content_fingerprint:
            documents.extend(
                previous_documents_by_file.get(corpus_file.file_path, ())
            )
            reused_file_count += 1
            continue

        source = read_document(project_root, corpus_file)
        chunks = chunk_document(source, config.max_chunk_size)
        documents.extend(build_bm25_documents(source, chunks))

    build = IndexBuild(
        index=BM25Index(documents, config.parameters),
        corpus_fingerprint=fingerprint_corpus(project_root, manifest),
        pipeline_fingerprint=pipeline_fingerprint,
    )
    return IncrementalIndexBuild(
        build=build,
        file_fingerprints=file_fingerprints,
        reused_file_count=reused_file_count,
        rebuilt_file_count=len(manifest) - reused_file_count,
        total_file_count=len(manifest),
    )
