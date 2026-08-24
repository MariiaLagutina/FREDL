"""Run deterministic chunk invariant audits over source documents."""

from collections.abc import Callable, Iterable
from pathlib import Path

from src.ingestion.audit.invariants import audit_chunk
from src.ingestion.audit.models import (
    ChunkAuditIssue,
    ChunkAuditIssueKind,
    ChunkAuditReport,
)
from src.ingestion.audit.statistics import summarize_sizes
from src.ingestion.chunking.orchestrator import (
    MAX_CHUNK_SIZE,
    chunk_document,
)
from src.ingestion.documents import Chunk, SourceDocument, read_document
from src.ingestion.files import discover_files


Chunker = Callable[[SourceDocument, int], list[Chunk]]


def audit_corpus(
    project_root: Path,
    corpus_root: Path,
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> ChunkAuditReport:
    """Discover, read, chunk, and audit every supported corpus file."""
    manifest = discover_files(project_root, corpus_root)
    documents = (
        read_document(project_root, corpus_file) for corpus_file in manifest
    )
    return audit_documents(documents, max_chunk_size)


def audit_documents(
    documents: Iterable[SourceDocument],
    max_chunk_size: int = MAX_CHUNK_SIZE,
    *,
    chunker: Chunker = chunk_document,
) -> ChunkAuditReport:
    """Audit chunk invariants and collect a deterministic size summary."""
    if max_chunk_size <= 0:
        message = "Maximum chunk size must be positive"
        raise ValueError(message)

    document_count = 0
    sizes: list[int] = []
    issues: list[ChunkAuditIssue] = []

    for document in documents:
        document_count += 1
        chunks = chunker(document, max_chunk_size)
        repeated_chunks = chunker(document, max_chunk_size)

        if chunks != repeated_chunks:
            issues.append(
                ChunkAuditIssue(
                    ChunkAuditIssueKind.NONDETERMINISTIC,
                    document.file_path,
                    None,
                    "Repeated chunking produced a different result",
                )
            )

        for index, chunk in enumerate(chunks):
            sizes.append(len(chunk.text))
            issues.extend(audit_chunk(document, chunk, index, max_chunk_size))

    return ChunkAuditReport(
        document_count=document_count,
        size_summary=summarize_sizes(sizes),
        issues=tuple(issues),
    )
