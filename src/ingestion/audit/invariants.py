"""Check invariant failures for individual chunks."""

from src.ingestion.audit.models import ChunkAuditIssue, ChunkAuditIssueKind
from src.ingestion.documents import Chunk, SourceDocument


def audit_chunk(
    document: SourceDocument,
    chunk: Chunk,
    index: int,
    max_chunk_size: int,
) -> list[ChunkAuditIssue]:
    """Return every failed invariant for one produced chunk."""
    issues: list[ChunkAuditIssue] = []
    valid_range = 0 <= chunk.start < chunk.end <= len(document.text)

    if not valid_range:
        issues.append(
            _issue(
                ChunkAuditIssueKind.INVALID_RANGE,
                document,
                index,
                f"Chunk range [{chunk.start}:{chunk.end}) is outside source",
            )
        )
    elif chunk.text != document.text[chunk.start:chunk.end]:
        issues.append(
            _issue(
                ChunkAuditIssueKind.TEXT_MISMATCH,
                document,
                index,
                "Chunk text does not match its exact source slice",
            )
        )

    if len(chunk.text) > max_chunk_size:
        issues.append(
            _issue(
                ChunkAuditIssueKind.OVERSIZED,
                document,
                index,
                (
                    f"Chunk length {len(chunk.text)} exceeds maximum "
                    f"{max_chunk_size}"
                ),
            )
        )

    if not chunk.text.strip():
        issues.append(
            _issue(
                ChunkAuditIssueKind.EMPTY_TEXT,
                document,
                index,
                "Chunk contains no non-whitespace text",
            )
        )

    return issues


def _issue(
    kind: ChunkAuditIssueKind,
    document: SourceDocument,
    index: int,
    detail: str,
) -> ChunkAuditIssue:
    """Create one issue tied to a document and chunk index."""
    return ChunkAuditIssue(kind, document.file_path, index, detail)
