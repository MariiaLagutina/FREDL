"""Immutable result models for chunk invariant audits."""

from dataclasses import dataclass
from enum import Enum


class ChunkAuditIssueKind(str, Enum):
    """Identify one failed chunk invariant."""

    INVALID_RANGE = "invalid_range"
    OVERSIZED = "oversized"
    TEXT_MISMATCH = "text_mismatch"
    EMPTY_TEXT = "empty_text"
    NONDETERMINISTIC = "nondeterministic"


@dataclass(frozen=True, slots=True)
class ChunkAuditIssue:
    """Describe one invariant failure with exact source context."""

    kind: ChunkAuditIssueKind
    file_path: str
    chunk_index: int | None
    detail: str

    def __post_init__(self) -> None:
        """Reject issues that cannot identify an actionable failure."""
        if not self.file_path:
            message = "Audit issue file path must not be empty"
            raise ValueError(message)
        if self.chunk_index is not None and self.chunk_index < 0:
            message = "Audit issue chunk index must not be negative"
            raise ValueError(message)
        if not self.detail:
            message = "Audit issue detail must not be empty"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ChunkSizeSummary:
    """Store bounded distribution statistics without retaining all sizes."""

    count: int
    minimum: int
    median: float
    p95: int
    maximum: int

    def __post_init__(self) -> None:
        """Require a positive and ordered chunk-size distribution."""
        if self.count <= 0:
            message = "Chunk size summary count must be positive"
            raise ValueError(message)
        if self.minimum <= 0:
            message = "Chunk sizes must be positive"
            raise ValueError(message)
        if not self.minimum <= self.median <= self.p95 <= self.maximum:
            message = "Chunk size summary values must be ordered"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ChunkAuditReport:
    """Summarize audited documents, chunk sizes, and invariant failures."""

    document_count: int
    size_summary: ChunkSizeSummary | None
    issues: tuple[ChunkAuditIssue, ...] = ()

    def __post_init__(self) -> None:
        """Reject report counts that cannot describe an audit run."""
        if self.document_count < 0:
            message = "Audit document count must not be negative"
            raise ValueError(message)

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks represented by the distribution."""
        return self.size_summary.count if self.size_summary is not None else 0

    @property
    def passed(self) -> bool:
        """Return whether the audit found no invariant failures."""
        return not self.issues
