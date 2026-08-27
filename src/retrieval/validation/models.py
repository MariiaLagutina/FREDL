"""Immutable result models for retrieved-source validation."""

from dataclasses import dataclass
from enum import Enum


class SourceValidationIssueKind(str, Enum):
    """Identify one failed retrieved-source invariant."""

    UNKNOWN_PATH = "unknown_path"
    INVALID_RANGE = "invalid_range"
    OVERSIZED = "oversized"


@dataclass(frozen=True, slots=True)
class SourceValidationIssue:
    """Describe one source failure with its retrieval-result context."""

    kind: SourceValidationIssueKind
    result_index: int
    source_index: int
    question_id: str
    file_path: str
    detail: str

    def __post_init__(self) -> None:
        """Reject issue records without actionable source context."""
        if self.result_index < 0:
            raise ValueError("Validation result index must not be negative")
        if self.source_index < 0:
            raise ValueError("Validation source index must not be negative")
        if not self.question_id:
            raise ValueError("Validation question ID must not be empty")
        if not self.file_path:
            raise ValueError("Validation file path must not be empty")
        if not self.detail:
            raise ValueError("Validation issue detail must not be empty")


@dataclass(frozen=True, slots=True)
class SourceValidationReport:
    """Summarize checked retrieval results and all source failures."""

    result_count: int
    source_count: int
    issues: tuple[SourceValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        """Require counts that can describe one validation run."""
        if self.result_count < 0:
            raise ValueError("Validation result count must not be negative")
        if self.source_count < 0:
            raise ValueError("Validation source count must not be negative")

    @property
    def invalid_source_count(self) -> int:
        """Count unique sources with one or more validation failures."""
        return len(
            {
                (issue.result_index, issue.source_index)
                for issue in self.issues
            }
        )

    @property
    def valid_source_count(self) -> int:
        """Count checked sources without any validation failure."""
        return self.source_count - self.invalid_source_count

    @property
    def passed(self) -> bool:
        """Return whether every checked source satisfied all invariants."""
        return not self.issues
