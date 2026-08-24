"""Internal models shared by text chunking strategies."""

from dataclasses import dataclass
from enum import Enum


class _BlockKind(str, Enum):
    """Identify one structural text block boundary."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    FENCED_CODE = "fenced_code"
    WHITESPACE = "whitespace"


@dataclass(frozen=True, slots=True)
class _TextBlock:
    """Store one exact text range and its active section path."""

    kind: _BlockKind
    start: int
    end: int
    section_path: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject ranges that cannot identify source content."""
        if self.start < 0 or self.end <= self.start:
            message = "Text block range must be positive"
            raise ValueError(message)
