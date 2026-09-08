"""Contracts for model-independent context construction."""

from dataclasses import dataclass
from typing import Protocol

from src.models import MinimalSource


class TokenCounter(Protocol):
    """Count model-specific tokens in an exact text value."""

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens the model tokenizer would consume."""


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    """Report the bounded context and exact sources used to construct it."""

    context: str
    sources: tuple[MinimalSource, ...]
    used_tokens: int
    skipped_source_count: int
