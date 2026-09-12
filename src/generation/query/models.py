"""Traceable results for one retrieval-augmented user query."""

from dataclasses import dataclass

from src.models import MinimalSource


@dataclass(frozen=True, slots=True)
class QueryAnswerResult:
    """Keep an answer linked to retrieval and prompt-context evidence."""

    answer: str
    retrieved_sources: tuple[MinimalSource, ...]
    context_sources: tuple[MinimalSource, ...]
    used_context_tokens: int
    skipped_source_count: int
    prompt_version: str
