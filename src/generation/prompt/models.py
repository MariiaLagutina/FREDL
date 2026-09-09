"""Contracts for internally traced grounded generation results."""

from dataclasses import dataclass

from src.models import MinimalSource


@dataclass(frozen=True, slots=True)
class GroundedAnswerResult:
    """Keep one answer linked to its prompt version and source locations."""

    answer: str
    sources: tuple[MinimalSource, ...]
    prompt_version: str
