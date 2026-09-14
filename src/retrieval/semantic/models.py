"""Immutable source identities returned by semantic retrieval."""

from dataclasses import dataclass

from src.ingestion import Chunk


@dataclass(frozen=True, slots=True)
class SemanticDocument:
    """Associate one embedding row with its exact source chunk."""

    chunk: Chunk

    @property
    def key(self) -> tuple[str, int, int]:
        """Return the deterministic identity of the source span."""
        return (self.chunk.file_path, self.chunk.start, self.chunk.end)


@dataclass(frozen=True, slots=True)
class SemanticHit:
    """Return one semantic match with its cosine similarity."""

    document: SemanticDocument
    score: float
