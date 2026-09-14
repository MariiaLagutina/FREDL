"""Immutable contracts for rank-based hybrid retrieval."""

from dataclasses import dataclass

from src.ingestion import Chunk


@dataclass(frozen=True, slots=True)
class RRFParameters:
    """Control reciprocal-rank contributions from both retrievers."""

    rank_constant: int = 60
    lexical_weight: float = 1.0
    semantic_weight: float = 1.0

    def __post_init__(self) -> None:
        """Reject parameters that cannot produce a meaningful ranking."""
        if self.rank_constant <= 0:
            raise ValueError("RRF rank constant must be greater than zero")
        if self.lexical_weight < 0 or self.semantic_weight < 0:
            raise ValueError("RRF weights must not be negative")
        if self.lexical_weight == 0 and self.semantic_weight == 0:
            raise ValueError(
                "At least one RRF weight must be greater than zero"
            )


@dataclass(frozen=True, slots=True)
class HybridHit:
    """Expose one fused source with its score and contributing ranks."""

    chunk: Chunk
    score: float
    lexical_rank: int | None
    semantic_rank: int | None

    @property
    def key(self) -> tuple[str, int, int]:
        """Return the deterministic identity of the exact source span."""
        return (self.chunk.file_path, self.chunk.start, self.chunk.end)
