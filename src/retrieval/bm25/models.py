"""Immutable inputs and parameters for BM25 retrieval."""

from dataclasses import dataclass

from src.ingestion.documents import Chunk


@dataclass(frozen=True, slots=True)
class BM25Parameters:
    """Control field scoring and the final metadata score multiplier."""

    k1: float = 1.5
    b: float = 0.75
    metadata_weight: float = 1.0

    def __post_init__(self) -> None:
        """Reject values outside the BM25 parameter domains."""
        if self.k1 <= 0:
            raise ValueError("BM25 k1 must be greater than zero")
        if not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be between zero and one")
        if self.metadata_weight < 0:
            raise ValueError("BM25 metadata weight must not be negative")


@dataclass(frozen=True, slots=True)
class BM25Document:
    """Keep exact source evidence separate from lexical search signals."""

    chunk: Chunk
    content_terms: tuple[str, ...]
    metadata_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require immutable collections of non-empty normalized terms."""
        _validate_terms("content_terms", self.content_terms)
        _validate_terms("metadata_terms", self.metadata_terms)

    @property
    def key(self) -> tuple[str, int, int]:
        """Return a deterministic identity from the exact source span."""
        return (self.chunk.file_path, self.chunk.start, self.chunk.end)


def _validate_terms(name: str, terms: tuple[str, ...]) -> None:
    """Validate one immutable lexical field without changing its terms."""
    if not isinstance(terms, tuple):
        raise TypeError(f"BM25 {name} must be a tuple")
    if any(not isinstance(term, str) or not term for term in terms):
        raise ValueError(f"BM25 {name} must contain non-empty strings")
