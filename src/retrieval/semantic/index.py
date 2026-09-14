"""Validated exact cosine index for normalized semantic embeddings."""

from collections.abc import Sequence
import math

import torch
from torch import Tensor

from src.retrieval.semantic.models import SemanticDocument, SemanticHit

_NORMALIZATION_TOLERANCE = 1e-5


class SemanticIndex:
    """Keep ordered source documents aligned with one embedding matrix."""

    def __init__(
        self,
        documents: Sequence[SemanticDocument],
        embeddings: Tensor,
    ) -> None:
        """Validate and retain a CPU matrix of normalized embeddings."""
        self._documents = tuple(documents)
        self._embeddings = _validate_embeddings(
            embeddings,
            expected_rows=len(self._documents),
        )
        _require_unique_documents(self._documents)

    @property
    def documents(self) -> tuple[SemanticDocument, ...]:
        """Expose documents in their embedding-row order."""
        return self._documents

    @property
    def embeddings(self) -> Tensor:
        """Return an isolated copy of the normalized embedding matrix."""
        return self._embeddings.clone()

    @property
    def dimension(self) -> int:
        """Return the fixed embedding width."""
        return self._embeddings.shape[1]

    def search(
        self,
        query_embedding: Tensor,
        top_k: int = 5,
    ) -> list[SemanticHit]:
        """Rank documents by exact cosine similarity with stable ties."""
        if top_k <= 0:
            raise ValueError("Semantic search top_k must be greater than zero")
        query = _validate_query(query_embedding, self.dimension)
        scores = torch.mv(self._embeddings, query)
        ranked_rows = sorted(
            range(len(self._documents)),
            key=lambda row: (-float(scores[row]), row),
        )
        return [
            SemanticHit(
                document=self._documents[row],
                score=float(scores[row]),
            )
            for row in ranked_rows[:top_k]
        ]


def _validate_embeddings(embeddings: Tensor, expected_rows: int) -> Tensor:
    """Require one finite normalized vector for every source document."""
    if embeddings.ndim != 2:
        raise ValueError(
            "Semantic embeddings must be a two-dimensional matrix"
        )
    if embeddings.shape[0] != expected_rows:
        raise ValueError("Semantic document and embedding counts must match")
    if embeddings.shape[1] <= 0:
        raise ValueError(
            "Semantic embedding dimension must be greater than zero"
        )
    if not embeddings.is_floating_point():
        raise TypeError("Semantic embeddings must use a floating-point dtype")
    matrix = embeddings.detach().to(device="cpu", dtype=torch.float32).clone()
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("Semantic embeddings must contain only finite values")
    _require_normalized(matrix, "Semantic embeddings")
    return matrix


def _validate_query(query: Tensor, expected_dimension: int) -> Tensor:
    """Require one finite normalized vector matching the index dimension."""
    if query.ndim != 1 or query.shape[0] != expected_dimension:
        raise ValueError(
            "Semantic query embedding dimension must match the index"
        )
    if not query.is_floating_point():
        raise TypeError(
            "Semantic query embedding must use a floating-point dtype"
        )
    vector = query.detach().to(device="cpu", dtype=torch.float32).clone()
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(
            "Semantic query embedding must contain only finite values"
        )
    _require_normalized(vector.unsqueeze(0), "Semantic query embedding")
    return vector


def _require_normalized(vectors: Tensor, label: str) -> None:
    """Reject zero-length or non-unit vectors before cosine scoring."""
    norms = torch.linalg.vector_norm(vectors, dim=1)
    if any(
        not math.isclose(float(norm), 1.0, abs_tol=_NORMALIZATION_TOLERANCE)
        for norm in norms
    ):
        raise ValueError(f"{label} must be L2-normalized")


def _require_unique_documents(documents: Sequence[SemanticDocument]) -> None:
    """Reject ambiguous duplicate source spans."""
    keys = [document.key for document in documents]
    if len(keys) != len(set(keys)):
        raise ValueError("Semantic documents must have unique source spans")
