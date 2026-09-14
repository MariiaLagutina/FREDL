"""Coordinate production lexical and semantic retrieval before fusion."""

from typing import Protocol

from torch import Tensor

from src.retrieval.bm25 import (
    AuxiliaryPathReranker,
    BM25Hit,
    BM25Index,
    BM25Retriever,
)
from src.retrieval.hybrid.models import HybridHit, RRFParameters
from src.retrieval.hybrid.rrf import reciprocal_rank_fusion
from src.retrieval.semantic import SemanticIndex, SemanticTextEncoder


class LexicalCandidateRetriever(Protocol):
    """Describe the ranked BM25 candidates required by hybrid search."""

    def search(self, query: str, top_k: int = 5) -> list[BM25Hit]:
        """Return ordered lexical candidates for one raw query."""


class ProductionBM25Retriever:
    """Preserve the selected auxiliary-path reranking BM25 baseline."""

    def __init__(
        self,
        index: BM25Index,
        *,
        auxiliary_path_penalty: float = 0.5,
        path_candidate_depth: int = 20,
    ) -> None:
        """Validate and retain the production lexical ranking policy."""
        if path_candidate_depth < 0:
            raise ValueError("Path candidate depth must not be negative")
        self._retriever = BM25Retriever(index)
        self._reranker = AuxiliaryPathReranker(auxiliary_path_penalty)
        self._path_candidate_depth = path_candidate_depth

    def search(self, query: str, top_k: int = 5) -> list[BM25Hit]:
        """Return BM25 candidates after the configured path reranking."""
        if top_k <= 0:
            raise ValueError("BM25 candidate top_k must be greater than zero")
        candidate_count = top_k
        if self._reranker.penalty_weight > 0:
            candidate_count = max(top_k, self._path_candidate_depth)
        hits = self._retriever.search(query, top_k=candidate_count)
        return [result.hit for result in self._reranker.rerank(hits)][:top_k]


class HybridRetriever:
    """Reuse two retrievers and one encoder for rank-based search."""

    def __init__(
        self,
        lexical_retriever: LexicalCandidateRetriever,
        semantic_index: SemanticIndex,
        semantic_encoder: SemanticTextEncoder,
        parameters: RRFParameters | None = None,
    ) -> None:
        """Keep loaded retrieval resources available across queries."""
        self._lexical_retriever = lexical_retriever
        self._semantic_index = semantic_index
        self._semantic_encoder = semantic_encoder
        self._parameters = parameters or RRFParameters()

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[HybridHit]:
        """Retrieve two candidate rankings and return their RRF fusion."""
        if top_k <= 0:
            raise ValueError("Hybrid search top_k must be greater than zero")
        if candidate_k < top_k:
            raise ValueError(
                "Hybrid candidate_k must be greater than or equal to top_k"
            )
        lexical_hits = self._lexical_retriever.search(
            query,
            top_k=candidate_k,
        )
        query_embedding = self._semantic_encoder.encode((query,))[0]
        return self.search_with_embedding(
            query,
            query_embedding,
            top_k=top_k,
            candidate_k=candidate_k,
            lexical_hits=lexical_hits,
        )

    def search_with_embedding(
        self,
        query: str,
        query_embedding: Tensor,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
        lexical_hits: list[BM25Hit] | None = None,
    ) -> list[HybridHit]:
        """Fuse one pre-encoded query without invoking the encoder again."""
        if top_k <= 0:
            raise ValueError("Hybrid search top_k must be greater than zero")
        if candidate_k < top_k:
            raise ValueError(
                "Hybrid candidate_k must be greater than or equal to top_k"
            )
        active_lexical_hits = lexical_hits
        if active_lexical_hits is None:
            active_lexical_hits = self._lexical_retriever.search(
                query,
                top_k=candidate_k,
            )
        semantic_hits = self._semantic_index.search(
            query_embedding,
            top_k=candidate_k,
        )
        return reciprocal_rank_fusion(
            active_lexical_hits,
            semantic_hits,
            top_k=top_k,
            parameters=self._parameters,
        )
