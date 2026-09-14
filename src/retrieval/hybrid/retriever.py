"""Coordinate lexical and semantic retrieval before rank fusion."""

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid.models import HybridHit, RRFParameters
from src.retrieval.hybrid.rrf import reciprocal_rank_fusion
from src.retrieval.semantic import SemanticIndex, SemanticTextEncoder


class HybridRetriever:
    """Reuse two retrievers and one encoder for rank-based search."""

    def __init__(
        self,
        lexical_retriever: BM25Retriever,
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
        semantic_hits = self._semantic_index.search(
            query_embedding,
            top_k=candidate_k,
        )
        return reciprocal_rank_fusion(
            lexical_hits,
            semantic_hits,
            top_k=top_k,
            parameters=self._parameters,
        )
