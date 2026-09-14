"""Public rank-based hybrid retrieval contracts."""

from src.retrieval.hybrid.models import HybridHit, RRFParameters
from src.retrieval.hybrid.retriever import (
    HybridRetriever,
    LexicalCandidateRetriever,
    ProductionBM25Retriever,
)
from src.retrieval.hybrid.rrf import reciprocal_rank_fusion
from src.retrieval.hybrid.workflow import (
    HybridBatchSearchReport,
    HybridSearchReport,
    run_stored_hybrid_retrieval,
    run_stored_hybrid_search,
)

__all__ = [
    "HybridHit",
    "HybridBatchSearchReport",
    "HybridRetriever",
    "HybridSearchReport",
    "LexicalCandidateRetriever",
    "ProductionBM25Retriever",
    "RRFParameters",
    "reciprocal_rank_fusion",
    "run_stored_hybrid_retrieval",
    "run_stored_hybrid_search",
]
