"""Fuse lexical and semantic rankings with reciprocal rank fusion."""

from collections.abc import Sequence
from dataclasses import dataclass

from src.ingestion import Chunk
from src.retrieval.bm25 import BM25Hit
from src.retrieval.hybrid.models import HybridHit, RRFParameters
from src.retrieval.semantic import SemanticHit

SourceKey = tuple[str, int, int]


@dataclass(slots=True)
class _Candidate:
    """Accumulate contributions for one exact source span."""

    chunk: Chunk
    first_seen: int
    score: float = 0.0
    lexical_rank: int | None = None
    semantic_rank: int | None = None


def reciprocal_rank_fusion(
    lexical_hits: Sequence[BM25Hit],
    semantic_hits: Sequence[SemanticHit],
    *,
    top_k: int = 5,
    parameters: RRFParameters | None = None,
) -> list[HybridHit]:
    """Combine two ordered rankings without comparing their raw scores."""
    if top_k <= 0:
        raise ValueError("Hybrid search top_k must be greater than zero")
    active_parameters = parameters or RRFParameters()
    candidates: dict[SourceKey, _Candidate] = {}
    _add_ranking(
        candidates,
        [(hit.document.key, hit.document.chunk) for hit in lexical_hits],
        source_name="lexical",
        weight=active_parameters.lexical_weight,
        rank_constant=active_parameters.rank_constant,
    )
    _add_ranking(
        candidates,
        [(hit.document.key, hit.document.chunk) for hit in semantic_hits],
        source_name="semantic",
        weight=active_parameters.semantic_weight,
        rank_constant=active_parameters.rank_constant,
    )
    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.score, candidate.first_seen),
    )
    return [
        HybridHit(
            chunk=candidate.chunk,
            score=candidate.score,
            lexical_rank=candidate.lexical_rank,
            semantic_rank=candidate.semantic_rank,
        )
        for candidate in ranked[:top_k]
    ]


def _add_ranking(
    candidates: dict[SourceKey, _Candidate],
    sources: Sequence[tuple[SourceKey, Chunk]],
    *,
    source_name: str,
    weight: float,
    rank_constant: int,
) -> None:
    """Add one ranking while enforcing unambiguous source identities."""
    if weight == 0:
        return
    seen: set[SourceKey] = set()
    for rank, (key, chunk) in enumerate(sources, start=1):
        if key in seen:
            raise ValueError(
                f"Hybrid {source_name} ranking contains duplicate source spans"
            )
        seen.add(key)
        candidate = candidates.get(key)
        if candidate is None:
            candidate = _Candidate(
                chunk=chunk,
                first_seen=len(candidates),
            )
            candidates[key] = candidate
        elif candidate.chunk != chunk:
            raise ValueError(
                "Hybrid rankings disagree about content for one source span"
            )
        candidate.score += weight / (rank_constant + rank)
        if source_name == "lexical":
            candidate.lexical_rank = rank
        else:
            candidate.semantic_rank = rank
