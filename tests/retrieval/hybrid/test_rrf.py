"""Tests for reciprocal rank fusion across exact source spans."""

import pytest

from src.ingestion import Chunk
from src.retrieval.bm25 import BM25Document, BM25Hit
from src.retrieval.hybrid import RRFParameters, reciprocal_rank_fusion
from src.retrieval.semantic import SemanticDocument, SemanticHit


def _chunk(path: str, text: str | None = None) -> Chunk:
    content = text or path
    return Chunk(file_path=path, start=0, end=len(content), text=content)


def _lexical_hit(chunk: Chunk, score: float = 1.0) -> BM25Hit:
    document = BM25Document(chunk=chunk, content_terms=("term",))
    return BM25Hit(document, score, score, 0.0)


def _semantic_hit(chunk: Chunk, score: float = 0.5) -> SemanticHit:
    return SemanticHit(SemanticDocument(chunk), score)


def test_rrf_rewards_sources_supported_by_both_rankings() -> None:
    shared = _chunk("shared.py")
    lexical_only = _chunk("lexical.py")
    semantic_only = _chunk("semantic.py")

    hits = reciprocal_rank_fusion(
        [_lexical_hit(lexical_only), _lexical_hit(shared)],
        [_semantic_hit(semantic_only), _semantic_hit(shared)],
        top_k=3,
    )

    assert [hit.chunk for hit in hits] == [
        shared,
        lexical_only,
        semantic_only,
    ]
    assert hits[0].lexical_rank == 2
    assert hits[0].semantic_rank == 2
    assert hits[1].semantic_rank is None
    assert hits[2].lexical_rank is None


def test_rrf_uses_weights_without_comparing_raw_retriever_scores() -> None:
    lexical = _chunk("lexical.py")
    semantic = _chunk("semantic.py")

    hits = reciprocal_rank_fusion(
        [_lexical_hit(lexical, score=0.001)],
        [_semantic_hit(semantic, score=0.999)],
        top_k=2,
        parameters=RRFParameters(lexical_weight=2.0),
    )

    assert [hit.chunk for hit in hits] == [lexical, semantic]


def test_zero_weight_excludes_that_retriever_from_fusion() -> None:
    lexical = _chunk("lexical.py")
    semantic = _chunk("semantic.py")

    hits = reciprocal_rank_fusion(
        [_lexical_hit(lexical)],
        [_semantic_hit(semantic)],
        parameters=RRFParameters(semantic_weight=0.0),
    )

    assert [hit.chunk for hit in hits] == [lexical]


def test_rrf_rejects_duplicate_source_spans_within_one_ranking() -> None:
    chunk = _chunk("duplicate.py")

    with pytest.raises(ValueError, match="lexical ranking contains duplicate"):
        reciprocal_rank_fusion(
            [_lexical_hit(chunk), _lexical_hit(chunk)],
            [],
        )


def test_rrf_rejects_content_mismatch_for_the_same_source_span() -> None:
    lexical = _chunk("source.py", "first")
    semantic = _chunk("source.py", "other")

    with pytest.raises(ValueError, match="disagree about content"):
        reciprocal_rank_fusion(
            [_lexical_hit(lexical)],
            [_semantic_hit(semantic)],
        )


def test_rrf_rejects_invalid_limits_and_parameters() -> None:
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([], [], top_k=0)
    with pytest.raises(ValueError, match="rank constant"):
        RRFParameters(rank_constant=0)
    with pytest.raises(ValueError, match="must not be negative"):
        RRFParameters(semantic_weight=-1.0)
    with pytest.raises(ValueError, match="At least one"):
        RRFParameters(lexical_weight=0.0, semantic_weight=0.0)
