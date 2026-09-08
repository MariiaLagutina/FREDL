"""Tests for bounded exact-identifier reranking of BM25 hits."""

import pytest

from src.ingestion import Chunk
from src.retrieval.bm25 import (
    BM25Document,
    BM25Hit,
    IdentifierReranker,
)


def _hit(
    file_path: str,
    score: float,
    *,
    content_terms: tuple[str, ...] = (),
    metadata_terms: tuple[str, ...] = (),
) -> BM25Hit:
    """Create one inspectable synthetic hit with an exact source span."""
    text = file_path
    return BM25Hit(
        document=BM25Document(
            chunk=Chunk(file_path, 0, len(text), text),
            content_terms=content_terms,
            metadata_terms=metadata_terms,
        ),
        score=score,
        content_score=score,
        metadata_score=0.0,
    )


@pytest.mark.parametrize(
    ("query", "term"),
    [
        ("Where is cache_key defined?", "cache_key"),
        ("How does CacheStore work?", "cachestore"),
        ("Where is _private_cache set?", "_private_cache"),
        ("What are FP8_MIN limits?", "fp8_min"),
        ("What does CacheStore.get_item return?", "cachestore.get_item"),
    ],
)
def test_complete_structural_identifier_receives_bonus(
    query: str,
    term: str,
) -> None:
    """Supported identifier forms match their complete normalized term."""
    result = IdentifierReranker().rerank(
        query,
        [_hit("match.py", 2.0, content_terms=(term,))],
    )[0]

    assert result.matched_identifiers == (term,)
    assert result.bonus == pytest.approx(0.2)
    assert result.score == pytest.approx(2.2)


def test_exact_identifier_can_promote_lower_bm25_hit() -> None:
    """A close lower hit may move first with complete identifier evidence."""
    higher = _hit("general.py", 10.0, content_terms=("fp8", "min"))
    exact = _hit("definition.py", 9.5, metadata_terms=("fp8_min",))

    results = IdentifierReranker().rerank(
        "Where is FP8_MIN defined?",
        [higher, exact],
    )

    assert [result.hit.document.chunk.file_path for result in results] == [
        "definition.py",
        "general.py",
    ]
    assert results[0].matched_identifiers == ("fp8_min",)


def test_words_and_identifier_components_receive_no_bonus() -> None:
    """Words and split components do not imitate an exact identifier."""
    result = IdentifierReranker().rerank(
        "Where is cache_key stored?",
        [_hit("partial.py", 3.0, content_terms=("cache", "key", "stored"))],
    )[0]

    assert result.matched_identifiers == ()
    assert result.bonus == 0.0
    assert result.score == 3.0


def test_bonus_is_capped_and_duplicate_query_identifiers_count_once() -> None:
    """Repeated identifiers cannot create an unbounded bonus."""
    result = IdentifierReranker(match_weight=0.1, max_matches=2).rerank(
        "FIRST_ID FIRST_ID SECOND_ID THIRD_ID",
        [
            _hit(
                "constants.py",
                5.0,
                content_terms=("first_id", "second_id", "third_id"),
            )
        ],
    )[0]

    assert result.matched_identifiers == ("first_id", "second_id")
    assert result.bonus == pytest.approx(1.0)


def test_unmatched_ties_use_deterministic_source_order() -> None:
    """Reranking preserves the existing source-key tie-breaking rule."""
    results = IdentifierReranker().rerank(
        "ordinary words",
        [_hit("z.py", 1.0), _hit("a.py", 1.0)],
    )

    assert [result.hit.document.chunk.file_path for result in results] == [
        "a.py",
        "z.py",
    ]


@pytest.mark.parametrize("match_weight", [-0.1, 1.1])
def test_invalid_match_weight_is_rejected(match_weight: float) -> None:
    """Identifier weight must stay inside its bounded domain."""
    with pytest.raises(ValueError, match="weight"):
        IdentifierReranker(match_weight=match_weight)


def test_invalid_max_matches_is_rejected() -> None:
    """The reranker requires a positive explicit match cap."""
    with pytest.raises(ValueError, match="max matches"):
        IdentifierReranker(max_matches=0)
