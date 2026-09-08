"""Tests for bounded auxiliary-source path reranking."""

import pytest

from src.ingestion import Chunk
from src.retrieval.bm25 import (
    AuxiliaryPathReranker,
    BM25Document,
    BM25Hit,
)


def _hit(file_path: str, score: float) -> BM25Hit:
    """Create one inspectable synthetic path hit."""
    return BM25Hit(
        BM25Document(
            Chunk(file_path, 0, len(file_path), file_path),
            ("term",),
        ),
        score,
        score,
        0.0,
    )


def test_auxiliary_penalty_promotes_close_documentation_hit() -> None:
    """Examples and tests may move below a close regular source."""
    results = AuxiliaryPathReranker(0.1).rerank(
        [
            _hit("data/raw/project/examples/demo.py", 10.0),
            _hit("data/raw/project/docs/guide.md", 9.5),
        ]
    )

    assert [result.hit.document.chunk.file_path for result in results] == [
        "data/raw/project/docs/guide.md",
        "data/raw/project/examples/demo.py",
    ]
    assert results[1].penalty == pytest.approx(1.0)


def test_path_segments_must_match_exactly() -> None:
    """A filename containing tests is not an auxiliary directory."""
    result = AuxiliaryPathReranker(0.5).rerank(
        [_hit("data/raw/project/src/contest.py", 4.0)]
    )[0]

    assert result.penalty == 0
    assert result.score == 4.0


@pytest.mark.parametrize("penalty", [-0.1, 1.1])
def test_invalid_auxiliary_penalty_is_rejected(penalty: float) -> None:
    """A path penalty remains a bounded fraction of the BM25 score."""
    with pytest.raises(ValueError, match="between zero and one"):
        AuxiliaryPathReranker(penalty)
