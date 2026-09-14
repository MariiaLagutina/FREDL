"""Tests for coordinating reusable lexical and semantic resources."""

import torch
from torch import Tensor

from src.ingestion import Chunk
from src.retrieval.bm25 import BM25Document, BM25Index, BM25Retriever
from src.retrieval.hybrid import HybridRetriever, ProductionBM25Retriever
from src.retrieval.semantic import SemanticDocument, SemanticIndex


class _RecordingEncoder:
    """Return one controlled query vector and record encoding calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: tuple[str, ...]) -> Tensor:
        self.calls.append(texts)
        return torch.tensor([[1.0, 0.0]])


def _document(path: str, term: str) -> BM25Document:
    return BM25Document(
        chunk=Chunk(path, 0, len(term), term),
        content_terms=(term,),
    )


def test_hybrid_retriever_reuses_resources_and_fuses_deeper_candidates(
) -> None:
    lexical_only = _document("lexical.py", "cache")
    shared = _document("shared.py", "cache")
    lexical_index = BM25Index((lexical_only, shared))
    semantic_index = SemanticIndex(
        (
            SemanticDocument(_document("semantic.py", "other").chunk),
            SemanticDocument(shared.chunk),
        ),
        torch.tensor([[1.0, 0.0], [0.8, 0.6]]),
    )
    encoder = _RecordingEncoder()
    retriever = HybridRetriever(
        BM25Retriever(lexical_index),
        semantic_index,
        encoder,
    )

    hits = retriever.search("cache", top_k=1, candidate_k=2)

    assert [hit.chunk.file_path for hit in hits] == ["shared.py"]
    assert hits[0].lexical_rank == 2
    assert hits[0].semantic_rank == 2
    assert encoder.calls == [("cache",)]


def test_hybrid_retriever_rejects_too_shallow_candidate_pool() -> None:
    encoder = _RecordingEncoder()
    retriever = HybridRetriever(
        BM25Retriever(BM25Index(())),
        SemanticIndex((), torch.empty((0, 2))),
        encoder,
    )

    try:
        retriever.search("cache", top_k=5, candidate_k=4)
    except ValueError as error:
        assert "candidate_k" in str(error)
    else:
        raise AssertionError("Expected an invalid candidate depth error")

    assert encoder.calls == []


def test_production_lexical_candidates_keep_selected_path_penalty() -> None:
    auxiliary = _document("tests/cache.py", "cache")
    production = _document("src/cache.py", "cache")
    retriever = ProductionBM25Retriever(
        BM25Index((auxiliary, production)),
        path_candidate_depth=2,
    )

    hits = retriever.search("cache", top_k=1)

    assert [hit.document.chunk.file_path for hit in hits] == [
        "src/cache.py"
    ]
