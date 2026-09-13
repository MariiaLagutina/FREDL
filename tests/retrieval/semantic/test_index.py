"""Tests for the exact semantic index."""

import pytest
import torch

from src.ingestion import Chunk
from src.retrieval.semantic import SemanticDocument, SemanticIndex


def _document(path: str, start: int = 0, end: int = 4) -> SemanticDocument:
    return SemanticDocument(
        Chunk(
            file_path=path,
            start=start,
            end=end,
            text="text",
            section_path=(),
        )
    )


def test_search_ranks_normalized_embeddings_by_cosine_similarity() -> None:
    documents = (_document("first.py"), _document("second.py"))
    index = SemanticIndex(
        documents,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )

    hits = index.search(torch.tensor([0.8, 0.6]), top_k=2)

    assert [hit.document for hit in hits] == [documents[0], documents[1]]
    assert [hit.score for hit in hits] == pytest.approx([0.8, 0.6])


def test_search_uses_document_order_to_break_equal_score_ties() -> None:
    documents = (_document("first.py"), _document("second.py"))
    index = SemanticIndex(
        documents,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )

    hits = index.search(torch.tensor([1.0, 0.0]), top_k=2)

    assert [hit.document for hit in hits] == list(documents)


@pytest.mark.parametrize(
    ("embeddings", "message"),
    [
        (torch.tensor([1.0, 0.0]), "two-dimensional"),
        (torch.tensor([[1.0, 0.0, 0.0]]), "counts must match"),
        (torch.tensor([[2.0, 0.0], [0.0, 1.0]]), "L2-normalized"),
        (torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]), "finite"),
    ],
)
def test_index_rejects_invalid_embedding_matrices(
    embeddings: torch.Tensor,
    message: str,
) -> None:
    documents = (_document("first.py"), _document("second.py"))

    with pytest.raises(ValueError, match=message):
        SemanticIndex(documents, embeddings)


def test_index_rejects_duplicate_source_spans() -> None:
    document = _document("same.py")

    with pytest.raises(ValueError, match="unique source spans"):
        SemanticIndex(
            (document, document),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )


@pytest.mark.parametrize(
    ("query", "message"),
    [
        (torch.tensor([1.0, 0.0, 0.0]), "dimension must match"),
        (torch.tensor([2.0, 0.0]), "L2-normalized"),
        (torch.tensor([float("inf"), 0.0]), "finite"),
    ],
)
def test_search_rejects_invalid_query_embeddings(
    query: torch.Tensor,
    message: str,
) -> None:
    index = SemanticIndex(
        (_document("source.py"),),
        torch.tensor([[1.0, 0.0]]),
    )

    with pytest.raises(ValueError, match=message):
        index.search(query)


def test_index_isolated_from_mutable_input_and_returned_tensors() -> None:
    embeddings = torch.tensor([[1.0, 0.0]])
    index = SemanticIndex((_document("source.py"),), embeddings)

    embeddings[0, 0] = 0.0
    returned = index.embeddings
    returned[0, 0] = 0.0

    hits = index.search(torch.tensor([1.0, 0.0]))

    assert hits[0].score == pytest.approx(1.0)
