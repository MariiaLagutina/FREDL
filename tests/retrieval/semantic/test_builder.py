"""Tests for alignment between lexical and semantic documents."""

from dataclasses import dataclass, field

import pytest
import torch
from torch import Tensor

from src.ingestion import Chunk
from src.retrieval.bm25 import BM25Document, BM25Index
from src.retrieval.semantic import build_semantic_index


@dataclass
class FakeEncoder:
    """Return controlled vectors while recording exact input text."""

    embeddings: Tensor
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def encode(self, texts: tuple[str, ...]) -> Tensor:
        self.calls.append(texts)
        return self.embeddings


def _bm25_document(
    path: str,
    text: str,
    start: int,
) -> BM25Document:
    return BM25Document(
        chunk=Chunk(
            file_path=path,
            start=start,
            end=start + len(text),
            text=text,
            section_path=("Section",),
        ),
        content_terms=("term",),
        metadata_terms=("metadata",),
    )


def test_builder_encodes_exact_bm25_chunk_text_in_document_order() -> None:
    first = _bm25_document("guide.md", "first text", 10)
    second = _bm25_document("source.py", "second text", 30)
    lexical_index = BM25Index((first, second))
    encoder = FakeEncoder(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    semantic_index = build_semantic_index(lexical_index, encoder)

    assert encoder.calls == [("first text", "second text")]
    assert [
        document.chunk for document in semantic_index.documents
    ] == [first.chunk, second.chunk]
    assert semantic_index.dimension == 2


def test_builder_relies_on_index_validation_for_encoder_row_count() -> None:
    lexical_index = BM25Index((_bm25_document("guide.md", "text", 0),))
    encoder = FakeEncoder(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="counts must match"):
        build_semantic_index(lexical_index, encoder)


def test_builder_supports_an_empty_lexical_index() -> None:
    lexical_index = BM25Index(())
    encoder = FakeEncoder(torch.empty((0, 384), dtype=torch.float32))

    semantic_index = build_semantic_index(lexical_index, encoder)

    assert encoder.calls == [()]
    assert semantic_index.documents == ()
    assert semantic_index.dimension == 384
