"""Build a semantic index from the exact documents already used by BM25."""

from typing import Protocol

from torch import Tensor

from src.retrieval.bm25 import BM25Index
from src.retrieval.semantic.index import SemanticIndex
from src.retrieval.semantic.models import SemanticDocument


class SemanticTextEncoder(Protocol):
    """Describe the embedding operation required by the index builder."""

    def encode(self, texts: tuple[str, ...]) -> Tensor:
        """Return one normalized vector per input text."""


def build_semantic_index(
    lexical_index: BM25Index,
    encoder: SemanticTextEncoder,
) -> SemanticIndex:
    """Encode BM25 chunks without creating a second corpus pipeline."""
    lexical_documents = lexical_index.documents
    semantic_documents = tuple(
        SemanticDocument(document.chunk) for document in lexical_documents
    )
    embeddings = encoder.encode(
        tuple(document.chunk.text for document in lexical_documents)
    )
    return SemanticIndex(semantic_documents, embeddings)
