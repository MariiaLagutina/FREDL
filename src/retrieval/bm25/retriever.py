"""Expose raw-query search over a prebuilt BM25 index."""

from src.retrieval.bm25.index import BM25Index
from src.retrieval.bm25.models import BM25Hit
from src.retrieval.tokenization import QueryTokenizer


class BM25Retriever:
    """Translate a user query into lexical terms before ranking."""

    def __init__(
        self,
        index: BM25Index,
        query_tokenizer: QueryTokenizer | None = None,
    ) -> None:
        """Keep query preparation outside the mathematical index."""
        self._index = index
        self._query_tokenizer = query_tokenizer or QueryTokenizer()

    @property
    def index(self) -> BM25Index:
        """Return the exact index used for retrieval and inspection."""
        return self._index

    def search(self, query: str, top_k: int = 5) -> list[BM25Hit]:
        """Tokenize one raw query and return its ranked matching chunks."""
        if not isinstance(query, str):
            raise TypeError("BM25 query must be a string")
        if not query.strip():
            raise ValueError("BM25 query must contain non-whitespace text")
        return self._index.search(
            tuple(self._query_tokenizer.tokenize(query)),
            top_k=top_k,
        )
