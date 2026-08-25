"""Tokenize mixed natural-language and source-code search queries."""

from src.retrieval.tokenization.code import CodeTokenizer
from src.retrieval.tokenization.text import TextTokenizer


class QueryTokenizer:
    """Combine text and code signals without repeating query terms."""

    def __init__(self) -> None:
        """Create the two deterministic lexical tokenizers once."""
        self._text_tokenizer = TextTokenizer()
        self._code_tokenizer = CodeTokenizer()

    def tokenize(self, query: str) -> list[str]:
        """Return stable unique terms for a mixed codebase query."""
        terms: list[str] = []
        seen: set[str] = set()
        for tokenizer in (self._text_tokenizer, self._code_tokenizer):
            for term in tokenizer.tokenize(query):
                if term not in seen:
                    seen.add(term)
                    terms.append(term)
        return terms
