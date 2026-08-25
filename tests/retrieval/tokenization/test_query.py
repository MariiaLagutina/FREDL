"""Tests for mixed natural-language and code query tokenization."""

from src.retrieval.tokenization import QueryTokenizer


def test_query_keeps_plain_terms_once() -> None:
    """Text and code tokenizers do not duplicate ordinary words."""
    assert QueryTokenizer().tokenize("Find cache") == ["find", "cache"]


def test_query_expands_qualified_code_identifier() -> None:
    """A symbol query keeps exact, qualified, and component signals."""
    assert QueryTokenizer().tokenize("CacheStore.get_item") == [
        "cachestore.get_item",
        "cachestore",
        "cache",
        "store",
        "get",
        "item",
    ]


def test_query_keeps_cautious_possessive_base() -> None:
    """Natural-language query rules remain available for possessives."""
    assert QueryTokenizer().tokenize("developer's cache") == [
        "developer's",
        "developer",
        "cache",
    ]
