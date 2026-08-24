"""Integration tests for complete source-code tokenization."""

from src.retrieval.tokenization import CodeTokenizer


def test_code_tokenizer_combines_exact_and_component_signals() -> None:
    """Complete code keeps lexical order across several identifiers."""
    source = "class HTTPServer2:\n    def load_model(self):\n        pass\n"

    assert CodeTokenizer().tokenize(source) == [
        "class",
        "httpserver2",
        "http",
        "server",
        "2",
        "def",
        "load_model",
        "load",
        "model",
        "self",
        "pass",
    ]


def test_code_tokenizer_preserves_frequency_across_lexemes() -> None:
    """Repeated source terms remain available to lexical ranking."""
    assert CodeTokenizer().tokenize("cache = cache.get()") == [
        "cache",
        "cache.get",
        "cache",
        "get",
    ]


def test_code_tokenizer_handles_unicode_snake_case() -> None:
    """Unicode identifier components survive normalization and expansion."""
    assert CodeTokenizer().tokenize("größen_limit = 1") == [
        "größen_limit",
        "größen",
        "limit",
        "1",
    ]


def test_code_tokenizer_returns_empty_list_without_lexemes() -> None:
    """Source punctuation alone produces no lexical search terms."""
    assert CodeTokenizer().tokenize("... ---") == []
