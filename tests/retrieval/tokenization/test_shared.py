"""Golden tests for shared lexical scanning."""

from src.retrieval.tokenization import scan_tokens
from src.retrieval.tokenization.shared import scan_lexemes


def test_scanner_normalizes_case_and_discards_outer_punctuation() -> None:
    """Ordinary sentence punctuation does not become a search term."""
    assert scan_tokens("Hello, WORLD! Search-ready.") == [
        "hello",
        "world",
        "search",
        "ready",
    ]


def test_scanner_preserves_unicode_words() -> None:
    """Lowercasing keeps non-ASCII letters searchable."""
    assert scan_tokens("ÜBER Größe, Straße.") == [
        "über",
        "größe",
        "straße",
    ]


def test_scanner_preserves_technical_lexical_units() -> None:
    """Later code expansion still receives complete identifiers."""
    assert scan_tokens(
        "gpu_memory_utilization vLLM.engine v0.10.1 __init__"
    ) == [
        "gpu_memory_utilization",
        "vllm.engine",
        "v0.10.1",
        "__init__",
    ]


def test_scanner_keeps_order_and_repeated_terms() -> None:
    """Ranking can later use occurrence order and term frequency."""
    assert scan_tokens("Cache cache CACHE") == [
        "cache",
        "cache",
        "cache",
    ]


def test_scanner_ignores_punctuation_only_input() -> None:
    """Separators alone do not produce empty or symbolic tokens."""
    assert scan_tokens("... ___ ---") == []


def test_lexeme_scanner_preserves_structural_capitalization() -> None:
    """Code expansion receives CamelCase boundaries before normalization."""
    assert scan_lexemes("SamplingParams HTTPServer2") == [
        "SamplingParams",
        "HTTPServer2",
    ]
