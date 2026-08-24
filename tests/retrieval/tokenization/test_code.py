"""Golden tests for source-code identifier expansion."""

from src.retrieval.tokenization.code import expand_code_lexeme


def test_expansion_preserves_snake_case_and_adds_components() -> None:
    """An exact identifier and its underscore-separated terms coexist."""
    assert expand_code_lexeme("gpu_memory_utilization") == [
        "gpu_memory_utilization",
        "gpu",
        "memory",
        "utilization",
    ]


def test_expansion_splits_camel_case() -> None:
    """CamelCase remains exact while its word boundaries become searchable."""
    assert expand_code_lexeme("SamplingParams") == [
        "samplingparams",
        "sampling",
        "params",
    ]


def test_expansion_splits_dotted_identifier_layers() -> None:
    """Qualified names retain both their path and component context."""
    assert expand_code_lexeme("PagedAttention.forward") == [
        "pagedattention.forward",
        "pagedattention",
        "paged",
        "attention",
        "forward",
    ]


def test_expansion_handles_acronyms_and_numbers() -> None:
    """Acronym boundaries and numeric suffixes remain explicit signals."""
    assert expand_code_lexeme("HTTPServer2") == [
        "httpserver2",
        "http",
        "server",
        "2",
    ]


def test_expansion_keeps_dunder_name_and_readable_component() -> None:
    """Python punctuation stays exact without hiding the base name."""
    assert expand_code_lexeme("__init__") == ["__init__", "init"]


def test_expansion_does_not_duplicate_plain_word() -> None:
    """A token already equal to its only component appears once."""
    assert expand_code_lexeme("cache") == ["cache"]


def test_expansion_rejects_symbol_only_lexeme() -> None:
    """A symbolic separator cannot become a searchable code token."""
    assert expand_code_lexeme("___") == []
