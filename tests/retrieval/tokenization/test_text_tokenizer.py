"""Golden tests for documentation tokenization."""

from src.retrieval.tokenization import TextTokenizer


def test_text_tokenizer_normalizes_an_ordinary_sentence() -> None:
    """Case and sentence punctuation do not alter ordinary terms."""
    assert TextTokenizer().tokenize(
        "The MODEL supports continuous batching."
    ) == [
        "the",
        "model",
        "supports",
        "continuous",
        "batching",
    ]


def test_text_tokenizer_splits_hyphens_and_preserves_apostrophes() -> None:
    """Word boundaries stay useful without contraction fragments."""
    assert TextTokenizer().tokenize(
        "The model's memory-aware scheduler isn’t blocked."
    ) == [
        "the",
        "model's",
        "model",
        "memory",
        "aware",
        "scheduler",
        "isn't",
        "blocked",
    ]


def test_text_tokenizer_keeps_short_contractions_unexpanded() -> None:
    """Short apostrophe forms do not create speculative base signals."""
    assert TextTokenizer().tokenize("It's clear she's ready.") == [
        "it's",
        "clear",
        "she's",
        "ready",
    ]


def test_text_tokenizer_expands_uppercase_technical_possessive() -> None:
    """A short uppercase acronym remains searchable without its suffix."""
    assert TextTokenizer().tokenize("The GPU's cache is full.") == [
        "the",
        "gpu's",
        "gpu",
        "cache",
        "is",
        "full",
    ]


def test_text_tokenizer_preserves_technical_units_in_documentation() -> None:
    """Documentation can still contain exact code and version references."""
    assert TextTokenizer().tokenize(
        "Use gpu_memory_utilization with vLLM.engine in v0.10.1."
    ) == [
        "use",
        "gpu_memory_utilization",
        "with",
        "vllm.engine",
        "in",
        "v0.10.1",
    ]


def test_text_tokenizer_does_not_apply_stemming() -> None:
    """Different technical word forms remain independently searchable."""
    assert TextTokenizer().tokenize(
        "generate generated generation generator"
    ) == [
        "generate",
        "generated",
        "generation",
        "generator",
    ]


def test_text_tokenizer_ignores_markdown_decoration() -> None:
    """Common markup characters do not become lexical terms."""
    assert TextTokenizer().tokenize("# **Model setup** `served_model`") == [
        "model",
        "setup",
        "served_model",
    ]
