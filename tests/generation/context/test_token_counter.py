"""Tests for model-tokenizer context token counting."""

from dataclasses import dataclass, field

from src.generation import HuggingFaceTokenCounter


@dataclass
class RecordingTokenizer:
    """Record adapter calls and return deterministic token identifiers."""

    calls: list[tuple[str, bool]] = field(default_factory=list)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        """Return one synthetic token per whitespace-separated word."""
        self.calls.append((text, add_special_tokens))
        return list(range(len(text.split())))


def test_counter_uses_tokenizer_without_special_tokens() -> None:
    """Context budgets use content tokens from the injected model tokenizer."""
    tokenizer = RecordingTokenizer()
    counter = HuggingFaceTokenCounter(tokenizer)

    count = counter.count_tokens("one two three")

    assert count == 3
    assert tokenizer.calls == [("one two three", False)]


def test_counter_supports_empty_context() -> None:
    """Empty context delegates to the tokenizer with no content tokens."""
    tokenizer = RecordingTokenizer()

    count = HuggingFaceTokenCounter(tokenizer).count_tokens("")

    assert count == 0
    assert tokenizer.calls == [("", False)]
