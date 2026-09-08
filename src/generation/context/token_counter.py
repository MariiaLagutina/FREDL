"""Adapt a Hugging Face-compatible tokenizer to context token counting."""

from dataclasses import dataclass
from typing import Protocol


class HuggingFaceTokenizer(Protocol):
    """Describe the small tokenizer surface required by context building."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        """Return token identifiers for one exact text value."""


@dataclass(frozen=True, slots=True)
class HuggingFaceTokenCounter:
    """Count content tokens with the selected model's tokenizer."""

    tokenizer: HuggingFaceTokenizer

    def count_tokens(self, text: str) -> int:
        """Count context content without model prompt-control tokens."""
        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )
