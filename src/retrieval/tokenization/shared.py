"""Shared lexical scanning without format-specific token expansion."""

import re


_TOKEN_PATTERN = re.compile(r"\w+(?:\.\w+)*")


def scan_tokens(text: str) -> list[str]:
    """Return normalized lexical units in their original occurrence order.

    Unicode words, underscores, and internal dots remain available for later
    format-specific expansion. Punctuation outside a token is discarded.
    Repeated terms stay repeated so later ranking can measure term frequency.
    """
    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
        if any(character.isalnum() for character in token)
    ]
