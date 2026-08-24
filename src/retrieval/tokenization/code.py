"""Expand source-code identifiers into exact and component search terms."""

import re

from src.retrieval.tokenization.shared import normalize_lexeme, scan_lexemes


_IDENTIFIER_SEPARATOR_PATTERN = re.compile(r"[._]+")
_CAMEL_PART_PATTERN = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|[A-Z]+|\d+"
)


class CodeTokenizer:
    """Create exact and component lexical signals from source code."""

    def tokenize(self, text: str) -> list[str]:
        """Tokenize complete source text without removing term frequency."""
        tokens: list[str] = []
        for lexeme in scan_lexemes(text):
            tokens.extend(expand_code_lexeme(lexeme))
        return tokens


def expand_code_lexeme(lexeme: str) -> list[str]:
    """Return one normalized identifier followed by unique subword signals."""
    if not any(character.isalnum() for character in lexeme):
        return []

    expanded: list[str] = []
    _append_unique(expanded, normalize_lexeme(lexeme))

    for component in _IDENTIFIER_SEPARATOR_PATTERN.split(lexeme):
        if not component:
            continue
        _append_unique(expanded, normalize_lexeme(component))
        if not component.isascii() or "'" in component:
            continue
        for match in _CAMEL_PART_PATTERN.finditer(component):
            _append_unique(expanded, normalize_lexeme(match.group()))

    return expanded


def _append_unique(tokens: list[str], candidate: str) -> None:
    """Append one non-empty token once while preserving signal order."""
    if candidate and candidate not in tokens:
        tokens.append(candidate)
