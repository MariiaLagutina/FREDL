"""Apply a bounded exact-identifier bonus to existing BM25 hits."""

from dataclasses import dataclass
import re
from collections.abc import Sequence

from src.retrieval.bm25.models import BM25Hit
from src.retrieval.tokenization.shared import normalize_lexeme, scan_lexemes


_CAMEL_CASE_PATTERN = re.compile(r"[a-z][A-Z]|[A-Z]{2,}[a-z]")


@dataclass(frozen=True, slots=True)
class IdentifierRerankHit:
    """Expose the original hit and every contribution to its new score."""

    hit: BM25Hit
    matched_identifiers: tuple[str, ...]
    bonus: float
    score: float


@dataclass(frozen=True, slots=True)
class IdentifierReranker:
    """Reward complete structural query identifiers without replacing BM25."""

    match_weight: float = 0.1
    max_matches: int = 3

    def __post_init__(self) -> None:
        """Require a small non-negative bonus with an explicit upper bound."""
        if not 0 <= self.match_weight <= 1:
            message = "Identifier match weight must be between zero and one"
            raise ValueError(message)
        if self.max_matches < 1:
            raise ValueError("Identifier max matches must be positive")

    def rerank(
        self,
        query: str,
        hits: Sequence[BM25Hit],
    ) -> list[IdentifierRerankHit]:
        """Rerank existing hits by bounded complete-identifier matches."""
        if not isinstance(query, str):
            raise TypeError("Identifier reranking query must be a string")
        identifiers = _query_identifiers(query)
        reranked = [self._score_hit(hit, identifiers) for hit in hits]
        reranked.sort(
            key=lambda result: (-result.score, result.hit.document.key)
        )
        return reranked

    def _score_hit(
        self,
        hit: BM25Hit,
        identifiers: tuple[str, ...],
    ) -> IdentifierRerankHit:
        terms = set(hit.document.content_terms)
        terms.update(hit.document.metadata_terms)
        matches = tuple(
            identifier for identifier in identifiers if identifier in terms
        )[: self.max_matches]
        bonus = hit.score * self.match_weight * len(matches)
        return IdentifierRerankHit(
            hit=hit,
            matched_identifiers=matches,
            bonus=bonus,
            score=hit.score + bonus,
        )


def _query_identifiers(query: str) -> tuple[str, ...]:
    """Return unique normalized lexemes with explicit identifier structure."""
    identifiers: list[str] = []
    for lexeme in scan_lexemes(query):
        if not _has_identifier_structure(lexeme):
            continue
        normalized = normalize_lexeme(lexeme)
        if normalized not in identifiers:
            identifiers.append(normalized)
    return tuple(identifiers)


def _has_identifier_structure(lexeme: str) -> bool:
    """Distinguish identifiers from ordinary natural-language words."""
    return (
        "_" in lexeme
        or "." in lexeme
        or bool(_CAMEL_CASE_PATTERN.search(lexeme))
        or (len(lexeme) > 1 and lexeme.isupper())
    )
