"""Build and query an explainable two-field BM25 index."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import log

from src.retrieval.bm25.models import (
    BM25CorpusStatistics,
    BM25Document,
    BM25Hit,
    BM25Parameters,
)


class BM25Index:
    """Rank immutable chunk documents with separate field statistics."""

    def __init__(
        self,
        documents: Sequence[BM25Document],
        parameters: BM25Parameters | None = None,
    ) -> None:
        """Calculate stable corpus statistics once at index construction."""
        self._documents = tuple(documents)
        self._parameters = parameters or BM25Parameters()
        _validate_unique_keys(self._documents)

        self._content_lengths = tuple(
            len(document.content_terms) for document in self._documents
        )
        self._metadata_lengths = tuple(
            len(document.metadata_terms) for document in self._documents
        )
        self._content_postings = _build_postings(
            document.content_terms for document in self._documents
        )
        self._metadata_postings = _build_postings(
            document.metadata_terms for document in self._documents
        )
        content_frequencies = _document_frequencies(
            self._content_postings
        )
        metadata_frequencies = _document_frequencies(
            self._metadata_postings
        )
        self._statistics = BM25CorpusStatistics(
            document_count=len(self._documents),
            average_content_length=_average_length(self._content_lengths),
            average_metadata_length=_average_length(self._metadata_lengths),
            content_document_frequencies=tuple(
                sorted(content_frequencies.items())
            ),
            metadata_document_frequencies=tuple(
                sorted(metadata_frequencies.items())
            ),
        )

    @property
    def parameters(self) -> BM25Parameters:
        """Return the exact scoring parameters stored with the index."""
        return self._parameters

    @property
    def documents(self) -> tuple[BM25Document, ...]:
        """Return immutable indexed documents for persistence and audits."""
        return self._documents

    @property
    def statistics(self) -> BM25CorpusStatistics:
        """Return immutable values calculated from the indexed corpus."""
        return self._statistics

    def search(
        self,
        query_terms: Sequence[str],
        top_k: int = 5,
    ) -> list[BM25Hit]:
        """Rank matching documents and expose both field contributions."""
        if top_k < 1:
            raise ValueError("BM25 top_k must be greater than zero")
        terms = _unique_query_terms(query_terms)
        if not terms:
            return []

        content_scores = self._score_field(
            terms,
            self._content_postings,
            self._content_lengths,
            self._statistics.average_content_length,
        )
        metadata_scores = self._score_field(
            terms,
            self._metadata_postings,
            self._metadata_lengths,
            self._statistics.average_metadata_length,
        )

        hits: list[BM25Hit] = []
        candidate_indexes = content_scores.keys() | metadata_scores.keys()
        for document_index in candidate_indexes:
            content_score = content_scores.get(document_index, 0.0)
            metadata_score = metadata_scores.get(document_index, 0.0)
            score = (
                content_score
                + self._parameters.metadata_weight * metadata_score
            )
            if score > 0:
                hits.append(
                    BM25Hit(
                        document=self._documents[document_index],
                        score=score,
                        content_score=content_score,
                        metadata_score=metadata_score,
                    )
                )

        hits.sort(key=lambda hit: (-hit.score, hit.document.key))
        return hits[:top_k]

    def _score_field(
        self,
        query_terms: tuple[str, ...],
        postings: Mapping[str, tuple[tuple[int, int], ...]],
        document_lengths: tuple[int, ...],
        average_length: float,
    ) -> dict[int, float]:
        """Calculate standard BM25 for one independent lexical field."""
        if average_length == 0:
            return {}

        scores: dict[int, float] = {}
        for term in query_terms:
            term_postings = postings.get(term)
            if term_postings is None:
                continue
            document_frequency = len(term_postings)
            inverse_document_frequency = log(
                1
                + (
                    self._statistics.document_count
                    - document_frequency
                    + 0.5
                )
                / (document_frequency + 0.5)
            )
            for document_index, term_frequency in term_postings:
                length_ratio = (
                    document_lengths[document_index] / average_length
                )
                normalization = 1 - self._parameters.b + (
                    self._parameters.b * length_ratio
                )
                contribution = inverse_document_frequency * (
                    term_frequency * (self._parameters.k1 + 1)
                ) / (
                    term_frequency + self._parameters.k1 * normalization
                )
                scores[document_index] = (
                    scores.get(document_index, 0.0) + contribution
                )
        return scores


def _build_postings(
    fields: Iterable[tuple[str, ...]],
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Store document indexes and term frequencies for fast lookup."""
    mutable_postings: dict[str, list[tuple[int, int]]] = {}
    for document_index, terms in enumerate(fields):
        for term, term_frequency in Counter(terms).items():
            mutable_postings.setdefault(term, []).append(
                (document_index, term_frequency)
            )
    return {
        term: tuple(term_postings)
        for term, term_postings in mutable_postings.items()
    }


def _document_frequencies(
    postings: Mapping[str, tuple[tuple[int, int], ...]],
) -> dict[str, int]:
    """Derive one inspectable document frequency per indexed term."""
    return {
        term: len(term_postings)
        for term, term_postings in postings.items()
    }


def _average_length(lengths: Sequence[int]) -> float:
    """Return mean field length, including empty metadata fields."""
    return sum(lengths) / len(lengths) if lengths else 0.0


def _unique_query_terms(query_terms: Sequence[str]) -> tuple[str, ...]:
    """Validate normalized terms and ignore accidental query repetition."""
    if isinstance(query_terms, str):
        raise TypeError("BM25 query terms must be a sequence of terms")
    if any(not isinstance(term, str) or not term for term in query_terms):
        raise ValueError("BM25 query terms must contain non-empty strings")
    return tuple(dict.fromkeys(query_terms))


def _validate_unique_keys(documents: Sequence[BM25Document]) -> None:
    """Reject ambiguous duplicate evidence spans before ranking."""
    keys = [document.key for document in documents]
    if len(keys) != len(set(keys)):
        raise ValueError("BM25 document keys must be unique")
