"""Run the existing local MiniLM encoder over verified PDF records."""

from collections.abc import Sequence
from dataclasses import dataclass

from src.evaluation.pdf.bm25_baseline import PdfBm25Baseline
from src.evaluation.pdf.models import PdfGoldenDataset
from src.evaluation.pdf.retrieval_metrics import PdfRetrievalResult
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.retrieval.semantic import (
    build_semantic_index,
    MiniLMEncoder,
    SemanticTextEncoder,
)


@dataclass(frozen=True, slots=True)
class PdfMiniLmHit:
    """Return one PDF record with its cosine similarity score."""

    record: PdfIndexRecord
    score: float


class PdfMiniLmBaseline:
    """Adapt verified PDF records to the existing semantic index."""

    def __init__(
        self,
        records: Sequence[PdfIndexRecord],
        encoder: SemanticTextEncoder | None = None,
    ) -> None:
        """Encode the exact PDF record text once and retain record identity."""
        self._records = tuple(records)
        self._encoder = encoder or MiniLMEncoder()
        lexical = PdfBm25Baseline(self._records)
        self._index = build_semantic_index(
            lexical.lexical_index,
            self._encoder,
        )
        self._records_by_key = {
            document.key: record
            for document, record in zip(self._index.documents, self._records)
        }

    @property
    def record_count(self) -> int:
        """Return the exact number of embedded PDF records."""
        return len(self._records)

    @property
    def dimension(self) -> int:
        """Return the stored semantic vector width."""
        return self._index.dimension

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[PdfMiniLmHit, ...]:
        """Rank the complete PDF corpus by cosine similarity."""
        if not isinstance(query, str):
            raise TypeError("PDF semantic query must be a string")
        if not query.strip():
            raise ValueError("PDF semantic query must not be empty")
        limit = self.record_count if top_k is None else top_k
        query_embedding = self._encoder.encode((query,))[0]
        return tuple(
            PdfMiniLmHit(
                record=self._records_by_key[hit.document.key],
                score=hit.score,
            )
            for hit in self._index.search(query_embedding, top_k=limit)
        )

    def run_dataset(
        self,
        dataset: PdfGoldenDataset,
    ) -> tuple[PdfRetrievalResult, ...]:
        """Rank all PDF records per question without an abstention guess."""
        return tuple(
            PdfRetrievalResult(
                case_id=case.id,
                ranked_record_ids=tuple(
                    hit.record.record_id
                    for hit in self.search(case.question)
                ),
            )
            for case in dataset.cases
        )
