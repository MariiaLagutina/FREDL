"""Fuse PDF BM25 and MiniLM rankings with existing project RRF."""

from collections.abc import Sequence
from dataclasses import dataclass

from src.evaluation.pdf.bm25_baseline import PdfBm25Baseline
from src.evaluation.pdf.models import PdfGoldenDataset
from src.evaluation.pdf.retrieval_metrics import PdfRetrievalResult
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever, RRFParameters
from src.retrieval.semantic import (
    build_semantic_index,
    MiniLMEncoder,
    SemanticTextEncoder,
)


PDF_HYBRID_PARAMETERS = RRFParameters(
    rank_constant=60,
    lexical_weight=1.0,
    semantic_weight=1.0,
)


@dataclass(frozen=True, slots=True)
class PdfHybridHit:
    """Return one fused PDF record and both contributing ranks."""

    record: PdfIndexRecord
    score: float
    lexical_rank: int | None
    semantic_rank: int | None


class PdfHybridBaseline:
    """Combine lexical and semantic PDF rankings without score mixing."""

    def __init__(
        self,
        records: Sequence[PdfIndexRecord],
        encoder: SemanticTextEncoder | None = None,
        parameters: RRFParameters = PDF_HYBRID_PARAMETERS,
        *,
        retrieval_texts: Sequence[str] | None = None,
    ) -> None:
        """Build aligned lexical and semantic indexes over PDF records."""
        self._records = tuple(records)
        self._encoder = encoder or MiniLMEncoder()
        lexical = PdfBm25Baseline(
            self._records, retrieval_texts=retrieval_texts,
        )
        semantic_index = build_semantic_index(
            lexical.lexical_index,
            self._encoder,
        )
        self._retriever = HybridRetriever(
            BM25Retriever(lexical.lexical_index),
            semantic_index,
            self._encoder,
            parameters,
        )
        self._parameters = parameters
        self._records_by_key = {
            document.key: record
            for document, record in zip(
                semantic_index.documents,
                self._records,
            )
        }

    @property
    def record_count(self) -> int:
        """Return the shared lexical and semantic corpus size."""
        return len(self._records)

    @property
    def parameters(self) -> RRFParameters:
        """Expose the exact fusion parameters used by this baseline."""
        return self._parameters

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[PdfHybridHit, ...]:
        """Fuse complete-corpus candidate rankings for one question."""
        limit = self.record_count if top_k is None else top_k
        hits = self._retriever.search(
            query,
            top_k=limit,
            candidate_k=self.record_count,
        )
        return tuple(
            PdfHybridHit(
                record=self._records_by_key[hit.key],
                score=hit.score,
                lexical_rank=hit.lexical_rank,
                semantic_rank=hit.semantic_rank,
            )
            for hit in hits
        )

    def run_dataset(
        self,
        dataset: PdfGoldenDataset,
    ) -> tuple[PdfRetrievalResult, ...]:
        """Return one complete fused ranking per golden question."""
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
