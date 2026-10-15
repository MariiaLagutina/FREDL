"""Run the existing lexical BM25 index over verified PDF records."""

from collections.abc import Sequence
from dataclasses import dataclass

from src.evaluation.pdf.models import PdfGoldenDataset
from src.evaluation.pdf.retrieval_metrics import PdfRetrievalResult
from src.ingestion.documents import Chunk
from src.ingestion.pdf.index_records import (
    pdf_record_order_key,
    PdfIndexRecord,
)
from src.retrieval.bm25 import (
    BM25Document,
    BM25Index,
    BM25Parameters,
    BM25Retriever,
)
from src.retrieval.tokenization import CodeTokenizer, TextTokenizer


@dataclass(frozen=True, slots=True)
class PdfBm25Hit:
    """Return one PDF record with its inspectable BM25 score."""

    record: PdfIndexRecord
    score: float
    content_score: float
    metadata_score: float


class PdfBm25Baseline:
    """Adapt verified PDF records to the existing project BM25 index."""

    def __init__(
        self,
        records: Sequence[PdfIndexRecord],
        parameters: BM25Parameters | None = None,
        *,
        retrieval_texts: Sequence[str] | None = None,
    ) -> None:
        """Build one immutable lexical index and preserve record identity.

        retrieval_texts, aligned with records, replaces only the searchable
        text, such as a policy view without boilerplate; records keep theirs.
        Records are indexed in canonical reading order, so input order
        cannot change rankings or tie-breaking.
        """
        supplied = tuple(records)
        _require_unique_record_ids(supplied)
        texts = _aligned_texts(supplied, retrieval_texts)
        pairs = sorted(
            zip(supplied, texts),
            key=lambda pair: pdf_record_order_key(pair[0]),
        )
        self._records = tuple(record for record, _ in pairs)
        documents = tuple(
            _build_document(record, text) for record, text in pairs
        )
        self._retriever = BM25Retriever(BM25Index(documents, parameters))
        self._records_by_key = {
            document.key: record
            for document, record in zip(documents, self._records)
        }

    @property
    def records(self) -> tuple[PdfIndexRecord, ...]:
        """Return indexed records in canonical reading order."""
        return self._records

    @property
    def lexical_index(self) -> BM25Index:
        """Expose the aligned lexical corpus for semantic reuse."""
        return self._retriever.index

    @property
    def record_count(self) -> int:
        """Return the exact number of indexed PDF records."""
        return len(self._records)

    @property
    def parameters(self) -> BM25Parameters:
        """Expose the untuned scoring parameters used by the baseline."""
        return self._retriever.index.parameters

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[PdfBm25Hit, ...]:
        """Rank matching PDF records, defaulting to the complete corpus."""
        limit = self.record_count if top_k is None else top_k
        hits = self._retriever.search(query, top_k=limit)
        return tuple(
            PdfBm25Hit(
                record=self._records_by_key[hit.document.key],
                score=hit.score,
                content_score=hit.content_score,
                metadata_score=hit.metadata_score,
            )
            for hit in hits
        )

    def run_dataset(
        self,
        dataset: PdfGoldenDataset,
    ) -> tuple[PdfRetrievalResult, ...]:
        """Rank the complete corpus once for every golden question."""
        return tuple(
            _to_retrieval_result(case.id, self.search(case.question))
            for case in dataset.cases
        )


def _aligned_texts(
    records: tuple[PdfIndexRecord, ...],
    retrieval_texts: Sequence[str] | None,
) -> tuple[str, ...]:
    if retrieval_texts is None:
        return tuple(record.text for record in records)
    if len(retrieval_texts) != len(records):
        raise ValueError("PDF retrieval texts must align with records")
    if any(
        len(text) != len(record.text)
        for record, text in zip(records, retrieval_texts)
    ):
        raise ValueError("PDF retrieval texts must keep record offsets")
    return tuple(retrieval_texts)


def _build_document(record: PdfIndexRecord, text: str) -> BM25Document:
    """Keep searchable PDF text and section metadata in separate fields."""
    text_tokenizer = TextTokenizer()
    metadata_tokenizer = CodeTokenizer()
    metadata_terms = tuple(
        term
        for title in record.section_path
        for term in metadata_tokenizer.tokenize(title)
    )
    chunk = Chunk(
        file_path=f"{record.book_path}#flow={record.flow_index}",
        start=record.flow_start,
        end=record.flow_end,
        text=text,
        section_path=record.section_path,
    )
    return BM25Document(
        chunk=chunk,
        content_terms=tuple(text_tokenizer.tokenize(text)),
        metadata_terms=metadata_terms,
    )


def _to_retrieval_result(
    case_id: str,
    hits: tuple[PdfBm25Hit, ...],
) -> PdfRetrievalResult:
    """Use zero lexical matches as the baseline's only abstention rule."""
    return PdfRetrievalResult(
        case_id=case_id,
        ranked_record_ids=tuple(hit.record.record_id for hit in hits),
        abstained=not hits,
    )


def _require_unique_record_ids(records: Sequence[PdfIndexRecord]) -> None:
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("PDF BM25 baseline requires unique record IDs")
