"""Map page-labelled PDF evidence to auditable index-record candidates."""

from collections.abc import Sequence
from dataclasses import dataclass

from src.evaluation.pdf.models import (
    PdfGoldenDataset,
    PdfGoldenEvidenceRole,
    PdfGoldenModality,
)
from src.ingestion.pdf.index_records import PdfIndexRecord


@dataclass(frozen=True, slots=True)
class PdfEvidenceCandidates:
    """Keep page-level candidates separate from confirmed relevance."""

    case_id: str
    evidence_index: int
    role: PdfGoldenEvidenceRole
    modality: PdfGoldenModality
    physical_pages: tuple[int, ...]
    page_record_ids: tuple[str, ...]
    missing_pages: tuple[int, ...]
    requires_visual_processing: bool


def find_pdf_evidence_candidates(
    dataset: PdfGoldenDataset,
    records: Sequence[PdfIndexRecord],
) -> tuple[PdfEvidenceCandidates, ...]:
    """Find records touching labelled pages without asserting relevance."""
    _require_unique_record_ids(records)
    book_records = tuple(
        record
        for record in records
        if record.book_path == dataset.book_path
    )
    if not book_records:
        raise ValueError("PDF index records do not contain the golden book")

    results = []
    for case in dataset.cases:
        for evidence_index, evidence in enumerate(case.evidence):
            labelled_pages = set(evidence.physical_pages)
            page_record_ids = tuple(
                record.record_id
                for record in book_records
                if labelled_pages.intersection(_record_pages(record))
            )
            covered_pages = {
                page
                for record in book_records
                for page in _record_pages(record)
                if page in labelled_pages
            }
            results.append(
                PdfEvidenceCandidates(
                    case_id=case.id,
                    evidence_index=evidence_index,
                    role=evidence.role,
                    modality=evidence.modality,
                    physical_pages=evidence.physical_pages,
                    page_record_ids=page_record_ids,
                    missing_pages=tuple(
                        page
                        for page in evidence.physical_pages
                        if page not in covered_pages
                    ),
                    requires_visual_processing=(
                        evidence.modality is PdfGoldenModality.FIGURE
                    ),
                )
            )
    return tuple(results)


def _record_pages(record: PdfIndexRecord) -> frozenset[int]:
    return frozenset(span.page_number for span in record.page_spans)


def _require_unique_record_ids(records: Sequence[PdfIndexRecord]) -> None:
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("PDF evidence mapping requires unique record IDs")
