"""Tests for canonical, versioned PDF record-set fingerprints."""

from collections.abc import Callable
from dataclasses import replace

import pytest

from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.fingerprints import (
    fingerprint_pdf_records,
    legacy_fingerprint_pdf_records_v1,
)
from src.ingestion.pdf.headings import PdfHeadingStatus
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.ingestion.pdf.sections import PdfSectionResolution


# Computed by the version 1 implementation at commit 4c4c225.
LEGACY_REFERENCE = (
    "df78657d7360588cc9e29aee362f2a82eca327a287eb189bbdec6669090d7cc2"
)


def _record(
    record_id: str = "pdf:record-1",
    *,
    text: str = "Evidence.",
    section_path: tuple[str, ...] = ("Part", "Section"),
    flow_index: int = 0,
    flow_start: int = 0,
    page: int = 7,
) -> PdfIndexRecord:
    return PdfIndexRecord(
        record_id=record_id,
        text=text,
        book_path="books/example.pdf",
        section_path=section_path,
        flow_index=flow_index,
        flow_start=flow_start,
        flow_end=flow_start + len(text),
        page_spans=(PdfChunkPageSpan(page, 0, len(text), 0, len(text)),),
        section_interval_id="pdf-section:a",
    )


def _pair() -> tuple[PdfIndexRecord, PdfIndexRecord]:
    return (
        _record(),
        _record("pdf:record-2", text="More.", section_path=("Part",),
                flow_index=1, page=8),
    )


def test_legacy_fingerprint_matches_the_version_1_implementation() -> None:
    first, second = _pair()
    legacy_pair = (
        replace(first, section_interval_id=None),
        replace(second, section_interval_id=None),
    )

    assert legacy_fingerprint_pdf_records_v1(legacy_pair) == LEGACY_REFERENCE


def test_fingerprint_ignores_input_order() -> None:
    first, second = _pair()

    assert fingerprint_pdf_records((first, second)) == (
        fingerprint_pdf_records((second, first))
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda record: replace(record, text="Changed."),
        lambda record: replace(record, section_path=("Part", "Other")),
        lambda record: replace(record, section_interval_id="pdf-section:b"),
        lambda record: replace(record, book_path="books/other.pdf"),
        lambda record: replace(record, flow_start=1),
        lambda record: replace(
            record, page_spans=(PdfChunkPageSpan(9, 0, 9, 0, 9),),
        ),
    ],
    ids=["text", "section_path", "interval", "book", "flow", "pages"],
)
def test_behavior_fields_change_the_fingerprint(
    change: Callable[[PdfIndexRecord], PdfIndexRecord],
) -> None:
    first, second = _pair()

    assert fingerprint_pdf_records((first, second)) != (
        fingerprint_pdf_records((change(first), second))
    )


def test_diagnostic_fields_do_not_change_the_fingerprint() -> None:
    first, second = _pair()
    annotated = replace(
        first,
        section_resolution=PdfSectionResolution.UNRESOLVED,
        section_hint=("Part", "Hint"),
        unresolved_reason=PdfHeadingStatus.NOT_FOUND,
    )

    assert fingerprint_pdf_records((first, second)) == (
        fingerprint_pdf_records((annotated, second))
    )


def test_canonical_encoding_separates_variable_length_paths() -> None:
    # Version 1 concatenates fields with one separator, so a shorter path
    # whose later fields shift into the gap hashes identically.
    long_path = replace(_record(), section_path=("Part", "0"),
                        flow_index=0, flow_start=0, text="Evidence.")
    shifted = PdfIndexRecord(
        record_id="pdf:record-1",
        text="9\0Evidence.",
        book_path="books/example.pdf",
        section_path=("Part",),
        flow_index=0,
        flow_start=0,
        flow_end=0,
        page_spans=long_path.page_spans,
    )
    long_path = replace(long_path, flow_end=9, section_interval_id=None)

    assert legacy_fingerprint_pdf_records_v1((long_path,)) == (
        legacy_fingerprint_pdf_records_v1((shifted,))
    )
    assert fingerprint_pdf_records((long_path,)) != (
        fingerprint_pdf_records((shifted,))
    )


def test_fingerprint_rejects_duplicate_record_ids() -> None:
    with pytest.raises(ValueError, match="unique record IDs"):
        fingerprint_pdf_records((_record(), _record()))
