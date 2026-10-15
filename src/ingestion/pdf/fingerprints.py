"""Fingerprint PDF index records with a canonical, versioned encoding."""

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json

from src.ingestion.pdf.index_records import (
    canonical_pdf_record_order,
    PdfIndexRecord,
)


PDF_RECORD_SET_SCHEMA = "fredl.pdf.record-set"
PDF_RECORD_SET_VERSION = 2


def fingerprint_canonical_json(payload: Mapping[str, object]) -> str:
    """Hash JSON with sorted keys and no insignificant whitespace."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def fingerprint_pdf_records(records: Sequence[PdfIndexRecord]) -> str:
    """Hash every record field that can change retrieval or evaluation.

    Records are hashed in canonical reading order, so input order cannot
    change the fingerprint; retrieval uses the same order. Section
    resolution, outline hints, and unresolved reasons are display-only and
    excluded, as are reliability findings, which records do not carry.
    """
    _require_unique_record_ids(records)
    return fingerprint_canonical_json({
        "schema": PDF_RECORD_SET_SCHEMA,
        "version": PDF_RECORD_SET_VERSION,
        "records": [
            {
                "record_id": record.record_id,
                "book_path": record.book_path,
                "text": record.text,
                "section_path": list(record.section_path),
                "section_interval_id": record.section_interval_id,
                "flow_index": record.flow_index,
                "flow_start": record.flow_start,
                "flow_end": record.flow_end,
                "page_spans": [
                    [
                        span.page_number,
                        span.chunk_start,
                        span.chunk_end,
                        span.page_source_start,
                        span.page_source_end,
                    ]
                    for span in record.page_spans
                ],
            }
            for record in canonical_pdf_record_order(records)
        ],
    })


def legacy_fingerprint_pdf_records_v1(
    records: Sequence[PdfIndexRecord],
) -> str:
    """Reproduce the version 1 fingerprint used by historical judgments.

    Its separator encoding is ambiguous for variable-length section paths;
    it exists only to verify artefacts created before version 2.
    """
    _require_unique_record_ids(records)
    digest = sha256()
    for record in sorted(records, key=lambda item: item.record_id):
        values = (
            record.record_id,
            record.book_path,
            *record.section_path,
            str(record.flow_index),
            str(record.flow_start),
            str(record.flow_end),
            record.text,
        )
        for value in values:
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        for span in record.page_spans:
            for numeric_value in (
                span.page_number,
                span.chunk_start,
                span.chunk_end,
                span.page_source_start,
                span.page_source_end,
            ):
                digest.update(str(numeric_value).encode("ascii"))
                digest.update(b"\0")
    return digest.hexdigest()


def _require_unique_record_ids(records: Sequence[PdfIndexRecord]) -> None:
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("PDF record fingerprint requires unique record IDs")
