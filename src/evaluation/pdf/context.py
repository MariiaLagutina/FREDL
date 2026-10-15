"""Expand ranked PDF hits with their immediate reading context."""

from collections import defaultdict
from collections.abc import Sequence

from src.ingestion.pdf.index_records import PdfIndexRecord


def find_pdf_neighbor_context(
    ranked_record_ids: Sequence[str],
    records: Sequence[PdfIndexRecord],
    *,
    top_k: int = 5,
) -> tuple[str, ...]:
    """Return adjacent records from the same section interval as each hit.

    Records match only by their exact section interval identity. Equal
    section paths, resolutions, or hints are not sufficient, because
    distinct intervals can share all of them. A record without an interval
    identity receives no context.
    """
    if top_k <= 0:
        raise ValueError("PDF context top_k must be positive")

    records_by_id = {record.record_id: record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("PDF context records must have unique IDs")
    if not set(ranked_record_ids).issubset(records_by_id):
        raise ValueError("PDF context ranking references an unknown record")

    records_by_flow: dict[int, list[PdfIndexRecord]] = defaultdict(list)
    for record in records:
        records_by_flow[record.flow_index].append(record)
    for flow_records in records_by_flow.values():
        flow_records.sort(key=lambda item: item.flow_start)

    positions = {
        record.record_id: (flow_records, index)
        for flow_records in records_by_flow.values()
        for index, record in enumerate(flow_records)
    }
    ranked = set(ranked_record_ids[:top_k])
    context: list[str] = []
    seen: set[str] = set()
    for record_id in ranked_record_ids[:top_k]:
        flow_records, index = positions[record_id]
        record = flow_records[index]
        for neighbor_index in (index - 1, index + 1):
            if not 0 <= neighbor_index < len(flow_records):
                continue
            neighbor = flow_records[neighbor_index]
            if not _same_interval(record, neighbor):
                continue
            if not _records_touch(record, neighbor):
                continue
            if neighbor.record_id in ranked or neighbor.record_id in seen:
                continue
            context.append(neighbor.record_id)
            seen.add(neighbor.record_id)
    return tuple(context)


def _same_interval(left: PdfIndexRecord, right: PdfIndexRecord) -> bool:
    """Return whether both records carry the same known interval."""
    return (
        left.section_interval_id is not None
        and left.section_interval_id == right.section_interval_id
    )


def _records_touch(
    left: PdfIndexRecord,
    right: PdfIndexRecord,
) -> bool:
    """Return whether two records form one uninterrupted text interval."""
    return (
        left.flow_end == right.flow_start
        or right.flow_end == left.flow_start
    )
