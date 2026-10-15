"""Migrate human PDF record judgments to a new record set conservatively.

A label moves automatically only when its record's text is unchanged
apart from whitespace and its pages, section path, and evidence pages still
agree. Every other label waits for an explicit human decision.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import re

from src.evaluation.pdf.judgments import (
    PdfJudgmentMapping,
    PdfJudgmentProvenance,
    PdfJudgmentSet,
    PdfJudgmentSource,
    PdfRecordJudgment,
    PdfRecordRelevance,
    validate_pdf_judgment_set,
)
from src.evaluation.pdf.models import PdfGoldenCase, PdfGoldenDataset
from src.ingestion.pdf.fingerprints import fingerprint_pdf_records
from src.ingestion.pdf.index_records import (
    canonical_pdf_record_order,
    PdfIndexRecord,
)


# A moved boundary still counts as one record when it keeps this share.
CHANGED_BOUNDARY_SHARE = 0.9
_WHITESPACE = re.compile(r"\s+")


class PdfRecordMappingKind(str, Enum):
    """Classify how one historical record relates to the new records."""

    BYTE_IDENTICAL = "byte_identical"
    WHITESPACE_ONLY = "whitespace_only"
    CHANGED_PROVENANCE = "changed_provenance"
    CHANGED_BOUNDARY = "changed_boundary"
    SPLIT = "split"
    MERGED = "merged"
    AMBIGUOUS = "ambiguous"
    UNMAPPABLE = "unmappable"


AUTOMATIC_MAPPINGS: Mapping[PdfRecordMappingKind, PdfJudgmentMapping] = {
    PdfRecordMappingKind.BYTE_IDENTICAL: PdfJudgmentMapping.BYTE_IDENTICAL,
    PdfRecordMappingKind.WHITESPACE_ONLY: PdfJudgmentMapping.WHITESPACE_ONLY,
}


@dataclass(frozen=True, slots=True)
class PdfMappingCandidate:
    """Describe how much of a historical record one new record holds."""

    record_id: str
    share_of_source: float
    share_of_target: float


@dataclass(frozen=True, slots=True)
class PdfRecordMapping:
    """Relate one historical record to its new candidates."""

    source_record_id: str
    kind: PdfRecordMappingKind
    target_record_id: str | None
    candidates: tuple[PdfMappingCandidate, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class PdfPendingJudgment:
    """Hold one historical label that needs a human decision."""

    judgment: PdfRecordJudgment
    mapping: PdfRecordMapping
    reason: str


@dataclass(frozen=True, slots=True)
class PdfReviewDecision:
    """Record a human decision for one pending historical label.

    A decision without a target keeps the label historical only.
    """

    case_id: str
    evidence_index: int
    source_record_id: str
    target_record_id: str | None
    relevance: PdfRecordRelevance | None


@dataclass(frozen=True, slots=True)
class PdfJudgmentMigration:
    """Account for every historical label exactly once."""

    source: PdfJudgmentSource
    target_dataset_id: str
    target_book_path: str
    target_record_set_fingerprint: str
    mappings: tuple[PdfRecordMapping, ...]
    migrated: tuple[PdfRecordJudgment, ...]
    pending: tuple[PdfPendingJudgment, ...]


def map_pdf_records(
    source_records: Sequence[PdfIndexRecord],
    target_records: Sequence[PdfIndexRecord],
    source_record_ids: Sequence[str],
) -> tuple[PdfRecordMapping, ...]:
    """Locate historical record text among new records, ignoring spaces.

    Whitespace is the only difference ignored, because separator repairs
    change it without changing the evidence a human judged.
    """
    sources = {record.record_id: record for record in source_records}
    ordered = canonical_pdf_record_order(target_records)
    owners: list[str] = []
    parts: list[str] = []
    for record in ordered:
        compact = _compact(record.text)
        parts.append(compact)
        owners.extend([record.record_id] * len(compact))
    stream = "".join(parts)
    targets = {record.record_id: record for record in ordered}
    sizes = {
        record_id: len(_compact(record.text))
        for record_id, record in targets.items()
    }

    mappings = [
        _map_one(sources[record_id], stream, owners, targets, sizes)
        for record_id in dict.fromkeys(source_record_ids)
    ]
    return _mark_shared_targets(mappings)


def migrate_pdf_judgments(
    *,
    source_set: PdfJudgmentSet,
    source_sha256: str,
    source_dataset: PdfGoldenDataset,
    source_records: Sequence[PdfIndexRecord],
    target_dataset: PdfGoldenDataset,
    target_records: Sequence[PdfIndexRecord],
) -> PdfJudgmentMigration:
    """Move unchanged labels and queue every other one for review."""
    validate_pdf_judgment_set(source_set, source_dataset, source_records)
    if target_dataset.book_path != source_set.book_path:
        raise ValueError("PDF target dataset belongs to another book")

    mappings = map_pdf_records(
        source_records,
        target_records,
        [judgment.record_id for judgment in source_set.judgments],
    )
    by_source = {mapping.source_record_id: mapping for mapping in mappings}
    targets = {record.record_id: record for record in target_records}
    source_cases = {case.id: case for case in source_dataset.cases}
    target_cases = {case.id: case for case in target_dataset.cases}

    migrated: list[PdfRecordJudgment] = []
    pending: list[PdfPendingJudgment] = []
    for judgment in source_set.judgments:
        mapping = by_source[judgment.record_id]
        reason = _blocking_reason(
            judgment, mapping, targets,
            source_cases[judgment.case_id],
            target_cases.get(judgment.case_id),
        )
        if reason is not None:
            pending.append(PdfPendingJudgment(judgment, mapping, reason))
            continue
        assert mapping.target_record_id is not None
        migrated.append(PdfRecordJudgment(
            case_id=judgment.case_id,
            evidence_index=judgment.evidence_index,
            record_id=mapping.target_record_id,
            relevance=judgment.relevance,
            provenance=PdfJudgmentProvenance(
                source_record_id=judgment.record_id,
                mapping=AUTOMATIC_MAPPINGS[mapping.kind],
            ),
        ))

    return PdfJudgmentMigration(
        source=PdfJudgmentSource(
            dataset_id=source_set.dataset_id,
            judgments_sha256=source_sha256,
            record_set_fingerprint=source_set.record_set_fingerprint,
            record_set_fingerprint_version=source_set.schema_version,
        ),
        target_dataset_id=target_dataset.dataset_id,
        target_book_path=target_dataset.book_path,
        target_record_set_fingerprint=fingerprint_pdf_records(target_records),
        mappings=mappings,
        migrated=tuple(migrated),
        pending=tuple(pending),
    )


def finalize_pdf_judgment_migration(
    migration: PdfJudgmentMigration,
    decisions: Sequence[PdfReviewDecision],
    target_dataset: PdfGoldenDataset,
    target_records: Sequence[PdfIndexRecord],
) -> PdfJudgmentSet:
    """Build the new judgment set only when every pending label is decided."""
    by_key = {
        (item.case_id, item.evidence_index, item.source_record_id): item
        for item in decisions
    }
    if len(by_key) != len(decisions):
        raise ValueError("PDF review decisions must be unique")
    pending_keys = {
        (
            item.judgment.case_id,
            item.judgment.evidence_index,
            item.judgment.record_id,
        )
        for item in migration.pending
    }
    if set(by_key) != pending_keys:
        raise ValueError("PDF review decisions must cover every pending label")

    judgments = list(migration.migrated)
    for key in sorted(pending_keys):
        decision = by_key[key]
        if decision.target_record_id is None:
            continue
        if decision.relevance is None:
            raise ValueError("PDF review decision must state its relevance")
        judgments.append(PdfRecordJudgment(
            case_id=decision.case_id,
            evidence_index=decision.evidence_index,
            record_id=decision.target_record_id,
            relevance=decision.relevance,
            provenance=PdfJudgmentProvenance(
                source_record_id=decision.source_record_id,
                mapping=PdfJudgmentMapping.MANUAL_REVIEW,
            ),
        ))

    judgment_set = PdfJudgmentSet(
        schema_version=2,
        dataset_id=migration.target_dataset_id,
        book_path=migration.target_book_path,
        record_set_fingerprint=migration.target_record_set_fingerprint,
        judgments=tuple(sorted(
            judgments,
            key=lambda item: (item.case_id, item.evidence_index,
                              item.record_id),
        )),
        migrated_from=migration.source,
    )
    validate_pdf_judgment_set(judgment_set, target_dataset, target_records)
    return judgment_set


def _map_one(
    source: PdfIndexRecord,
    stream: str,
    owners: list[str],
    targets: Mapping[str, PdfIndexRecord],
    sizes: Mapping[str, int],
) -> PdfRecordMapping:
    compact = _compact(source.text)
    starts = _occurrences(stream, compact)
    if not starts:
        return PdfRecordMapping(
            source.record_id, PdfRecordMappingKind.UNMAPPABLE, None, (),
            "historical text does not occur in the new records",
        )
    if len(starts) > 1:
        return PdfRecordMapping(
            source.record_id, PdfRecordMappingKind.AMBIGUOUS, None, (),
            f"historical text occurs {len(starts)} times",
        )

    counts = Counter(owners[starts[0]:starts[0] + len(compact)])
    candidates = tuple(
        PdfMappingCandidate(
            record_id,
            round(count / len(compact), 4),
            round(count / sizes[record_id], 4),
        )
        for record_id, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0]),
        )
    )
    main = candidates[0]
    target = targets[main.record_id]
    if len(candidates) == 1 and main.share_of_target == 1.0:
        if _pages(target) != _pages(source):
            return PdfRecordMapping(
                source.record_id, PdfRecordMappingKind.CHANGED_PROVENANCE,
                main.record_id, candidates, "physical pages differ",
            )
        if target.section_path != source.section_path:
            return PdfRecordMapping(
                source.record_id, PdfRecordMappingKind.CHANGED_PROVENANCE,
                main.record_id, candidates, "section path differs",
            )
        kind = (
            PdfRecordMappingKind.BYTE_IDENTICAL
            if target.text == source.text
            else PdfRecordMappingKind.WHITESPACE_ONLY
        )
        return PdfRecordMapping(
            source.record_id, kind, main.record_id, candidates,
            "same text, pages, and section path",
        )
    if main.share_of_source >= CHANGED_BOUNDARY_SHARE and (
        main.share_of_target >= CHANGED_BOUNDARY_SHARE
    ):
        return PdfRecordMapping(
            source.record_id, PdfRecordMappingKind.CHANGED_BOUNDARY,
            main.record_id, candidates,
            "record boundary moved; text was added or removed",
        )
    if main.share_of_source >= CHANGED_BOUNDARY_SHARE:
        return PdfRecordMapping(
            source.record_id, PdfRecordMappingKind.MERGED, main.record_id,
            candidates, "historical text is part of a larger record",
        )
    return PdfRecordMapping(
        source.record_id, PdfRecordMappingKind.SPLIT, None, candidates,
        "historical text is split across new records",
    )


def _mark_shared_targets(
    mappings: list[PdfRecordMapping],
) -> tuple[PdfRecordMapping, ...]:
    """Treat a new record claimed by several historical records as merged."""
    claims = Counter(
        mapping.target_record_id for mapping in mappings
        if mapping.target_record_id is not None
    )
    return tuple(
        PdfRecordMapping(
            mapping.source_record_id,
            PdfRecordMappingKind.MERGED,
            mapping.target_record_id,
            mapping.candidates,
            "several historical records map to this record",
        )
        if mapping.target_record_id is not None
        and claims[mapping.target_record_id] > 1
        else mapping
        for mapping in mappings
    )


def _blocking_reason(
    judgment: PdfRecordJudgment,
    mapping: PdfRecordMapping,
    targets: Mapping[str, PdfIndexRecord],
    source_case: PdfGoldenCase,
    target_case: PdfGoldenCase | None,
) -> str | None:
    """Return why a label cannot move automatically, or None."""
    if mapping.kind not in AUTOMATIC_MAPPINGS:
        return f"{mapping.kind.value}: {mapping.reason}"
    if target_case is None:
        return "question is absent from the target golden set"
    if target_case != source_case:
        return "question or evidence changed in the target golden set"
    assert mapping.target_record_id is not None
    evidence_pages = set(
        target_case.evidence[judgment.evidence_index].physical_pages
    )
    if not evidence_pages & _pages(targets[mapping.target_record_id]):
        return "new record does not overlap the evidence pages"
    return None


def _pages(record: PdfIndexRecord) -> set[int]:
    return {span.page_number for span in record.page_spans}


def _occurrences(stream: str, needle: str) -> list[int]:
    if not needle:
        return []
    starts = []
    position = stream.find(needle)
    while position != -1:
        starts.append(position)
        position = stream.find(needle, position + 1)
    return starts


def _compact(text: str) -> str:
    return _WHITESPACE.sub("", text)
