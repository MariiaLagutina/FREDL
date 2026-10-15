"""Tests for conservative migration of human PDF judgments."""

from dataclasses import replace
import json

import pytest

from src.evaluation.pdf.judgments import (
    PdfJudgmentMapping,
    PdfJudgmentSet,
    PdfRecordRelevance,
)
from src.evaluation.pdf.migration import (
    finalize_pdf_judgment_migration,
    map_pdf_records,
    migrate_pdf_judgments,
    PdfRecordMappingKind,
    PdfReviewDecision,
)
from src.evaluation.pdf.models import PdfGoldenDataset
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.fingerprints import (
    fingerprint_pdf_records,
    legacy_fingerprint_pdf_records_v1,
)
from src.ingestion.pdf.index_records import PdfIndexRecord


BOOK = "books/example.pdf"


def _record(
    record_id: str,
    text: str,
    *,
    start: int,
    page: int = 3,
    section: tuple[str, ...] = ("Guide", "Caching"),
) -> PdfIndexRecord:
    return PdfIndexRecord(
        record_id=record_id,
        text=text,
        book_path=BOOK,
        section_path=section,
        flow_index=0,
        flow_start=start,
        flow_end=start + len(text),
        page_spans=(PdfChunkPageSpan(page, 0, len(text), start,
                                     start + len(text)),),
    )


def _kind(
    source: PdfIndexRecord,
    targets: tuple[PdfIndexRecord, ...],
) -> PdfRecordMappingKind:
    return map_pdf_records((source,), targets, (source.record_id,))[0].kind


def test_identical_text_pages_and_section_are_byte_identical() -> None:
    source = _record("old", "Caches expire.\nThen refill.", start=0)
    target = _record("new", "Caches expire.\nThen refill.", start=0)

    assert _kind(source, (target,)) is PdfRecordMappingKind.BYTE_IDENTICAL


def test_separator_repair_is_whitespace_only() -> None:
    source = _record("old", "Caches expire.Then refill.", start=0)
    target = _record("new", "Caches expire.\nThen refill.", start=0)

    assert _kind(source, (target,)) is PdfRecordMappingKind.WHITESPACE_ONLY


def test_glued_words_are_not_whitespace_only_changes() -> None:
    source = _record("old", "Caches expire. Then refill.", start=0)
    target = _record("new", "Caches expired. Then refill.", start=0)

    assert _kind(source, (target,)) is PdfRecordMappingKind.UNMAPPABLE


@pytest.mark.parametrize(
    ("page", "section", "reason"),
    [
        (4, ("Guide", "Caching"), "physical pages differ"),
        (3, ("Guide", "Queues"), "section path differs"),
    ],
)
def test_same_text_with_changed_provenance_needs_review(
    page: int,
    section: tuple[str, ...],
    reason: str,
) -> None:
    source = _record("old", "Caches expire.", start=0)
    target = _record("new", "Caches expire.", start=0, page=page,
                     section=section)

    mapping = map_pdf_records((source,), (target,), ("old",))[0]

    assert mapping.kind is PdfRecordMappingKind.CHANGED_PROVENANCE
    assert mapping.reason == reason


def test_small_boundary_move_is_a_changed_boundary() -> None:
    body = "Caches expire after a configured interval of time. " * 3
    source = _record("old", body, start=0)
    target = _record("new", body + "See below.", start=0)

    assert _kind(source, (target,)) is PdfRecordMappingKind.CHANGED_BOUNDARY


def test_record_inside_a_much_larger_one_is_merged() -> None:
    source = _record("old", "Caches expire.", start=0)
    target = _record("new", "Caches expire. " + "Queues hold work. " * 5,
                     start=0)

    assert _kind(source, (target,)) is PdfRecordMappingKind.MERGED


def test_two_historical_records_in_one_new_record_are_merged() -> None:
    first = _record("old-1", "Caches expire.", start=0)
    second = _record("old-2", "Then refill.", start=14)
    target = _record("new", "Caches expire. Then refill.", start=0)

    mappings = map_pdf_records((first, second), (target,),
                               ("old-1", "old-2"))

    assert {item.kind for item in mappings} == {PdfRecordMappingKind.MERGED}


def test_text_divided_between_new_records_is_split() -> None:
    source = _record("old", "Caches expire. Then refill.", start=0)
    targets = (
        _record("new-1", "Caches expire. ", start=0),
        _record("new-2", "Then refill.", start=15),
    )

    mapping = map_pdf_records((source,), targets, ("old",))[0]

    assert mapping.kind is PdfRecordMappingKind.SPLIT
    assert mapping.target_record_id is None


def test_repeated_text_is_ambiguous() -> None:
    source = _record("old", "Caches expire.", start=0)
    targets = (
        _record("new-1", "Caches expire.", start=0),
        _record("new-2", "Caches expire.", start=14, page=9),
    )

    assert _kind(source, targets) is PdfRecordMappingKind.AMBIGUOUS


def _dataset(dataset_id: str, *, pages: list[int] | None = None,
             extra_case: bool = False) -> PdfGoldenDataset:
    case = {
        "id": "case-1",
        "question": "When do caches expire?",
        "why_now": "I tune a cache.",
        "expected_answer": "After the interval.",
        "answerability": "complete",
        "evidence": [{"physical_pages": pages or [3], "role": "required",
                      "modality": "text"}],
        "tags": ["cache"],
    }
    negative = {
        "id": "case-2", "question": "What is absent?", "why_now": "Check.",
        "expected_answer": "Not stated.", "answerability": "not_found",
        "evidence": [], "tags": ["negative"],
    }
    return PdfGoldenDataset.model_validate_json(json.dumps({
        "schema_version": 1, "dataset_id": dataset_id, "book_path": BOOK,
        "page_numbering": "physical_pdf_page", "language": "en",
        "cases": [case, negative] if extra_case else [case],
    }))


def _source_set(records: tuple[PdfIndexRecord, ...],
                labels: list[tuple[str, str]]) -> PdfJudgmentSet:
    return PdfJudgmentSet.model_validate_json(json.dumps({
        "schema_version": 1, "dataset_id": "golden-v1", "book_path": BOOK,
        "record_set_fingerprint": legacy_fingerprint_pdf_records_v1(records),
        "judgments": [
            {"case_id": "case-1", "evidence_index": 0, "record_id": record,
             "relevance": relevance}
            for record, relevance in labels
        ],
    }))


def _migration_inputs() -> tuple[
    tuple[PdfIndexRecord, ...], tuple[PdfIndexRecord, ...],
]:
    body = "Caches expire after a configured interval of time. " * 3
    source = (
        _record("old-a", "Caches expire.", start=0),
        _record("old-b", "Entries refill.Lazily.", start=14),
        _record("old-c", body, start=36),
    )
    target = (
        _record("new-a", "Caches expire.", start=0),
        _record("new-b", "Entries refill.\nLazily.", start=14),
        _record("new-c", body + "See below.", start=37),
    )
    return source, target


def test_migration_accounts_for_every_label_once() -> None:
    source, target = _migration_inputs()
    labels = [("old-a", "required"), ("old-b", "supporting"),
              ("old-c", "irrelevant")]

    migration = migrate_pdf_judgments(
        source_set=_source_set(source, labels),
        source_sha256="a" * 64,
        source_dataset=_dataset("golden-v1"),
        source_records=source,
        target_dataset=_dataset("golden-v2", extra_case=True),
        target_records=target,
    )

    migrated = [
        (item.record_id, item.relevance.value) for item in migration.migrated
    ]
    mappings = [
        item.provenance.mapping
        for item in migration.migrated if item.provenance
    ]
    assert migrated == [("new-a", "required"), ("new-b", "supporting")]
    assert mappings == [
        PdfJudgmentMapping.BYTE_IDENTICAL,
        PdfJudgmentMapping.WHITESPACE_ONLY,
    ]
    assert [item.judgment.record_id for item in migration.pending] == [
        "old-c",
    ]
    assert migration.pending[0].reason.startswith("changed_boundary")
    assert migration.target_record_set_fingerprint == (
        fingerprint_pdf_records(target)
    )


def test_changed_question_blocks_automatic_migration() -> None:
    source, target = _migration_inputs()
    changed = _dataset("golden-v2")
    question = changed.cases[0].model_copy(
        update={"question": "How long do caches live?"},
    )
    changed = changed.model_copy(update={"cases": (question,)})

    migration = migrate_pdf_judgments(
        source_set=_source_set(source, [("old-a", "required")]),
        source_sha256="a" * 64,
        source_dataset=_dataset("golden-v1"),
        source_records=source,
        target_dataset=changed,
        target_records=target,
    )

    assert migration.migrated == ()
    assert migration.pending[0].reason == (
        "question or evidence changed in the target golden set"
    )


def test_migration_rejects_mismatched_historical_records() -> None:
    source, target = _migration_inputs()
    labels = _source_set(source, [("old-a", "required")])
    altered = (replace(source[0], text="Caches expired."), *source[1:])

    with pytest.raises(ValueError, match="fingerprint differs"):
        migrate_pdf_judgments(
            source_set=labels,
            source_sha256="a" * 64,
            source_dataset=_dataset("golden-v1"),
            source_records=altered,
            target_dataset=_dataset("golden-v2"),
            target_records=target,
        )


def test_finalization_requires_a_decision_for_every_pending_label() -> None:
    source, target = _migration_inputs()
    migration = migrate_pdf_judgments(
        source_set=_source_set(source, [("old-a", "required"),
                                        ("old-c", "irrelevant")]),
        source_sha256="a" * 64,
        source_dataset=_dataset("golden-v1"),
        source_records=source,
        target_dataset=_dataset("golden-v2"),
        target_records=target,
    )

    with pytest.raises(ValueError, match="cover every pending label"):
        finalize_pdf_judgment_migration(
            migration, (), _dataset("golden-v2"), target,
        )

    final = finalize_pdf_judgment_migration(
        migration,
        (PdfReviewDecision("case-1", 0, "old-c", "new-c",
                           PdfRecordRelevance.IRRELEVANT),),
        _dataset("golden-v2"),
        target,
    )

    assert final.schema_version == 2
    assert final.dataset_id == "golden-v2"
    assert final.record_set_fingerprint == fingerprint_pdf_records(target)
    assert final.migrated_from == migration.source
    reviewed = [item for item in final.judgments if item.record_id == "new-c"]
    assert reviewed[0].provenance is not None
    assert reviewed[0].provenance.mapping is PdfJudgmentMapping.MANUAL_REVIEW


def test_decision_without_target_keeps_the_label_historical() -> None:
    source, target = _migration_inputs()
    migration = migrate_pdf_judgments(
        source_set=_source_set(source, [("old-a", "required"),
                                        ("old-c", "irrelevant")]),
        source_sha256="a" * 64,
        source_dataset=_dataset("golden-v1"),
        source_records=source,
        target_dataset=_dataset("golden-v2"),
        target_records=target,
    )

    final = finalize_pdf_judgment_migration(
        migration,
        (PdfReviewDecision("case-1", 0, "old-c", None, None),),
        _dataset("golden-v2"),
        target,
    )

    assert [item.record_id for item in final.judgments] == ["new-a"]
