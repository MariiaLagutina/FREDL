"""Tests for the multi-book library corpus and golden-set contracts."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.evaluation.pdf.library import (
    build_pdf_library_corpus,
    PdfLibraryBookIdentity,
    PdfLibraryBookInput,
    PdfLibraryManifest,
    require_library_matches_manifest,
)
from src.evaluation.pdf.loader import load_pdf_golden_dataset
from src.evaluation.pdf.multibook_golden import (
    load_pdf_multibook_golden,
    PdfExpectedOutcome,
    PdfMultibookGolden,
)
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy


BOOK_A = "books/alpha.pdf"
BOOK_B = "books/beta.pdf"
BOOK_C = "books/delta.pdf"
NOTE = (
    "A NOTE FROM THE PUBLISHER ABOUT THIS DRAFT\n"
    "Draft chapters arrive before editing so readers can follow along.\n"
    "Send comments about missing material to the address on the site.\n"
)


def _record(
    book: str,
    record_id: str,
    text: str,
    *,
    chapter: str = "Chapter 1",
    start: int = 0,
    page: int = 1,
) -> PdfIndexRecord:
    return PdfIndexRecord(
        record_id=record_id,
        text=text,
        book_path=book,
        section_path=(chapter,),
        flow_index=0,
        flow_start=start,
        flow_end=start + len(text),
        page_spans=(PdfChunkPageSpan(page, 0, len(text), 0, len(text)),),
        section_interval_id=f"{book}:{chapter}",
    )


def _book(
    book: str,
    records: tuple[PdfIndexRecord, ...],
) -> PdfLibraryBookInput:
    return PdfLibraryBookInput(
        book_path=book, pdf_sha256=book[6] * 64, records=records, outline=(),
    )


def _library_inputs() -> list[PdfLibraryBookInput]:
    # Alpha repeats the publisher note in three chapters; beta has the same
    # note once, so only alpha's copies are boilerplate.
    alpha = tuple(
        _record(BOOK_A, f"a{index}",
                f"Chapter {index}\n{NOTE}Chapter {index} covers queues.\n",
                chapter=f"Chapter {index}", start=index * 1000, page=index)
        for index in range(1, 4)
    )
    beta = (
        _record(BOOK_B, "b1", f"Intro\n{NOTE}Caches keep answers.\n"),
        _record(BOOK_B, "b2", "Retries repeat requests.\n",
                chapter="Chapter 2", start=1000, page=2),
    )
    return [_book(BOOK_A, alpha), _book(BOOK_B, beta)]


def test_library_applies_policy_per_book_and_keeps_provenance() -> None:
    corpus = build_pdf_library_corpus(
        _library_inputs(),
        PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE,
    )

    texts = dict(zip(
        (record.record_id for record in corpus.records),
        corpus.retrieval_texts,
    ))
    assert "PUBLISHER" not in texts["a1"]
    assert len(texts["a1"]) == len(corpus.book(BOOK_A).records[0].text)
    assert "PUBLISHER" in texts["b1"]
    assert [record.book_path for record in corpus.records] == [
        BOOK_A, BOOK_A, BOOK_A, BOOK_B, BOOK_B,
    ]
    assert len(corpus.source_records) == 5
    assert corpus.manifest.book_paths == (BOOK_A, BOOK_B)


def test_library_fingerprint_ignores_input_order_but_not_policy() -> None:
    inputs = _library_inputs()
    reordered = [
        PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, tuple(reversed(book.records)),
            book.outline,
        )
        for book in reversed(inputs)
    ]
    policy = PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE

    first = build_pdf_library_corpus(inputs, policy)
    second = build_pdf_library_corpus(reordered, policy)
    other = build_pdf_library_corpus(inputs, PdfRetrievalPolicy.ALL_RECORDS)

    assert first.fingerprint == second.fingerprint
    assert first.manifest.fingerprint == second.manifest.fingerprint
    assert first.fingerprint != other.fingerprint
    assert first.manifest.fingerprint == other.manifest.fingerprint


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (lambda books: [books[0], books[0]], "book paths must be unique"),
        (lambda books: [books[0], _book(BOOK_B, (
            _record(BOOK_B, "a1", "Duplicate identity.\n"),))],
         "unique across books"),
        (lambda books: [_book(BOOK_B, books[0].records)], "must belong"),
        (lambda books: [], "at least one book"),
    ],
)
def test_library_rejects_ambiguous_identities(
    inputs: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_pdf_library_corpus(
            inputs(_library_inputs()), PdfRetrievalPolicy.ALL_RECORDS,
        )


def test_library_manifest_detects_changed_files_and_records() -> None:
    corpus = build_pdf_library_corpus(
        _library_inputs(), PdfRetrievalPolicy.ALL_RECORDS,
    )
    manifest = corpus.manifest
    require_library_matches_manifest(corpus, manifest)

    changed_file = manifest.model_copy(update={"books": (
        manifest.books[0].model_copy(update={"pdf_sha256": "0" * 64}),
        manifest.books[1],
    )})
    with pytest.raises(ValueError, match="PDF file differs"):
        require_library_matches_manifest(corpus, changed_file)
    with pytest.raises(ValidationError, match="sorted"):
        PdfLibraryManifest(books=tuple(reversed(manifest.books)))


# Golden-set contract -----------------------------------------------------

def _locator(book: str, page: int, anchor: str, start: int = 0) -> object:
    return {
        "book_path": book, "physical_page": page, "exact_anchor": anchor,
        "page_text_start": start, "page_text_end": start + len(anchor),
        "page_text_sha256": "f" * 64,
    }


def _unit(
    unit_id: str,
    book: str,
    page: int,
    modality: str = "text",
) -> object:
    return {
        "unit_id": unit_id, "required_content": f"Content of {unit_id}.",
        "modality": modality, "locators": [_locator(book, page, unit_id)],
    }


def _case(case_id: str, category: str, outcome: str, **fields: Any) -> object:
    support = {
        "complete": ("complete", "complete"),
        "partial": ("partial", "partial"),
        "ambiguous": ("conditional", "conditional"),
        "unsupported": ("not_found", "not_found"),
        "extraction_limited": ("complete", "extraction_limited"),
        "source_incomplete": ("incomplete", "incomplete"),
    }[outcome]
    return {
        "case_id": case_id, "question": f"Synthetic question {case_id}?",
        "category": category, "expected_outcome": outcome,
        "source_support": support[0], "text_context_support": support[1],
        **fields,
    }


def _golden() -> dict[str, Any]:
    return {
        "schema": "fredl.pdf.multibook-golden",
        "version": 1,
        "dataset_id": "synthetic-multibook-draft",
        "status": "draft",
        "language": "en",
        "library": {"books": [
            {"book_path": book, "pdf_sha256": book[6] * 64,
             "record_set_fingerprint": "e" * 64}
            for book in (BOOK_A, BOOK_B, BOOK_C)
        ]},
        "evidence_units": [
            _unit("a-queue", BOOK_A, 3),
            _unit("a-worker", BOOK_A, 9),
            _unit("b-cache", BOOK_B, 4),
            _unit("c-cache", BOOK_C, 7),
            _unit("c-figure", BOOK_C, 8, "figure"),
            _unit("c-clipped", BOOK_C, 11, "structured_text"),
        ],
        "cases": [
            _case("single", "single_book", "complete",
                  sufficient_sets=[["a-queue"]]),
            _case("sections", "multiple_sections_single_book", "complete",
                  sufficient_sets=[["a-queue", "a-worker"]]),
            _case("either", "independent_multi_book", "complete",
                  sufficient_sets=[["b-cache"], ["c-cache"]]),
            _case("both", "complementary_multi_book", "complete",
                  sufficient_sets=[["a-queue", "b-cache"]]),
            _case("which", "ambiguous_terminology", "ambiguous",
                  conditional_units=["b-cache", "c-cache"]),
            _case("absent", "hard_negative", "unsupported",
                  related_units=["a-worker"]),
            _case("figure", "reliability_limited", "extraction_limited",
                  limitation_reason="visual_evidence_unavailable",
                  sufficient_sets=[["c-figure"]]),
            _case("clipped", "reliability_limited", "source_incomplete",
                  limitation_reason="source_incomplete",
                  related_units=["c-clipped"]),
        ],
    }


def _validate(data: dict[str, Any]) -> PdfMultibookGolden:
    return PdfMultibookGolden.model_validate_json(json.dumps(data))


def _case_named(data: dict[str, Any], case_id: str) -> dict[str, Any]:
    return next(case for case in data["cases"] if case["case_id"] == case_id)


def test_valid_golden_covers_every_outcome_and_category() -> None:
    golden = _validate(_golden())

    outcomes = {case.expected_outcome for case in golden.cases}
    assert outcomes == set(PdfExpectedOutcome) - {PdfExpectedOutcome.PARTIAL}
    assert golden.units["c-figure"].book_path == BOOK_C


def test_content_fingerprint_is_canonical_and_ignores_only_status() -> None:
    draft = _validate(_golden())
    frozen_data = _golden()
    frozen_data["status"] = "frozen"
    frozen_data["freeze_record"] = {
        "approved_by": "Reviewer", "approval_note": "Synthetic approval.",
    }
    reordered = json.loads(json.dumps(_golden()), object_pairs_hook=lambda
                           pairs: dict(reversed(pairs)))

    assert _validate(frozen_data).content_fingerprint == (
        draft.content_fingerprint
    )
    assert _validate(reordered).content_fingerprint == (
        draft.content_fingerprint
    )
    changed = _golden()
    _case_named(changed, "single")["question"] = "A different question?"
    assert _validate(changed).content_fingerprint != draft.content_fingerprint


@pytest.mark.parametrize(
    ("case_id", "update", "message"),
    [
        ("single", {"sufficient_sets": [["missing"]]}, "unknown evidence"),
        ("single", {"sufficient_sets": [["a-queue"], ["a-queue",
                                                      "a-worker"]]},
         "distinct and minimal"),
        ("single", {"sufficient_sets": [["a-queue", "a-queue"]]},
         "without repeats"),
        ("single", {"sufficient_sets": [["a-queue", "b-cache"]]},
         "from one book"),
        ("both", {"sufficient_sets": [["a-queue", "a-worker"]]},
         "span two books"),
        ("either", {"sufficient_sets": [["b-cache", "c-cache"]]},
         "one-book sets"),
        ("sections", {"sufficient_sets": [["a-queue"]]}, "two or more"),
        ("absent", {"sufficient_sets": [["a-worker"]], "related_units": []},
         "no sufficient evidence set"),
        ("which", {"conditional_units": ["b-cache"]},
         "two conditional units|Unreferenced"),
        ("single", {"related_units": ["a-queue"]}, "more than one role"),
        ("single", {"source_support": "partial"}, "support fields"),
        ("single", {"expected_outcome": "unsupported", "sufficient_sets": [],
                    "source_support": "not_found",
                    "text_context_support": "not_found",
                    "related_units": ["a-queue"]},
         "cannot expect"),
    ],
)
def test_invalid_evidence_and_outcome_combinations_are_rejected(
    case_id: str,
    update: dict[str, Any],
    message: str,
) -> None:
    data = _golden()
    _case_named(data, case_id).update(deepcopy(update))

    with pytest.raises(ValidationError, match=message):
        _validate(data)


def test_frozen_status_requires_a_freeze_record() -> None:
    frozen_without_record = _golden()
    frozen_without_record["status"] = "frozen"
    draft_with_record = _golden()
    draft_with_record["freeze_record"] = {
        "approved_by": "Reviewer", "approval_note": "Synthetic approval.",
    }

    for data in (frozen_without_record, draft_with_record):
        with pytest.raises(ValidationError, match="freeze record"):
            _validate(data)


def test_limitation_reasons_keep_three_failure_kinds_apart() -> None:
    clipped_as_extraction = _golden()
    _case_named(clipped_as_extraction, "clipped").update({
        "limitation_reason": "mathematical_structure_lost",
    })
    figure_as_source = _golden()
    _case_named(figure_as_source, "figure").update({
        "limitation_reason": "source_incomplete",
    })
    negative_with_reason = _golden()
    _case_named(negative_with_reason, "absent").update({
        "limitation_reason": "source_incomplete",
    })
    clipped_as_unsupported = _golden()
    _case_named(clipped_as_unsupported, "clipped").update({
        "category": "hard_negative", "expected_outcome": "unsupported",
        "source_support": "not_found", "text_context_support": "not_found",
    })

    for data, message in (
        (clipped_as_extraction, "needs reason source_incomplete"),
        (figure_as_source, "needs an extraction reason"),
        (negative_with_reason, "takes no limitation"),
        (clipped_as_unsupported, "takes no limitation"),
    ):
        with pytest.raises(ValidationError, match=message):
            _validate(data)


def test_units_and_locators_must_be_verifiable() -> None:
    mixed = _golden()
    mixed["evidence_units"][0]["supporting_locators"] = [
        _locator(BOOK_B, 2, "elsewhere"),
    ]
    outside = _golden()
    outside["library"]["books"] = outside["library"]["books"][:2]
    bad_span = _golden()
    bad_span["evidence_units"][0]["locators"][0]["page_text_end"] += 1

    for data, message in (
        (mixed, "exactly one book"),
        (outside, "outside the library"),
        (bad_span, "match its anchor"),
    ):
        with pytest.raises(ValidationError, match=message):
            _validate(data)


def test_single_book_and_multibook_formats_stay_separate(
    tmp_path: Path,
) -> None:
    multibook = tmp_path / "multibook.json"
    multibook.write_text(json.dumps(_golden()), encoding="utf-8")
    single = tmp_path / "single.json"
    single.write_text(json.dumps({
        "schema_version": 1, "dataset_id": "single", "book_path": BOOK_A,
        "page_numbering": "physical_pdf_page", "language": "en",
        "cases": [{
            "id": "q1", "question": "What do queues hold?",
            "why_now": "Testing.", "expected_answer": "Work.",
            "answerability": "complete", "tags": ["queues"],
            "evidence": [{"physical_pages": [3], "role": "required",
                          "modality": "text"}],
        }],
    }), encoding="utf-8")

    assert load_pdf_multibook_golden(multibook).dataset_id == (
        "synthetic-multibook-draft"
    )
    assert load_pdf_golden_dataset(single).book_path == BOOK_A
    with pytest.raises(ValueError, match="invalid"):
        load_pdf_multibook_golden(single)
    with pytest.raises(ValueError):
        load_pdf_golden_dataset(multibook)


def test_library_identity_matches_golden_library_pin() -> None:
    corpus = build_pdf_library_corpus(
        _library_inputs(), PdfRetrievalPolicy.ALL_RECORDS,
    )
    pins = corpus.manifest.books

    assert all(isinstance(pin, PdfLibraryBookIdentity) for pin in pins)
    assert [pin.record_set_fingerprint for pin in pins] == [
        book.corpus.record_set_fingerprint for book in corpus.books
    ]
