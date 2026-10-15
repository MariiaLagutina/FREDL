"""Tests for multi-book judgment pools, worksheets, and validation."""

import csv
import io
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.cli import CliError
from src.evaluation.pdf.judgments import PdfRecordRelevance
from src.evaluation.pdf.library import (
    build_pdf_library_corpus,
    load_pdf_library_corpus,
    PdfLibraryBookInput,
    PdfLibraryCorpus,
)
from src.evaluation.pdf.multibook_cli import (
    pdf_collect_judgments,
    pdf_import_multibook_golden,
    pdf_prepare_judgment_pilot,
)
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook_judgments import (
    load_pdf_multibook_judgments,
    PdfMultibookJudgment,
    PdfMultibookJudgmentSet,
    PdfPoolSource,
    validate_multibook_judgments,
)
from src.evaluation.pdf.pooling import (
    anchor_pool_entries,
    build_judgment_pool,
    read_judgment_worksheet,
    render_judgment_worksheet,
    retrieval_pool_entries,
    summarize_pool,
)
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.ingestion.pdf.reading_document import prepare_pdf_reading_document
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy
from tests.evaluation.pdf.test_multibook_import import (
    _decisions,
    _package_files,
    _write_package,
    _write_pdf,
    BOOKS,
    PAGES,
)


ALPHA = "books/alpha.pdf"
BETA = "books/beta.pdf"
# Two records share alpha page 1; one record continues onto page 2.
ALPHA_RECORDS = (
    ("a1", "Queues hold work.", 1, 0),
    ("a2", "Workers poll the queue.", 1, 18),
    ("a3", "Backpressure slows producers.", 2, 0),
)
BETA_RECORDS = (
    ("b1", "Caches keep answers.", 1, 0),
    ("b2", "Eviction removes old entries.", 2, 0),
)


def _records(book: str, rows: Any) -> tuple[PdfIndexRecord, ...]:
    return tuple(
        PdfIndexRecord(
            record_id=record_id, text=text, book_path=book,
            section_path=("Chapter",), flow_index=0,
            flow_start=index * 100, flow_end=index * 100 + len(text),
            page_spans=(PdfChunkPageSpan(page, 0, len(text), start,
                                         start + len(text)),),
            section_interval_id=f"{book}:chapter",
        )
        for index, (record_id, text, page, start) in enumerate(rows)
    )


def _library() -> PdfLibraryCorpus:
    return build_pdf_library_corpus([
        PdfLibraryBookInput(ALPHA, "a" * 64,
                            _records(ALPHA, ALPHA_RECORDS), ()),
        PdfLibraryBookInput(BETA, "b" * 64, _records(BETA, BETA_RECORDS), ()),
    ], PdfRetrievalPolicy.ALL_RECORDS)


def _locator(book: str, page: int, anchor: str, start: int) -> object:
    return {
        "book_path": book, "physical_page": page, "exact_anchor": anchor,
        "page_text_start": start, "page_text_end": start + len(anchor),
        "page_text_sha256": "f" * 64,
    }


def _golden(library: PdfLibraryCorpus) -> PdfMultibookGolden:
    def case(case_id: str, category: str, outcome: str, support: tuple[
        str, str,
    ], **fields: Any) -> object:
        return {"case_id": case_id, "question": f"Question {case_id}?",
                "category": category, "expected_outcome": outcome,
                "source_support": support[0],
                "text_context_support": support[1], **fields}

    return PdfMultibookGolden.model_validate_json(json.dumps({
        "dataset_id": "synthetic", "status": "draft", "language": "en",
        "library": library.manifest.model_dump(mode="json"),
        "evidence_units": [
            {"unit_id": "queue", "required_content": "Queues hold work.",
             "modality": "text",
             "locators": [_locator(ALPHA, 1, "Queues hold", 0)]},
            {"unit_id": "cache", "required_content": "Caches keep answers.",
             "modality": "text",
             "locators": [_locator(BETA, 1, "Caches keep", 0)],
             "supporting_locators": [_locator(BETA, 2, "Eviction", 0)]},
            {"unit_id": "backpressure", "required_content": "Producers slow.",
             "modality": "text",
             "locators": [_locator(ALPHA, 2, "Backpressure", 0)]},
        ],
        "cases": [
            case("both", "complementary_multi_book", "complete",
                 ("complete", "complete"),
                 sufficient_sets=[["queue", "cache"]]),
            case("which", "ambiguous_terminology", "ambiguous",
                 ("conditional", "conditional"),
                 conditional_units=["queue", "backpressure"]),
            case("absent", "hard_negative", "unsupported",
                 ("not_found", "not_found"), related_units=["cache"]),
        ],
    }))


def test_anchor_pool_offers_locator_pages_without_labels() -> None:
    library = _library()
    golden = _golden(library)

    pool = build_judgment_pool(golden, library, anchor_pool_entries(
        golden, library, ("absent", "which", "both"),
    ))

    entries = {entry.key: entry.sources for entry in pool.entries}
    assert entries[("both", "queue", "a1")] == (
        PdfPoolSource.ANCHOR_OVERLAP, PdfPoolSource.LOCATOR_PAGE,
    )
    assert entries[("both", "queue", "a2")] == (PdfPoolSource.LOCATOR_PAGE,)
    assert entries[("both", "cache", "b2")] == (PdfPoolSource.LOCATOR_PAGE,)
    assert ("which", "backpressure", "a3") in entries
    assert not any(key[0] == "absent" for key in entries)
    assert [entry.case_id for entry in pool.entries][:3] == ["both"] * 3
    summary = summarize_pool(pool, library)
    assert (summary.entries, summary.unique_records) == (len(entries), 5)


def test_retrieval_pool_is_blind_deduplicated_and_complete() -> None:
    library = _library()
    golden = _golden(library)
    rankings = {
        PdfPoolSource.BM25: {"both": ["a1", "b2", "a3"]},
        PdfPoolSource.MINILM: {"both": ["b1", "a1"]},
        PdfPoolSource.HYBRID: {"both": ["a1", "b1"]},
    }

    entries = retrieval_pool_entries(golden, rankings, 2, ("both",))
    pool = build_judgment_pool(
        golden, library,
        (*anchor_pool_entries(golden, library, ("both",)), *entries),
        retrieval_depth=2,
    )

    sources = {entry.key: entry.sources for entry in pool.entries}
    assert sources[("both", "queue", "a1")] == (
        PdfPoolSource.ANCHOR_OVERLAP, PdfPoolSource.LOCATOR_PAGE,
        PdfPoolSource.BM25, PdfPoolSource.MINILM, PdfPoolSource.HYBRID,
    )
    assert ("both", "queue", "a3") not in sources
    worksheet, sheet = render_judgment_worksheet(golden, library, pool)
    for hidden in ("bm25", "minilm", "hybrid", "anchor_overlap",
                   "locator_page", "BM25"):
        assert hidden not in worksheet and hidden not in sheet
    with pytest.raises(ValueError, match="BM25, MiniLM, and hybrid"):
        retrieval_pool_entries(
            golden, {PdfPoolSource.BM25: rankings[PdfPoolSource.BM25]}, 2,
            ("both",),
        )


def _filled(sheet: str, labels: dict[str, tuple[str, str]]) -> str:
    rows = list(csv.DictReader(io.StringIO(sheet)))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]),
                            lineterminator="\n")
    writer.writeheader()
    for row in rows:
        relevance, supplies = labels.get(row["record_id"], ("", ""))
        row.update(relevance=relevance, supplies_evidence=supplies)
        writer.writerow(row)
    return output.getvalue()


def _pilot() -> tuple[Any, Any, Any, str]:
    library = _library()
    golden = _golden(library)
    pool = build_judgment_pool(golden, library, anchor_pool_entries(
        golden, library, ("both",),
    ))
    return library, golden, pool, render_judgment_worksheet(
        golden, library, pool,
    )[1]


def test_worksheet_round_trip_keeps_unjudged_rows_unjudged() -> None:
    library, golden, pool, sheet = _pilot()
    filled = _filled(sheet, {"a1": ("required", "yes"),
                             "b1": ("required", "no"),
                             "a2": ("irrelevant", "")})

    judgments = read_judgment_worksheet(
        filled, pool, dataset_id="synthetic", reviewer="Reviewer",
        status="pilot",
    )
    warnings = validate_multibook_judgments(judgments, golden, library, pool)

    labels = {judgment.record_id: judgment for judgment in judgments.judgments}
    assert labels["b1"].supplies_evidence is False
    assert labels["a2"].relevance is PdfRecordRelevance.IRRELEVANT
    assert "b2" not in labels
    assert warnings == ("1 pooled entries are unjudged",)
    frozen = judgments.model_copy(update={"status": "frozen"})
    with pytest.raises(ValueError, match="Frozen judgments miss"):
        validate_multibook_judgments(frozen, golden, library, pool)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ({"a1": ("required", "")}, "need supplies_evidence"),
        ({"a2": ("irrelevant", "yes")}, "take no supplies_evidence"),
        ({"a1": ("relevant", "yes")}, "relevant"),
        ({"a1": ("required", "maybe")}, "yes or no"),
        ({"a1": ("", "yes")}, "without relevance"),
    ],
)
def test_worksheet_rejects_invalid_labels(
    labels: dict[str, tuple[str, str]],
    message: str,
) -> None:
    _, _, pool, sheet = _pilot()

    with pytest.raises((ValueError, ValidationError), match=message):
        read_judgment_worksheet(
            _filled(sheet, labels), pool, dataset_id="synthetic",
            reviewer="Reviewer", status="pilot",
        )


def test_worksheet_rows_must_match_the_pool() -> None:
    _, _, pool, sheet = _pilot()
    tampered = sheet.replace(",a1,", ",a3,", 1)
    shortened = "\n".join(sheet.splitlines()[:-1]) + "\n"

    with pytest.raises(ValueError, match="does not match"):
        read_judgment_worksheet(tampered, pool, dataset_id="synthetic",
                                reviewer="Reviewer", status="pilot")
    with pytest.raises(ValueError, match="missing"):
        read_judgment_worksheet(shortened, pool, dataset_id="synthetic",
                                reviewer="Reviewer", status="pilot")


def test_validation_binds_pool_and_flags_cross_book_relevance() -> None:
    library, golden, pool, _ = _pilot()
    cross_book = PdfMultibookJudgmentSet(
        dataset_id="synthetic",
        golden_content_fingerprint=golden.content_fingerprint,
        library_fingerprint=library.manifest.fingerprint,
        pool_fingerprint=pool.fingerprint,
        status="pilot",
        judgments=(PdfMultibookJudgment(
            case_id="both", unit_id="cache", record_id="a1",
            relevance=PdfRecordRelevance.SUPPORTING, supplies_evidence=True,
            reviewer="Reviewer",
        ),),
    )

    with pytest.raises(ValueError, match="outside the pool"):
        validate_multibook_judgments(cross_book, golden, library, pool)
    other_pool = cross_book.model_copy(update={"pool_fingerprint": "0" * 64})
    with pytest.raises(ValueError, match="different pool"):
        validate_multibook_judgments(other_pool, golden, library, pool)


def test_cli_prepares_pilot_and_collects_judgments(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    for alias, path in BOOKS.items():
        _write_pdf(library_root / path, PAGES[alias])
    texts = {
        alias: tuple(
            page.reading_text
            for page in prepare_pdf_reading_document(
                library_root, path,
            ).reading_pages
        )
        for alias, path in BOOKS.items()
    }
    library = load_pdf_library_corpus(
        library_root, tuple(BOOKS.values()), PdfRetrievalPolicy.ALL_RECORDS,
    )
    files = _package_files(texts)
    for book in files["source-manifest.json"]["books"].values():
        pinned = library.book(book["book_path"])
        book["pdf_sha256"] = pinned.pdf_sha256
        book["record_fingerprint"] = pinned.corpus.record_set_fingerprint
    private = tmp_path / "private"
    package = _write_package(private / "package", files)
    decisions = private / "decisions.json"
    decisions.write_text(_decisions().model_dump_json(by_alias=True))
    pdf_import_multibook_golden(str(package), str(library_root),
                                str(decisions), str(private / "draft"))
    golden = str(private / "draft" / "multibook-golden-draft.json")
    pilot = private / "pilot"

    summary = pdf_prepare_judgment_pilot(golden, str(library_root),
                                         "q-both,q-figure", str(pilot))

    assert "entries" in summary
    sheet = (pilot / "worksheet.csv").read_text()
    assert "anchor_overlap" not in sheet
    assert "q-both" in (pilot / "worksheet.md").read_text()
    filled = pilot / "filled.csv"
    filled.write_text(_filled(sheet, {}))
    result = pdf_collect_judgments(
        golden, str(library_root), str(pilot / "judgment-pool.json"),
        str(filled), "Reviewer", str(pilot / "judgments.json"),
    )
    assert result.startswith("0 judgments")
    assert "unjudged" in result
    stored = load_pdf_multibook_judgments(pilot / "judgments.json")
    assert stored.status == "pilot"
    with pytest.raises(CliError, match="Unknown cases"):
        pdf_prepare_judgment_pilot(golden, str(library_root), "missing",
                                   str(private / "other"))
