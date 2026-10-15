"""Tests for multi-book judgment pools, worksheets, and validation."""

import csv
import io
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.cli import CliError
from src.evaluation.pdf.judgments import PdfRecordRelevance
from src.evaluation.pdf.library import load_pdf_library_corpus
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
from src.ingestion.pdf.reading_document import prepare_pdf_reading_document
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy
from tests.evaluation.pdf.multibook_golden_helpers import _golden, _library
from tests.evaluation.pdf.multibook_package_helpers import (
    _decisions,
    _package_files,
    _write_package,
    _write_pdf,
    BOOKS,
    PAGES,
)


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


def test_worksheet_names_the_interpretation_of_each_unit() -> None:
    library = _library()
    golden = _golden(library)
    pool = build_judgment_pool(golden, library, anchor_pool_entries(
        golden, library, ("which",),
    ))

    worksheet, _ = render_judgment_worksheet(golden, library, pool)

    assert "### Interpretation `queues`: Meaning queues" in worksheet
    assert worksheet.index("Interpretation `producers`") < worksheet.index(
        "Evidence unit `backpressure`",
    )


def _relabelled(golden: PdfMultibookGolden) -> PdfMultibookGolden:
    # A content change outside the pool (reference notes) gives a new
    # golden fingerprint, as a schema migration does.
    cases = tuple(
        case.model_copy(update={"reference_notes": "Revised notes."})
        for case in golden.cases
    )
    return PdfMultibookGolden.model_validate(
        golden.model_copy(update={"cases": cases}).model_dump(
            mode="json", by_alias=True,
        ),
    )


def test_carry_over_moves_labels_unchanged_to_a_regenerated_pool() -> None:
    from hashlib import sha256

    from src.evaluation.pdf.pooling import carry_over_worksheet

    library, golden, pool, sheet = _pilot()
    labelled = _filled(sheet, {"a1": ("required", "yes"),
                               "a2": ("irrelevant", ""),
                               "b1": ("supporting", "no")})
    revised = _relabelled(golden)
    new_pool = build_judgment_pool(revised, library, anchor_pool_entries(
        revised, library, ("both",),
    ))
    digest = sha256(labelled.encode("utf-8")).hexdigest()

    moved = carry_over_worksheet(pool, new_pool, labelled, digest)

    assert new_pool.fingerprint != pool.fingerprint
    assert moved.sheet == labelled
    assert [old for old, new, _ in moved.row_map] == [
        new for _, new, _ in moved.row_map
    ]
    judgments = read_judgment_worksheet(
        moved.sheet, new_pool, dataset_id="synthetic", reviewer="Reviewer",
        status="pilot",
    )
    assert len(judgments.judgments) == 3
    with pytest.raises(ValueError, match="recorded hash"):
        carry_over_worksheet(pool, new_pool, labelled, "0" * 64)
    other_pool = build_judgment_pool(revised, library, anchor_pool_entries(
        revised, library, ("which",),
    ))
    with pytest.raises(ValueError, match="pool differs"):
        carry_over_worksheet(pool, other_pool, labelled, digest)
    tampered = labelled.replace(",a1,", ",a3,", 1)
    with pytest.raises(ValueError, match="does not match"):
        carry_over_worksheet(
            pool, new_pool, tampered,
            sha256(tampered.encode("utf-8")).hexdigest(),
        )


def test_cli_migrates_a_labelled_pilot_from_v1_to_v2(tmp_path: Path) -> None:
    from hashlib import sha256

    from src.evaluation.pdf.multibook_cli import pdf_migrate_judgment_pilot
    from src.evaluation.pdf.multibook_golden import load_pdf_multibook_golden
    from src.evaluation.pdf.multibook_import import PdfImportDecisions
    from tests.evaluation.pdf.multibook_package_helpers import (
        _as_v1,
        _interpretations,
        _with_ambiguous,
    )

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
    files = _with_ambiguous(texts)
    for book in files["source-manifest.json"]["books"].values():
        pinned = library.book(book["book_path"])
        book["pdf_sha256"] = pinned.pdf_sha256
        book["record_fingerprint"] = pinned.corpus.record_set_fingerprint
    private = tmp_path / "private"
    package = _write_package(private / "package", files)
    decisions = private / "decisions.json"
    decisions.write_text(PdfImportDecisions(
        decisions=(*_decisions().decisions, _interpretations()),
    ).model_dump_json(by_alias=True))
    pdf_import_multibook_golden(str(package), str(library_root),
                                str(decisions), str(private / "first"))
    first = load_pdf_multibook_golden(
        private / "first" / "multibook-golden-draft.json",
    )
    v1 = _as_v1(first)
    v1_path = private / "v1.json"
    v1_path.write_text(v1.model_dump_json(by_alias=True))
    pdf_import_multibook_golden(str(package), str(library_root),
                                str(decisions), str(private / "v2"),
                                migrated_from=str(v1_path))
    v2_path = private / "v2" / "multibook-golden-draft.json"
    v2 = load_pdf_multibook_golden(v2_path)
    assert "v1_preserved" in (private / "v2" / "import-report.md").read_text()
    pool = build_judgment_pool(v2, library, anchor_pool_entries(
        v2, library, ("q-both", "q-which"),
    ))
    old_pool = pool.model_copy(
        update={"golden_content_fingerprint": v1.content_fingerprint},
    )
    (private / "old-pool.json").write_text(
        old_pool.model_dump_json(by_alias=True),
    )
    sheet = render_judgment_worksheet(v2, library, pool)[1]
    records = [row["record_id"] for row in csv.DictReader(io.StringIO(sheet))]
    labelled = _filled(sheet, {records[0]: ("required", "yes"),
                               records[-1]: ("irrelevant", "")})
    (private / "labels.csv").write_text(labelled)
    digest = sha256(labelled.encode("utf-8")).hexdigest()

    def migrate(output: str, expected: str) -> str:
        return pdf_migrate_judgment_pilot(
            str(v1_path), str(v2_path), str(library_root),
            str(private / "old-pool.json"), str(private / "labels.csv"),
            expected, "Reviewer", str(private / output),
        )

    summary = migrate("pilot-v2", digest)

    labelled_rows = sum(
        bool(row["relevance"]) for row in csv.DictReader(io.StringIO(labelled))
    )
    assert summary.startswith(f"{labelled_rows} judgments moved")
    out = private / "pilot-v2"
    assert (out / "worksheet-labels.csv").read_text() == labelled
    assert v2.content_fingerprint in (out / "migration-report.md").read_text()
    assert load_pdf_multibook_judgments(out / "judgments.json").status == (
        "pilot"
    )
    assert migrate("pilot-v2", digest) == summary
    with pytest.raises(CliError, match="recorded hash"):
        migrate("refused", "0" * 64)
    assert not (private / "refused").exists()
