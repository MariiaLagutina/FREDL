"""Tests for importing a reviewed candidate package as a draft golden set."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.cli import CliError
from src.evaluation.pdf.multibook.library import (
    build_pdf_library_corpus,
    load_pdf_library_corpus,
    PdfLibraryBookInput,
)
from src.evaluation.pdf.multibook.curation.cli import (
    pdf_freeze_multibook_golden,
    pdf_import_multibook_golden,
)
from src.evaluation.pdf.multibook.curation.golden import (
    load_pdf_multibook_golden,
    PdfExpectedOutcome,
    PdfLimitationReason,
)
from src.evaluation.pdf.multibook.curation.importer import (
    freeze_pdf_multibook_golden,
    import_pdf_multibook_candidates,
    load_pdf_review_package,
    PdfImportDecision,
    PdfImportDecisions,
    render_pdf_import_report,
    verify_pdf_multibook_golden,
    write_pdf_multibook_golden,
)
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.ingestion.pdf.reading_document import prepare_pdf_reading_document
from src.ingestion.pdf.reading_text import PdfReadingPage
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy
from tests.evaluation.pdf.multibook.curation.package_helpers import (
    _as_v1,
    _decisions,
    _interpretations,
    _package_files,
    _with_ambiguous,
    _write_package,
    _write_pdf,
    BOOKS,
    PAGES,
)


def _import(tmp_path: Path, files: dict[str, Any] | None = None) -> Any:
    package = _write_package(tmp_path / "package", files or _package_files(
        PAGES,
    ))
    return import_pdf_multibook_candidates(
        load_pdf_review_package(package), _decisions(),
    )


def test_import_uses_approved_values_and_reports_every_change(
    tmp_path: Path,
) -> None:
    imported = _import(tmp_path)
    golden = imported.golden
    cases = {case.case_id: case for case in golden.cases}
    kinds = {(item.subject, item.kind) for item in imported.findings}

    assert golden.status == "draft"
    assert golden.dataset_id == "fredl-multibook-golden-v1"
    assert cases["q-queue"].question == "Approved question about queues?"
    assert cases["q-clipped"].expected_outcome is (
        PdfExpectedOutcome.SOURCE_INCOMPLETE
    )
    assert cases["q-clipped"].limitation_reason is (
        PdfLimitationReason.SOURCE_INCOMPLETE
    )
    assert cases["q-clipped"].related_units == ("bet-clipped",)
    assert cases["q-figure"].limitation_reason is (
        PdfLimitationReason.VISUAL_EVIDENCE_UNAVAILABLE
    )
    assert cases["q-absent"].sufficient_sets == ()
    assert {
        ("q-queue", "approved_question"),
        ("q-clipped", "decision_applied"),
        ("q-clipped", "proposal_differs"),
        ("q-figure", "decision_applied"),
        ("q-figure", "normalized_support"),
        ("q-absent", "normalized_support"),
    } <= kinds
    assert golden.provenance is not None
    assert golden.provenance.decisions[0].startswith("q-clipped:")
    report = render_pdf_import_report(imported)
    assert "source_incomplete" in report and "not frozen" in report


def test_import_is_deterministic(tmp_path: Path) -> None:
    first = _import(tmp_path / "one").golden
    second = _import(tmp_path / "two").golden

    assert first.content_fingerprint == second.content_fingerprint


def test_import_requires_explicit_decision_for_missing_outcome(
    tmp_path: Path,
) -> None:
    package = load_pdf_review_package(
        _write_package(tmp_path / "package", _package_files(PAGES)),
    )
    only_figure = PdfImportDecisions(decisions=_decisions().decisions[1:])

    with pytest.raises(ValueError, match="no approved expected outcome"):
        import_pdf_multibook_candidates(package, only_figure)


def test_decision_cannot_override_an_approved_value(tmp_path: Path) -> None:
    package = load_pdf_review_package(
        _write_package(tmp_path / "package", _package_files(PAGES)),
    )
    overriding = PdfImportDecisions(decisions=(
        *_decisions().decisions,
        PdfImportDecision(
            case_id="q-absent",
            expected_outcome=PdfExpectedOutcome.SOURCE_INCOMPLETE,
            provenance="Contradiction",
        ),
    ))

    with pytest.raises(ValueError, match="contradicts approved"):
        import_pdf_multibook_candidates(package, overriding)


@pytest.mark.parametrize("change", ["tamper", "unlisted", "missing"])
def test_package_checksums_must_match(tmp_path: Path, change: str) -> None:
    package = _write_package(tmp_path / "package", _package_files(PAGES))
    if change == "tamper":
        (package / "review-decisions.csv").write_text("id,decision\n")
    elif change == "unlisted":
        (package / "notes.md").write_text("Extra file.")
    else:
        (package / "source-manifest.json").unlink()

    with pytest.raises(ValueError, match="changed|differ"):
        load_pdf_review_package(package)


def test_unaccepted_or_conflicting_candidates_stop_import(
    tmp_path: Path,
) -> None:
    rejected = _package_files(PAGES)
    rejected["candidates.json"]["candidates"][0]["human_review"][
        "decision"
    ] = "investigate"
    conflicting = _package_files(PAGES)
    conflicting["candidates.json"]["candidates"][2][
        "candidate_source_support"
    ] = "complete"

    with pytest.raises(ValueError, match="not accepted"):
        _import(tmp_path / "rejected", rejected)
    with pytest.raises(ValueError, match="conflicts with the expected"):
        _import(tmp_path / "conflicting", conflicting)


def test_freeze_keeps_content_and_never_overwrites(tmp_path: Path) -> None:
    draft = _import(tmp_path).golden
    frozen = freeze_pdf_multibook_golden(
        draft, approved_by="Reviewer", approval_note="Approved for testing.",
    )
    path = tmp_path / "frozen.json"
    write_pdf_multibook_golden(path, frozen)
    write_pdf_multibook_golden(path, frozen)

    assert frozen.status == "frozen"
    assert frozen.content_fingerprint == draft.content_fingerprint
    assert load_pdf_multibook_golden(path).freeze_record is not None
    with pytest.raises(ValueError, match="Only a draft"):
        freeze_pdf_multibook_golden(
            frozen, approved_by="Reviewer", approval_note="Again.",
        )
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        write_pdf_multibook_golden(path, draft)


def _library_and_pages(
    texts: dict[str, tuple[str, ...]],
) -> tuple[Any, dict[str, list[PdfReadingPage]]]:
    inputs = []
    pages = {}
    for alias, path in BOOKS.items():
        records = tuple(
            PdfIndexRecord(
                record_id=f"{alias}-{number}", text=text, book_path=path,
                section_path=("Chapter",), flow_index=0,
                flow_start=number * 100, flow_end=number * 100 + len(text),
                page_spans=(PdfChunkPageSpan(number, 0, len(text), 0,
                                             len(text)),),
                section_interval_id=f"{alias}:chapter",
            )
            for number, text in enumerate(texts[alias], start=1)
        )
        inputs.append(PdfLibraryBookInput(path, "a" * 64, records, ()))
        pages[path] = [
            PdfReadingPage(path, number, text, text, ())
            for number, text in enumerate(texts[alias], start=1)
        ]
    return (
        build_pdf_library_corpus(inputs, PdfRetrievalPolicy.ALL_RECORDS),
        pages,
    )


def test_verification_rechecks_hashes_anchors_and_records(
    tmp_path: Path,
) -> None:
    library, pages = _library_and_pages(PAGES)
    files = _package_files(PAGES)
    for book in files["source-manifest.json"]["books"].values():
        book["record_fingerprint"] = library.book(
            book["book_path"],
        ).corpus.record_set_fingerprint
    golden = _import(tmp_path, files).golden

    findings = verify_pdf_multibook_golden(golden, library, pages)

    assert len(findings) == 4
    assert all("overlap 1 record" in item.detail for item in findings)
    changed = dict(pages)
    changed[BOOKS["ALP"]] = [
        PdfReadingPage(BOOKS["ALP"], 1, "Edited.", "Edited.", ()),
        pages[BOOKS["ALP"]][1],
    ]
    with pytest.raises(ValueError, match="page text changed"):
        verify_pdf_multibook_golden(golden, library, changed)


def test_cli_imports_a_draft_and_freezes_only_on_request(
    tmp_path: Path,
) -> None:
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
    package = _write_package(tmp_path / "private" / "package", files)
    decisions = tmp_path / "private" / "decisions.json"
    decisions.write_text(_decisions().model_dump_json(by_alias=True))
    output = tmp_path / "private" / "draft"
    before = (package / "candidates.json").read_bytes()

    summary = pdf_import_multibook_golden(
        str(package), str(library_root), str(decisions), str(output),
    )

    assert summary.startswith("draft fredl-multibook-golden-v1: 5 cases")
    draft = load_pdf_multibook_golden(output / "multibook-golden-draft.json")
    assert draft.status == "draft"
    assert "locators_verified" in (output / "import-report.md").read_text()
    assert (package / "candidates.json").read_bytes() == before
    assert not (output / "frozen.json").exists()
    pdf_freeze_multibook_golden(
        str(output / "multibook-golden-draft.json"),
        str(output / "frozen.json"), "Reviewer", "Approved.",
    )
    assert load_pdf_multibook_golden(output / "frozen.json").status == (
        "frozen"
    )
    with pytest.raises(CliError, match="Only a draft"):
        pdf_freeze_multibook_golden(
            str(output / "frozen.json"), str(output / "again.json"),
            "Reviewer", "Again.",
        )


# Version 2: clarification decisions and migration from version 1 --------


def test_ambiguous_question_needs_explicit_interpretations(
    tmp_path: Path,
) -> None:
    package = load_pdf_review_package(
        _write_package(tmp_path / "package", _with_ambiguous(PAGES)),
    )

    with pytest.raises(ValueError, match="needs approved interpretations"):
        import_pdf_multibook_candidates(package, _decisions())
    imported = import_pdf_multibook_candidates(package, PdfImportDecisions(
        decisions=(*_decisions().decisions, _interpretations()),
    ))

    cases = {case.case_id: case for case in imported.golden.cases}
    assert cases["q-which"].expected_behavior.value == (
        "clarification_required"
    )
    assert cases["q-which"].conditional_units == ("alp-queue", "bet-cache")
    assert all(case.expected_behavior.value == "answer"
               for case_id, case in cases.items() if case_id != "q-which")
    kinds = {(item.subject, item.kind) for item in imported.findings}
    assert ("q-which", "interpretations_applied") in kinds
    assert ("all cases", "expected_behavior") in kinds
    assert "## Interpretations" in render_pdf_import_report(imported)


def test_clear_question_rejects_interpretations(tmp_path: Path) -> None:
    package = load_pdf_review_package(
        _write_package(tmp_path / "package", _package_files(PAGES)),
    )
    wrong = _interpretations().model_copy(update={"case_id": "q-queue"})

    with pytest.raises(ValueError, match="takes no interpretations"):
        import_pdf_multibook_candidates(package, PdfImportDecisions(
            decisions=(*_decisions().decisions, wrong),
        ))


def test_version_1_decision_files_still_load() -> None:
    decisions = PdfImportDecisions.model_validate_json(json.dumps({
        "schema": "fredl.pdf.multibook-import-decisions", "version": 1,
        "decisions": [{"case_id": "q-clipped",
                       "expected_outcome": "source_incomplete",
                       "provenance": "Reviewer decision D3"}],
    }))

    assert decisions.version == 1


def test_v1_to_v2_preservation_detects_any_changed_value(
    tmp_path: Path,
) -> None:
    from src.evaluation.pdf.multibook.curation.golden import PdfGoldenLineage
    from src.evaluation.pdf.multibook.curation.importer import (
        require_v1_to_v2_preserved,
    )

    package = load_pdf_review_package(
        _write_package(tmp_path / "package", _with_ambiguous(PAGES)),
    )
    v1_decisions = _decisions()
    v2_decisions = PdfImportDecisions(
        decisions=(*v1_decisions.decisions, _interpretations()),
    )
    v1 = _as_v1(import_pdf_multibook_candidates(
        package, v2_decisions,
    ).golden)
    v1 = v1.model_copy(update={"provenance": v1.provenance.model_copy(
        update={"decisions": v1.provenance.decisions[:2]},
    )})
    lineage = PdfGoldenLineage(
        schema_version=1, dataset_id=v1.dataset_id,
        content_fingerprint=v1.content_fingerprint,
    )
    v2 = import_pdf_multibook_candidates(
        package, v2_decisions, migrated_from=lineage,
    ).golden

    findings = require_v1_to_v2_preserved(v1, v2)

    assert {item.subject for item in findings} >= {"q-which", "all cases"}
    changed = v1.model_copy(update={"cases": (
        v1.cases[0].model_copy(update={"question": "Changed?"}),
        *v1.cases[1:],
    )})
    with pytest.raises(ValueError, match="does not name"):
        require_v1_to_v2_preserved(changed, v2)
    relinked = v2.model_copy(update={"migrated_from": lineage.model_copy(
        update={"content_fingerprint": changed.content_fingerprint},
    )})
    with pytest.raises(ValueError, match="changes q-queue"):
        require_v1_to_v2_preserved(changed, relinked)
