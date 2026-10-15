"""Commands for the private multi-book golden set.

Private inputs are path arguments, and outputs are created once. Import
writes a draft only; freezing is a separate command that requires a
named human approval.
"""

from collections.abc import Sequence
from pathlib import Path

from src.evaluation.pdf.cli import _user_errors, _write_new
from src.evaluation.pdf.library import (
    load_pdf_library_corpus,
    PdfLibraryCorpus,
    require_library_matches_manifest,
)
from src.evaluation.pdf.multibook_golden import (
    load_pdf_multibook_golden,
    PdfMultibookGolden,
)
from src.evaluation.pdf.multibook_import import (
    freeze_pdf_multibook_golden,
    import_pdf_multibook_candidates,
    load_pdf_import_decisions,
    load_pdf_review_package,
    render_pdf_import_report,
    verify_pdf_multibook_golden,
    write_pdf_multibook_golden,
)
from src.evaluation.pdf.multibook_judgments import (
    judgment_counts,
    load_pdf_judgment_pool,
    validate_multibook_judgments,
)
from src.evaluation.pdf.pooling import (
    anchor_pool_entries,
    build_judgment_pool,
    read_judgment_worksheet,
    render_judgment_worksheet,
    summarize_pool,
)
from src.ingestion.pdf.reading_document import prepare_pdf_reading_document
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy


DRAFT_FILE = "multibook-golden-draft.json"
REPORT_FILE = "import-report.md"
POOL_FILE = "judgment-pool.json"
WORKSHEET_FILE = "worksheet.md"
SHEET_FILE = "worksheet.csv"
SUMMARY_FILE = "pool-summary.md"
# A rough reading pace for effort estimates, not a measurement.
_WORDS_PER_MINUTE = 200
_SECONDS_PER_DECISION = 20


@_user_errors
def pdf_import_multibook_golden(
    package_dir: str,
    library_root: str,
    decisions: str,
    output_dir: str,
) -> str:
    """Import an accepted review package as a draft and verify locators.

    The draft is never frozen. decisions lists explicit human decisions
    for values the package does not encode; it cannot override any
    approved value.
    """
    imported = import_pdf_multibook_candidates(
        load_pdf_review_package(Path(package_dir)),
        load_pdf_import_decisions(Path(decisions)),
    )
    golden = imported.golden
    root = Path(library_root)
    book_paths = golden.library.book_paths
    library = load_pdf_library_corpus(
        root, book_paths, PdfRetrievalPolicy.ALL_RECORDS,
    )
    pages = {
        book_path: prepare_pdf_reading_document(root, book_path).reading_pages
        for book_path in book_paths
    }
    verification = verify_pdf_multibook_golden(golden, library, pages)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_pdf_multibook_golden(out / DRAFT_FILE, golden)
    _write_new(
        out / REPORT_FILE, render_pdf_import_report(imported, verification),
    )
    return (
        f"draft {golden.dataset_id}: {len(golden.cases)} cases, "
        f"{len(golden.evidence_units)} evidence units, "
        f"{len(imported.findings)} findings; content "
        f"{golden.content_fingerprint}"
    )


@_user_errors
def pdf_freeze_multibook_golden(
    draft: str,
    output: str,
    approved_by: str,
    approval_note: str,
) -> str:
    """Write the frozen copy of a reviewed draft without changing content."""
    source = load_pdf_multibook_golden(Path(draft))
    frozen = freeze_pdf_multibook_golden(
        source, approved_by=approved_by, approval_note=approval_note,
    )
    write_pdf_multibook_golden(Path(output), frozen)
    return f"frozen {frozen.dataset_id}: content {frozen.content_fingerprint}"


@_user_errors
def pdf_prepare_judgment_pilot(
    golden: str,
    library_root: str,
    cases: str | Sequence[str],
    output_dir: str,
) -> str:
    """Prepare an anchor-based pool and blind worksheet for chosen cases.

    No retrieval runs. The reviewer fills worksheet.csv; nothing here
    assigns a relevance label.
    """
    dataset = load_pdf_multibook_golden(Path(golden))
    library = _library(library_root, dataset)
    selected = _case_list(cases)
    pool = build_judgment_pool(
        dataset, library, anchor_pool_entries(dataset, library, selected),
    )
    worksheet, sheet = render_judgment_worksheet(dataset, library, pool)
    summary = summarize_pool(pool, library)
    minutes = (
        summary.words / _WORDS_PER_MINUTE
        + summary.entries * _SECONDS_PER_DECISION / 60
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_new(out / POOL_FILE, pool.model_dump_json(by_alias=True, indent=1))
    _write_new(out / WORKSHEET_FILE, worksheet)
    _write_new(out / SHEET_FILE, sheet)
    _write_new(out / SUMMARY_FILE, "\n".join([
        "# Judgment pool summary", "",
        "Anchor-based pool only; retrieval pooling is not included.", "",
        f"- Pool fingerprint: `{pool.fingerprint}`",
        f"- Entries (record x unit): {summary.entries}",
        f"- Distinct records: {summary.unique_records}",
        f"- Words to read: {summary.words}",
        f"- Rough effort: {minutes:.0f} minutes at {_WORDS_PER_MINUTE} "
        f"words per minute and {_SECONDS_PER_DECISION} s per decision",
        "", "| Case | Entries | Distinct records |", "| --- | --- | --- |",
        *(f"| {case_id} | {count} | {records} |"
          for case_id, count, records in summary.by_case),
    ]))
    return (
        f"pool {pool.fingerprint[:12]}: {summary.entries} entries, "
        f"{summary.unique_records} records, {summary.words} words"
    )


@_user_errors
def pdf_collect_judgments(
    golden: str,
    library_root: str,
    pool: str,
    worksheet: str,
    reviewer: str,
    output: str,
    status: str = "pilot",
) -> str:
    """Validate a filled worksheet and store it as a judgment set."""
    dataset = load_pdf_multibook_golden(Path(golden))
    library = _library(library_root, dataset)
    offered = load_pdf_judgment_pool(Path(pool))
    judgments = read_judgment_worksheet(
        Path(worksheet).read_text(encoding="utf-8"), offered,
        dataset_id=dataset.dataset_id, reviewer=reviewer, status=status,
    )
    warnings = validate_multibook_judgments(
        judgments, dataset, library, offered,
    )
    _write_new(
        Path(output), judgments.model_dump_json(by_alias=True, indent=1),
    )
    counts = ", ".join(
        f"{label} {count}"
        for label, count in judgment_counts(judgments.judgments).items()
    )
    return "\n".join([
        f"{len(judgments.judgments)} judgments ({counts})",
        *(f"warning: {item}" for item in warnings),
    ])


def _library(
    library_root: str,
    dataset: PdfMultibookGolden,
) -> PdfLibraryCorpus:
    library = load_pdf_library_corpus(
        Path(library_root), dataset.library.book_paths,
        PdfRetrievalPolicy.ALL_RECORDS,
    )
    require_library_matches_manifest(library, dataset.library)
    return library


def _case_list(cases: str | Sequence[str]) -> tuple[str, ...]:
    items = cases.split(",") if isinstance(cases, str) else cases
    selected = tuple(item.strip() for item in items if item.strip())
    if not selected:
        raise ValueError("Name at least one case")
    return selected
