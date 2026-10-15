"""Commands for the private multi-book golden set.

Private inputs are path arguments, and outputs are created once. Import
writes a draft only; freezing is a separate command that requires a
named human approval.
"""

from pathlib import Path

from src.evaluation.pdf.cli import _user_errors, _write_new
from src.evaluation.pdf.library import load_pdf_library_corpus
from src.evaluation.pdf.multibook_golden import load_pdf_multibook_golden
from src.evaluation.pdf.multibook_import import (
    freeze_pdf_multibook_golden,
    import_pdf_multibook_candidates,
    load_pdf_import_decisions,
    load_pdf_review_package,
    render_pdf_import_report,
    verify_pdf_multibook_golden,
    write_pdf_multibook_golden,
)
from src.ingestion.pdf.reading_document import prepare_pdf_reading_document
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy


DRAFT_FILE = "multibook-golden-draft.json"
REPORT_FILE = "import-report.md"


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
