"""Create private review packages from three saved global runs."""

from hashlib import sha256
import os
from pathlib import Path

from pypdf.errors import PyPdfError

from src.cli import CliError
from src.evaluation.pdf.library import load_pdf_library_corpus, sha256_file
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook_pool_package import (
    write_review_package,
)
from src.evaluation.pdf.multibook_pool_models import (
    PdfPoolingInputs, PdfPoolingRun,
)
from src.evaluation.pdf.multibook_pooling import (
    build_pdf_retrieval_judgment_pool,
)
from src.evaluation.pdf.multibook_pool_worksheet import (
    render_retrieval_pool_worksheet,
)
from src.evaluation.pdf.multibook_runs import PdfMultibookRetrievalRun
from src.ingestion.pdf.pages import _resolve_book_path
from src.ingestion.pdf.reading_document import prepare_pdf_reading_document


def load_pdf_pooling_runs(path: Path) -> tuple[PdfPoolingRun, ...]:
    """Check exact run-file hashes before parsing; resolve relative references.

    Input manifest order is immaterial. Duplicate resolved paths and mismatched
    retriever declarations are errors, never silently recovered.
    """
    manifest = PdfPoolingInputs.model_validate_json(path.read_bytes())
    paths = [Path(ref.path) if Path(ref.path).is_absolute()
             else path.parent / ref.path for ref in manifest.runs]
    if len({ref.resolve() for ref in paths}) != len(paths):
        raise ValueError("Duplicate run artifact references")
    inputs = []
    for ref, source in zip(manifest.runs, paths):
        raw = source.read_bytes()
        if sha256(raw).hexdigest() != ref.sha256:
            raise ValueError("Run file SHA256 differs")
        run = PdfMultibookRetrievalRun.model_validate_json(raw)
        if run.manifest.retrieval.retriever != ref.retriever:
            raise ValueError("Declared retriever differs from artifact")
        inputs.append(PdfPoolingRun(artifact_sha256=ref.sha256, run=run))
    return tuple(inputs)


def pdf_pool_multibook_judgments(
    golden: str, library_root: str, runs: str, output_dir: str,
    depth: int = 10,
) -> str:
    """Pool three compatible runs into blank, blind evidence-unit tasks.

    runs is a version 1 fredl.pdf.multibook-pooling-inputs JSON manifest.
    Depth is 5 or 10. The output directory must be new, with an existing
    parent. No retrieval, model loading, label transfer or metrics run.
    """
    target = Path(output_dir)
    try:
        if type(depth) is not int or depth not in (5, 10):
            raise ValueError("Invalid depth")
        if os.path.lexists(target) or not target.parent.is_dir():
            raise ValueError("Output directory must be new")
    except (OSError, ValueError):
        raise CliError(
            "Use depth=5 or 10 and a new output directory with an existing "
            "parent. Existing outputs are never reused."
        ) from None
    try:
        raw = Path(golden).read_bytes()
        dataset = PdfMultibookGolden.model_validate_json(raw)
        inputs = load_pdf_pooling_runs(Path(runs))
        root = Path(library_root)
        for pin in dataset.library.books:
            if sha256_file(_resolve_book_path(root, pin.book_path)) != (
                pin.pdf_sha256
            ):
                raise ValueError("Pinned PDF hash differs")
        policy = inputs[0].run.manifest.policy
        corpus = load_pdf_library_corpus(root, dataset.library.book_paths,
                                         policy)
        pages = {
            book: prepare_pdf_reading_document(root, book).reading_pages
            for book in dataset.library.book_paths
        }
        pool = build_pdf_retrieval_judgment_pool(
            golden=dataset, golden_sha256=sha256(raw).hexdigest(),
            corpus=corpus, runs=inputs, reading_pages=pages, depth=depth,
        )
        worksheet, sheet = render_retrieval_pool_worksheet(
            dataset, corpus, pool,
        )
    except (OSError, ValueError, TypeError, PyPdfError):
        raise CliError(
            "Cannot pool inputs. Check Golden Set v2, all three run hashes "
            "and fingerprints, compatible query/context protocols, pinned "
            "PDFs, extracted records and evidence locators."
        ) from None
    files = {
        "judgment-pool.json": pool.model_dump_json(indent=1) + "\n",
        "worksheet.md": worksheet,
        "worksheet.csv": sheet,
        "pool-summary.md": (
            "# Retrieval judgment pool\n\n"
            "Private provenance summary; keep separate from blind review.\n\n"
            f"- Query variants: {len(pool.query_plan.queries)}\n"
            f"- Query-record candidates: {len(pool.candidates)}\n"
            f"- Evidence-unit tasks: {len(pool.tasks)}\n"
            f"- Depth: {pool.depth}\n"
            f"- Pool fingerprint: `{pool.fingerprint}`\n\n"
            "All labels are blank. Candidates without evidence-unit tasks "
            "remain in the inventory; no relevance is inferred.\n"
        ),
    }
    try:
        write_review_package(target, files, pool.fingerprint)
    except (OSError, ValueError):
        raise CliError(
            "Review package was not completed. Check permissions and choose "
            "a new output directory. Linux/filesystem support for atomic "
            "no-replace directory rename is required."
        ) from None
    return (f"{len(pool.candidates)} query-record candidates; "
            f"{len(pool.tasks)} blank tasks; pool {pool.fingerprint}")
