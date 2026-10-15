"""Offline multi-book runner CLI and atomic, exclusive run publication."""

from hashlib import sha256
import os
from pathlib import Path
import tempfile

from pypdf.errors import PyPdfError

from src.cli import CliError
from src.evaluation.pdf.library import (
    load_pdf_library_corpus,
    sha256_file,
)
from src.evaluation.pdf.manifest import PdfRetrieverKind
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook_queries import (
    build_pdf_multibook_query_plan,
    PdfQueryMode,
)
from src.evaluation.pdf.multibook_runner import run_pdf_multibook_retrieval
from src.evaluation.pdf.multibook_runs import PdfMultibookRetrievalRun
from src.ingestion.pdf.pages import _resolve_book_path
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy


def pdf_run_multibook_retrieval(
    library_root: str,
    golden: str,
    retriever: str,
    output: str,
    policy: str = PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE.value,
    query_mode: str = PdfQueryMode.ALL.value,
) -> str:
    """Search exactly the pinned books, globally, without model downloads.

    MiniLM/Hybrid need the pinned model already in the local cache.
    Outputs are new files only. This command neither labels records nor
    evaluates relevance, answerability or clarification behavior.
    """
    try:
        kind = PdfRetrieverKind(retriever)
        selected_policy = PdfRetrievalPolicy(policy)
        mode = PdfQueryMode(query_mode)
    except ValueError:
        raise CliError(
            "Invalid retriever, policy or query_mode; use command --help. "
            "Retrievers: bm25/minilm/hybrid; modes: original/clarified/all."
        ) from None
    target = Path(output)
    try:
        _require_new_output(target)
    except (OSError, ValueError):
        raise CliError(
            "Output must be a new file in an existing writable directory."
        ) from None
    try:
        raw = Path(golden).read_bytes()
        dataset = PdfMultibookGolden.model_validate_json(raw)
        build_pdf_multibook_query_plan(dataset, mode)
    except (OSError, ValueError):
        raise CliError(
            "Cannot read a valid Golden Set v2/query plan. Check the input "
            "and approved clarified questions, or select query_mode=original."
        ) from None
    try:
        root = Path(library_root)
        # Reject changed PDFs before extraction or any encoder loading.
        for pin in dataset.library.books:
            source = _resolve_book_path(root, pin.book_path)
            if sha256_file(source) != pin.pdf_sha256:
                raise ValueError("Pinned PDF hash differs")
        corpus = load_pdf_library_corpus(
            root, dataset.library.book_paths, selected_policy,
        )
    except (OSError, ValueError, PyPdfError):
        raise CliError(
            "Cannot build the pinned library. Check canonical book paths, "
            "PDF hashes and readable text-based PDFs."
        ) from None
    try:
        run = run_pdf_multibook_retrieval(
            golden=dataset, golden_sha256=sha256(raw).hexdigest(),
            corpus=corpus, retriever=kind, query_mode=mode,
        )
    except (OSError, ValueError, RuntimeError, TypeError):
        raise CliError(
            "Retrieval failed. Check record fingerprints, selected queries "
            "and corpus eligibility. MiniLM/Hybrid require the pinned model "
            "in the local cache; downloads are disabled."
        ) from None
    try:
        file_hash = write_pdf_multibook_run(target, run)
    except (OSError, ValueError):
        raise CliError(
            "Run was not published. Check output permissions, existing files "
            "and filesystem hard-link support; choose a new output path."
        ) from None
    empty = sum(not result.ranking for result in run.results)
    lexical = (
        f"; BM25 queries without lexical matches: {empty}"
        if kind is PdfRetrieverKind.BM25 else ""
    )
    return (
        f"{kind.value}: {len(run.results)} queries{lexical}\n"
        f"configuration identity: {run.manifest_identity}\n"
        f"comparison identity: {run.manifest.comparison_identity}\n"
        f"results fingerprint: {run.results_fingerprint}\n"
        f"artifact SHA256: {file_hash}"
    )


def _require_new_output(path: Path) -> None:
    if os.path.lexists(path) or not path.parent.is_dir() or not os.access(
        path.parent, os.W_OK | os.X_OK,
    ):
        raise ValueError("Output already exists or its directory is missing")


def write_pdf_multibook_run(path: Path, run: PdfMultibookRetrievalRun) -> str:
    """Publish complete UTF-8 bytes atomically, without replacing any file.

    The caller must validate against its current corpus first. A same-directory
    temporary file is flushed before an exclusive hard link publishes it;
    link failure cannot truncate an existing destination. No rename fallback
    is used because rename could overwrite a concurrently created output.
    """
    _require_new_output(path)
    checked = PdfMultibookRetrievalRun.model_validate(run.model_dump())
    payload = (checked.model_dump_json(indent=1) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".fredl-run-", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256(payload).hexdigest()
