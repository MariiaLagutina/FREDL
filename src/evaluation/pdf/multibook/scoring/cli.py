"""Collect human CSV labels and evaluate saved global runs without models."""

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import tempfile

from pypdf.errors import PyPdfError

from src.cli import CliError
from src.evaluation.pdf.multibook.library import (
    load_pdf_library_corpus,
    sha256_file,
)
from src.evaluation.pdf.multibook.scoring.collector import (
    collect_pdf_multibook_judgments, PdfMissingJudgments,
)
from src.evaluation.pdf.multibook.curation.golden import PdfMultibookGolden
from src.evaluation.pdf.multibook.scoring.judgment_models import (
    PdfCollectedJudgments, PdfJudgmentInputs, PdfWorksheetSubmission,
)
from src.evaluation.pdf.multibook.scoring.report import (
    build_multibook_metric_report,
)
from src.evaluation.pdf.multibook.scoring.metrics import (
    score_pdf_multibook_retrieval,
)
from src.evaluation.pdf.multibook.scoring.metric_models import (
    PdfQueryRetrievalMetrics,
)
from src.evaluation.pdf.multibook.pooling.cli import load_pdf_pooling_runs
from src.evaluation.pdf.multibook.pooling.models import (
    PdfRetrievalJudgmentPool,
)
from src.ingestion.pdf.pages import _resolve_book_path
from src.ingestion.pdf.reading_document import prepare_pdf_reading_document


@dataclass(frozen=True)
class _EvaluationInput:
    """Keep only fully validated inputs and their consumed-byte hashes."""

    pool: PdfRetrievalJudgmentPool
    judgments: PdfCollectedJudgments
    scores: tuple[PdfQueryRetrievalMetrics, ...]
    hashes: dict[str, str]


def _inputs(
    golden: str, library_root: str, runs: str, inputs: str, *, evaluate: bool,
) -> _EvaluationInput:
    raw = Path(golden).read_bytes()
    dataset = PdfMultibookGolden.model_validate_json(raw)
    manifest_path = Path(inputs)
    manifest_raw = manifest_path.read_bytes()
    manifest = PdfJudgmentInputs.model_validate_json(manifest_raw)

    def resolve(path: str) -> Path:
        value = Path(path)
        return value if value.is_absolute() else manifest_path.parent / value

    sources = [resolve(ref.path) for ref in manifest.worksheets]
    if len(set(path.resolve() for path in sources)) != len(sources):
        raise ValueError('Duplicate worksheet paths')
    pool_raw = resolve(manifest.pool.path).read_bytes()
    if sha256(pool_raw).hexdigest() != manifest.pool.sha256:
        raise ValueError('Pool file SHA256 differs')
    pool = PdfRetrievalJudgmentPool.model_validate_json(pool_raw)
    if pool.fingerprint != manifest.pool_fingerprint:
        raise ValueError('Pool fingerprint differs')
    submissions = tuple(PdfWorksheetSubmission(
        content=path.read_bytes(), sha256=ref.sha256, reviewer=ref.reviewer,
    ) for path, ref in zip(sources, manifest.worksheets))
    # Reject changed human input bytes before extracting any PDFs.
    if any(sha256(item.content).hexdigest() != item.sha256
           for item in submissions):
        raise ValueError('Worksheet file SHA256 differs')
    pinned_runs = load_pdf_pooling_runs(Path(runs))
    root = Path(library_root)
    for pin in dataset.library.books:
        source = _resolve_book_path(root, pin.book_path)
        if sha256_file(source) != pin.pdf_sha256:
            raise ValueError('Pinned PDF hash differs')
    corpus = load_pdf_library_corpus(root, dataset.library.book_paths,
                                     pinned_runs[0].run.manifest.policy)
    pages = {book: prepare_pdf_reading_document(root, book).reading_pages
             for book in dataset.library.book_paths}
    judgments = collect_pdf_multibook_judgments(
        golden=dataset, golden_sha256=sha256(raw).hexdigest(), corpus=corpus,
        pool=pool, runs=pinned_runs, reading_pages=pages,
        submissions=submissions,
    )
    scores = score_pdf_multibook_retrieval(
        golden=dataset, golden_sha256=sha256(raw).hexdigest(), corpus=corpus,
        pool=pool, runs=pinned_runs, reading_pages=pages, judgments=judgments,
    ) if evaluate else ()
    hashes = {'golden': sha256(raw).hexdigest(), 'pool': manifest.pool.sha256,
              'judgment_inputs': sha256(manifest_raw).hexdigest(),
              **{f'worksheet:{index}': item.sha256
                 for index, item in enumerate(submissions)}}
    return _EvaluationInput(pool, judgments, scores, hashes)


def pdf_collect_multibook_judgments(
    golden: str, library_root: str, runs: str, inputs: str, output: str,
) -> str:
    """Collect exact Slice 3 CSV bindings as a new draft artifact.

    inputs pins pool path/hash/fingerprint and CSV paths/hashes/reviewers.
    Blank labels are retained. No source file is rewritten or label migrated.
    """
    return _execute(golden, library_root, runs, inputs, output, evaluate=False)


def pdf_evaluate_multibook_retrieval(
    golden: str, library_root: str, runs: str, inputs: str, output: str,
) -> str:
    """Recollect pinned human CSVs and score complete top-5/top-10 bindings.

    inputs is the same read-only manifest used by the collector. Final metrics
    are blocked on any unjudged contributing record/unit/interpretation.
    """
    return _execute(golden, library_root, runs, inputs, output, evaluate=True)


def _execute(
    golden: str, library_root: str, runs: str, inputs: str, output: str,
    *, evaluate: bool,
) -> str:
    target = Path(output)
    if os.path.lexists(target) or not target.parent.is_dir():
        raise CliError('Output must be a new file with an existing parent.')
    try:
        validated = _inputs(golden, library_root, runs, inputs,
                            evaluate=evaluate)
        artifact = build_multibook_metric_report(
            validated.scores, validated.pool, validated.judgments,
            validated.hashes,
        ) if evaluate else validated.judgments
        payload = (artifact.model_dump_json(indent=1) + '\n').encode('utf-8')
        _write_new_artifact(target, payload)
    except PdfMissingJudgments as error:
        # Exact binding identifiers are requested diagnostics, never book text.
        raise CliError(str(error)) from None
    except (OSError, ValueError, TypeError, PyPdfError):
        raise CliError(
            'Invalid or incompatible evaluation inputs, labels, hashes or '
            'output; no final artifact was published.'
        ) from None
    return (f'Written {artifact.schema_name}; '
            f'fingerprint {artifact.fingerprint}; '
            f'artifact SHA256 {sha256(payload).hexdigest()}')


def _write_new_artifact(target: Path, payload: bytes) -> None:
    """Publish closed complete bytes exclusively, as in the Slice 2 runner."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb', dir=target.parent, prefix='.fredl-eval-', delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
