"""Assignment-compatible command functions for the public CLI."""

from pathlib import Path

from src.ingestion import discover_files
from src.retrieval import run_stored_retrieval, run_stored_search
from src.retrieval.index_store import (
    IndexStore,
    PipelineConfig,
    SCHEMA_VERSION,
    build_index,
    fingerprint_corpus,
    fingerprint_pipeline,
)
from src.retrieval.validation import (
    MAX_SOURCE_LENGTH,
    SourceValidationReport,
    validate_retrieval_file,
)


DEFAULT_INDEX_PATH = Path("data/processed/bm25-index.json")
DEFAULT_CORPUS_ROOT = Path("data/raw")


class CliError(Exception):
    """Represent an expected user-facing command failure."""


def index(
    index_path: str = str(DEFAULT_INDEX_PATH),
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
) -> dict[str, object]:
    """Build and save the default production BM25 index."""
    try:
        root = Path(project_root)
        config = PipelineConfig()
        build = build_index(
            root,
            _below_root(root, Path(corpus_root)),
            config,
            index_schema_version=SCHEMA_VERSION,
        )
        output_path = _below_root(root, Path(index_path))
        IndexStore(output_path).save(
            build.index,
            build.corpus_fingerprint,
            build.pipeline_fingerprint,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CliError(_error_message(error)) from None
    return {
        "index_path": str(output_path),
        "schema_version": SCHEMA_VERSION,
        "document_count": len(build.index.documents),
        "corpus_fingerprint": build.corpus_fingerprint,
        "pipeline_fingerprint": build.pipeline_fingerprint,
    }


def search(
    query: str,
    k: int = 5,
    index_path: str = str(DEFAULT_INDEX_PATH),
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
) -> list[dict[str, object]]:
    """Return the top-k exact source locations for one raw query."""
    try:
        _require_positive_k(k)
        root = Path(project_root)
        fingerprint = _current_corpus_fingerprint(root, Path(corpus_root))
        sources = run_stored_search(
            _below_root(root, Path(index_path)),
            fingerprint,
            _current_pipeline_fingerprint(),
            query,
            k,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CliError(_error_message(error)) from None
    return [source.model_dump() for source in sources]


def search_dataset(
    dataset_path: str,
    save_directory: str,
    k: int = 5,
    index_path: str = str(DEFAULT_INDEX_PATH),
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
) -> str:
    """Search one question dataset and save its validated result JSON."""
    try:
        _require_positive_k(k)
        root = Path(project_root)
        dataset = _below_root(root, Path(dataset_path))
        output = _below_root(root, Path(save_directory)) / dataset.name
        fingerprint = _current_corpus_fingerprint(root, Path(corpus_root))
        run_stored_retrieval(
            _below_root(root, Path(index_path)),
            fingerprint,
            _current_pipeline_fingerprint(),
            dataset,
            output,
            k,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CliError(_error_message(error)) from None
    return str(output)


def validate_sources(
    results_path: str,
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
    max_source_length: int = MAX_SOURCE_LENGTH,
) -> dict[str, object]:
    """Audit every source in one retrieval-results JSON file."""
    try:
        root = Path(project_root)
        report = validate_retrieval_file(
            _below_root(root, Path(results_path)),
            root,
            _below_root(root, Path(corpus_root)),
            max_source_length=max_source_length,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CliError(_error_message(error)) from None
    return _validation_report_dict(report)


def _current_corpus_fingerprint(
    project_root: Path,
    corpus_root: Path,
) -> str:
    """Identify the currently discovered assignment corpus."""
    resolved_corpus = _below_root(project_root, corpus_root)
    manifest = discover_files(project_root, resolved_corpus)
    return fingerprint_corpus(project_root, manifest)


def _current_pipeline_fingerprint() -> str:
    """Identify the default production index build pipeline."""
    return fingerprint_pipeline(
        PipelineConfig(),
        index_schema_version=SCHEMA_VERSION,
    )


def _below_root(project_root: Path, path: Path) -> Path:
    """Resolve a relative CLI path from the configured project root."""
    if path.is_absolute():
        return path
    return project_root / path


def _require_positive_k(k: int) -> None:
    """Reject an invalid limit before corpus or index I/O begins."""
    if k <= 0:
        raise ValueError("Search k must be greater than zero")


def _error_message(error: Exception) -> str:
    """Format an expected boundary failure without implementation details."""
    if isinstance(error, FileNotFoundError):
        missing_path = error.filename or str(error)
        return f"File not found: {missing_path}"
    if isinstance(error, NotADirectoryError):
        invalid_path = error.filename or str(error)
        return f"Directory not found: {invalid_path}"
    return str(error) or error.__class__.__name__


def _validation_report_dict(
    report: SourceValidationReport,
) -> dict[str, object]:
    """Convert an internal audit report into terminal-friendly values."""
    return {
        "result_count": report.result_count,
        "source_count": report.source_count,
        "valid_source_count": report.valid_source_count,
        "invalid_source_count": report.invalid_source_count,
        "passed": report.passed,
        "issues": [
            {
                "kind": issue.kind.value,
                "result_index": issue.result_index,
                "source_index": issue.source_index,
                "question_id": issue.question_id,
                "file_path": issue.file_path,
                "detail": issue.detail,
            }
            for issue in report.issues
        ],
    }
