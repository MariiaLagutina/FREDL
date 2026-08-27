"""Assignment-compatible command functions for the public CLI."""

from pathlib import Path

from src.ingestion import discover_files
from src.retrieval import run_stored_retrieval, run_stored_search
from src.retrieval.index_store import fingerprint_corpus


DEFAULT_INDEX_PATH = Path("data/processed/bm25-index.json")
DEFAULT_CORPUS_ROOT = Path("data/raw")


def search(
    query: str,
    k: int = 5,
    index_path: str = str(DEFAULT_INDEX_PATH),
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
) -> list[dict[str, object]]:
    """Return the top-k exact source locations for one raw query."""
    _require_positive_k(k)
    root = Path(project_root)
    fingerprint = _current_corpus_fingerprint(root, Path(corpus_root))
    sources = run_stored_search(
        _below_root(root, Path(index_path)),
        fingerprint,
        query,
        k,
    )
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
    _require_positive_k(k)
    root = Path(project_root)
    dataset = _below_root(root, Path(dataset_path))
    output = _below_root(root, Path(save_directory)) / dataset.name
    fingerprint = _current_corpus_fingerprint(root, Path(corpus_root))
    run_stored_retrieval(
        _below_root(root, Path(index_path)),
        fingerprint,
        dataset,
        output,
        k,
    )
    return str(output)


def _current_corpus_fingerprint(
    project_root: Path,
    corpus_root: Path,
) -> str:
    """Identify the currently discovered assignment corpus."""
    resolved_corpus = _below_root(project_root, corpus_root)
    manifest = discover_files(project_root, resolved_corpus)
    return fingerprint_corpus(project_root, manifest)


def _below_root(project_root: Path, path: Path) -> Path:
    """Resolve a relative CLI path from the configured project root."""
    if path.is_absolute():
        return path
    return project_root / path


def _require_positive_k(k: int) -> None:
    """Reject an invalid limit before corpus or index I/O begins."""
    if k <= 0:
        raise ValueError("Search k must be greater than zero")
