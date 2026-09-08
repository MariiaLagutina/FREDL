"""Load only retrieved source files through a safe corpus boundary."""

from collections.abc import Sequence
from pathlib import Path

from src.models import MinimalSource


def load_ranked_source_texts(
    project_root: Path,
    corpus_root: Path,
    ranked_sources: Sequence[MinimalSource],
) -> dict[str, str]:
    """Return exact text for unique retrieved paths inside the corpus root."""
    resolved_project_root = project_root.resolve(strict=True)
    resolved_corpus_root = corpus_root.resolve(strict=True)
    _validate_roots(resolved_project_root, resolved_corpus_root)

    source_texts: dict[str, str] = {}
    for source in ranked_sources:
        if source.file_path in source_texts:
            continue
        source_texts[source.file_path] = _read_source(
            resolved_project_root,
            resolved_corpus_root,
            source.file_path,
        )
    return source_texts


def _validate_roots(project_root: Path, corpus_root: Path) -> None:
    """Require an existing corpus directory strictly below the project root."""
    if not project_root.is_dir():
        raise NotADirectoryError(str(project_root))
    if not corpus_root.is_dir():
        raise NotADirectoryError(str(corpus_root))
    if corpus_root == project_root:
        raise ValueError("Corpus root must be below the project root")
    try:
        corpus_root.relative_to(project_root)
    except ValueError as error:
        message = "Corpus root must be inside the project root"
        raise ValueError(message) from error


def _read_source(
    project_root: Path,
    corpus_root: Path,
    file_path: str,
) -> str:
    """Read one canonical project-relative file without changing newlines."""
    resolved_source_path = (project_root / file_path).resolve(strict=True)
    try:
        resolved_source_path.relative_to(corpus_root)
    except ValueError as error:
        message = "Context source must be inside the corpus root"
        raise ValueError(message) from error

    canonical_file_path = resolved_source_path.relative_to(
        project_root
    ).as_posix()
    if file_path != canonical_file_path:
        raise ValueError(
            "Context source path must be project-relative canonical POSIX"
        )
    if not resolved_source_path.is_file():
        raise ValueError("Context source must be a regular file")

    with resolved_source_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        return stream.read()
