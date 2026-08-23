"""Discover indexable files in a corpus safely and deterministically."""

import codecs
import os
from enum import Enum
from pathlib import Path
from stat import S_ISREG

from pydantic import BaseModel


DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
BINARY_SAMPLE_SIZE_BYTES = 8192


class FileKind(str, Enum):
    """Identify the chunking strategy required for a corpus file."""

    PYTHON = "python"
    TEXT = "text"


class CorpusFile(BaseModel):
    """Describe one portable, project-relative corpus file."""

    file_path: str
    kind: FileKind


SUFFIX_TO_KIND: dict[str, FileKind] = {
    ".md": FileKind.TEXT,
    ".py": FileKind.PYTHON,
    ".rst": FileKind.TEXT,
    ".txt": FileKind.TEXT,
}

IGNORED_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {
        "__pycache__",
        "node_modules",
    }
)


def discover_files(
    project_root: Path,
    corpus_root: Path,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> list[CorpusFile]:
    """Return a stable manifest of supported files below a corpus root.

    Args:
        project_root: Root directory used to produce evaluation-facing paths.
        corpus_root: Directory containing the corpus to inspect.
        max_file_size_bytes: Largest file accepted into the manifest.

    Returns:
        Supported corpus files sorted by their project-relative POSIX paths.

    Raises:
        FileNotFoundError: If either supplied path does not exist.
        NotADirectoryError: If either supplied path is not a directory.
        ValueError: If the corpus is outside the project root or the size
            limit is not positive.
    """
    if max_file_size_bytes <= 0:
        message = "Maximum file size must be positive"
        raise ValueError(message)

    resolved_project_root = project_root.resolve(strict=True)
    resolved_corpus_root = corpus_root.resolve(strict=True)

    if not resolved_project_root.is_dir():
        raise NotADirectoryError(str(project_root))
    if not resolved_corpus_root.is_dir():
        raise NotADirectoryError(str(corpus_root))

    if resolved_corpus_root == resolved_project_root:
        message = "Corpus root must be below the project root"
        raise ValueError(message)

    try:
        resolved_corpus_root.relative_to(resolved_project_root)
    except ValueError as error:
        message = "Corpus root must be inside the project root"
        raise ValueError(message) from error

    manifest: list[CorpusFile] = []

    for current_root, directory_names, file_names in os.walk(
        resolved_corpus_root,
        followlinks=False,
    ):
        current_path = Path(current_root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if _is_safe_directory(current_path / name)
        )

        for file_name in sorted(file_names):
            candidate = current_path / file_name
            kind = _file_kind(candidate)
            if kind is None:
                continue

            try:
                file_status = candidate.stat(follow_symlinks=False)
            except OSError:
                continue

            if (
                not S_ISREG(file_status.st_mode)
                or file_status.st_size > max_file_size_bytes
            ):
                continue

            try:
                is_binary = _is_binary(candidate)
            except OSError:
                continue
            if is_binary:
                continue

            relative_path = candidate.relative_to(
                resolved_project_root
            ).as_posix()
            manifest.append(CorpusFile(file_path=relative_path, kind=kind))

    return sorted(manifest, key=lambda item: item.file_path)


def _is_safe_directory(path: Path) -> bool:
    """Return whether a directory is suitable for recursive discovery."""
    return (
        not path.name.startswith(".")
        and path.name not in IGNORED_DIRECTORY_NAMES
        and not path.is_symlink()
    )


def _file_kind(path: Path) -> FileKind | None:
    """Return the chunking kind for a supported regular file."""
    if path.name.startswith("."):
        return None
    return SUFFIX_TO_KIND.get(path.suffix.lower())


def _is_binary(path: Path) -> bool:
    """Return whether a supported file sample is not UTF-8 text."""
    with path.open("rb") as stream:
        sample = stream.read(BINARY_SAMPLE_SIZE_BYTES)

    if b"\x00" in sample:
        return True

    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        decoder.decode(sample, final=False)
    except UnicodeDecodeError:
        return True
    return False
