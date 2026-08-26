"""Calculate a stable identity for an exact discovered corpus."""

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from src.ingestion import CorpusFile


def fingerprint_corpus(
    project_root: Path,
    corpus_files: Sequence[CorpusFile],
) -> str:
    """Hash canonical relative paths and bytes in deterministic order."""
    resolved_root = project_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(str(project_root))

    paths = [corpus_file.file_path for corpus_file in corpus_files]
    if len(paths) != len(set(paths)):
        raise ValueError("Corpus fingerprint paths must be unique")

    digest = sha256()
    for file_path in sorted(paths):
        source_path = (
            resolved_root / file_path
        ).resolve(strict=True)
        try:
            canonical_path = source_path.relative_to(resolved_root).as_posix()
        except ValueError as error:
            message = "Corpus file must be inside project root"
            raise ValueError(message) from error
        if canonical_path != file_path:
            raise ValueError(
                "Corpus fingerprint path must be canonical and relative"
            )
        digest.update(file_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
