"""Read corpus documents and create chunks with exact source offsets."""

from dataclasses import dataclass
from pathlib import Path

from src.ingestion.files import CorpusFile, FileKind


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """Store one corpus file without changing its source text."""

    file_path: str
    kind: FileKind
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """Store an exact half-open character span from a source document."""

    file_path: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        """Require coordinates to describe the stored text exactly."""
        if self.start < 0:
            message = "Chunk start must not be negative"
            raise ValueError(message)
        if self.end <= self.start:
            message = "Chunk end must be greater than start"
            raise ValueError(message)
        if len(self.text) != self.end - self.start:
            message = "Chunk text length must match its character span"
            raise ValueError(message)


def read_document(
    project_root: Path,
    corpus_file: CorpusFile,
) -> SourceDocument:
    """Read one discovered corpus file without newline normalization.

    Args:
        project_root: Root directory used by the manifest path.
        corpus_file: Discovered file with a project-relative POSIX path.

    Returns:
        The exact decoded source text and its manifest metadata.

    Raises:
        FileNotFoundError: If the project root or source file does not exist.
        NotADirectoryError: If the project root is not a directory.
        ValueError: If the source resolves outside the project root.
        UnicodeDecodeError: If the complete source is not valid UTF-8.
    """
    resolved_project_root = project_root.resolve(strict=True)
    if not resolved_project_root.is_dir():
        raise NotADirectoryError(str(project_root))

    source_path = resolved_project_root / corpus_file.file_path
    resolved_source_path = source_path.resolve(strict=True)

    try:
        canonical_file_path = resolved_source_path.relative_to(
            resolved_project_root
        ).as_posix()
    except ValueError as error:
        message = "Source file must be inside the project root"
        raise ValueError(message) from error

    if corpus_file.file_path != canonical_file_path:
        message = "Source path must be project-relative canonical POSIX"
        raise ValueError(message)

    with resolved_source_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        text = stream.read()

    return SourceDocument(
        file_path=corpus_file.file_path,
        kind=corpus_file.kind,
        text=text,
    )


def make_chunk(
    document: SourceDocument,
    start: int,
    end: int,
) -> Chunk:
    """Create a chunk from an exact half-open document slice."""
    if start < 0:
        message = "Chunk start must not be negative"
        raise ValueError(message)
    if end <= start:
        message = "Chunk end must be greater than start"
        raise ValueError(message)
    if end > len(document.text):
        message = "Chunk end must not exceed document length"
        raise ValueError(message)

    return Chunk(
        file_path=document.file_path,
        start=start,
        end=end,
        text=document.text[start:end],
    )
