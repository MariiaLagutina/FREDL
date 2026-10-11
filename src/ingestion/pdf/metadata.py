"""Read book identity without copying cover artwork or page content."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True, slots=True)
class PdfBookIdentity:
    """Store metadata suitable for a generated FREDL book card."""

    title: str | None
    author: str | None
    page_count: int


def read_pdf_identity(path: Path) -> PdfBookIdentity:
    """Read identity metadata without extracting page text or artwork."""
    reader = PdfReader(path)
    metadata = reader.metadata

    return PdfBookIdentity(
        title=_clean_metadata_value(metadata.title if metadata else None),
        author=_clean_metadata_value(metadata.author if metadata else None),
        page_count=len(reader.pages),
    )


def _clean_metadata_value(value: str | None) -> str | None:
    """Return normalized metadata text or None for an empty value."""
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None
