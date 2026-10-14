"""Read embedded PDF navigation without inferring chapter boundaries."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast, TypeAlias

from pypdf import PdfReader
from pypdf.generic import Destination

from src.ingestion.pdf.pages import _resolve_book_path


PdfOutlineItem: TypeAlias = Destination | list["PdfOutlineItem"]


@dataclass(frozen=True, slots=True)
class PdfOutlineEntry:
    """Preserve a bookmark's hierarchy and optional physical destination.

    destination_top is the declared vertical scroll target in PDF points,
    when the destination type provides one.
    """

    title: str
    section_path: tuple[str, ...]
    page_number: int | None
    destination_top: float | None = None


def read_pdf_outline(
    library_root: Path,
    book_path: str,
) -> tuple[PdfOutlineEntry, ...]:
    """Read bookmarks in document order from a canonical library path.

    Destinations identify pages, not exact section text boundaries. Missing
    or unresolved destinations remain explicit instead of being guessed.
    """
    reader = PdfReader(_resolve_book_path(library_root, book_path))
    entries: list[PdfOutlineEntry] = []

    def visit(
        items: list[PdfOutlineItem],
        parent_path: tuple[str, ...],
    ) -> None:
        previous_path = parent_path
        for item in items:
            if isinstance(item, list):
                visit(item, previous_path)
                continue
            title = item.title or ""
            section_path = (*parent_path, title)
            page_index = reader.get_destination_page_number(item)
            page_number = (
                page_index + 1
                if page_index is not None
                and 0 <= page_index < len(reader.pages)
                else None
            )
            top = item.top if page_number is not None else None
            entries.append(PdfOutlineEntry(
                title,
                section_path,
                page_number,
                float(top) if top is not None else None,
            ))
            previous_path = section_path

    visit(cast(list[PdfOutlineItem], reader.outline), ())
    return tuple(entries)
