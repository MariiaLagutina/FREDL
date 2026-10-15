"""Run the verified PDF pipeline from library file to index records."""

from dataclasses import dataclass
from pathlib import Path

from src.ingestion.pdf.chunker import chunk_pdf_reading_document
from src.ingestion.pdf.headings import match_pdf_headings
from src.ingestion.pdf.index_records import (
    build_pdf_index_records,
    PdfIndexRecord,
)
from src.ingestion.pdf.outline import PdfOutlineEntry, read_pdf_outline
from src.ingestion.pdf.reading_document import prepare_pdf_reading_document
from src.ingestion.pdf.sections import (
    build_pdf_section_intervals,
    map_pdf_section_boundaries,
)


@dataclass(frozen=True, slots=True)
class PdfBookRecords:
    """Hold one book's index records and the outline that classified them."""

    book_path: str
    outline: tuple[PdfOutlineEntry, ...]
    records: tuple[PdfIndexRecord, ...]


def build_pdf_book_records(
    library_root: Path,
    book_path: str,
) -> PdfBookRecords:
    """Build index records with verified sections for one library book."""
    document = prepare_pdf_reading_document(library_root, book_path)
    outline = read_pdf_outline(library_root, book_path)
    matches = match_pdf_headings(
        outline,
        document.reading_pages,
        physical_pages=document.physical_pages,
    )
    intervals = build_pdf_section_intervals(
        map_pdf_section_boundaries(matches, document.reading_flows),
        document.reading_flows,
    )
    chunks = chunk_pdf_reading_document(
        document, section_intervals=intervals,
    )
    return PdfBookRecords(
        book_path=book_path,
        outline=outline,
        records=build_pdf_index_records(chunks),
    )
