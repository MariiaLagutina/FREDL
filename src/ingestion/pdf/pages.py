"""Extract text from one physical PDF page with stable provenance."""

from dataclasses import dataclass
from pathlib import Path

import pypdf
from pypdf import PageObject, PdfReader
from pypdf.generic import ContentStream

from src.ingestion.pdf.layout import PdfTextFragment, extract_pdf_text_layout


DRAWING_OPERATORS: frozenset[bytes] = frozenset(
    {
        b"B",
        b"B*",
        b"F",
        b"S",
        b"b",
        b"b*",
        b"c",
        b"f",
        b"f*",
        b"h",
        b"l",
        b"m",
        b"re",
        b"s",
        b"v",
        b"y",
    }
)


@dataclass(frozen=True, slots=True)
class PdfPage:
    """Store text and provenance for one physical page in a PDF file."""

    book_path: str
    page_number: int
    width_points: float
    height_points: float
    image_count: int
    drawing_operator_count: int
    text: str
    text_fragments: tuple[PdfTextFragment, ...]
    extractor: str


def extract_pdf_page(
    library_root: Path,
    book_path: str,
    page_number: int,
) -> PdfPage:
    """Extract one page from a library-relative PDF path."""
    if page_number < 1:
        message = "PDF page number must be at least 1"
        raise ValueError(message)

    resolved_book_path = _resolve_book_path(library_root, book_path)
    reader = PdfReader(resolved_book_path)
    if page_number > len(reader.pages):
        message = "PDF page number exceeds the document length"
        raise ValueError(message)

    page = reader.pages[page_number - 1]
    return _extract_page(reader, book_path, page_number, page)


def extract_pdf_pages(
    library_root: Path,
    book_path: str,
) -> tuple[PdfPage, ...]:
    """Extract every physical page while opening the PDF only once."""
    resolved_book_path = _resolve_book_path(library_root, book_path)
    reader = PdfReader(resolved_book_path)

    return tuple(
        _extract_page(reader, book_path, page_number, page)
        for page_number, page in enumerate(reader.pages, start=1)
    )


def _extract_page(
    reader: PdfReader,
    book_path: str,
    page_number: int,
    page: PageObject,
) -> PdfPage:
    """Convert one already-open PDF page into observable data."""
    text_layout = extract_pdf_text_layout(page)
    drawing_operator_count = _count_drawing_operators(page, reader)

    return PdfPage(
        book_path=book_path,
        page_number=page_number,
        width_points=float(page.mediabox.width),
        height_points=float(page.mediabox.height),
        image_count=len(page.images),
        drawing_operator_count=drawing_operator_count,
        text=text_layout.raw_text,
        text_fragments=text_layout.fragments,
        extractor=f"pypdf-{pypdf.__version__}",
    )


def _resolve_book_path(library_root: Path, book_path: str) -> Path:
    """Resolve one canonical book path inside the configured library."""
    resolved_library_root = library_root.resolve(strict=True)
    if not resolved_library_root.is_dir():
        raise NotADirectoryError(str(library_root))

    resolved_book_path = (resolved_library_root / book_path).resolve(
        strict=True
    )
    try:
        canonical_book_path = resolved_book_path.relative_to(
            resolved_library_root
        ).as_posix()
    except ValueError as error:
        message = "PDF book must be inside the library root"
        raise ValueError(message) from error

    if book_path != canonical_book_path:
        message = "PDF book path must be library-relative canonical POSIX"
        raise ValueError(message)

    return resolved_book_path


def _count_drawing_operators(
    page: PageObject,
    reader: PdfReader,
) -> int:
    """Count PDF path-painting commands without inferring their meaning."""
    contents = page.get_contents()
    if contents is None:
        return 0

    operations = ContentStream(contents, reader).operations
    return sum(
        operator in DRAWING_OPERATORS for _, operator in operations
    )
