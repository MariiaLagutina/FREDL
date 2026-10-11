"""Extract text from one physical PDF page with stable provenance."""

from dataclasses import dataclass
from pathlib import Path

import pypdf
from pypdf import PdfReader


@dataclass(frozen=True, slots=True)
class PdfPage:
    """Store text and provenance for one physical page in a PDF file."""

    book_path: str
    page_number: int
    width_points: float
    height_points: float
    image_count: int
    text: str
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

    reader = PdfReader(resolved_book_path)
    if page_number > len(reader.pages):
        message = "PDF page number exceeds the document length"
        raise ValueError(message)

    page = reader.pages[page_number - 1]
    text = page.extract_text() or ""

    return PdfPage(
        book_path=book_path,
        page_number=page_number,
        width_points=float(page.mediabox.width),
        height_points=float(page.mediabox.height),
        image_count=len(page.images),
        text=text,
        extractor=f"pypdf-{pypdf.__version__}",
    )
