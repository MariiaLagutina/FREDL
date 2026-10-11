"""PDF-specific ingestion utilities."""

from src.ingestion.pdf.assessment import (
    assess_pdf_page,
    PdfPageAssessment,
    PdfPageContentKind,
)
from src.ingestion.pdf.book_card import BookCard, build_book_card
from src.ingestion.pdf.chunks import (
    make_pdf_chunk,
    PdfChunk,
    PdfChunkPageJoin,
    PdfChunkPageSpan,
)
from src.ingestion.pdf.chunker import chunk_pdf_reading_document
from src.ingestion.pdf.layout import (
    extract_pdf_text_layout,
    PdfTextFragment,
    PdfTextLayout,
)
from src.ingestion.pdf.metadata import PdfBookIdentity, read_pdf_identity
from src.ingestion.pdf.margins import (
    find_margin_fragment_indexes,
    find_repeated_margin_patterns,
    PdfMarginLocation,
    PdfMarginPattern,
)
from src.ingestion.pdf.pages import (
    extract_pdf_page,
    extract_pdf_pages,
    PdfPage,
)
from src.ingestion.pdf.planning import PdfBookPagePlan, plan_pdf_pages
from src.ingestion.pdf.reading_text import (
    build_pdf_reading_pages,
    PdfReadingPage,
)
from src.ingestion.pdf.reading_flow import (
    build_pdf_reading_flows,
    PdfPageJoin,
    PdfPageJoinKind,
    PdfReadingFlow,
    PdfReadingPageSpan,
)
from src.ingestion.pdf.reading_document import (
    PdfReadingDocument,
    prepare_pdf_reading_document,
)
from src.ingestion.pdf.text_flow import (
    build_pdf_text_flows,
    PdfPageTextSpan,
    PdfTextFlow,
)

__all__ = [
    "BookCard",
    "PdfBookIdentity",
    "PdfBookPagePlan",
    "PdfOutlineEntry",
    "read_pdf_outline",
    "PdfChunk",
    "PdfChunkPageJoin",
    "PdfChunkPageSpan",
    "PdfMarginLocation",
    "PdfMarginPattern",
    "PdfPage",
    "PdfPageAssessment",
    "PdfPageContentKind",
    "PdfPageJoin",
    "PdfPageJoinKind",
    "PdfPageTextSpan",
    "PdfReadingPage",
    "PdfReadingFlow",
    "PdfReadingDocument",
    "PdfReadingPageSpan",
    "PdfTextFragment",
    "PdfTextFlow",
    "PdfTextLayout",
    "assess_pdf_page",
    "build_book_card",
    "build_pdf_text_flows",
    "chunk_pdf_reading_document",
    "build_pdf_reading_pages",
    "build_pdf_reading_flows",
    "extract_pdf_page",
    "extract_pdf_pages",
    "extract_pdf_text_layout",
    "find_margin_fragment_indexes",
    "find_repeated_margin_patterns",
    "make_pdf_chunk",
    "plan_pdf_pages",
    "prepare_pdf_reading_document",
    "read_pdf_identity",
]

from src.ingestion.pdf.outline import PdfOutlineEntry, read_pdf_outline
