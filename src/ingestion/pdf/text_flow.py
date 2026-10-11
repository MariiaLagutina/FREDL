"""Build continuous PDF text flows with exact physical-page provenance."""

from dataclasses import dataclass

from src.ingestion.pdf.pages import PdfPage


PAGE_SEPARATOR = "\n"


@dataclass(frozen=True, slots=True)
class PdfPageTextSpan:
    """Map one page's exact extracted text into a continuous flow."""

    page_number: int
    flow_start: int
    flow_end: int


@dataclass(frozen=True, slots=True)
class PdfTextFlow:
    """Store one uninterrupted run of text-bearing physical pages."""

    book_path: str
    text: str
    page_spans: tuple[PdfPageTextSpan, ...]


def build_pdf_text_flows(
    pages: tuple[PdfPage, ...],
) -> tuple[PdfTextFlow, ...]:
    """Join adjacent text pages and stop at non-text page boundaries."""
    if not pages:
        return ()

    _validate_page_sequence(pages)
    flows: list[PdfTextFlow] = []
    flow_parts: list[str] = []
    page_spans: list[PdfPageTextSpan] = []
    flow_length = 0

    for page in pages:
        if not page.text.strip():
            flow_length = _finish_flow(
                flows,
                flow_parts,
                page_spans,
                page.book_path,
            )
            continue

        if flow_parts:
            flow_parts.append(PAGE_SEPARATOR)
            flow_length += len(PAGE_SEPARATOR)

        flow_start = flow_length
        flow_parts.append(page.text)
        flow_length += len(page.text)
        page_spans.append(
            PdfPageTextSpan(
                page_number=page.page_number,
                flow_start=flow_start,
                flow_end=flow_length,
            )
        )

    _finish_flow(
        flows,
        flow_parts,
        page_spans,
        pages[-1].book_path,
    )
    return tuple(flows)


def _finish_flow(
    flows: list[PdfTextFlow],
    flow_parts: list[str],
    page_spans: list[PdfPageTextSpan],
    book_path: str,
) -> int:
    """Store and reset one accumulated text flow."""
    if flow_parts:
        flows.append(
            PdfTextFlow(
                book_path=book_path,
                text="".join(flow_parts),
                page_spans=tuple(page_spans),
            )
        )
        flow_parts.clear()
        page_spans.clear()
    return 0


def _validate_page_sequence(pages: tuple[PdfPage, ...]) -> None:
    """Require one book in strictly increasing physical page order."""
    book_path = pages[0].book_path
    previous_page_number = 0

    for page in pages:
        if page.book_path != book_path:
            message = "PDF text flow requires pages from one book"
            raise ValueError(message)
        if page.page_number <= previous_page_number:
            message = "PDF pages must be in increasing physical order"
            raise ValueError(message)
        previous_page_number = page.page_number
