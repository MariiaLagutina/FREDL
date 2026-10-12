"""Map verified PDF headings into continuous reading-flow coordinates."""

from dataclasses import dataclass

from src.ingestion.pdf.headings import PdfHeadingMatch, PdfHeadingStatus
from src.ingestion.pdf.reading_flow import PdfReadingFlow


@dataclass(frozen=True, slots=True)
class PdfSectionBoundary:
    """Locate one verified section heading in page and flow coordinates."""

    title: str
    section_path: tuple[str, ...]
    page_number: int
    page_source_start: int
    page_source_end: int
    flow_index: int
    flow_start: int
    flow_end: int


@dataclass(frozen=True, slots=True)
class _FlowPageLocation:
    flow_index: int
    flow_start: int
    flow_end: int
    source_start: int
    source_end: int


def map_pdf_section_boundaries(
    matches: tuple[PdfHeadingMatch, ...],
    flows: tuple[PdfReadingFlow, ...],
) -> tuple[PdfSectionBoundary, ...]:
    """Translate verified page ranges into reading-flow ranges."""
    page_locations = _index_flow_pages(flows)
    boundaries = []

    for match in matches:
        if match.status is not PdfHeadingStatus.MATCHED:
            continue
        if (
            match.entry.page_number is None
            or match.source_start is None
            or match.source_end is None
        ):
            message = "Matched PDF heading must contain exact page coordinates"
            raise ValueError(message)

        page_number = match.entry.page_number
        location = page_locations.get(page_number)
        if location is None:
            message = "Matched PDF heading page must belong to a reading flow"
            raise ValueError(message)
        if (
            match.source_start < location.source_start
            or match.source_end > location.source_end
            or match.source_end <= match.source_start
        ):
            message = "PDF heading range must be inside its reading page span"
            raise ValueError(message)

        relative_start = match.source_start - location.source_start
        relative_end = match.source_end - location.source_start
        boundaries.append(
            PdfSectionBoundary(
                title=match.entry.title,
                section_path=match.entry.section_path,
                page_number=page_number,
                page_source_start=match.source_start,
                page_source_end=match.source_end,
                flow_index=location.flow_index,
                flow_start=location.flow_start + relative_start,
                flow_end=location.flow_start + relative_end,
            )
        )

    return tuple(
        sorted(
            boundaries,
            key=lambda boundary: (
                boundary.flow_index,
                boundary.flow_start,
                boundary.flow_end,
            ),
        )
    )


def _index_flow_pages(
    flows: tuple[PdfReadingFlow, ...],
) -> dict[int, _FlowPageLocation]:
    page_locations = {}
    book_paths = {flow.book_path for flow in flows}
    if len(book_paths) > 1:
        message = "PDF section boundaries require flows from one book"
        raise ValueError(message)

    for flow_index, flow in enumerate(flows):
        for span in flow.page_spans:
            if span.page_number in page_locations:
                message = "PDF page may belong to only one reading flow"
                raise ValueError(message)
            page_locations[span.page_number] = _FlowPageLocation(
                flow_index=flow_index,
                flow_start=span.flow_start,
                flow_end=span.flow_end,
                source_start=span.source_start,
                source_end=span.source_end,
            )
    return page_locations
