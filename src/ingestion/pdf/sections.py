"""Map verified PDF headings into continuous reading-flow coordinates."""

from dataclasses import dataclass

from src.ingestion.pdf.headings import PdfHeadingMatch, PdfHeadingStatus
from src.ingestion.pdf.reading_flow import PdfReadingFlow


@dataclass(frozen=True, slots=True)
class PdfSectionBoundary:
    """Locate one verified section heading in page and flow coordinates."""

    book_path: str
    title: str
    section_path: tuple[str, ...]
    page_number: int
    page_source_start: int
    page_source_end: int
    flow_index: int
    flow_start: int
    flow_end: int


@dataclass(frozen=True, slots=True)
class PdfSectionInterval:
    """Describe content from one verified heading to the next."""

    book_path: str
    title: str
    section_path: tuple[str, ...]
    start_page_number: int
    start_flow_index: int
    start_flow_offset: int
    heading_end_flow_offset: int
    end_flow_index: int
    end_flow_offset: int

    @property
    def spans_multiple_flows(self) -> bool:
        """Return whether an empty page divides this section's text."""
        return self.start_flow_index != self.end_flow_index


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
                book_path=flows[location.flow_index].book_path,
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


def build_pdf_section_intervals(
    boundaries: tuple[PdfSectionBoundary, ...],
    flows: tuple[PdfReadingFlow, ...],
) -> tuple[PdfSectionInterval, ...]:
    """Build half-open section ranges from verified heading boundaries."""
    if not boundaries:
        return ()
    if not flows:
        message = "PDF section intervals require reading flows"
        raise ValueError(message)

    book_paths = {flow.book_path for flow in flows}
    if len(book_paths) != 1:
        message = "PDF section intervals require flows from one book"
        raise ValueError(message)
    book_path = flows[0].book_path

    ordered = tuple(
        sorted(
            boundaries,
            key=lambda boundary: (
                boundary.flow_index,
                boundary.flow_start,
                boundary.flow_end,
            ),
        )
    )
    _validate_section_boundaries(ordered, flows, book_path)

    intervals = []
    for index, boundary in enumerate(ordered):
        if index + 1 < len(ordered):
            next_boundary = ordered[index + 1]
            end_flow_index = next_boundary.flow_index
            end_flow_offset = next_boundary.flow_start
        else:
            end_flow_index = len(flows) - 1
            end_flow_offset = len(flows[-1].text)
        intervals.append(
            PdfSectionInterval(
                book_path=boundary.book_path,
                title=boundary.title,
                section_path=boundary.section_path,
                start_page_number=boundary.page_number,
                start_flow_index=boundary.flow_index,
                start_flow_offset=boundary.flow_start,
                heading_end_flow_offset=boundary.flow_end,
                end_flow_index=end_flow_index,
                end_flow_offset=end_flow_offset,
            )
        )
    return tuple(intervals)


def _validate_section_boundaries(
    boundaries: tuple[PdfSectionBoundary, ...],
    flows: tuple[PdfReadingFlow, ...],
    book_path: str,
) -> None:
    previous_coordinate: tuple[int, int] | None = None
    for boundary in boundaries:
        if boundary.book_path != book_path:
            message = "PDF section boundary must match its reading flow book"
            raise ValueError(message)
        if not 0 <= boundary.flow_index < len(flows):
            message = "PDF section boundary flow index is outside the document"
            raise ValueError(message)
        flow = flows[boundary.flow_index]
        if not (
            0 <= boundary.flow_start
            < boundary.flow_end
            <= len(flow.text)
        ):
            message = "PDF section boundary range is outside its reading flow"
            raise ValueError(message)
        coordinate = (boundary.flow_index, boundary.flow_start)
        if (
            previous_coordinate is not None
            and coordinate <= previous_coordinate
        ):
            message = "PDF section boundaries must have unique positions"
            raise ValueError(message)
        previous_coordinate = coordinate


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
