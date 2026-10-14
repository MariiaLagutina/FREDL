"""Map verified PDF headings into continuous reading-flow coordinates."""

from dataclasses import dataclass
from enum import Enum

from src.ingestion.pdf.headings import PdfHeadingMatch, PdfHeadingStatus
from src.ingestion.pdf.reading_flow import PdfReadingFlow


UNRESOLVED_HEADING_STATUSES: frozenset[PdfHeadingStatus] = frozenset(
    {
        PdfHeadingStatus.NOT_FOUND,
        PdfHeadingStatus.AMBIGUOUS,
        PdfHeadingStatus.PAGE_UNAVAILABLE,
    }
)


class PdfSectionResolution(str, Enum):
    """Describe the evidence behind a span's section attribution.

    VERIFIED spans start at a heading confirmed on the page. UNRESOLVED
    spans follow an outline heading that could not be confirmed; only the
    ancestors shared with the last verified heading remain verified.
    NONE spans have no section evidence, such as text before every heading.
    """

    VERIFIED = "verified"
    UNRESOLVED = "unresolved"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class PdfSectionBoundary:
    """Locate one section start in page and flow coordinates.

    section_path is always verified. For an unresolved boundary it holds
    only the verified ancestors, the outline path is kept separately as an
    unverified hint, and the boundary has no heading text of its own.
    """

    book_path: str
    title: str
    section_path: tuple[str, ...]
    page_number: int
    page_source_start: int
    page_source_end: int
    flow_index: int
    flow_start: int
    flow_end: int
    resolution: PdfSectionResolution = PdfSectionResolution.VERIFIED
    outline_hint: tuple[str, ...] = ()
    unresolved_reason: PdfHeadingStatus | None = None


@dataclass(frozen=True, slots=True)
class PdfSectionInterval:
    """Describe content from one section start to the next."""

    book_path: str
    title: str
    section_path: tuple[str, ...]
    start_page_number: int
    start_flow_index: int
    start_flow_offset: int
    heading_end_flow_offset: int
    end_flow_index: int
    end_flow_offset: int
    resolution: PdfSectionResolution = PdfSectionResolution.VERIFIED
    outline_hint: tuple[str, ...] = ()
    unresolved_reason: PdfHeadingStatus | None = None

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
    """Translate heading evidence into ordered reading-flow boundaries.

    Matches must be in outline order. A heading that could not be verified
    on a known page ends the previous verified section at that page instead
    of letting the previous section path continue across it.
    """
    page_locations = _index_flow_pages(flows)
    boundaries = []
    last_verified: PdfSectionBoundary | None = None

    for match in matches:
        if match.status in UNRESOLVED_HEADING_STATUSES:
            unresolved = _unresolved_boundary(
                match, flows, page_locations, last_verified,
            )
            if unresolved is not None:
                boundaries.append(unresolved)
            continue
        if match.status is not PdfHeadingStatus.MATCHED:
            continue
        page_number = match.heading_page_number
        if (
            page_number is None
            or match.source_start is None
            or match.source_end is None
        ):
            message = "Matched PDF heading must contain exact page coordinates"
            raise ValueError(message)

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
        last_verified = PdfSectionBoundary(
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
        boundaries.append(last_verified)

    return _without_superseded_unresolved(
        tuple(
            sorted(
                boundaries,
                key=lambda boundary: (
                    boundary.flow_index,
                    boundary.flow_start,
                    boundary.resolution
                    is not PdfSectionResolution.VERIFIED,
                    boundary.flow_end,
                ),
            )
        ),
    )


def _unresolved_boundary(
    match: PdfHeadingMatch,
    flows: tuple[PdfReadingFlow, ...],
    page_locations: dict[int, "_FlowPageLocation"],
    last_verified: PdfSectionBoundary | None,
) -> PdfSectionBoundary | None:
    """Close the previous section where an unverified heading may start.

    With a destination page, the boundary is placed at the start of that
    text page, or right after the previous verified heading when that
    heading is on the same page. Without a usable destination the heading
    may start anywhere after the previous verified heading, so the boundary
    follows that heading directly.
    """
    destination = match.entry.page_number
    page_number = (
        next(
            (
                number for number in sorted(page_locations)
                if number >= destination
            ),
            None,
        )
        if destination is not None
        else None
    )
    if page_number is None:
        position = _position_after(last_verified, flows)
    else:
        position = _destination_position(
            page_locations[page_number], page_number, flows, last_verified,
        )
    if position is None:
        return None
    flow_index, flow_offset, boundary_page, source_offset = position
    verified_path = _common_prefix(
        last_verified.section_path if last_verified else (),
        match.entry.section_path,
    )
    return PdfSectionBoundary(
        book_path=flows[flow_index].book_path,
        title=match.entry.title,
        section_path=verified_path,
        page_number=boundary_page,
        page_source_start=source_offset,
        page_source_end=source_offset,
        flow_index=flow_index,
        flow_start=flow_offset,
        flow_end=flow_offset,
        resolution=PdfSectionResolution.UNRESOLVED,
        outline_hint=match.entry.section_path,
        unresolved_reason=match.status,
    )


def _destination_position(
    location: "_FlowPageLocation",
    page_number: int,
    flows: tuple[PdfReadingFlow, ...],
    last_verified: PdfSectionBoundary | None,
) -> tuple[int, int, int, int] | None:
    """Return the start of a destination page, after any earlier heading."""
    if last_verified is not None and (
        last_verified.flow_index > location.flow_index
    ):
        return None
    flow_offset = location.flow_start
    if (
        last_verified is not None
        and last_verified.flow_index == location.flow_index
        and last_verified.flow_end > flow_offset
    ):
        flow_offset = last_verified.flow_end
    if flow_offset >= len(flows[location.flow_index].text):
        return None
    source_offset = location.source_start + flow_offset - location.flow_start
    if source_offset > location.source_end:
        # The previous verified heading ends on a later page; the order of
        # the outline and the pages disagree, so no boundary is invented.
        return None
    return location.flow_index, flow_offset, page_number, source_offset


def _position_after(
    last_verified: PdfSectionBoundary | None,
    flows: tuple[PdfReadingFlow, ...],
) -> tuple[int, int, int, int] | None:
    """Return the first reading position after the last verified heading."""
    if last_verified is None:
        return None
    flow_index = last_verified.flow_index
    flow_offset = last_verified.flow_end
    if flow_offset >= len(flows[flow_index].text):
        if flow_index + 1 >= len(flows):
            return None
        flow_index, flow_offset = flow_index + 1, 0
    for span in flows[flow_index].page_spans:
        if span.flow_start <= flow_offset < span.flow_end:
            source_offset = span.source_start + flow_offset - span.flow_start
            return flow_index, flow_offset, span.page_number, source_offset
    # The offset is a page-join space; the next page starts the position.
    following = next(
        span for span in flows[flow_index].page_spans
        if span.flow_start > flow_offset
    )
    return (
        flow_index,
        following.flow_start,
        following.page_number,
        following.source_start,
    )


def _common_prefix(
    first: tuple[str, ...],
    second: tuple[str, ...],
) -> tuple[str, ...]:
    """Return outline ancestors shared by two section paths.

    The outline lists sections depth-first in reading order, so text
    between a verified heading and a later unverified sibling still lies
    inside every ancestor the two paths share.
    """
    prefix = []
    for left, right in zip(first, second):
        if left != right:
            break
        prefix.append(left)
    return tuple(prefix)


def _without_superseded_unresolved(
    boundaries: tuple[PdfSectionBoundary, ...],
) -> tuple[PdfSectionBoundary, ...]:
    """Drop unresolved boundaries that coincide with stronger evidence.

    Verified boundaries sort first at a shared position. An unresolved
    boundary is also dropped when it would precede the
    verified boundary it was derived after, which only happens when the
    outline order disagrees with page order.
    """
    kept: list[PdfSectionBoundary] = []
    for boundary in boundaries:
        if boundary.resolution is PdfSectionResolution.UNRESOLVED and kept:
            previous = kept[-1]
            if previous.flow_index == boundary.flow_index and (
                previous.flow_start == boundary.flow_start
                or previous.flow_end > boundary.flow_start
            ):
                continue
        kept.append(boundary)
    return tuple(kept)


def build_pdf_section_intervals(
    boundaries: tuple[PdfSectionBoundary, ...],
    flows: tuple[PdfReadingFlow, ...],
) -> tuple[PdfSectionInterval, ...]:
    """Build half-open section ranges from section boundaries."""
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
                resolution=boundary.resolution,
                outline_hint=boundary.outline_hint,
                unresolved_reason=boundary.unresolved_reason,
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
        # Only an unresolved boundary lacks verified heading text.
        heading_length_valid = (
            boundary.flow_start < boundary.flow_end
            if boundary.resolution is PdfSectionResolution.VERIFIED
            else boundary.flow_start == boundary.flow_end
        )
        if not (
            0 <= boundary.flow_start < len(flow.text)
            and boundary.flow_end <= len(flow.text)
            and heading_length_valid
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
