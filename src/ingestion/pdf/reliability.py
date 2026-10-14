"""Report PDF structures whose extracted text is not fully reliable."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
import re
import statistics

from src.ingestion.pdf.assessment import assess_pdf_page, PdfPageContentKind
from src.ingestion.pdf.chunks import PdfChunk
from src.ingestion.pdf.headings import PdfHeadingMatch, PdfHeadingStatus
from src.ingestion.pdf.layout import PdfTextFragment
from src.ingestion.pdf.margins import find_margin_fragment_indexes
from src.ingestion.pdf.pages import PdfPage
from src.ingestion.pdf.reading_document import PdfReadingDocument
from src.ingestion.pdf.reading_text import PdfReadingPage


TABLE_LABEL = re.compile(r"\s*Table [0-9]+[-.][0-9]+\.")
CAPTION_SIZE_TOLERANCE = 0.05
MATH_SYMBOLS = re.compile(r"[∑∏∫√∂∇]")
FOOTNOTE_MARKER = re.compile(r"[0-9]{1,3}")
SMALL_FONT_RATIO = 0.8
GLYPH_LINE_MIN_LINES = 20
GLYPH_LINE_RATIO = 0.6


class PdfStructureKind(str, Enum):
    """Name an observable structure that affects extraction reliability."""

    CODE = "code"
    CODE_OVERFLOW = "code_overflow"
    CODE_SPLIT_BY_CHUNK = "code_split_by_chunk"
    TABLE = "table"
    FORMULA = "formula"
    FIGURE_IMAGE = "figure_image"
    VISUAL_ONLY_PAGE = "visual_only_page"
    FOOTNOTE_MARKER = "footnote_marker"
    GLYPH_PER_LINE_TEXT = "glyph_per_line_text"


class PdfExtractionSupport(str, Enum):
    """Describe how far extracted text can be trusted for a structure.

    VERIFIED text matches the page; DEGRADED text is present but loses
    structure or may be incomplete; UNSUPPORTED content is not extracted;
    SOURCE_CORRUPTION is a defect already visible in the PDF itself.
    """

    VERIFIED = "verified"
    DEGRADED = "degraded"
    UNSUPPORTED = "unsupported"
    SOURCE_CORRUPTION = "source_corruption"


@dataclass(frozen=True, slots=True)
class PdfStructureFinding:
    """Record one structure observation on one physical page."""

    kind: PdfStructureKind
    support: PdfExtractionSupport
    page_number: int
    detail: str
    chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class PdfReliabilityReport:
    """Summarize structure findings and section evidence for one book."""

    book_path: str
    findings: tuple[PdfStructureFinding, ...]
    heading_statuses: dict[str, int]
    heading_page_sources: dict[str, int]
    chunk_section_resolutions: dict[str, int]

    def count(
        self,
        kind: PdfStructureKind | None = None,
        support: PdfExtractionSupport | None = None,
    ) -> int:
        """Count findings filtered by structure kind and support level."""
        return sum(
            1 for finding in self.findings
            if (kind is None or finding.kind is kind)
            and (support is None or finding.support is support)
        )


def assess_pdf_reliability(
    document: PdfReadingDocument,
    matches: tuple[PdfHeadingMatch, ...] = (),
    chunks: tuple[PdfChunk, ...] = (),
) -> PdfReliabilityReport:
    """Report unsupported and degraded structures without changing text."""
    pages = document.physical_pages
    body_size = _body_size(pages)
    overflow = {
        page.page_number: _overflowing_code_lines(page) for page in pages
    }
    code_may_be_clipped = any(overflow.values())
    findings: list[PdfStructureFinding] = []
    for page, reading in zip(pages, document.reading_pages):
        findings.extend(_page_findings(
            page,
            reading,
            body_size,
            overflow[page.page_number],
            code_may_be_clipped,
            document,
        ))
    findings.extend(_code_split_findings(document, chunks))
    return PdfReliabilityReport(
        book_path=document.book_path,
        findings=tuple(sorted(
            findings,
            key=lambda item: (
                item.page_number, item.kind.value, item.chunk_id or "",
            ),
        )),
        heading_statuses=_counts(match.status.value for match in matches),
        heading_page_sources=_counts(
            match.page_source.value for match in matches
            if match.status is PdfHeadingStatus.MATCHED
        ),
        chunk_section_resolutions=_counts(
            chunk.section_resolution.value for chunk in chunks
        ),
    )


def _page_findings(
    page: PdfPage,
    reading: PdfReadingPage,
    body_size: float | None,
    overflow_lines: int,
    code_may_be_clipped: bool,
    document: PdfReadingDocument,
) -> list[PdfStructureFinding]:
    findings = []

    def add(
        kind: PdfStructureKind,
        support: PdfExtractionSupport,
        detail: str,
    ) -> None:
        findings.append(
            PdfStructureFinding(kind, support, page.page_number, detail)
        )

    content_kind = assess_pdf_page(page).content_kind
    if content_kind in (
        PdfPageContentKind.IMAGE_ONLY,
        PdfPageContentKind.POSSIBLE_VECTOR_GRAPHICS,
    ):
        add(
            PdfStructureKind.VISUAL_ONLY_PAGE,
            PdfExtractionSupport.UNSUPPORTED,
            "page has no extractable text; visual content needs OCR",
        )
    elif page.image_count:
        add(
            PdfStructureKind.FIGURE_IMAGE,
            PdfExtractionSupport.UNSUPPORTED,
            f"{page.image_count} embedded image(s); image content is not "
            "extracted",
        )

    code_lines = len(_code_lines(page))
    if code_lines:
        add(
            PdfStructureKind.CODE,
            PdfExtractionSupport.DEGRADED if code_may_be_clipped
            else PdfExtractionSupport.VERIFIED,
            f"{code_lines} monospaced line(s)"
            + (
                "; this PDF does not wrap code, so lines may be clipped"
                if code_may_be_clipped else ""
            ),
        )
    if overflow_lines:
        add(
            PdfStructureKind.CODE_OVERFLOW,
            PdfExtractionSupport.SOURCE_CORRUPTION,
            f"{overflow_lines} code line(s) extend beyond the page edge",
        )
    tables = _table_captions(page, body_size)
    if tables:
        add(
            PdfStructureKind.TABLE,
            PdfExtractionSupport.DEGRADED,
            f"{tables} table caption(s); column structure is not preserved",
        )
    formulas = _formula_lines(page)
    if formulas:
        add(
            PdfStructureKind.FORMULA,
            PdfExtractionSupport.UNSUPPORTED,
            f"{formulas} line(s) with raised or lowered math; layout is "
            "flattened",
        )
    markers = _footnote_markers(page, body_size, document)
    if markers:
        add(
            PdfStructureKind.FOOTNOTE_MARKER,
            PdfExtractionSupport.DEGRADED,
            f"{markers} raised footnote marker(s) extracted as plain digits",
        )
    if _is_glyph_per_line(reading.reading_text):
        add(
            PdfStructureKind.GLYPH_PER_LINE_TEXT,
            PdfExtractionSupport.SOURCE_CORRUPTION,
            "most lines hold one or two glyphs; the PDF text is laid out "
            "one character per line",
        )
    return findings


def _body_size(pages: tuple[PdfPage, ...]) -> float | None:
    sizes = [
        fragment.font_size
        for page in pages
        for fragment in page.text_fragments
    ]
    return statistics.median(sizes) if sizes else None


def _code_lines(page: PdfPage) -> list[list[PdfTextFragment]]:
    """Group monospaced fragments by baseline."""
    lines: dict[float, list[PdfTextFragment]] = {}
    for fragment in page.text_fragments:
        if fragment.uniform_advance is not None and fragment.text.strip():
            lines.setdefault(round(fragment.page_y, 1), []).append(fragment)
    return list(lines.values())


def _overflowing_code_lines(page: PdfPage) -> int:
    """Count code lines whose exact advance passes the page edge."""
    count = 0
    for line in _code_lines(page):
        last = max(line, key=lambda fragment: fragment.page_x)
        advance = last.uniform_advance or 0.0
        right = (
            last.page_x
            + len(last.text.rstrip()) * advance * last.font_size
        )
        if right > page.width_points + 1:
            count += 1
    return count


def _visual_lines(page: PdfPage) -> list[list[PdfTextFragment]]:
    """Group consecutive fragments whose baselines stay within one line."""
    lines: list[list[PdfTextFragment]] = []
    for fragment in page.text_fragments:
        if lines:
            line = lines[-1]
            tolerance = max(line[0].font_size, fragment.font_size) * 0.6
            if abs(fragment.page_y - line[0].page_y) <= tolerance:
                line.append(fragment)
                continue
        lines.append([fragment])
    return lines


def _formula_lines(page: PdfPage) -> int:
    """Count lines mixing an equation with raised or lowered small text."""
    count = 0
    for line in _visual_lines(page):
        largest = max(fragment.font_size for fragment in line)
        baseline = statistics.median(
            fragment.page_y for fragment in line
            if fragment.font_size == largest
        )
        text = "".join(fragment.text for fragment in line)
        shifted = any(
            fragment.font_size <= largest * SMALL_FONT_RATIO
            and abs(fragment.page_y - baseline) > 1
            and fragment.uniform_advance is None
            for fragment in line
        )
        if shifted and ("=" in text or MATH_SYMBOLS.search(text)):
            count += 1
    return count


def _footnote_markers(
    page: PdfPage,
    body_size: float | None,
    document: PdfReadingDocument,
) -> int:
    """Count small raised digits that follow body text."""
    if body_size is None:
        return 0
    margins = set(
        find_margin_fragment_indexes(page, document.margin_patterns)
    )
    fragments = page.text_fragments
    return sum(
        1
        for index, fragment in enumerate(fragments)
        if index > 0
        and index not in margins
        and FOOTNOTE_MARKER.fullmatch(fragment.text.strip())
        and fragment.font_size <= body_size * SMALL_FONT_RATIO
        and fragments[index - 1].font_size >= body_size * 0.95
        and fragment.page_y - fragments[index - 1].page_y > 1
    )


def _table_captions(page: PdfPage, body_size: float | None) -> int:
    """Count table labels set apart from body text by their font size.

    A reference such as `as shown in Table 3-1.` can wrap to the start of
    a line, so a label in the body font is not treated as a caption.
    """
    if body_size is None:
        return 0
    return sum(
        1
        for fragment in page.text_fragments
        if TABLE_LABEL.match(fragment.text)
        and abs(fragment.font_size - body_size)
        > body_size * CAPTION_SIZE_TOLERANCE
    )


def _is_glyph_per_line(text: str) -> bool:
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < GLYPH_LINE_MIN_LINES:
        return False
    short = sum(1 for line in lines if len(line.strip()) <= 2)
    return short / len(lines) >= GLYPH_LINE_RATIO


def _code_split_findings(
    document: PdfReadingDocument,
    chunks: tuple[PdfChunk, ...],
) -> list[PdfStructureFinding]:
    """Report chunk boundaries that fall inside a contiguous code block."""
    blocks = {
        page.page_number: _code_blocks(page, reading, document)
        for page, reading in zip(
            document.physical_pages, document.reading_pages,
        )
    }
    findings = []
    ordered = sorted(
        chunks, key=lambda item: (item.flow_index, item.flow_start),
    )
    for left, right in zip(ordered, ordered[1:]):
        if (
            left.flow_index != right.flow_index
            or left.flow_end != right.flow_start
        ):
            continue
        span = right.page_spans[0]
        offset = span.page_source_start
        if any(
            start < offset < end
            for start, end in blocks.get(span.page_number, ())
        ):
            findings.append(PdfStructureFinding(
                PdfStructureKind.CODE_SPLIT_BY_CHUNK,
                PdfExtractionSupport.DEGRADED,
                span.page_number,
                f"boundary after {left.chunk_id} splits a code block",
                chunk_id=right.chunk_id,
            ))
    return findings


def _code_blocks(
    page: PdfPage,
    reading: PdfReadingPage,
    document: PdfReadingDocument,
) -> tuple[tuple[int, int], ...]:
    """Locate runs of consecutive monospaced fragments in reading text.

    Kept fragments are located in order; only whitespace may separate
    them because reading text deletes margins from the raw text. Pages
    whose fragments cannot be located report no blocks.
    """
    margins = set(
        find_margin_fragment_indexes(page, document.margin_patterns)
    )
    blocks: list[tuple[int, int]] = []
    position = 0
    current: tuple[int, int] | None = None
    for index, fragment in enumerate(page.text_fragments):
        if index in margins:
            continue
        start = reading.reading_text.find(fragment.text, position)
        if start < 0 or reading.reading_text[position:start].strip():
            return ()
        end = start + len(fragment.text)
        position = end
        if fragment.uniform_advance is None or not fragment.text.strip():
            if current is not None:
                blocks.append(current)
                current = None
            continue
        current = (current[0], end) if current is not None else (start, end)
    if current is not None:
        blocks.append(current)
    return tuple(blocks)


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))
