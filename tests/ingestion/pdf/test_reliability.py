"""Verify that unreliable PDF structures are reported, not trusted."""

from src.ingestion.pdf.chunker import chunk_pdf_reading_document
from src.ingestion.pdf.layout import PdfTextFragment
from src.ingestion.pdf.pages import PdfPage
from src.ingestion.pdf.reading_document import PdfReadingDocument
from src.ingestion.pdf.reading_flow import build_pdf_reading_flows
from src.ingestion.pdf.reading_text import build_pdf_reading_pages
from src.ingestion.pdf.reliability import (
    assess_pdf_reliability,
    PdfExtractionSupport,
    PdfReliabilityReport,
    PdfStructureKind,
)


BOOK = "books/test.pdf"
BODY = 10.0
MONO = 0.6
Part = PdfTextFragment | str


def _fragment(
    text: str,
    page_y: float,
    *,
    size: float = BODY,
    page_x: float = 40,
    advance: float | None = None,
) -> PdfTextFragment:
    return PdfTextFragment(
        text, 0, 0, 0, 0, page_x, page_y, size, uniform_advance=advance,
    )


def _code(text: str, page_y: float, page_x: float = 40) -> PdfTextFragment:
    return _fragment(text, page_y, page_x=page_x, advance=MONO)


def _page(
    page_number: int,
    *parts: Part,
    images: int = 0,
    drawings: int = 0,
) -> PdfPage:
    text = "".join(
        part if isinstance(part, str) else part.text for part in parts
    )
    return PdfPage(
        BOOK, page_number, 300, 400, images, drawings, text,
        tuple(part for part in parts if not isinstance(part, str)),
        "test",
    )


def _prose(page_number: int) -> PdfPage:
    parts: list[Part] = []
    for index in range(6):
        parts.extend((_fragment("Plain body prose.", 300 - index * 12), "\n"))
    return _page(page_number, *parts)


def _report(*pages: PdfPage, chunk: bool = False) -> PdfReliabilityReport:
    reading = build_pdf_reading_pages(pages, ())
    document = PdfReadingDocument(
        BOOK, pages, (), reading, build_pdf_reading_flows(reading),
    )
    chunks = (
        chunk_pdf_reading_document(
            document, max_chunk_size=40, min_chunk_size=20,
        )
        if chunk else ()
    )
    return assess_pdf_reliability(document, (), chunks)


def _kinds(report: PdfReliabilityReport) -> set[tuple[str, str]]:
    return {
        (finding.kind.value, finding.support.value)
        for finding in report.findings
    }


def test_plain_prose_has_no_findings() -> None:
    assert _report(_prose(1), _prose(2)).findings == ()


def test_code_within_page_is_verified() -> None:
    page = _page(1, _code("x = 1", 300), "\n", _code("y = 2", 288))

    report = _report(page, _prose(2))

    assert _kinds(report) == {("code", "verified")}


def test_code_beyond_page_edge_degrades_all_code_in_the_book() -> None:
    wide = _page(1, _code("result = model(" + "a" * 40 + ")", 300))
    narrow = _page(2, _code("x = 1", 300))

    report = _report(wide, narrow, _prose(3))

    assert report.count(
        PdfStructureKind.CODE_OVERFLOW,
        PdfExtractionSupport.SOURCE_CORRUPTION,
    ) == 1
    assert report.count(
        PdfStructureKind.CODE, PdfExtractionSupport.DEGRADED,
    ) == 2
    assert report.count(support=PdfExtractionSupport.VERIFIED) == 0


def test_table_caption_is_degraded_but_reference_is_ignored() -> None:
    caption = _page(
        1,
        _fragment("Table 2-4. Latency by region", 300, size=8), "\n",
        _fragment("Region Latency", 288),
    )
    reference = _page(
        2,
        _fragment("as summarized in", 300), "\n",
        _fragment("Table 2-4.", 288),
    )

    report = _report(caption, reference, _prose(3))

    assert _kinds(report) == {("table", "degraded")}
    assert report.findings[0].page_number == 1


def test_lowered_math_with_equation_is_unsupported() -> None:
    page = _page(
        1,
        _fragment("y", 300, size=14), _fragment("i", 296, size=10),
        _fragment(" = w x", 300, size=14), _fragment("i", 296, size=10),
        _fragment(" + b", 300, size=14),
    )

    report = _report(page, _prose(2), _prose(3))

    assert _kinds(report) == {("formula", "unsupported")}


def test_subscript_without_equation_is_not_a_formula() -> None:
    page = _page(
        1,
        _fragment("H", 300, size=14), _fragment("2", 296, size=10),
        _fragment("O is water", 300, size=14),
    )

    assert _report(page, _prose(2), _prose(3)).count(
        PdfStructureKind.FORMULA,
    ) == 0


def test_raised_small_digit_after_body_is_a_footnote_marker() -> None:
    page = _page(
        1,
        _fragment("This claim needs a source.", 300), "\n",
        _fragment("1", 304, size=6),
    )

    assert _kinds(_report(page, _prose(2))) == {
        ("footnote_marker", "degraded"),
    }


def test_one_glyph_per_line_text_is_source_corruption() -> None:
    letters = "Sample caption set one glyph per line"
    parts: list[Part] = []
    for index, letter in enumerate(letters.replace(" ", "")):
        parts.extend((_fragment(letter, 380 - index * 5), "\n"))

    report = _report(_page(1, *parts), _prose(2))

    assert ("glyph_per_line_text", "source_corruption") in _kinds(report)


def test_visual_pages_and_images_are_unsupported() -> None:
    image_only = _page(1, images=1)
    vector_only = _page(2, drawings=12)
    figure = _page(3, _fragment("Figure 1-1. A chart", 300), images=1)

    report = _report(image_only, vector_only, figure, _prose(4))

    assert report.count(
        PdfStructureKind.VISUAL_ONLY_PAGE, PdfExtractionSupport.UNSUPPORTED,
    ) == 2
    assert report.count(
        PdfStructureKind.FIGURE_IMAGE, PdfExtractionSupport.UNSUPPORTED,
    ) == 1


def test_chunk_boundary_inside_code_block_is_reported() -> None:
    code_lines: list[Part] = []
    for index in range(6):
        code_lines.extend((_code(f"call_{index}(value)", 300 - index * 12),
                           "\n"))
    page = _page(1, _fragment("Run this:", 320), "\n", *code_lines)

    report = _report(page, _prose(2), chunk=True)

    splits = [
        finding for finding in report.findings
        if finding.kind is PdfStructureKind.CODE_SPLIT_BY_CHUNK
    ]
    assert splits
    assert all(finding.chunk_id for finding in splits)
    assert all(
        finding.support is PdfExtractionSupport.DEGRADED for finding in splits
    )


def test_chunk_boundary_between_prose_is_not_a_code_split() -> None:
    report = _report(_prose(1), _prose(2), chunk=True)

    assert report.count(PdfStructureKind.CODE_SPLIT_BY_CHUNK) == 0
