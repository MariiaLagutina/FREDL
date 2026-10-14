"""Tests for auditable PDF reading-text preparation."""

from src.ingestion.pdf.layout import PdfTextFragment
from src.ingestion.pdf.margins import find_repeated_margin_patterns
from src.ingestion.pdf.pages import PdfPage
from src.ingestion.pdf.reading_flow import build_pdf_reading_flows
from src.ingestion.pdf.reading_text import build_pdf_reading_pages


FOOTER_Y = 42.0


def _fragment(
    text: str,
    page_y: float,
    *,
    page_x: float = 40.0,
    font_size: float = 9.0,
    font_name: str | None = None,
) -> PdfTextFragment:
    return PdfTextFragment(
        text=text,
        container_x=0.0,
        container_y=0.0,
        text_x=0.0,
        text_y=0.0,
        page_x=page_x,
        page_y=page_y,
        font_size=font_size,
        font_name=font_name,
    )


def _page(page_number: int) -> PdfPage:
    body = _fragment(f"Body {page_number}.\n", 400.0)
    footer = _fragment(f"{page_number} | Chapter\n", FOOTER_Y)
    raw_text = body.text + footer.text
    return PdfPage(
        book_path="books/example.pdf",
        page_number=page_number,
        width_points=600.0,
        height_points=800.0,
        image_count=0,
        drawing_operator_count=0,
        text=raw_text,
        text_fragments=(body, footer),
        extractor="test",
    )


def _extracted_page(
    page_number: int,
    parts: tuple[str | PdfTextFragment, ...],
) -> PdfPage:
    """Mimic pypdf: separators exist in raw text but not as fragments."""
    raw_text = "".join(
        part if isinstance(part, str) else part.text for part in parts
    )
    return PdfPage(
        book_path="books/example.pdf",
        page_number=page_number,
        width_points=600.0,
        height_points=800.0,
        image_count=0,
        drawing_operator_count=0,
        text=raw_text,
        text_fragments=tuple(
            part for part in parts if isinstance(part, PdfTextFragment)
        ),
        extractor="test",
    )


def _footer(page_number: int) -> PdfTextFragment:
    return _fragment(f"{page_number} | Chapter 1: Basics", FOOTER_Y)


def _other_page(page_number: int) -> tuple[str | PdfTextFragment, ...]:
    return (_fragment("Other page.", 400.0), "\n", _footer(page_number))


def _reading_text(*parts: str | PdfTextFragment) -> str:
    """Return reading text for page 2 of a book with a repeated footer."""
    pages = (
        _extracted_page(1, _other_page(1)),
        _extracted_page(2, parts),
        _extracted_page(3, _other_page(3)),
    )
    reading_pages = build_pdf_reading_pages(pages)
    assert reading_pages[1].removed_margin_signatures
    return reading_pages[1].reading_text


def test_removes_confirmed_margins_and_preserves_raw_text() -> None:
    pages = tuple(_page(page_number) for page_number in range(1, 4))
    patterns = find_repeated_margin_patterns(pages)

    reading_pages = build_pdf_reading_pages(pages, patterns)

    assert tuple(page.raw_text for page in reading_pages) == tuple(
        page.text for page in pages
    )
    assert tuple(page.reading_text for page in reading_pages) == (
        "Body 1.\n",
        "Body 2.\n",
        "Body 3.\n",
    )
    assert all(
        page.removed_margin_signatures == (patterns[0].signature,)
        for page in reading_pages
    )


def test_preserves_page_text_when_no_pattern_is_confirmed() -> None:
    page = _page(1)

    reading_page = build_pdf_reading_pages((page,))[0]

    assert reading_page.reading_text == page.text
    assert reading_page.removed_margin_signatures == ()


def test_keeps_line_breaks_between_prose_lines() -> None:
    reading_text = _reading_text(
        _fragment("Readers who skim a long", 420.0),
        "\n",
        _fragment("chapter often miss its point.", 408.0),
        "\n",
        _footer(2),
    )

    assert reading_text == (
        "Readers who skim a long\nchapter often miss its point.\n"
    )


def test_keeps_heading_and_body_on_separate_lines() -> None:
    reading_text = _reading_text(
        _fragment("returns early.", 500.0, font_size=10.5),
        "\n",
        _fragment("Practical Patterns: A Guide for", 470.0, font_size=20),
        "\n",
        _fragment("Reliable Pipelines", 446.0, font_size=20),
        "\n",
        _fragment("Start with the basics.", 420.0, font_size=10.5),
        "\n",
        _footer(2),
    )

    assert reading_text == (
        "returns early.\n"
        "Practical Patterns: A Guide for\n"
        "Reliable Pipelines\n"
        "Start with the basics.\n"
    )


def test_keeps_multiline_code_lines_and_indentation() -> None:
    reading_text = _reading_text(
        _fragment("pipeline:", 520.0),
        "\n",
        _fragment("steps = Pipeline([", 500.0, font_name="/Mono"),
        "\n    ",
        _fragment("Scale(factor=2)])", 488.0, font_name="/Mono"),
        "\n",
        _fragment("Each step receives the previous output.", 470.0),
        "\n",
        _footer(2),
    )

    assert reading_text == (
        "pipeline:\n"
        "steps = Pipeline([\n"
        "    Scale(factor=2)])\n"
        "Each step receives the previous output.\n"
    )


def test_does_not_insert_spaces_between_same_line_fragments() -> None:
    reading_text = _reading_text(
        _fragment("We call a ", 400.0, page_x=40.0),
        _fragment("parse", 400.0, page_x=90.0, font_name="/Mono"),
        _fragment(" helper.", 400.0, page_x=120.0),
        "\n",
        _footer(2),
    )

    assert reading_text == "We call a parse helper.\n"


def test_keeps_the_line_break_when_a_margin_is_inside_raw_text() -> None:
    reading_text = _reading_text(
        _fragment("Last body line.", 400.0),
        "\n",
        _footer(2),
        "\n",
        _fragment("Sidebar line.", 300.0),
    )

    assert reading_text == "Last body line.\nSidebar line."


def test_keeps_the_newline_gap_when_a_margin_follows_on_the_same_line(
) -> None:
    reading_text = _reading_text(
        _fragment("Before", 400.0),
        " ",
        _footer(2),
        "\n",
        _fragment("After", 380.0),
    )

    assert reading_text == "Before\nAfter"


def test_does_not_remove_text_that_cannot_be_located() -> None:
    pages = (
        _extracted_page(1, _other_page(1)),
        _extracted_page(2, ("Body.\n", _footer(2))),
        _extracted_page(3, _other_page(3)),
    )
    patterns = find_repeated_margin_patterns(pages)
    unlocatable = PdfPage(
        book_path=pages[1].book_path,
        page_number=2,
        width_points=600.0,
        height_points=800.0,
        image_count=0,
        drawing_operator_count=0,
        text="Body.\nunexpected 2 | Chapter 1: Basics",
        text_fragments=(_fragment("Body.", 400.0), _footer(2)),
        extractor="test",
    )

    reading_page = build_pdf_reading_pages(
        (pages[0], unlocatable, pages[2]),
        patterns,
    )[1]

    assert reading_page.reading_text == unlocatable.text
    assert reading_page.removed_margin_signatures == ()


def test_pages_without_margins_keep_raw_text_exactly() -> None:
    pages = (
        _extracted_page(1, _other_page(1)),
        _extracted_page(2, (_fragment("No footer", 400.0), "\n  ")),
        _extracted_page(3, _other_page(3)),
    )

    reading_page = build_pdf_reading_pages(pages)[1]

    assert reading_page.reading_text == "No footer\n  "
    assert reading_page.removed_margin_signatures == ()


def test_reading_text_crosses_pages_with_exact_source_offsets() -> None:
    pages = (
        _extracted_page(1, (
            _fragment("A sentence", 420.0),
            "\n",
            _fragment("continues", 408.0),
            "\n",
            _footer(1),
        )),
        _extracted_page(2, (
            _fragment("on the next", 700.0),
            "\n",
            _fragment("page.", 688.0),
            "\n",
            _footer(2),
        )),
        _extracted_page(3, _other_page(3)),
    )
    reading_pages = build_pdf_reading_pages(pages)

    flow = build_pdf_reading_flows(reading_pages)[0]

    assert flow.text == (
        "A sentence\ncontinues on the next\npage. Other page."
    )
    for span in flow.page_spans:
        page = reading_pages[span.page_number - 1]
        assert flow.text[span.flow_start:span.flow_end] == (
            page.reading_text[span.source_start:span.source_end]
        )
    assert "Chapter" not in flow.text
