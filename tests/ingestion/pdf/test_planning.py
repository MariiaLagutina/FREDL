"""Tests for routing extracted PDF pages before chunking."""

from src.ingestion.pdf import PdfPage, plan_pdf_pages


def pdf_page(
    page_number: int,
    *,
    text: str = "",
    image_count: int = 0,
    drawing_operator_count: int = 0,
) -> PdfPage:
    """Create one synthetic physical page observation."""
    return PdfPage(
        book_path="books/synthetic.pdf",
        page_number=page_number,
        width_points=300,
        height_points=400,
        image_count=image_count,
        drawing_operator_count=drawing_operator_count,
        text=text,
        extractor="pypdf-test",
    )


def test_plan_pdf_pages_preserves_text_order_and_physical_numbers() -> None:
    """Empty pages do not renumber the remaining text flow."""
    pages = (
        pdf_page(1, text="First page"),
        pdf_page(2),
        pdf_page(3, text="Third page"),
    )

    plan = plan_pdf_pages(pages)

    assert tuple(page.page_number for page in plan.text_pages) == (1, 3)
    assert plan.empty_page_numbers == (2,)
    assert plan.visual_page_numbers == ()


def test_plan_pdf_pages_keeps_text_and_flags_its_images() -> None:
    """A mixed page remains searchable and enters visual processing."""
    pages = (pdf_page(1, text="Caption and body", image_count=2),)

    plan = plan_pdf_pages(pages)

    assert tuple(page.page_number for page in plan.text_pages) == (1,)
    assert plan.visual_page_numbers == (1,)


def test_plan_pdf_pages_routes_non_text_visuals_without_fake_text() -> None:
    """Image and vector pages enter only the visual route."""
    pages = (
        pdf_page(1, image_count=1),
        pdf_page(2, drawing_operator_count=5),
    )

    plan = plan_pdf_pages(pages)

    assert plan.text_pages == ()
    assert plan.empty_page_numbers == ()
    assert plan.visual_page_numbers == (1, 2)
