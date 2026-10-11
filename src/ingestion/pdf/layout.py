"""Preserve positioned PDF text fragments for later layout analysis."""

from dataclasses import dataclass

from pypdf import PageObject
from pypdf.generic import DictionaryObject


@dataclass(frozen=True, slots=True)
class PdfTextFragment:
    """Store one extracted text fragment and its raw PDF coordinates."""

    text: str
    container_x: float
    container_y: float
    text_x: float
    text_y: float
    page_x: float
    page_y: float
    font_size: float
    font_name: str | None = None
    font_weight: float | None = None
    font_flags: int | None = None


@dataclass(frozen=True, slots=True)
class PdfTextLayout:
    """Store raw page text together with its positioned fragments."""

    raw_text: str
    fragments: tuple[PdfTextFragment, ...]


def extract_pdf_text_layout(page: PageObject) -> PdfTextLayout:
    """Extract raw text and positioned fragments in one page pass."""
    fragments: list[PdfTextFragment] = []

    def collect_fragment(
        text: str,
        container_matrix: list[float],
        text_matrix: list[float],
        font_dictionary: object,
        font_size: float,
    ) -> None:
        if not text.strip():
            return

        page_x, page_y = _transform_text_origin(
            container_matrix,
            text_matrix,
        )
        font_name, font_weight, font_flags = _font_metadata(font_dictionary)
        fragments.append(
            PdfTextFragment(
                text=text,
                container_x=float(container_matrix[4]),
                container_y=float(container_matrix[5]),
                text_x=float(text_matrix[4]),
                text_y=float(text_matrix[5]),
                page_x=page_x,
                page_y=page_y,
                font_size=float(font_size),
                font_name=font_name,
                font_weight=font_weight,
                font_flags=font_flags,
            )
        )

    raw_text = page.extract_text(visitor_text=collect_fragment) or ""
    return PdfTextLayout(raw_text=raw_text, fragments=tuple(fragments))


def _transform_text_origin(
    container_matrix: list[float],
    text_matrix: list[float],
) -> tuple[float, float]:
    """Transform a text origin into physical page coordinates."""
    text_x = text_matrix[4]
    text_y = text_matrix[5]
    page_x = (
        text_x * container_matrix[0]
        + text_y * container_matrix[2]
        + container_matrix[4]
    )
    page_y = (
        text_x * container_matrix[1]
        + text_y * container_matrix[3]
        + container_matrix[5]
    )
    return float(page_x), float(page_y)


def _font_metadata(
    font: object,
) -> tuple[str | None, float | None, int | None]:
    """Preserve declared font facts; missing metadata remains unknown.

    Type 0 fonts store their descriptor on a descendant font. Names may
    contain subset prefixes and are retained exactly as declared. Neither
    a name nor a weight alone establishes that text is a heading.
    """
    if not isinstance(font, DictionaryObject):
        return None, None, None
    name = font.get("/BaseFont")
    descriptor = font.get("/FontDescriptor")
    if descriptor is None:
        descendants = font.get("/DescendantFonts")
        if descendants is not None:
            descendants = descendants.get_object()
            if descendants:
                descendant = descendants[0].get_object()
                descriptor = descendant.get("/FontDescriptor")
    if descriptor is None:
        return str(name) if name is not None else None, None, None
    descriptor = descriptor.get_object()
    weight = descriptor.get("/FontWeight")
    flags = descriptor.get("/Flags")
    return (
        str(name) if name is not None else None,
        float(weight) if weight is not None else None,
        int(flags) if flags is not None else None,
    )
