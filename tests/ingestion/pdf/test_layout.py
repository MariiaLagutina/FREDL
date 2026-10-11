"""Verify declared font metadata without inferring semantic headings."""

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

from src.ingestion.pdf.layout import _font_metadata, extract_pdf_text_layout


def test_extraction_preserves_font_facts_and_text() -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=400)
    descriptor = DictionaryObject({
        NameObject("/FontWeight"): NumberObject(700),
        NameObject("/Flags"): NumberObject(64),
    })
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica-Bold"),
        NameObject("/FontDescriptor"): writer._add_object(descriptor),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): writer._add_object(font),
        }),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 18 Tf 20 200 Td (A heading candidate) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)

    layout = extract_pdf_text_layout(page)

    fragment = layout.fragments[0]
    assert layout.raw_text == "A heading candidate"
    assert fragment.font_name == "/Helvetica-Bold"
    assert fragment.font_weight == 700
    assert fragment.font_flags == 64
    assert fragment.font_size == 18
    assert (fragment.page_x, fragment.page_y) == (20, 200)


def test_reads_indirect_descendant_font_descriptor() -> None:
    writer = PdfWriter()
    descriptor = DictionaryObject({
        NameObject("/FontWeight"): NumberObject(400),
    })
    descendant = DictionaryObject({
        NameObject("/FontDescriptor"): writer._add_object(descriptor),
    })
    font = DictionaryObject({
        NameObject("/BaseFont"): NameObject("/ABCDEF+Example"),
        NameObject("/DescendantFonts"): ArrayObject([
            writer._add_object(descendant),
        ]),
    })

    assert _font_metadata(font) == ("/ABCDEF+Example", 400, None)


@pytest.mark.parametrize("font", [None, DictionaryObject()])
def test_missing_font_metadata_stays_unknown(font: object) -> None:
    assert _font_metadata(font) == (None, None, None)


def test_font_name_does_not_invent_weight() -> None:
    font = DictionaryObject({
        NameObject("/BaseFont"): NameObject("/Helvetica-Bold"),
    })

    assert _font_metadata(font) == ("/Helvetica-Bold", None, None)
