"""Tests for reversible PDF retrieval eligibility policies."""

from copy import deepcopy

import pytest

from src.evaluation.pdf.bm25_baseline import PdfBm25Baseline
from src.ingestion.pdf.chunker import chunk_pdf_reading_document
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.headings import PdfHeadingMatch, PdfHeadingStatus
from src.ingestion.pdf.index_records import (
    build_pdf_index_records,
    PdfIndexRecord,
)
from src.ingestion.pdf.outline import PdfOutlineEntry
from src.ingestion.pdf.reading_document import PdfReadingDocument
from src.ingestion.pdf.reading_flow import build_pdf_reading_flows
from src.ingestion.pdf.reading_text import PdfReadingPage
from src.ingestion.pdf.retrieval_corpus import (
    build_pdf_retrieval_corpus,
    classify_pdf_records,
    PdfRecordClassification,
    PdfRecordRole,
    PdfRetrievalPolicy,
)
from src.ingestion.pdf.sections import (
    build_pdf_section_intervals,
    map_pdf_section_boundaries,
    PdfSectionResolution,
)


BOOK = "books/example.pdf"
NOTE = (
    "A NOTE FROM THE PUBLISHER ABOUT THIS DRAFT\n"
    "Draft chapters arrive before editing so readers can follow along.\n"
    "Send comments about missing material to the address on the site.\n"
)
TERMS = (
    "adapters", "batching", "caching", "datasets", "embeddings", "filters",
    "graphs", "hashing", "indexes", "joins", "kernels", "logging",
    "metrics", "nodes", "outputs", "parsers", "queues", "retries",
    "schemas", "tokens", "updates", "vectors", "workers", "xml", "yaml",
    "zones",
)
SECTION_TITLES = tuple(f"Working with {term.title()}" for term in TERMS[:12])
OUTLINE = tuple(
    PdfOutlineEntry(title, (title,), None)
    for title in (
        "Preface", "1. Getting Started", "Installing the Tools",
        "2. Reading Data", "Parsing Records", *SECTION_TITLES,
        "3. Shipping", "Index",
    )
)


def _record(
    record_id: str,
    text: str,
    *,
    section: tuple[str, ...] = ("1. Getting Started",),
    interval: str | None = None,
    start: int = 0,
    resolution: PdfSectionResolution | None = None,
    page: int = 1,
) -> PdfIndexRecord:
    return PdfIndexRecord(
        record_id=record_id,
        text=text,
        book_path=BOOK,
        section_path=section,
        flow_index=0,
        flow_start=start,
        flow_end=start + len(text),
        page_spans=(PdfChunkPageSpan(page, 0, len(text), 0, len(text)),),
        section_resolution=(
            resolution if resolution is not None
            else PdfSectionResolution.VERIFIED if section
            else PdfSectionResolution.NONE
        ),
        section_interval_id=interval or f"interval:{section}",
    )


def _roles(*records: PdfIndexRecord) -> dict[str, PdfRecordRole]:
    return {
        item.record_id: item.role
        for item in classify_pdf_records(records, OUTLINE)
    }


def _prose(start: int = 1000) -> PdfIndexRecord:
    return _record(
        "prose",
        "Installing the Tools\nRun the installer, then check the version.\n",
        start=start,
    )


def _index_lines(count: int, terms: tuple[str, ...] = TERMS) -> list[str]:
    """Return alphabetical entries with locators inside a 250-page book."""
    return [
        f"{terms[index * len(terms) // count]} {index}, "
        f"{10 + index}, {14 + index}\n"
        for index in range(count)
    ]


def _index(lines: list[str], record_id: str = "index") -> PdfIndexRecord:
    return _record(record_id, "".join(lines), section=("Index",),
                   start=5000, page=250)


def test_page_locator_index_is_navigation() -> None:
    assert _roles(_prose(), _index(_index_lines(24))) == {
        "prose": PdfRecordRole.CONTENT,
        "index": PdfRecordRole.INDEX,
    }


def test_section_reference_index_is_navigation() -> None:
    lines = [
        f"{TERMS[index * 26 // 24]} {index}, "
        "Parsing Records-Installing the Tools\n"
        for index in range(24)
    ]

    assert _roles(_index(lines))["index"] is PdfRecordRole.INDEX


def test_contents_lines_repeating_outline_titles_are_navigation() -> None:
    entries = ("Preface", "1. Getting Started", "Installing the Tools",
               "2. Reading Data", "Parsing Records", *SECTION_TITLES,
               "3. Shipping")
    lines = "".join(
        f"{title} . . . . {page}\n"
        for page, title in enumerate(entries + entries[:4], start=1)
    )
    contents = _record("contents", lines, section=("Contents",))

    assert _roles(contents)["contents"] is PdfRecordRole.TABLE_OF_CONTENTS


def test_classification_and_corpus_ignore_input_order() -> None:
    lines = _index_lines(24)
    records = (
        _prose(),
        _index(lines[:12], "first"),
        _record("second", "".join(lines[12:]), section=("Index",),
                start=6000, page=250),
    )

    def corpus(order: tuple[PdfIndexRecord, ...]) -> tuple[
        dict[str, PdfRecordRole], str, tuple[PdfIndexRecord, ...],
    ]:
        classified = classify_pdf_records(order, OUTLINE)
        built = build_pdf_retrieval_corpus(
            order, classified, PdfRetrievalPolicy.EXCLUDE_NAVIGATION,
        )
        roles = {item.record_id: item.role for item in classified}
        return (roles, built.fingerprint, built.records)

    assert corpus(tuple(reversed(records))) == corpus(records)
    assert corpus(records)[0]["second"] is PdfRecordRole.INDEX


def test_navigation_shape_spans_all_records_of_an_interval() -> None:
    lines = _index_lines(24)
    first = _index(lines[:12], "first")
    second = _record("second", "".join(lines[12:]), section=("Index",),
                     start=6000, page=250)

    roles = _roles(first, second)

    assert roles == {"first": PdfRecordRole.INDEX,
                     "second": PdfRecordRole.INDEX}


def test_short_record_of_outline_titles_stays_content() -> None:
    heading = _record("heading", "Parsing Records\n2. Reading Data\n")

    assert _roles(heading)["heading"] is PdfRecordRole.CONTENT


def test_prose_with_commas_stays_content() -> None:
    lines = "".join(
        f"Readers compare tools, formats, and costs in step {index}.\n"
        for index in range(30)
    )
    prose = _record("prose", lines, section=("Preface",))

    assert _roles(prose)["prose"] is PdfRecordRole.CONTENT


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param(
            [f"Region {index}, {100 + index}, {300 - index}"
             for index in range(30)],
            id="numeric-table",
        ),
        pytest.param(
            [f"{index},{index * 2},{index * 3}" for index in range(30)],
            id="csv-listing",
        ),
        pytest.param(
            [f"{TERMS[index * 26 // 30].title()}, A. Data Systems, "
             f"{1990 + index}" for index in range(30)],
            id="sorted-bibliography-with-years",
        ),
        pytest.param(
            [f"Step {index} is explained in the next part, Parsing Records"
             for index in range(30)],
            id="prose-section-references",
        ),
        pytest.param(
            [line.rstrip("\n") for line in _index_lines(30, TERMS[::-1])],
            id="entries-not-alphabetical",
        ),
    ],
)
def test_content_with_numbers_or_references_stays_content(
    lines: list[str],
) -> None:
    record = _record("content", "\n".join(lines) + "\n",
                     section=("2. Reading Data",), page=250)

    assert _roles(record)["content"] is PdfRecordRole.CONTENT


def test_roadmap_of_outline_titles_with_descriptions_stays_content() -> None:
    lines = "".join(
        f"{title}\nThis part explains {title.casefold()} in depth.\n"
        for title in SECTION_TITLES
    )
    roadmap = _record("roadmap", lines, section=("Preface",))

    assert _roles(roadmap)["roadmap"] is PdfRecordRole.CONTENT


def test_few_repeated_outline_titles_stay_content() -> None:
    titles = ("1. Getting Started", "Installing the Tools",
              "2. Reading Data", "Parsing Records", "3. Shipping")
    lines = "".join(f"{title}\n" for title in titles * 6)
    summary = _record("summary", lines, section=("Preface",))

    assert _roles(summary)["summary"] is PdfRecordRole.CONTENT


def test_index_shape_below_the_line_minimum_stays_content() -> None:
    assert _roles(_index(_index_lines(19)))["index"] is PdfRecordRole.CONTENT


def test_text_before_the_first_verified_heading_is_front_matter() -> None:
    cover = _record("cover", "Cover blurb and copyright notice.\n",
                    section=(), interval="leading")

    roles = _roles(cover, _prose())

    assert roles["cover"] is PdfRecordRole.UNSECTIONED_FRONT_MATTER
    corpus = build_pdf_retrieval_corpus(
        (cover, _prose()),
        classify_pdf_records((cover, _prose()), OUTLINE),
        PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE,
    )
    assert [record.record_id for record in corpus.records] == [
        "cover", "prose",
    ]


def test_book_without_verified_sections_has_no_front_matter() -> None:
    record = _record("only", "Text of a book without an outline.\n",
                     section=(), interval="leading")

    assert _roles(record)["only"] is PdfRecordRole.CONTENT


def _chapter(number: int, body: str) -> PdfIndexRecord:
    title = f"Chapter {number}"
    return _record(
        f"chapter-{number}",
        f"{title}\n{NOTE}{body}\n",
        section=(title,),
        start=number * 1000,
    )


def test_boilerplate_repeated_across_sections_is_removed_from_search() -> (
    None
):
    chapters = tuple(
        _chapter(number, f"Chapter {number} explains topic {number}.")
        for number in range(1, 4)
    )

    classifications = classify_pdf_records(chapters, OUTLINE)
    corpus = build_pdf_retrieval_corpus(
        chapters, classifications,
        PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE,
    )

    assert all(item.boilerplate_ranges for item in classifications)
    assert corpus.records == chapters
    searchable = corpus.retrieval_texts[0]
    assert len(searchable) == len(chapters[0].text)
    assert searchable.count("\n") == chapters[0].text.count("\n")
    assert searchable.split() == [
        "Chapter", "1", "Chapter", "1", "explains", "topic", "1.",
    ]
    assert chapters[0].text.startswith("Chapter 1\nA NOTE FROM")


def test_text_repeated_in_too_few_sections_is_kept() -> None:
    chapters = (
        _chapter(1, "First body."),
        _chapter(2, "Second body."),
        _record("other-3", "Chapter three has its own text.\n",
                section=("Chapter 3",), start=3000),
        _record("other-4", "Chapter four has its own text.\n",
                section=("Chapter 4",), start=4000),
    )

    classifications = classify_pdf_records(chapters, OUTLINE)

    assert not any(item.boilerplate_ranges for item in classifications)


def test_boilerplate_only_record_leaves_search_but_not_the_corpus() -> None:
    chapters = tuple(
        _chapter(number, "Body.") for number in range(1, 4)
    )
    only_note = _record("only-note", NOTE, section=("Chapter 1",),
                        start=1500, interval="interval:('Chapter 1',)")
    records = (*chapters, only_note)
    classifications = classify_pdf_records(records, OUTLINE)

    navigation = build_pdf_retrieval_corpus(
        records, classifications, PdfRetrievalPolicy.EXCLUDE_NAVIGATION,
    )
    cleaned = build_pdf_retrieval_corpus(
        records, classifications,
        PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE,
    )

    assert "only-note" in {record.record_id for record in navigation.records}
    assert cleaned.excluded_record_ids == ("only-note",)
    assert only_note.text == NOTE


def test_policies_are_reversible_and_reproducible() -> None:
    records = (_prose(), _index(_index_lines(24)))

    def corpus(policy: PdfRetrievalPolicy) -> tuple[str, ...]:
        built = build_pdf_retrieval_corpus(
            records, classify_pdf_records(records, OUTLINE), policy,
        )
        return (built.fingerprint, *built.excluded_record_ids)

    full = build_pdf_retrieval_corpus(
        records, classify_pdf_records(records, OUTLINE),
        PdfRetrievalPolicy.ALL_RECORDS,
    )
    assert full.records == records
    assert full.retrieval_texts == tuple(record.text for record in records)
    assert corpus(PdfRetrievalPolicy.EXCLUDE_NAVIGATION) == corpus(
        PdfRetrievalPolicy.EXCLUDE_NAVIGATION
    )
    assert corpus(PdfRetrievalPolicy.EXCLUDE_NAVIGATION)[1:] == ("index",)
    assert len({corpus(policy)[0] for policy in PdfRetrievalPolicy}) == 3


def test_rejects_classifications_for_other_records() -> None:
    stale = (PdfRecordClassification("other", PdfRecordRole.CONTENT),)

    with pytest.raises(ValueError, match="must match the records"):
        build_pdf_retrieval_corpus(
            (_prose(),), stale, PdfRetrievalPolicy.ALL_RECORDS,
        )


def test_masking_keeps_offsets_source_text_and_physical_pages() -> None:
    page_texts = tuple(
        f"Chapter {number}\n{NOTE}Chapter {number} covers topic {number}.\n"
        for number in range(1, 4)
    )
    reading_pages = tuple(
        PdfReadingPage(BOOK, number, text, text, ())
        for number, text in enumerate(page_texts, start=10)
    )
    flows = build_pdf_reading_flows(reading_pages)
    document = PdfReadingDocument(BOOK, (), (), reading_pages, flows)
    matches = tuple(
        PdfHeadingMatch(
            PdfOutlineEntry(f"Chapter {number}", (f"Chapter {number}",),
                            number + 9),
            PdfHeadingStatus.MATCHED,
            0,
            len(f"Chapter {number}"),
        )
        for number in range(1, 4)
    )
    intervals = build_pdf_section_intervals(
        map_pdf_section_boundaries(matches, flows), flows,
    )
    records = build_pdf_index_records(chunk_pdf_reading_document(
        document, section_intervals=intervals,
    ))
    snapshot = deepcopy(records)

    classifications = classify_pdf_records(records, OUTLINE)
    corpus = build_pdf_retrieval_corpus(
        records, classifications,
        PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE,
    )
    hit = PdfBm25Baseline(
        corpus.records, retrieval_texts=corpus.retrieval_texts,
    ).search("topic 2")[0].record

    assert records == snapshot
    for record, text, item in zip(
        corpus.records, corpus.retrieval_texts, classifications,
    ):
        assert item.boilerplate_ranges
        assert len(text) == len(record.text)
        masked = {
            offset for start, end in item.boilerplate_ranges
            for offset in range(start, end)
        }
        for offset, character in enumerate(text):
            original = record.text[offset]
            if offset in masked and original not in "\r\n":
                assert character == " "
            else:
                assert character == original
    assert hit is records[1]
    pages = {page.page_number: page for page in reading_pages}
    for span in hit.page_spans:
        assert hit.text[span.chunk_start:span.chunk_end] == (
            pages[span.page_number].reading_text[
                span.page_source_start:span.page_source_end
            ]
        )
    assert [span.page_number for span in hit.page_spans] == [11]


def test_bm25_searches_policy_text_and_returns_original_record() -> None:
    chapters = tuple(
        _chapter(number, f"Chapter {number} covers topic {number}.")
        for number in range(1, 4)
    )
    corpus = build_pdf_retrieval_corpus(
        chapters, classify_pdf_records(chapters, OUTLINE),
        PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE,
    )
    baseline = PdfBm25Baseline(
        corpus.records, retrieval_texts=corpus.retrieval_texts,
    )

    assert baseline.search("publisher draft comments") == ()
    hit = baseline.search("topic 2")[0]
    assert hit.record is chapters[1]
    with pytest.raises(ValueError, match="align with records"):
        PdfBm25Baseline(chapters, retrieval_texts=("one",))
    with pytest.raises(ValueError, match="keep record offsets"):
        PdfBm25Baseline(chapters[:1], retrieval_texts=("shorter",))
