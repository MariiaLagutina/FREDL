"""Synthetic two-book library and version 2 golden set for multi-book tests."""

import json
from typing import Any

from src.evaluation.pdf.multibook.library import (
    build_pdf_library_corpus,
    PdfLibraryBookInput,
    PdfLibraryCorpus,
)
from src.evaluation.pdf.multibook.curation.golden import PdfMultibookGolden
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy


ALPHA = "books/alpha.pdf"
BETA = "books/beta.pdf"
# Two records share alpha page 1; one record continues onto page 2.
ALPHA_RECORDS = (
    ("a1", "Queues hold work.", 1, 0),
    ("a2", "Workers poll the queue.", 1, 18),
    ("a3", "Backpressure slows producers.", 2, 0),
)
BETA_RECORDS = (
    ("b1", "Caches keep answers.", 1, 0),
    ("b2", "Eviction removes old entries.", 2, 0),
)


def _records(book: str, rows: Any) -> tuple[PdfIndexRecord, ...]:
    return tuple(
        PdfIndexRecord(
            record_id=record_id, text=text, book_path=book,
            section_path=("Chapter",), flow_index=0,
            flow_start=index * 100, flow_end=index * 100 + len(text),
            page_spans=(PdfChunkPageSpan(page, 0, len(text), start,
                                         start + len(text)),),
            section_interval_id=f"{book}:chapter",
        )
        for index, (record_id, text, page, start) in enumerate(rows)
    )


def _library() -> PdfLibraryCorpus:
    return build_pdf_library_corpus([
        PdfLibraryBookInput(ALPHA, "a" * 64,
                            _records(ALPHA, ALPHA_RECORDS), ()),
        PdfLibraryBookInput(BETA, "b" * 64, _records(BETA, BETA_RECORDS), ()),
    ], PdfRetrievalPolicy.ALL_RECORDS)


def _locator(book: str, page: int, anchor: str, start: int) -> object:
    return {
        "book_path": book, "physical_page": page, "exact_anchor": anchor,
        "page_text_start": start, "page_text_end": start + len(anchor),
        "page_text_sha256": "f" * 64,
    }


def _golden(library: PdfLibraryCorpus) -> PdfMultibookGolden:
    def case(case_id: str, category: str, outcome: str, support: tuple[
        str, str,
    ], **fields: Any) -> object:
        behavior = (
            "clarification_required" if outcome == "ambiguous" else "answer"
        )
        return {"case_id": case_id, "question": f"Question {case_id}?",
                "category": category, "expected_outcome": outcome,
                "source_support": support[0],
                "text_context_support": support[1],
                "expected_behavior": behavior, **fields}

    def meaning(name: str, unit: str) -> object:
        return {"interpretation_id": name, "label": f"Meaning {name}",
                "clarified_question": f"Clarified {name}?",
                "expected_outcome": "complete", "source_support": "complete",
                "text_context_support": "complete",
                "sufficient_sets": [[unit]]}

    return PdfMultibookGolden.model_validate_json(json.dumps({
        "version": 2,
        "dataset_id": "synthetic", "status": "draft", "language": "en",
        "library": library.manifest.model_dump(mode="json"),
        "evidence_units": [
            {"unit_id": "queue", "required_content": "Queues hold work.",
             "modality": "text",
             "locators": [_locator(ALPHA, 1, "Queues hold", 0)]},
            {"unit_id": "cache", "required_content": "Caches keep answers.",
             "modality": "text",
             "locators": [_locator(BETA, 1, "Caches keep", 0)],
             "supporting_locators": [_locator(BETA, 2, "Eviction", 0)]},
            {"unit_id": "backpressure", "required_content": "Producers slow.",
             "modality": "text",
             "locators": [_locator(ALPHA, 2, "Backpressure", 0)]},
        ],
        "cases": [
            case("both", "complementary_multi_book", "complete",
                 ("complete", "complete"),
                 sufficient_sets=[["queue", "cache"]]),
            case("which", "ambiguous_terminology", "ambiguous",
                 ("conditional", "conditional"),
                 conditional_units=["queue", "backpressure"],
                 interpretations=[meaning("queues", "queue"),
                                  meaning("producers", "backpressure")]),
            case("absent", "hard_negative", "unsupported",
                 ("not_found", "not_found"), related_units=["cache"]),
        ],
    }))
