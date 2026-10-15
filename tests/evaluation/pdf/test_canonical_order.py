"""Tests that PDF rankings do not depend on incidental record order."""

from collections.abc import Callable
from dataclasses import replace
import random

import pytest
import torch
from torch import Tensor

from src.evaluation.pdf.bm25_baseline import PdfBm25Baseline
from src.evaluation.pdf.context import find_pdf_neighbor_context
from src.evaluation.pdf.hybrid_baseline import PdfHybridBaseline
from src.evaluation.pdf.minilm_baseline import PdfMiniLmBaseline
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.headings import PdfHeadingStatus
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.ingestion.pdf.sections import PdfSectionResolution


VOCABULARY = ("cache", "queue", "index", "token", "retry")
# Duplicate texts force exact score ties in every retriever.
TEXTS = (
    "cache entries expire after a retry.",
    "queue workers retry failed jobs.",
    "cache entries expire after a retry.",
    "index pages and token counts.",
    "queue workers retry failed jobs.",
    "token limits shape the cache.",
)


class _WordEncoder:
    def encode(self, texts: tuple[str, ...]) -> Tensor:
        vectors = torch.tensor(
            [
                [1.0 if word in text else 0.0 for word in VOCABULARY]
                for text in texts
            ],
            dtype=torch.float32,
        )
        return torch.nn.functional.normalize(vectors + 1e-3, dim=1)


def _records() -> tuple[PdfIndexRecord, ...]:
    records = []
    start = 0
    for index, text in enumerate(TEXTS):
        records.append(PdfIndexRecord(
            record_id=f"pdf:{index:02d}",
            text=text,
            book_path="books/example.pdf",
            section_path=("Section",),
            flow_index=0,
            flow_start=start,
            flow_end=start + len(text),
            page_spans=(PdfChunkPageSpan(1, 0, len(text), start,
                                         start + len(text)),),
            section_interval_id="pdf-section:a",
        ))
        start += len(text)
    return tuple(records)


def _orders() -> list[tuple[PdfIndexRecord, ...]]:
    records = _records()
    shuffled = list(records)
    random.Random(7).shuffle(shuffled)
    return [records, tuple(reversed(records)), tuple(shuffled)]


def _rankings(records: tuple[PdfIndexRecord, ...]) -> dict[str, list[str]]:
    query = "cache retry queue"
    encoder = _WordEncoder()
    return {
        "bm25": [hit.record.record_id
                 for hit in PdfBm25Baseline(records).search(query)],
        "minilm": [hit.record.record_id
                   for hit in PdfMiniLmBaseline(records, encoder)
                   .search(query)],
        "hybrid": [hit.record.record_id
                   for hit in PdfHybridBaseline(records, encoder)
                   .search(query)],
    }


def test_rankings_and_ties_ignore_input_order() -> None:
    expected = _rankings(_orders()[0])

    for records in _orders()[1:]:
        assert _rankings(records) == expected


def test_exact_ties_follow_canonical_reading_order() -> None:
    ranking = _rankings(tuple(reversed(_records())))["minilm"]

    assert ranking.index("pdf:00") < ranking.index("pdf:02")
    assert ranking.index("pdf:01") < ranking.index("pdf:04")


def test_neighbor_context_ignores_input_order() -> None:
    ranked = ("pdf:03", "pdf:01")
    expected = find_pdf_neighbor_context(ranked, _orders()[0], top_k=2)

    for records in _orders()[1:]:
        assert find_pdf_neighbor_context(ranked, records, top_k=2) == (
            expected
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda record: replace(
            record, section_resolution=PdfSectionResolution.UNRESOLVED,
        ),
        lambda record: replace(record, section_hint=("Section", "Hint")),
        lambda record: replace(
            record, unresolved_reason=PdfHeadingStatus.AMBIGUOUS,
        ),
    ],
    ids=["resolution", "hint", "reason"],
)
def test_diagnostic_fields_do_not_change_behavior(
    change: Callable[[PdfIndexRecord], PdfIndexRecord],
) -> None:
    records = _records()
    annotated = tuple(change(record) for record in records)
    ranked = ("pdf:03",)

    assert _rankings(annotated) == _rankings(records)
    assert find_pdf_neighbor_context(ranked, annotated, top_k=1) == (
        find_pdf_neighbor_context(ranked, records, top_k=1)
    )
