"""Tests for the PDF MiniLM semantic baseline adapter."""

import json

import torch
from torch import Tensor

from src.evaluation.pdf.minilm_baseline import PdfMiniLmBaseline
from src.evaluation.pdf.models import PdfGoldenDataset
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord


class _KeywordEncoder:
    """Create normalized two-dimensional vectors for adapter tests."""

    def encode(self, texts: tuple[str, ...]) -> Tensor:
        vectors = []
        for text in texts:
            if "review" in text.casefold():
                vectors.append((1.0, 0.0))
            else:
                vectors.append((0.0, 1.0))
        return torch.tensor(vectors, dtype=torch.float32)


def _record(record_id: str, text: str, flow_index: int) -> PdfIndexRecord:
    return PdfIndexRecord(
        record_id=record_id,
        text=text,
        book_path="books/example.pdf",
        section_path=("Section",),
        flow_index=flow_index,
        flow_start=0,
        flow_end=len(text),
        page_spans=(PdfChunkPageSpan(1, 0, len(text), 0, len(text)),),
    )


def _dataset() -> PdfGoldenDataset:
    return PdfGoldenDataset.model_validate_json(json.dumps({
        "schema_version": 1,
        "dataset_id": "golden-v1",
        "book_path": "books/example.pdf",
        "page_numbering": "physical_pdf_page",
        "language": "en",
        "cases": [{
            "id": "review",
            "question": "How should I review generated code?",
            "why_now": "I need evidence.",
            "expected_answer": "Review it.",
            "answerability": "complete",
            "evidence": [{
                "physical_pages": [1],
                "role": "required",
                "modality": "text",
            }],
            "tags": ["positive"],
        }],
    }))


def test_ranks_pdf_records_with_the_injected_semantic_encoder() -> None:
    baseline = PdfMiniLmBaseline(
        (
            _record("database", "Store data safely.", 0),
            _record("review", "Review generated code.", 1),
        ),
        encoder=_KeywordEncoder(),
    )

    hits = baseline.search("review this output", top_k=2)

    assert baseline.record_count == 2
    assert baseline.dimension == 2
    assert [hit.record.record_id for hit in hits] == ["review", "database"]
    assert hits[0].score == 1.0


def test_dataset_run_returns_the_complete_semantic_ranking() -> None:
    baseline = PdfMiniLmBaseline(
        (
            _record("database", "Store data safely.", 0),
            _record("review", "Review generated code.", 1),
        ),
        encoder=_KeywordEncoder(),
    )

    result = baseline.run_dataset(_dataset())[0]

    assert result.ranked_record_ids == ("review", "database")
    assert result.abstained is False
