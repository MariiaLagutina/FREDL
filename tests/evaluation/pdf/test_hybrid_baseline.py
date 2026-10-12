"""Tests for equal-weight PDF BM25 and MiniLM fusion."""

import torch
from torch import Tensor

from src.evaluation.pdf.hybrid_baseline import (
    PDF_HYBRID_PARAMETERS,
    PdfHybridBaseline,
)
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord


class _SemanticEncoder:
    def encode(self, texts: tuple[str, ...]) -> Tensor:
        vectors = [
            (1.0, 0.0)
            if "meaning" in text or "concept" in text
            else (0.0, 1.0)
            for text in texts
        ]
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


def test_fuses_lexical_and_semantic_ranks_over_one_pdf_corpus() -> None:
    baseline = PdfHybridBaseline(
        (
            _record("lexical", "question exact words", 0),
            _record("semantic", "meaning", 1),
            _record("other", "unrelated", 2),
        ),
        encoder=_SemanticEncoder(),
    )

    hits = baseline.search("question concept", top_k=3)

    assert baseline.record_count == 3
    assert baseline.parameters == PDF_HYBRID_PARAMETERS
    assert {hit.record.record_id for hit in hits} == {
        "lexical",
        "semantic",
        "other",
    }
    lexical = next(hit for hit in hits if hit.record.record_id == "lexical")
    semantic = next(hit for hit in hits if hit.record.record_id == "semantic")
    assert lexical.lexical_rank == 1
    assert semantic.semantic_rank == 1
    assert all(hit.score > 0 for hit in hits)
