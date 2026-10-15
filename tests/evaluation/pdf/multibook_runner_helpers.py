"""Shared synthetic queries, fake encoder and PDF inputs for B0 tests."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from src.evaluation.pdf.library import (
    PdfLibraryCorpus,
    load_pdf_library_corpus,
)
from src.evaluation.pdf.manifest import PdfRetrieverKind
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook_queries import PdfQueryMode
from src.evaluation.pdf.multibook_runner import run_pdf_multibook_retrieval
from src.evaluation.pdf.multibook_runs import PdfMultibookRetrievalRun
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy

from tests.evaluation.pdf.multibook_contract_helpers import ENV, HASH
from tests.evaluation.pdf.test_multibook_import import _write_pdf
from tests.evaluation.pdf.test_multibook_judgments import _golden


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: tuple[str, ...]) -> Tensor:
        self.calls.append(texts)
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "caches" in lowered:
                vectors.append((0.0, 1.0, 0.0))
            elif "queues" in lowered:
                vectors.append((1.0, 0.0, 0.0))
            else:
                vectors.append((0.0, 0.0, 1.0))
        return torch.tensor(vectors, dtype=torch.float32)


@pytest.fixture(autouse=True)
def forbid_real_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Real model loading is forbidden in runner tests")

    monkeypatch.setattr(
        "src.retrieval.semantic.encoder.load_semantic_backend", forbidden,
    )


def _queries(
    corpus: PdfLibraryCorpus, first: str = "queues",
) -> PdfMultibookGolden:
    data = _golden(corpus).model_dump(mode="json")
    data["cases"][0]["question"] = first
    data["cases"][1]["question"] = "queues or caches?"
    for item, text in zip(data["cases"][1]["interpretations"],
                          ("queues", "caches")):
        item["clarified_question"] = text
    data["cases"][2]["question"] = "zxqmissing"
    return PdfMultibookGolden.model_validate_json(json.dumps(data))


def _run(
    corpus: PdfLibraryCorpus,
    kind: str = "bm25",
    first: str = "queues",
    mode: PdfQueryMode = PdfQueryMode.ALL,
    encoder: FakeEncoder | None = None,
) -> PdfMultibookRetrievalRun:
    return run_pdf_multibook_retrieval(
        golden=_queries(corpus, first), golden_sha256=HASH, corpus=corpus,
        retriever=PdfRetrieverKind(kind), query_mode=mode, environment=ENV,
        encoder_factory=lambda config: encoder or FakeEncoder(),
    )


@pytest.fixture()
def pdf_inputs(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "library"
    books = {
        "books/alpha.pdf": (
            "Queues hold work.", "Backpressure slows producers.",
        ),
        "books/beta.pdf": (
            "Caches keep answers.", "Eviction removes entries.",
        ),
        "books/delta.pdf": ("Synthetic delta background.",),
        "books/gamma.pdf": ("Synthetic gamma background.",),
    }
    for path, pages in books.items():
        _write_pdf(root / path, pages)
    corpus = load_pdf_library_corpus(root, sorted(books),
                                     PdfRetrievalPolicy.ALL_RECORDS)
    golden = tmp_path / "golden.json"
    golden.write_text(_queries(corpus).model_dump_json(indent=1))
    # Discovery of this unpinned, invalid file would break the workflow.
    (root / "books" / "not-pinned.pdf").write_bytes(b"not a PDF")
    return root, golden
