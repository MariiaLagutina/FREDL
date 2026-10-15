"""Synthetic pinned runs and pages shared by pooling and worksheet tests."""

from hashlib import sha256
import json
from typing import Any

import pytest

from src.evaluation.pdf.library import PdfLibraryCorpus
from src.evaluation.pdf.manifest import PdfRetrieverKind
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook_pool_models import (
    PdfPoolingRun, PdfRetrievalJudgmentPool,
)
from src.evaluation.pdf.multibook_runner import run_pdf_multibook_retrieval
from src.evaluation.pdf.multibook_queries import PdfQueryMode
from src.ingestion.pdf.reading_text import PdfReadingPage
from tests.evaluation.pdf.multibook_contract_helpers import ENV, HASH
from tests.evaluation.pdf.multibook_runner_helpers import FakeEncoder, _queries


PoolInputs = tuple[
    PdfMultibookGolden, PdfLibraryCorpus, tuple[PdfPoolingRun, ...],
    dict[str, tuple[PdfReadingPage, ...]],
]


def pin_page_hashes(
    golden: PdfMultibookGolden,
    pages: dict[str, tuple[PdfReadingPage, ...]],
) -> PdfMultibookGolden:
    texts = {(book, page.page_number): page.reading_text
             for book, values in pages.items() for page in values}
    data = golden.model_dump(mode="json")
    for unit in data["evidence_units"]:
        for field in ("locators", "supporting_locators"):
            for locator in unit[field]:
                locator["page_text_sha256"] = sha256(texts[
                    locator["book_path"], locator["physical_page"]
                ].encode()).hexdigest()
    return PdfMultibookGolden.model_validate_json(json.dumps(data))


def saved_runs(
    golden: PdfMultibookGolden, corpus: PdfLibraryCorpus,
    golden_sha256: str = HASH, mode: PdfQueryMode = PdfQueryMode.ALL,
) -> tuple[PdfPoolingRun, ...]:
    result = []
    for kind in PdfRetrieverKind:
        run = run_pdf_multibook_retrieval(
            golden=golden, golden_sha256=golden_sha256, corpus=corpus,
            retriever=kind, environment=ENV, query_mode=mode,
            encoder_factory=lambda config: FakeEncoder(),
        )
        result.append(PdfPoolingRun(
            artifact_sha256=sha256(run.model_dump_json().encode()).hexdigest(),
            run=run,
        ))
    return tuple(result)


def synthetic_inputs(corpus: PdfLibraryCorpus) -> PoolInputs:
    pages = {}
    for book in corpus.books:
        buffers: dict[int, list[str]] = {}
        for record in book.records:
            for span in record.page_spans:
                buffer = buffers.setdefault(span.page_number, [])
                padding = max(0, span.page_source_end - len(buffer))
                buffer.extend(' ' * padding)
                text = record.text[span.chunk_start:span.chunk_end]
                buffer[span.page_source_start:span.page_source_end] = text
        pages[book.book_path] = tuple(PdfReadingPage(
            book.book_path, number, ''.join(text), ''.join(text), (),
        ) for number, text in sorted(buffers.items()))
    golden = pin_page_hashes(_queries(corpus), pages)
    return golden, corpus, saved_runs(golden, corpus), pages


def build_pool(
    inputs: PoolInputs, **kwargs: Any,
) -> PdfRetrievalJudgmentPool:
    from src.evaluation.pdf.multibook_pooling import (
        build_pdf_retrieval_judgment_pool,
    )
    golden, corpus, runs, pages = inputs
    return build_pdf_retrieval_judgment_pool(
        golden=golden, golden_sha256=HASH, corpus=corpus, runs=runs,
        reading_pages=pages, **kwargs,
    )


@pytest.fixture()
def pool_inputs(corpus: PdfLibraryCorpus) -> PoolInputs:
    return synthetic_inputs(corpus)
