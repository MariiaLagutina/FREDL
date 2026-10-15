"""Shared synthetic corpus and builders for B0 retrieval contracts."""

from dataclasses import replace
from typing import Any

import pytest

from src.evaluation.pdf.library import (
    PdfLibraryBookInput,
    PdfLibraryCorpus,
    build_pdf_library_corpus,
)
from src.evaluation.pdf.manifest import (
    PdfRetrieverKind,
    PdfRunEnvironment,
    bm25_configuration,
    rrf_configuration,
    semantic_configuration,
)
from src.evaluation.pdf.multibook.retrieval.manifest import (
    PdfMultibookRetrievalConfiguration,
    PdfMultibookRetrievalManifest,
    build_pdf_multibook_manifest,
)
from src.evaluation.pdf.multibook.retrieval.queries import (
    build_pdf_multibook_query_plan,
)
from src.evaluation.pdf.multibook.retrieval.runs import (
    PdfMultibookContextAtCutoff,
    PdfMultibookContextHit,
    PdfMultibookQueryResult,
    PdfMultibookRankedHit,
    PdfMultibookRetrievalRun,
    build_pdf_multibook_run,
    require_pdf_multibook_run_matches,
)
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy
from src.retrieval.bm25 import BM25Parameters
from src.retrieval.hybrid import RRFParameters
from src.retrieval.semantic.config import SemanticEncoderConfig

from tests.evaluation.pdf.multibook_golden_helpers import (
    _golden,
    _library,
    _records,
)


HASH = "e" * 64
ENV = PdfRunEnvironment("synthetic-revision", False, {"python": "synthetic"})


@pytest.fixture(autouse=True)
def no_retrieval_or_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Contract tests must not retrieve or load models")

    monkeypatch.setattr(
        "src.retrieval.semantic.encoder.load_semantic_backend", forbidden,
    )
    monkeypatch.setattr(
        "src.evaluation.pdf.bm25_baseline.PdfBm25Baseline.search", forbidden,
    )


@pytest.fixture()
def corpus() -> PdfLibraryCorpus:
    original = _library()
    inputs = []
    for book in original.books:
        records = book.records
        if book.book_path == "books/alpha.pdf":
            records = (records[0], replace(
                records[1], flow_start=records[0].flow_end,
                flow_end=records[0].flow_end + len(records[1].text),
            ), records[2])
        inputs.append(PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, records, (),
        ))
    for index, path in enumerate(("books/delta.pdf", "books/gamma.pdf")):
        inputs.append(PdfLibraryBookInput(
            path, str(index) * 64,
            _records(path, (
                (f"extra-{index}", "Synthetic background.", 1, 0),
            )),
            (),
        ))
    return build_pdf_library_corpus(inputs, PdfRetrievalPolicy.ALL_RECORDS)


def _config(kind: str = "bm25") -> PdfMultibookRetrievalConfiguration:
    return PdfMultibookRetrievalConfiguration(
        retriever=PdfRetrieverKind(kind),
        bm25=(bm25_configuration(BM25Parameters())
              if kind != "minilm" else None),
        semantic=semantic_configuration(SemanticEncoderConfig())
        if kind != "bm25" else None,
        rrf=rrf_configuration(RRFParameters(60, 1.0, 1.0))
        if kind == "hybrid" else None,
    )


def _manifest(
    corpus: PdfLibraryCorpus, kind: str = "bm25",
) -> PdfMultibookRetrievalManifest:
    golden = _golden(corpus)
    return build_pdf_multibook_manifest(
        golden=golden, golden_sha256=HASH, corpus=corpus,
        query_plan=build_pdf_multibook_query_plan(golden),
        retrieval=_config(kind), environment=ENV,
    )


def _results(
    manifest: PdfMultibookRetrievalManifest,
    hits: tuple[PdfMultibookRankedHit, ...] = (),
    context: tuple[PdfMultibookContextHit, ...] = (),
) -> tuple[PdfMultibookQueryResult, ...]:
    return tuple(PdfMultibookQueryResult(
        query=query.identity, ranking=hits,
        context=tuple(PdfMultibookContextAtCutoff(
            cutoff=cutoff, records=context,
        ) for cutoff in (5, 10)),
    ) for query in manifest.query_plan.queries)


def _run(corpus: PdfLibraryCorpus) -> PdfMultibookRetrievalRun:
    manifest = _manifest(corpus)
    return build_pdf_multibook_run(manifest, _results(
        manifest, (PdfMultibookRankedHit(record_id="a1", score=2.0),),
        (PdfMultibookContextHit(record_id="a2", parent_record_ids=("a1",)),),
    ))


def _check(run: PdfMultibookRetrievalRun, corpus: PdfLibraryCorpus) -> None:
    golden = _golden(corpus)
    require_pdf_multibook_run_matches(
        run, golden=golden, golden_sha256=HASH, corpus=corpus,
        query_plan=build_pdf_multibook_query_plan(golden),
        retrieval=run.manifest.retrieval,
    )
