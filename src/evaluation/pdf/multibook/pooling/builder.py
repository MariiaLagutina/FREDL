"""Deterministic query-isolated pooling from validated global run artifacts."""

from collections.abc import Mapping, Sequence
from typing import Literal

from src.evaluation.pdf.multibook.library import PdfLibraryCorpus
from src.evaluation.pdf.manifest import PdfRetrieverKind
from src.evaluation.pdf.multibook.curation.golden import (
    PdfEvidenceLocator, PdfMultibookGolden,
)
from src.evaluation.pdf.multibook.curation.importer import (
    anchor_records, verify_pdf_multibook_golden,
)
from src.evaluation.pdf.multibook.curation.pilot_judgments import pool_units
from src.evaluation.pdf.multibook.pooling.models import (
    PdfCandidateOrigin, PdfPoolCandidate, PdfPoolingRun,
    PdfPoolRunIdentity, PdfRetrievalJudgmentPool, PdfRetrievalJudgmentTask,
)
from src.evaluation.pdf.multibook.retrieval.queries import PdfQueryIdentity
from src.evaluation.pdf.multibook.retrieval.runner import (
    _require_consistent_corpus,
)
from src.evaluation.pdf.multibook.retrieval.runs import (
    require_pdf_multibook_run_matches,
)
from src.ingestion.pdf.reading_text import PdfReadingPage


def query_pool_units(
    golden: PdfMultibookGolden, identity: PdfQueryIdentity,
    *, include_related: bool = False,
) -> tuple[str, ...]:
    """Select this meaning's targets, or all its anchor units when requested.

    Original ambiguous questions retain separate conditional-unit tasks.
    Nothing here combines their evidence or assesses sufficient sets.
    """
    case = next(case for case in golden.cases
                if case.case_id == identity.case_id)
    if identity.interpretation_id is None:
        return case.unit_ids if include_related else pool_units(case)
    meaning = next(item for item in case.interpretations
                   if item.interpretation_id == identity.interpretation_id)
    return meaning.unit_ids if include_related else meaning.set_units


def build_pdf_retrieval_judgment_pool(
    *, golden: PdfMultibookGolden, golden_sha256: str,
    corpus: PdfLibraryCorpus, runs: Sequence[PdfPoolingRun],
    reading_pages: Mapping[str, Sequence[PdfReadingPage]], depth: int = 10,
) -> PdfRetrievalJudgmentPool:
    """Union global top-k, direct context and verified anchors, without labels.

    Exact file hashes are checked by the I/O caller. Revalidate even copied
    Pydantic objects before inspecting any fields. No retrieval/model executes.
    """
    if type(depth) is not int or depth not in (5, 10):
        raise ValueError("Depth must be 5 or 10 (available context cutoffs)")
    golden = PdfMultibookGolden.model_validate(golden.model_dump())
    checked = tuple(PdfPoolingRun.model_validate(item.model_dump())
                    for item in runs)
    _require_consistent_corpus(corpus)
    verify_pdf_multibook_golden(golden, corpus, reading_pages)
    by_kind = {item.run.manifest.retrieval.retriever: item for item in checked}
    if len(checked) != 3 or set(by_kind) != set(PdfRetrieverKind):
        raise ValueError("Need exactly BM25, MiniLM and Hybrid runs")
    ordered = tuple(by_kind[kind] for kind in PdfRetrieverKind)
    manifest = ordered[0].run.manifest
    for item in ordered:
        run = item.run
        require_pdf_multibook_run_matches(
            run, golden=golden, golden_sha256=golden_sha256, corpus=corpus,
            query_plan=manifest.query_plan, retrieval=run.manifest.retrieval,
        )
        if run.manifest.comparison_identity != manifest.comparison_identity:
            raise ValueError("Runs have different comparison identities")
    candidates: list[PdfPoolCandidate] = []
    tasks: list[PdfRetrievalJudgmentTask] = []
    records = corpus.source_records
    for index, query in enumerate(manifest.query_plan.queries):
        units = query_pool_units(golden, query.identity)
        origins: dict[str, list[PdfCandidateOrigin]] = {}
        for item in ordered:
            result = item.run.results[index]
            kind = item.run.manifest.retrieval.retriever
            for rank, hit in enumerate(result.ranking[:depth], start=1):
                origins.setdefault(hit.record_id, []).append(
                    PdfCandidateOrigin(kind="ranked", retriever=kind,
                                       rank=rank, score=hit.score),
                )
            context = next(part for part in result.context
                           if part.cutoff == depth)
            for neighbor in context.records:
                origins.setdefault(neighbor.record_id, []).append(
                    PdfCandidateOrigin(
                        kind="context", retriever=kind,
                        cutoff=5 if depth == 5 else 10,
                        parent_record_ids=neighbor.parent_record_ids,
                    ),
                )
        anchor_units = query_pool_units(
            golden, query.identity, include_related=True,
        )
        for unit_id in anchor_units:
            unit = golden.units[unit_id]
            locator_groups: tuple[tuple[
                Literal["anchor", "supporting_anchor"],
                tuple[PdfEvidenceLocator, ...],
            ], ...] = (("anchor", unit.locators),
                       ("supporting_anchor", unit.supporting_locators))
            for source, locators in locator_groups:
                for locator_index, locator in enumerate(locators):
                    for record_id in anchor_records(locator, records):
                        origins.setdefault(record_id, []).append(
                            PdfCandidateOrigin(
                                kind=source, unit_id=unit_id,
                                locator_index=locator_index,
                            ),
                        )
        selected = [record for record in records
                    if record.record_id in origins]
        candidates.extend(PdfPoolCandidate(
            query=query.identity, book_path=record.book_path,
            record_id=record.record_id,
            origins=tuple(origins[record.record_id]),
        ) for record in selected)
        tasks.extend(PdfRetrievalJudgmentTask(
            query=query.identity, unit_id=unit_id, record_id=record.record_id,
        ) for unit_id in units for record in selected)
    return PdfRetrievalJudgmentPool(
        golden_sha256=golden_sha256,
        golden_content_fingerprint=golden.content_fingerprint,
        library_fingerprint=corpus.manifest.fingerprint,
        corpus_fingerprint=corpus.fingerprint,
        comparison_identity=manifest.comparison_identity,
        query_plan=manifest.query_plan, depth=5 if depth == 5 else 10,
        runs=tuple(PdfPoolRunIdentity(
            retriever=item.run.manifest.retrieval.retriever,
            artifact_sha256=item.artifact_sha256,
            manifest_identity=item.run.manifest_identity,
            results_fingerprint=item.run.results_fingerprint,
        ) for item in ordered),
        candidates=tuple(candidates), tasks=tuple(tasks),
    )


def require_pdf_retrieval_pool_matches(
    pool: PdfRetrievalJudgmentPool, *, golden: PdfMultibookGolden,
    golden_sha256: str, corpus: PdfLibraryCorpus,
    runs: Sequence[PdfPoolingRun],
    reading_pages: Mapping[str, Sequence[PdfReadingPage]],
) -> None:
    """Rebuild and compare; never repair order, provenance or unit bindings."""
    pool = PdfRetrievalJudgmentPool.model_validate(pool.model_dump())
    expected = build_pdf_retrieval_judgment_pool(
        golden=golden, golden_sha256=golden_sha256, corpus=corpus,
        runs=runs, reading_pages=reading_pages, depth=pool.depth,
    )
    if pool != expected:
        raise ValueError("Retrieval pool differs from validated inputs")
