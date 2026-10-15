"""Ranking artifacts, context provenance and run compatibility."""

from dataclasses import replace
import json
from typing import Any

from pydantic import ValidationError
import pytest

from src.evaluation.pdf.multibook.library import (
    PdfLibraryBookInput,
    PdfLibraryCorpus,
    build_pdf_library_corpus,
)
from src.evaluation.pdf.multibook.retrieval.runs import (
    PdfMultibookContextAtCutoff,
    PdfMultibookContextHit,
    PdfMultibookQueryResult,
    PdfMultibookRankedHit,
    PdfMultibookRetrievalRun,
    build_pdf_multibook_run,
)
from src.ingestion.pdf.retrieval_corpus import (
    PdfRecordClassification,
    PdfRecordRole,
    PdfRetrievalPolicy,
    build_pdf_retrieval_corpus,
)

from tests.evaluation.pdf.multibook.retrieval import (
    contract_helpers as contract_fixtures,
)
from tests.evaluation.pdf.multibook.retrieval.contract_helpers import (
    _check,
    _manifest,
    _results,
    _run,
)


corpus = contract_fixtures.corpus
no_retrieval_or_model_loading = contract_fixtures.no_retrieval_or_model_loading


@pytest.mark.parametrize("schema", ["query_plan", "manifest", "run"])
@pytest.mark.parametrize("field,value", [
    ("version", 999), ("schema_name", "unsupported"), ("timestamp", "now"),
    ("version", True), ("version", 1.0),
])
def test_unsupported_artifacts_and_extra_metadata_are_rejected(
    corpus: PdfLibraryCorpus, schema: str, field: str, value: Any,
) -> None:
    run = _run(corpus)
    obj: Any = {"query_plan": run.manifest.query_plan,
                "manifest": run.manifest, "run": run}[schema]
    data = obj.model_dump(mode="json")
    data[field] = value
    with pytest.raises(ValidationError):
        type(obj).model_validate_json(json.dumps(data))


@pytest.mark.parametrize("change", ["missing", "duplicate", "reorder", "hash"])
def test_run_rejects_missing_duplicate_queries_and_result_tampering(
    corpus: PdfLibraryCorpus, change: str,
) -> None:
    data = _run(corpus).model_dump(mode="json")
    if change == "missing":
        data["results"].pop()
    elif change == "duplicate":
        data["results"][1] = data["results"][0]
    elif change == "reorder":
        data["results"].reverse()
    else:
        data["results"][0]["ranking"][0]["score"] += 1.0
    with pytest.raises(ValidationError):
        PdfMultibookRetrievalRun.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("score", [
    float("inf"), float("nan"), -float("inf"),
])
def test_scores_must_be_finite(score: float) -> None:
    with pytest.raises(ValidationError):
        PdfMultibookRankedHit(record_id="synthetic", score=score)


def test_duplicate_ranking_references_are_rejected(
    corpus: PdfLibraryCorpus,
) -> None:
    manifest = _manifest(corpus)
    hit = PdfMultibookRankedHit(record_id="a1", score=2.0)
    with pytest.raises(ValidationError, match="Duplicate ranking"):
        _results(manifest, (hit, hit))


@pytest.mark.parametrize("change", [
    "unknown", "order", "zero_score", "source_rank", "cross_book_context",
    "gap_context", "missing_context", "recursive_context", "ranked_context",
])
def test_reference_ranking_and_context_validation(
    corpus: PdfLibraryCorpus, change: str,
) -> None:
    manifest = _manifest(corpus)
    data = _results(manifest, (
        PdfMultibookRankedHit(record_id="a1", score=2.0),
    ), (PdfMultibookContextHit(
        record_id="a2", parent_record_ids=("a1",),
    ),))
    payload = [result.model_dump(mode="json") for result in data]
    for result in payload:
        if change == "unknown":
            result["ranking"][0]["record_id"] = "unknown"
        elif change == "order":
            result["ranking"].append({"record_id": "b1", "score": 3.0})
        elif change == "zero_score":
            result["ranking"][0]["score"] = 0.0
        elif change == "source_rank":
            result["ranking"][0]["lexical_rank"] = 1
        else:
            for context in result["context"]:
                entry = context["records"][0]
                if change == "missing_context":
                    context["records"] = []
                elif change == "recursive_context":
                    entry["parent_record_ids"] = ["a2"]
                else:
                    entry["record_id"] = {
                        "cross_book_context": "b1", "gap_context": "a3",
                        "ranked_context": "a1",
                    }[change]
    results = tuple(
        PdfMultibookQueryResult.model_validate_json(json.dumps(item))
        for item in payload
    )
    with pytest.raises(ValueError):
        _check(build_pdf_multibook_run(manifest, results), corpus)


@pytest.mark.parametrize("kind", ["minilm", "hybrid"])
def test_semantic_and_fused_contracts_without_loading_an_encoder(
    corpus: PdfLibraryCorpus, kind: str,
) -> None:
    manifest = _manifest(corpus, kind)
    hits = tuple(PdfMultibookRankedHit(
        record_id=record.record_id,
        score=(0.9 - index * 0.1) if kind == "minilm" else (
            1 / (61 + index) + (1 / (61 + index) if index < 2 else 0)
        ),
        lexical_rank=index + 1 if kind == "hybrid" and index < 2 else None,
        semantic_rank=index + 1 if kind == "hybrid" else None,
    ) for index, record in enumerate(corpus.records))
    run = build_pdf_multibook_run(manifest, _results(manifest, hits))
    _check(run, corpus)
    shortened = build_pdf_multibook_run(
        manifest, _results(manifest, hits[:-1]),
    )
    with pytest.raises(ValueError):
        _check(shortened, corpus)


@pytest.mark.parametrize("field,value", [
    ("score", 100.0), ("semantic_rank", 999), ("lexical_rank", 999),
])
def test_fused_scores_and_source_ranks_are_verified(
    corpus: PdfLibraryCorpus, field: str, value: float | int,
) -> None:
    manifest = _manifest(corpus, "hybrid")
    hits = [PdfMultibookRankedHit(
        record_id=record.record_id, score=1 / (61 + index),
        semantic_rank=index + 1,
    ) for index, record in enumerate(corpus.records)]
    hits[0] = PdfMultibookRankedHit.model_validate({
        **hits[0].model_dump(), field: value,
    })
    run = build_pdf_multibook_run(manifest, _results(manifest, tuple(hits)))
    with pytest.raises(ValueError):
        _check(run, corpus)


def test_bm25_ties_preserve_textual_flow_key_not_numeric_flow_order(
    corpus: PdfLibraryCorpus,
) -> None:
    inputs = []
    for book in corpus.books:
        records = book.records
        if book.book_path == "books/alpha.pdf":
            records = (replace(records[0], flow_index=2),
                       replace(records[1], flow_index=10), records[2])
        inputs.append(PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, records, (),
        ))
    corpus = build_pdf_library_corpus(inputs, corpus.policy)
    manifest = _manifest(corpus)
    hits = (PdfMultibookRankedHit(record_id="a2", score=1.0),
            PdfMultibookRankedHit(record_id="a1", score=1.0))
    _check(build_pdf_multibook_run(manifest, _results(manifest, hits)), corpus)
    with pytest.raises(ValueError, match="tie order"):
        _check(build_pdf_multibook_run(
            manifest, _results(manifest, tuple(reversed(hits))),
        ), corpus)


@pytest.mark.parametrize("change", ["flow", "interval", "unknown_interval"])
def test_context_requires_known_same_flow_and_interval(
    corpus: PdfLibraryCorpus, change: str,
) -> None:
    old_run = _run(corpus)
    inputs = []
    for book in corpus.books:
        records = book.records
        if book.book_path == "books/alpha.pdf":
            changes: dict[str, dict[str, Any]] = {
                "flow": {"flow_index": 9},
                "interval": {"section_interval_id": "another"},
                "unknown_interval": {"section_interval_id": None},
            }
            fields = changes[change]
            records = (records[0], replace(records[1], **fields), records[2])
        inputs.append(PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, records, (),
        ))
    corpus = build_pdf_library_corpus(inputs, corpus.policy)
    run = build_pdf_multibook_run(_manifest(corpus), old_run.results)
    with pytest.raises(ValueError, match="context provenance"):
        _check(run, corpus)


def test_empty_lexical_ranking_is_valid_without_inventing_answerability(
    corpus: PdfLibraryCorpus,
) -> None:
    manifest = _manifest(corpus)
    _check(build_pdf_multibook_run(manifest, _results(manifest)), corpus)


def test_known_but_excluded_record_cannot_rank_or_supply_context(
    corpus: PdfLibraryCorpus,
) -> None:
    policy = PdfRetrievalPolicy.EXCLUDE_NAVIGATION
    books = tuple(replace(book, corpus=build_pdf_retrieval_corpus(
        book.records,
        tuple(PdfRecordClassification(
            record.record_id,
            PdfRecordRole.INDEX if record.record_id == "a2"
            else PdfRecordRole.CONTENT,
        ) for record in book.records), policy,
    )) for book in corpus.books)
    corpus = replace(corpus, policy=policy, books=books)
    assert "a2" in {record.record_id for record in corpus.source_records}
    assert "a2" not in {record.record_id for record in corpus.records}
    manifest = _manifest(corpus)
    hit = PdfMultibookRankedHit(record_id="a1", score=1.0)
    _check(build_pdf_multibook_run(
        manifest, _results(manifest, (hit,)),
    ), corpus)
    with pytest.raises(ValueError, match="ineligible"):
        _check(build_pdf_multibook_run(manifest, _results(manifest, (
            PdfMultibookRankedHit(record_id="a2", score=1.0),
        ))), corpus)
    with pytest.raises(ValueError, match="context provenance"):
        _check(_run(corpus), corpus)


def test_context_is_bound_to_each_cutoff_not_the_full_ranking(
    corpus: PdfLibraryCorpus,
) -> None:
    inputs = []
    for book in corpus.books:
        records = book.records
        if book.book_path == "books/alpha.pdf":
            extended = []
            cursor = 0
            for index in range(11):
                template = records[min(index, 2)]
                extended.append(replace(
                    template,
                    record_id=template.record_id if index < 3
                    else f"chain-{index + 1}",
                    flow_start=cursor, flow_end=cursor + len(template.text),
                ))
                cursor += len(template.text)
            records = tuple(extended)
        inputs.append(PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, records, (),
        ))
    corpus = build_pdf_library_corpus(inputs, corpus.policy)
    manifest = _manifest(corpus)
    records = corpus.books[0].records
    hits = tuple(PdfMultibookRankedHit(
        record_id=record.record_id, score=float(20 - index),
    ) for index, record in enumerate(records[:10]))
    contexts = tuple(PdfMultibookContextAtCutoff(
        cutoff=cutoff, records=(PdfMultibookContextHit(
            record_id=records[cutoff].record_id,
            parent_record_ids=(records[cutoff - 1].record_id,),
        ),),
    ) for cutoff in (5, 10))
    results = tuple(PdfMultibookQueryResult(
        query=query.identity, ranking=hits, context=contexts,
    ) for query in manifest.query_plan.queries)
    _check(build_pdf_multibook_run(manifest, results), corpus)
    wrong = tuple(result.model_copy(update={"context": (
        contexts[0], contexts[1].model_copy(update={
            "records": contexts[0].records,
        }),
    )}) for result in results)
    with pytest.raises(ValueError, match="context provenance"):
        _check(build_pdf_multibook_run(manifest, wrong), corpus)


@pytest.mark.parametrize("field,value", [
    ("results_fingerprint", "0" * 64),
    ("manifest_identity", "0" * 64),
    ("results", ()),
])
def test_compatibility_revalidates_copied_run_envelopes(
    corpus: PdfLibraryCorpus, field: str, value: Any,
) -> None:
    run = _run(corpus)
    copied = run.model_copy(update={field: value})
    with pytest.raises(ValidationError):
        _check(copied, corpus)
    _check(run, corpus)
