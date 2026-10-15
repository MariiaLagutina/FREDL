"""Synthetic acceptance tests for B0 identities and artifact validation."""

from dataclasses import replace
import json
from typing import Any

import pytest
from pydantic import ValidationError

from src.evaluation.pdf.library import (
    build_pdf_library_corpus,
    PdfLibraryBookInput,
    PdfLibraryCorpus,
)
from src.evaluation.pdf.manifest import (
    bm25_configuration,
    PdfRetrieverKind,
    PdfRunEnvironment,
    rrf_configuration,
    semantic_configuration,
)
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook_manifest import (
    build_pdf_multibook_manifest,
    PdfMultibookRetrievalConfiguration,
    PdfMultibookRetrievalManifest,
    require_pdf_multibook_manifest_matches,
)
from src.evaluation.pdf.multibook_queries import (
    build_pdf_multibook_query_plan,
    PdfMultibookQueryPlan,
    PdfQueryMode,
    require_pdf_query_plan_matches,
)
from src.evaluation.pdf.multibook_runs import (
    build_pdf_multibook_run,
    PdfMultibookContextAtCutoff,
    PdfMultibookContextHit,
    PdfMultibookQueryResult,
    PdfMultibookRankedHit,
    PdfMultibookRetrievalRun,
    require_pdf_multibook_run_matches,
)
from src.ingestion.pdf.retrieval_corpus import (
    build_pdf_retrieval_corpus,
    PdfRecordClassification,
    PdfRecordRole,
    PdfRetrievalPolicy,
)
from src.retrieval.bm25 import BM25Parameters
from src.retrieval.hybrid import RRFParameters
from src.retrieval.semantic.config import SemanticEncoderConfig
from tests.evaluation.pdf.test_multibook_judgments import (
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


def test_query_modes_keep_source_order_and_structured_identity(
    corpus: PdfLibraryCorpus,
) -> None:
    golden = _golden(corpus)
    expected = {
        PdfQueryMode.ORIGINAL: [("both", None), ("which", None),
                                ("absent", None)],
        PdfQueryMode.CLARIFIED: [("which", "queues"), ("which", "producers")],
        PdfQueryMode.ALL: [
            ("both", None), ("which", None),
            ("which", "queues"), ("which", "producers"), ("absent", None),
        ],
    }
    before = golden.model_dump_json()
    for mode, keys in expected.items():
        plan = build_pdf_multibook_query_plan(golden, mode)
        assert [query.identity.key for query in plan.queries] == keys
        require_pdf_query_plan_matches(plan, golden)
        assert plan == PdfMultibookQueryPlan.model_validate_json(
            plan.model_dump_json(),
        )
    assert golden.model_dump_json() == before
    assert build_pdf_multibook_query_plan(golden).queries[1].track == (
        "original_ambiguous"
    )


@pytest.mark.parametrize("missing", [None, "", "  "])
def test_selected_clarified_question_cannot_be_invented(
    corpus: PdfLibraryCorpus, missing: str | None,
) -> None:
    data = _golden(corpus).model_dump(mode="json")
    data["cases"][1]["interpretations"][0]["clarified_question"] = missing
    golden = PdfMultibookGolden.model_validate_json(json.dumps(data))
    build_pdf_multibook_query_plan(golden, PdfQueryMode.ORIGINAL)
    for mode in (PdfQueryMode.CLARIFIED, PdfQueryMode.ALL):
        with pytest.raises(ValueError, match="Missing clarified"):
            build_pdf_multibook_query_plan(golden, mode)


@pytest.mark.parametrize("change", ["missing", "order", "text", "identity"])
def test_query_plan_must_match_exact_golden_variants(
    corpus: PdfLibraryCorpus, change: str,
) -> None:
    golden = _golden(corpus)
    data = build_pdf_multibook_query_plan(golden).model_dump(mode="json")
    if change == "missing":
        data["queries"].pop()
    elif change == "order":
        data["queries"].reverse()
    elif change == "text":
        data["queries"][0]["text"] = "Unapproved query text."
    else:
        data["queries"][2]["identity"]["interpretation_id"] = "unknown"
    plan = PdfMultibookQueryPlan.model_validate_json(json.dumps(data))
    with pytest.raises(ValueError, match="query plan differs"):
        require_pdf_query_plan_matches(plan, golden)


@pytest.mark.parametrize("change", ["duplicate", "track", "mode", "blank"])
def test_query_plan_rejects_inconsistent_metadata(
    corpus: PdfLibraryCorpus, change: str,
) -> None:
    data = _manifest(corpus).query_plan.model_dump(mode="json")
    if change == "duplicate":
        data["queries"].append(data["queries"][0])
    elif change == "track":
        data["queries"][0]["track"] = "clarified"
    elif change == "mode":
        data["mode"] = "original"
    else:
        data["queries"][0]["text"] = "  "
    with pytest.raises(ValidationError):
        PdfMultibookQueryPlan.model_validate_json(json.dumps(data))


def test_algorithm_identity_is_distinct_but_comparison_is_shared(
    corpus: PdfLibraryCorpus,
) -> None:
    manifests = [_manifest(corpus, kind)
                 for kind in ("bm25", "minilm", "hybrid")]
    assert len({manifest.identity for manifest in manifests}) == 3
    assert len({manifest.comparison_identity for manifest in manifests}) == 1
    assert len(corpus.books) == 4


def test_canonical_hashes_ignore_json_layout_and_environment(
    corpus: PdfLibraryCorpus,
) -> None:
    run = _run(corpus)
    data = run.model_dump(mode="json")
    loaded = PdfMultibookRetrievalRun.model_validate_json(
        json.dumps(data, sort_keys=True, indent=4),
    )
    assert loaded.results_fingerprint == run.results_fingerprint
    assert loaded.manifest.query_plan.fingerprint == (
        run.manifest.query_plan.fingerprint
    )
    data["manifest"]["code_revision"] = "different-revision"
    data["manifest"]["code_dirty"] = True
    data["manifest"]["environment"] = {"python": "different"}
    changed = PdfMultibookRetrievalRun.model_validate_json(json.dumps(data))
    assert changed.manifest.identity == run.manifest.identity
    assert changed.results_fingerprint == run.results_fingerprint
    _check(changed, corpus)


def test_configuration_does_not_determine_results(
    corpus: PdfLibraryCorpus,
) -> None:
    first = _run(corpus)
    changed = tuple(result.model_copy(update={"ranking": (
        PdfMultibookRankedHit(record_id="a1", score=3.0),
    )}) for result in first.results)
    second = build_pdf_multibook_run(first.manifest, changed)
    assert first.manifest_identity == second.manifest_identity
    assert first.results_fingerprint != second.results_fingerprint
    _check(first, corpus)
    _check(second, corpus)


@pytest.mark.parametrize("field", [
    "golden_sha256", "golden_content_fingerprint", "corpus_fingerprint",
    "library_fingerprint", "query_plan_fingerprint", "dataset_id",
    "eligible_record_count", "books",
])
def test_manifest_rejects_drift_and_inconsistent_metadata(
    corpus: PdfLibraryCorpus, field: str,
) -> None:
    manifest = _manifest(corpus)
    data = manifest.model_dump(mode="json")
    if field == "eligible_record_count":
        data[field] += 1
    elif field == "books":
        data[field][0]["corpus_fingerprint"] = "0" * 64
    else:
        data[field] = "different" if field == "dataset_id" else "0" * 64
    with pytest.raises(ValueError):
        changed = PdfMultibookRetrievalManifest.model_validate_json(
            json.dumps(data),
        )
        require_pdf_multibook_manifest_matches(
            changed, golden=_golden(corpus), golden_sha256=HASH,
            corpus=corpus, query_plan=manifest.query_plan,
            retrieval=manifest.retrieval,
        )


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
    ("bm25", None), ("semantic", {}),
    ("rrf", {"rank_constant": 60, "lexical_weight": 1.0,
             "semantic_weight": 1.0}),
])
def test_incompatible_algorithm_fields_are_rejected(
    field: str, value: Any,
) -> None:
    data = _config().model_dump(mode="json")
    data[field] = value
    with pytest.raises(ValidationError):
        PdfMultibookRetrievalConfiguration.model_validate_json(
            json.dumps(data),
        )


def test_expected_retrieval_configuration_cannot_silently_change(
    corpus: PdfLibraryCorpus,
) -> None:
    manifest = _manifest(corpus)
    with pytest.raises(ValueError, match="manifest differs"):
        require_pdf_multibook_manifest_matches(
            manifest, golden=_golden(corpus), golden_sha256=HASH,
            corpus=corpus, query_plan=manifest.query_plan,
            retrieval=_config("hybrid"),
        )


def test_library_order_is_canonical_but_policy_drift_is_incompatible(
    corpus: PdfLibraryCorpus,
) -> None:
    inputs = [PdfLibraryBookInput(
        book.book_path, book.pdf_sha256, tuple(reversed(book.records)), (),
    ) for book in reversed(corpus.books)]
    reordered = build_pdf_library_corpus(inputs, corpus.policy)
    assert _manifest(reordered).identity == _manifest(corpus).identity
    changed = build_pdf_library_corpus(
        inputs, PdfRetrievalPolicy.EXCLUDE_NAVIGATION,
    )
    assert changed.manifest == corpus.manifest
    assert _manifest(changed).comparison_identity != (
        _manifest(corpus).comparison_identity
    )
    with pytest.raises(ValueError, match="manifest differs"):
        _check(_run(corpus), changed)


@pytest.mark.parametrize("field,value", [
    ("k1", 0.0), ("b", 1.1), ("metadata_weight", -1.0),
    ("identifier_weight", float("nan")), ("tokenization", "unknown"),
])
def test_invalid_lexical_configuration_is_rejected(
    field: str, value: Any,
) -> None:
    data = _config().model_dump(mode="json")
    data["bm25"][field] = value
    with pytest.raises(ValidationError):
        PdfMultibookRetrievalConfiguration.model_validate_json(
            json.dumps(data),
        )


@pytest.mark.parametrize("field,value", [
    ("runtime", "unknown"), ("model_revision", "  "), ("max_length", 0),
])
def test_invalid_semantic_configuration_is_rejected(
    field: str, value: Any,
) -> None:
    data = _config("minilm").model_dump(mode="json")
    data["semantic"][field] = value
    with pytest.raises(ValidationError):
        PdfMultibookRetrievalConfiguration.model_validate_json(
            json.dumps(data),
        )


@pytest.mark.parametrize("field,value", [
    ("rank_constant", 0), ("lexical_weight", -1.0),
    ("semantic_weight", float("inf")),
])
def test_invalid_fusion_configuration_is_rejected(
    field: str, value: Any,
) -> None:
    data = _config("hybrid").model_dump(mode="json")
    data["rrf"][field] = value
    with pytest.raises(ValidationError):
        PdfMultibookRetrievalConfiguration.model_validate_json(
            json.dumps(data),
        )


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


def test_compatibility_revalidates_copied_nested_configuration(
    corpus: PdfLibraryCorpus,
) -> None:
    manifest = _manifest(corpus, "minilm")
    assert manifest.retrieval.semantic is not None
    semantic = manifest.retrieval.semantic.model_copy(update={"max_length": 0})
    retrieval = manifest.retrieval.model_copy(update={"semantic": semantic})
    copied = manifest.model_copy(update={"retrieval": retrieval})
    with pytest.raises(ValidationError):
        require_pdf_multibook_manifest_matches(
            copied, golden=_golden(corpus), golden_sha256=HASH,
            corpus=corpus, query_plan=manifest.query_plan, retrieval=retrieval,
        )
