"""Pooling provenance, interpretation isolation and fail-closed inputs."""

from dataclasses import replace
import json
from typing import Any

import pytest
from pydantic import ValidationError

from src.evaluation.pdf.library import PdfLibraryCorpus
from src.evaluation.pdf.multibook_pool_models import (
    PdfCandidateOrigin, PdfPoolingInputs, PdfRetrievalJudgmentPool,
)
from src.evaluation.pdf.multibook_pooling import (
    require_pdf_retrieval_pool_matches,
)
from tests.evaluation.pdf import multibook_contract_helpers as contracts
from tests.evaluation.pdf import multibook_pool_helpers as helpers
from tests.evaluation.pdf import multibook_runner_helpers as runner
from tests.evaluation.pdf.multibook_pool_helpers import PoolInputs, build_pool


corpus = contracts.corpus
pool_inputs = helpers.pool_inputs
forbid_real_model_loading = runner.forbid_real_model_loading


def test_union_retains_rank_anchor_context_and_all_retriever_origins(
    pool_inputs: PoolInputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Pooling must never rerun retrieval")
    monkeypatch.setattr(
        "src.evaluation.pdf.bm25_baseline.PdfBm25Baseline.search", forbidden,
    )
    pool = build_pool(pool_inputs)
    assert len(pool.runs) == 3
    a1 = next(item for item in pool.candidates
              if item.key == ("both", None, "a1"))
    assert {origin.retriever.value for origin in a1.origins
            if origin.kind == "ranked" and origin.retriever is not None} == {
        "bm25", "minilm", "hybrid",
    }
    assert any(origin.kind == "anchor" and origin.unit_id == "queue"
               for origin in a1.origins)
    a2 = next(item for item in pool.candidates
              if item.key == ("both", None, "a2"))
    assert any(origin.kind == "context" and origin.parent_record_ids == (
        "a1",
    ) for origin in a2.origins)
    assert {task.unit_id for task in pool.tasks
            if task.query.case_id == "both" and task.record_id == "a1"} == {
        "queue", "cache",
    }
    assert not any('relevance' in item.model_dump() for item in pool.tasks)


def test_variants_never_share_tasks_and_negative_inventory_is_kept(
    pool_inputs: PoolInputs,
) -> None:
    pool = build_pool(pool_inputs)
    for interpretation, units in [(None, {"queue", "backpressure"}),
                                  ("queues", {"queue"}),
                                  ("producers", {"backpressure"})]:
        assert {task.unit_id for task in pool.tasks
                if task.query.key == ("which", interpretation)} == units
    assert any(item.query.case_id == "absent" for item in pool.candidates)
    assert not any(task.query.case_id == "absent" for task in pool.tasks)
    negative_anchor = next(candidate for candidate in pool.candidates
                           if candidate.key == ("absent", None, "b1"))
    assert any(origin.kind == "anchor" and origin.unit_id == "cache"
               for origin in negative_anchor.origins)
    assert len({task.key for task in pool.tasks}) == len(pool.tasks)
    # Cross-book competitors are judged for every target unit, without quotas.
    assert any(task.unit_id == "queue" and task.record_id == "b1"
               for task in pool.tasks)


def test_run_input_order_does_not_change_pool_or_fingerprint(
    pool_inputs: PoolInputs,
) -> None:
    golden, corpus, runs, pages = pool_inputs
    pool = build_pool(pool_inputs)
    reverse = build_pool((golden, corpus, tuple(reversed(runs)), pages))
    assert reverse == pool
    assert reverse.fingerprint == pool.fingerprint
    restored = PdfRetrievalJudgmentPool.model_validate_json(
        pool.model_dump_json(indent=3),
    )
    assert restored.fingerprint == pool.fingerprint
    ids = {record.record_id: index
           for index, record in enumerate(corpus.source_records)}
    for query in pool.query_plan.queries:
        selected = [ids[item.record_id] for item in pool.candidates
                    if item.query == query.identity]
        assert selected == sorted(selected)


@pytest.mark.parametrize("depth", [0, -1, 1, 6, 20, True, 10.0])
def test_depth_requires_available_integer_context_cutoff(
    pool_inputs: PoolInputs, depth: Any,
) -> None:
    with pytest.raises(ValueError, match="Depth"):
        build_pool(pool_inputs, depth=depth)


@pytest.mark.parametrize("change", [
    "missing", "duplicate", "hash", "copied_run", "copied_config",
    "plan", "context", "corpus", "locator", "page_hash",
])
def test_incompatible_or_copied_inputs_fail_closed(
    pool_inputs: PoolInputs, change: str,
) -> None:
    golden, corpus, runs, pages = pool_inputs
    if change == "missing":
        runs = runs[:-1]
    elif change == "duplicate":
        runs = (runs[0], runs[0], runs[2])
    elif change == "hash":
        run = runs[0].run
        manifest = run.manifest.model_copy(update={"golden_sha256": "0" * 64})
        from src.evaluation.pdf.multibook_runs import build_pdf_multibook_run
        runs = (runs[0].model_copy(update={
            "run": build_pdf_multibook_run(manifest, run.results),
        }), *runs[1:])
    elif change.startswith("copied"):
        item = runs[0]
        if change == "copied_run":
            run = item.run.model_copy(update={"results": ()})
        else:
            config = item.run.manifest.retrieval.model_copy(update={
                "candidate_scope": "top20",
            })
            run = item.run.model_copy(update={"manifest": (
                item.run.manifest.model_copy(update={"retrieval": config})
            )})
        runs = (item.model_copy(update={"run": run}), *runs[1:])
    elif change in {"plan", "context"}:
        manifest = runs[0].run.manifest
        field = "query_plan" if change == "plan" else "context"
        value = manifest.query_plan.model_copy(update={"queries": ()}) if (
            change == "plan"
        ) else manifest.context.model_copy(update={"cutoffs": (10, 5)})
        run = runs[0].run.model_copy(update={"manifest": (
            manifest.model_copy(update={field: value})
        )})
        runs = (runs[0].model_copy(update={"run": run}), *runs[1:])
    elif change == "corpus":
        corpus = replace(corpus, books=tuple(reversed(corpus.books)))
    else:
        data = golden.model_dump(mode="json")
        locator = data['evidence_units'][0]['locators'][0]
        locator['exact_anchor' if change == 'locator'
                else 'page_text_sha256'] = (
            'X' * len(locator['exact_anchor']) if change == 'locator'
            else '0' * 64
        )
        golden = type(golden).model_validate_json(json.dumps(data))
    with pytest.raises(ValueError):
        build_pool((golden, corpus, runs, pages))


@pytest.mark.parametrize("change", ["order", "unit", "origin", "book"])
def test_pool_compatibility_rebuilds_without_repairing(
    pool_inputs: PoolInputs, change: str,
) -> None:
    golden, corpus, runs, pages = pool_inputs
    pool = build_pool(pool_inputs)
    if change == "order":
        pool = pool.model_copy(update={"tasks": tuple(reversed(pool.tasks))})
    elif change == "unit":
        pool = pool.model_copy(update={"tasks": (
            pool.tasks[0].model_copy(update={"unit_id": "unknown"}),
            *pool.tasks[1:],
        )})
    else:
        candidate = pool.candidates[0]
        candidate = candidate.model_copy(update=(
            {"book_path": "books/other.pdf"} if change == "book" else
            {"origins": (PdfCandidateOrigin(
                kind="ranked", retriever=pool.runs[0].retriever,
                rank=99, score=1.0,
            ),)}
        ))
        pool = pool.model_copy(update={"candidates": (
            candidate, *pool.candidates[1:],
        )})
    with pytest.raises(ValueError):
        require_pdf_retrieval_pool_matches(
            pool, golden=golden, golden_sha256=contracts.HASH,
            corpus=corpus, runs=runs, reading_pages=pages,
        )


@pytest.mark.parametrize("payload", [
    {"kind": "ranked"},
    {"kind": "anchor", "unit_id": "queue"},
    {"kind": "context", "retriever": "bm25", "rank": 1},
    {"kind": "ranked", "retriever": "bm25", "rank": 1,
     "relevance": "required"},
])
def test_origin_rejects_invalid_fields(payload: dict[str, Any]) -> None:
    import json
    with pytest.raises(ValidationError):
        PdfCandidateOrigin.model_validate_json(json.dumps(payload))


def test_input_manifest_rejects_duplicate_retrievers() -> None:
    with pytest.raises(ValidationError):
        PdfPoolingInputs.model_validate_json('''{"runs": [
          {"retriever":"bm25","path":"a.json","sha256":"%s"},
          {"retriever":"bm25","path":"b.json","sha256":"%s"},
          {"retriever":"hybrid","path":"c.json","sha256":"%s"}
        ]}''' % (('a' * 64,) * 3))


@pytest.mark.parametrize('mode', ['original', 'clarified', 'all'])
def test_pool_tracks_exact_run_query_selection(
    pool_inputs: PoolInputs, mode: str,
) -> None:
    from src.evaluation.pdf.multibook_queries import PdfQueryMode
    golden, corpus, _, pages = pool_inputs
    runs = helpers.saved_runs(golden, corpus, mode=PdfQueryMode(mode))
    pool = build_pool((golden, corpus, runs, pages))
    assert pool.query_plan == runs[0].run.manifest.query_plan
    assert {candidate.query.key for candidate in pool.candidates} == {
        query.identity.key for query in pool.query_plan.queries
    }


@pytest.mark.parametrize('depth', [5, 10])
def test_large_corpus_uses_exact_top_depth_and_cutoff_context(
    corpus: PdfLibraryCorpus, depth: int,
) -> None:
    from src.evaluation.pdf.library import (
        PdfLibraryBookInput, build_pdf_library_corpus,
    )
    from src.evaluation.pdf.multibook_import import anchor_records
    from src.evaluation.pdf.multibook_pooling import query_pool_units
    from tests.evaluation.pdf.multibook_golden_helpers import _records
    inputs = []
    for book in corpus.books:
        records = book.records
        if book.book_path == 'books/gamma.pdf':
            extra = _records(book.book_path, tuple(
                (f'large-{n}', 'queues ' * (30 - n), n + 2, 0)
                for n in range(20)
            ))
            records = (*records, *(
                replace(record, flow_index=n + 1)
                for n, record in enumerate(extra)
            ))
        inputs.append(PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, records, (),
        ))
    corpus = build_pdf_library_corpus(inputs, corpus.policy)
    golden, corpus, runs, pages = helpers.synthetic_inputs(corpus)
    pool = build_pool((golden, corpus, runs, pages), depth=depth)
    assert len(corpus.records) > 20
    for index, query in enumerate(pool.query_plan.queries):
        expected: set[str] = set()
        for item in runs:
            result = item.run.results[index]
            expected.update(hit.record_id for hit in result.ranking[:depth])
            expected.update(hit.record_id for context in result.context
                            if context.cutoff == depth
                            for hit in context.records)
        for unit_id in query_pool_units(
            golden, query.identity, include_related=True,
        ):
            unit = golden.units[unit_id]
            for locator in (*unit.locators, *unit.supporting_locators):
                expected.update(anchor_records(locator, corpus.source_records))
        assert {candidate.record_id for candidate in pool.candidates
                if candidate.query == query.identity} == expected


def test_excluded_anchor_stays_a_candidate_without_inventing_retrieval_origin(
    pool_inputs: PoolInputs,
) -> None:
    from src.ingestion.pdf.retrieval_corpus import (
        PdfRecordClassification, PdfRecordRole, PdfRetrievalPolicy,
        build_pdf_retrieval_corpus,
    )
    golden, corpus, _, pages = pool_inputs
    policy = PdfRetrievalPolicy.EXCLUDE_NAVIGATION
    books = tuple(replace(book, corpus=build_pdf_retrieval_corpus(
        book.records, tuple(PdfRecordClassification(
            record.record_id,
            PdfRecordRole.INDEX if record.record_id == 'a1'
            else PdfRecordRole.CONTENT,
        ) for record in book.records), policy,
    )) for book in corpus.books)
    corpus = replace(corpus, policy=policy, books=books)
    runs = helpers.saved_runs(golden, corpus)
    pool = build_pool((golden, corpus, runs, pages))
    a1 = next(candidate for candidate in pool.candidates
              if candidate.key == ('both', None, 'a1'))
    assert {origin.kind for origin in a1.origins} == {'anchor'}


@pytest.mark.parametrize('change', ['query_mode', 'policy', 'bad_context'])
def test_individually_valid_runs_must_share_pooling_protocol(
    pool_inputs: PoolInputs, change: str,
) -> None:
    from src.evaluation.pdf.multibook_queries import PdfQueryMode
    from src.evaluation.pdf.multibook_runs import build_pdf_multibook_run
    from src.evaluation.pdf.library import (
        PdfLibraryBookInput, build_pdf_library_corpus,
    )
    from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy
    golden, corpus, runs, pages = pool_inputs
    if change == 'query_mode':
        replacement = helpers.saved_runs(
            golden, corpus, mode=PdfQueryMode.ORIGINAL,
        )[0]
    elif change == 'policy':
        inputs = [PdfLibraryBookInput(book.book_path, book.pdf_sha256,
                                      book.records, ())
                  for book in corpus.books]
        other = build_pdf_library_corpus(
            inputs, PdfRetrievalPolicy.EXCLUDE_NAVIGATION,
        )
        replacement = helpers.saved_runs(golden, other)[0]
    else:
        run = runs[0].run
        result = run.results[0]
        contexts = tuple(part.model_copy(update={'records': ()})
                         for part in result.context)
        changed = result.model_copy(update={'context': contexts})
        replacement = runs[0].model_copy(update={'run': (
            build_pdf_multibook_run(run.manifest,
                                    (changed, *run.results[1:]))
        )})
    runs = (replacement, *runs[1:])
    with pytest.raises(ValueError):
        build_pool((golden, corpus, runs, pages))
