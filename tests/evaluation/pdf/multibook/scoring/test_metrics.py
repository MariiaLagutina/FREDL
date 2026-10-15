"""Synthetic evidence supply, OR-of-AND sets and ranked/context semantics."""

import json

import pytest

from src.evaluation.pdf.multibook.scoring.report import (
    aggregate_multibook_metrics, build_multibook_metric_report,
)
from tests.evaluation.pdf import multibook_contract_helpers as contracts
from tests.evaluation.pdf.multibook.pooling import pool_helpers as helpers
from tests.evaluation.pdf import multibook_runner_helpers as runner
from tests.evaluation.pdf.multibook.scoring.metric_helpers import (
    collect, fill, score, worksheet,
)
from tests.evaluation.pdf.multibook.pooling.pool_helpers import (
    PoolInputs,
    build_pool,
)


corpus = contracts.corpus
pool_inputs = helpers.pool_inputs
forbid_real_model_loading = runner.forbid_real_model_loading


def test_ber5_vs_ber10_and_mrr_truncation(
    pool_inputs: PoolInputs,
) -> None:
    pool = build_pool(pool_inputs)
    sheet = fill(worksheet(pool_inputs, pool), lambda row: (
        ('required', 'yes') if (row['unit_id'], row['record_id']) in {
            ('queue', 'extra-0'), ('cache', 'extra-1'),
        } else ('irrelevant', '')
    ))
    results = score(pool_inputs, pool, collect(pool_inputs, pool, sheet))
    semantic = next(row for row in results
                    if row.retriever.value == 'minilm'
                    and row.query.key == ('both', None))
    assert [row.ber for row in semantic.ranked_only] == [0.0, 1.0]
    assert [row.sufficient_set_success for row in semantic.ranked_only] == [
        False, True,
    ]
    assert semantic.mrr_at_10_supply == pytest.approx(1 / 6)
    assert semantic.mrr_at_10_localization == pytest.approx(1 / 6)
    lexical = next(row for row in results if row.retriever.value == 'bm25'
                   and row.query.key == ('both', None))
    assert lexical.mrr_at_10_supply == 0


def test_context_gain_never_gives_a_rank_to_context_only_evidence(
    pool_inputs: PoolInputs,
) -> None:
    pool = build_pool(pool_inputs)
    sheet = fill(worksheet(pool_inputs, pool), lambda row: (
        ('required', 'yes') if (row['unit_id'], row['record_id']) == (
            'queue', 'a2',
        ) else ('irrelevant', '')
    ))
    results = score(pool_inputs, pool, collect(pool_inputs, pool, sheet))
    row = next(row for row in results if row.retriever.value == 'bm25'
               and row.query.key == ('both', None))
    assert row.ranked_only[0].ber == 0
    assert row.with_context[0].ber == 0.5
    assert [gain.ber for gain in row.context_only_gain] == [0.5, 0.5]
    assert [gain.localization_coverage
            for gain in row.context_only_gain] == [0.5, 0.5]
    assert [gain.sufficient_set_success
            for gain in row.context_only_gain] == [0.0, 0.0]
    assert row.mrr_at_10_supply == row.mrr_at_10_localization == 0
    assert row.with_context[0].context[0].parent_record_ids == ('a1',)
    assert not row.ranked_only[0].context


def test_localization_is_not_supply_and_supporting_cannot_close_a_unit(
    pool_inputs: PoolInputs,
) -> None:
    pool = build_pool(pool_inputs)
    for relevance, supply, localized, supplied in [
        ('required', 'no', 0.5, 0.0),
        ('supporting', 'yes', 0.0, 0.0),
    ]:
        sheet = fill(worksheet(pool_inputs, pool), lambda row: (
            (relevance, supply) if (row['unit_id'], row['record_id']) == (
                'queue', 'a1',
            ) else ('irrelevant', '')
        ))
        result = score(pool_inputs, pool, collect(pool_inputs, pool, sheet))[0]
        assert result.ranked_only[0].localization_coverage == localized
        assert result.ranked_only[0].ber == supplied
        assert result.mrr_at_10_localization == (1 if localized else 0)
        assert result.mrr_at_10_supply == (1 if supplied else 0)


def test_original_and_clarified_units_do_not_share_labels(
    pool_inputs: PoolInputs,
) -> None:
    pool = build_pool(pool_inputs)
    sheet = fill(worksheet(pool_inputs, pool), lambda row: (
        ('required', 'yes') if row['interpretation_id'] == 'queues'
        else ('irrelevant', '')
    ))
    rows = score(pool_inputs, pool, collect(pool_inputs, pool, sheet))
    lexical = {row.query.key: row for row in rows
               if row.retriever.value == 'bm25'}
    assert lexical['which', 'queues'].ranked_only[0].ber == 1
    assert lexical['which', 'producers'].ranked_only[0].ber == 0
    assert lexical['which', None].ranked_only[0].ber is None
    assert lexical['absent', None].ranked_only[0].ber is None
    assert lexical['absent', None].mrr_at_10_supply is None


def test_alternative_sets_use_best_coverage_and_or_of_and_success(
    pool_inputs: PoolInputs,
) -> None:
    golden, corpus, _, pages = pool_inputs
    data = golden.model_dump(mode='json')
    data['cases'][0]['category'] = 'independent_multi_book'
    data['cases'][0]['sufficient_sets'] = [['queue'], ['cache']]
    golden = type(golden).model_validate_json(json.dumps(data))
    inputs = golden, corpus, helpers.saved_runs(golden, corpus), pages
    pool = build_pool(inputs)
    sheet = fill(worksheet(inputs, pool), lambda row: (
        ('required', 'yes') if (row['unit_id'], row['record_id']) == (
            'queue', 'a1',
        ) else ('irrelevant', '')
    ))
    row = score(inputs, pool, collect(inputs, pool, sheet))[0]
    assert row.ranked_only[0].ber == 1
    assert row.ranked_only[0].sufficient_set_success is True


@pytest.mark.parametrize('status,reason', [
    ('partial', None), ('extraction_limited', 'visual_evidence_unavailable'),
    ('source_incomplete', 'source_incomplete'),
])
def test_outcome_slices_and_headline_denominators(
    pool_inputs: PoolInputs, status: str, reason: str | None,
) -> None:
    golden, corpus, _, pages = pool_inputs
    data = golden.model_dump(mode='json')
    case = data['cases'][0]
    case.update(category='reliability_limited', expected_outcome=status,
                limitation_reason=reason)
    if status == 'partial':
        case.update(category='partially_answerable', source_support='partial',
                    text_context_support='partial')
    elif status == 'extraction_limited':
        case.update(text_context_support='extraction_limited')
    else:
        case.update(source_support='incomplete',
                    text_context_support='incomplete',
                    related_units=['queue', 'cache'], sufficient_sets=[])
    golden = type(golden).model_validate_json(json.dumps(data))
    inputs = golden, corpus, helpers.saved_runs(golden, corpus), pages
    pool = build_pool(inputs)
    sheet = fill(worksheet(inputs, pool), lambda row: ('irrelevant', ''))
    judgments = collect(inputs, pool, sheet)
    rows = score(inputs, pool, judgments)
    groups = {group.group: group for group in aggregate_multibook_metrics(rows)
              if group.retriever.value == 'bm25'}
    diagnostic = groups[f'outcome:{status}']
    expected = 0 if status == 'source_incomplete' else 1
    assert diagnostic.metrics['ranked_only.ber@5'].denominator == expected
    assert diagnostic.query_count == 1
    assert groups['headline_complete'].query_count == 2
    assert groups['headline_complete'].interpretation_count == 2
    assert groups['track:original_ambiguous'].eligible_query_count == 0


def test_report_fingerprint_excludes_environment_and_has_explicit_denominators(
    pool_inputs: PoolInputs,
) -> None:
    pool = build_pool(pool_inputs)
    sheet = fill(worksheet(pool_inputs, pool), lambda row: ('irrelevant', ''))
    judgments = collect(pool_inputs, pool, sheet)
    rows = score(pool_inputs, pool, judgments)
    report = build_multibook_metric_report(rows, pool, judgments, {})
    changed = report.model_copy(update={
        'diagnostics': {'python': 'different'},
    })
    assert changed.fingerprint == report.fingerprint
    assert report == type(report).model_validate_json(report.model_dump_json())
    assert len(report.original_queries) == 9
    assert len(report.interpretations) == 6
    assert all(metric.denominator >= 0 for group in report.aggregates
               for metric in group.metrics.values())


def test_top5_pool_cannot_silently_drop_unjudged_top10_records(
    pool_inputs: PoolInputs,
) -> None:
    from src.evaluation.pdf.multibook.scoring.collector import (
        PdfMissingJudgments,
    )
    pool = build_pool(pool_inputs, depth=5)
    sheet = fill(worksheet(pool_inputs, pool), lambda row: ('irrelevant', ''))
    judgments = collect(pool_inputs, pool, sheet)
    with pytest.raises(PdfMissingJudgments) as error:
        score(pool_inputs, pool, judgments)
    assert any(key[-1] in {'extra-0', 'extra-1'}
               for key in error.value.bindings)


def test_evidence_at_rank11_is_not_unrestricted_mrr(
    pool_inputs: PoolInputs,
) -> None:
    from dataclasses import replace
    from src.evaluation.pdf.library import (
        PdfLibraryBookInput, build_pdf_library_corpus,
    )
    from src.evaluation.pdf.multibook_runner import _neighbor_context
    from src.evaluation.pdf.multibook_runs import (
        PdfMultibookRankedHit, build_pdf_multibook_run,
    )
    from tests.evaluation.pdf.multibook_golden_helpers import _records
    _, corpus, _, _ = pool_inputs
    inputs = []
    for book in corpus.books:
        records = book.records
        if book.book_path == 'books/gamma.pdf':
            extra = _records(book.book_path, tuple(
                (f'far-{n}', 'Synthetic extra.', n + 2, 0) for n in range(6)
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
    semantic = runs[1].run
    ids = [record.record_id for record in corpus.records
           if record.record_id != 'a1']
    ids.insert(10, 'a1')
    hits = tuple(PdfMultibookRankedHit(
        record_id=record, score=1 - index * 0.05,
    ) for index, record in enumerate(ids))
    results = tuple(result.model_copy(update={
        'ranking': hits, 'context': _neighbor_context(hits, corpus),
    }) for result in semantic.results)
    semantic = build_pdf_multibook_run(semantic.manifest, results)
    runs = (runs[0], runs[1].model_copy(update={'run': semantic}), runs[2])
    inputs_tuple = golden, corpus, runs, pages
    pool = build_pool(inputs_tuple)
    sheet = fill(worksheet(inputs_tuple, pool), lambda row: (
        ('required', 'yes') if (row['unit_id'], row['record_id']) == (
            'queue', 'a1',
        ) else ('irrelevant', '')
    ))
    rows = score(inputs_tuple, pool, collect(inputs_tuple, pool, sheet))
    row = next(row for row in rows if row.retriever.value == 'minilm'
               and row.query.key == ('both', None))
    assert row.mrr_at_10_localization == row.mrr_at_10_supply == 0
    assert row.ranked_only[1].ber == 0
    assert row.with_context[1].ber == 0.5


@pytest.mark.parametrize('field,value', [
    ('pool_fingerprint', '0' * 64), ('golden_sha256', '0' * 64),
    ('judgments', ()),
])
def test_copied_judgment_artifacts_cannot_bypass_compatibility(
    pool_inputs: PoolInputs, field: str, value: object,
) -> None:
    pool = build_pool(pool_inputs)
    sheet = fill(worksheet(pool_inputs, pool), lambda row: ('irrelevant', ''))
    labels = collect(pool_inputs, pool, sheet)
    with pytest.raises(ValueError):
        score(pool_inputs, pool, labels.model_copy(update={field: value}))
