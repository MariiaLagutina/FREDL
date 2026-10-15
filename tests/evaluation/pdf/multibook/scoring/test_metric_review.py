"""Critical metric semantics and completeness regressions from final review."""

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from src.evaluation.pdf.library import (
    PdfLibraryBookInput, build_pdf_library_corpus,
)
from src.evaluation.pdf.multibook.scoring.collector import PdfMissingJudgments
from src.evaluation.pdf.multibook.scoring.report import (
    aggregate_multibook_metrics,
)
from src.evaluation.pdf.multibook.pooling.models import PdfPoolingRun
from src.evaluation.pdf.multibook.retrieval.runner import _neighbor_context
from src.evaluation.pdf.multibook.retrieval.runs import (
    PdfMultibookRankedHit, build_pdf_multibook_run,
)
from tests.evaluation.pdf.multibook.retrieval import (
    contract_helpers as contracts,
)
from tests.evaluation.pdf.multibook.pooling import pool_helpers as helpers
from tests.evaluation.pdf.multibook.retrieval import runner_helpers as runner
from tests.evaluation.pdf.multibook.scoring.metric_helpers import (
    collect, fill, score, worksheet,
)
from tests.evaluation.pdf.multibook.pooling.pool_helpers import (
    PoolInputs,
    build_pool,
)
from tests.evaluation.pdf.multibook_golden_helpers import _records


corpus = contracts.corpus
pool_inputs = helpers.pool_inputs
forbid_real_model_loading = runner.forbid_real_model_loading


def test_different_sized_alternatives_and_duplicate_supply_records(
    pool_inputs: PoolInputs,
) -> None:
    golden, corpus, _, pages = pool_inputs
    data = golden.model_dump(mode='json')
    data['cases'][0]['category'] = 'independent_multi_book'
    data['cases'][0]['sufficient_sets'] = [
        ['queue', 'backpressure'], ['cache'],
    ]
    golden = type(golden).model_validate_json(json.dumps(data))
    inputs = golden, corpus, helpers.saved_runs(golden, corpus), pages
    pool = build_pool(inputs)
    for cache, expected, success in [(False, 0.5, False), (True, 1.0, True)]:
        sheet = fill(worksheet(inputs, pool), lambda row: (
            ('required', 'yes') if (
                row['unit_id'] == 'queue' and row['record_id'] in {'a1', 'a2'}
                or cache and row['unit_id'] == 'cache'
                and row['record_id'] == 'b1'
            ) else ('irrelevant', '')
        ))
        rows = score(inputs, pool, collect(inputs, pool, sheet))
        row = next(row for row in rows if row.retriever.value == 'minilm'
                   and row.query.key == ('both', None))
        assert row.with_context[1].ber == expected
        assert row.with_context[1].sufficient_set_success is success
        assert row.with_context[1].supplied_units.count('queue') == 1


def test_reordering_equivalent_rows_preserves_metrics_and_exact_missing_list(
    pool_inputs: PoolInputs,
) -> None:
    pool = build_pool(pool_inputs)
    blank = worksheet(pool_inputs, pool)
    lines = blank.splitlines()
    reordered = '\n'.join([lines[0], *reversed(lines[1:])]) + '\n'
    missing = []
    for sheet in (blank, reordered):
        with pytest.raises(PdfMissingJudgments) as error:
            score(pool_inputs, pool, collect(pool_inputs, pool, sheet))
        missing.append(error.value.bindings)
    assert missing[0] == missing[1]
    assert set(missing[0]) == {task.key for task in pool.tasks}
    first = collect(pool_inputs, pool, fill(
        blank, lambda row: ('irrelevant', ''),
    ))
    second = collect(pool_inputs, pool, fill(
        reordered, lambda row: ('irrelevant', ''),
    ))
    assert score(pool_inputs, pool, first) == score(pool_inputs, pool, second)
    # Byte provenance changes intentionally even when the decisions do not.
    assert first.sources != second.sources
    assert first.fingerprint != second.fingerprint


def test_headline_counts_variants_without_ambiguous_original_double_counting(
    pool_inputs: PoolInputs,
) -> None:
    pool = build_pool(pool_inputs)
    sheet = fill(worksheet(pool_inputs, pool), lambda row: ('required', 'yes'))
    rows = score(pool_inputs, pool, collect(pool_inputs, pool, sheet))
    groups = {group.group: group for group in aggregate_multibook_metrics(rows)
              if group.retriever.value == 'bm25'}
    headline = groups['headline_complete']
    assert (headline.query_count, headline.interpretation_count,
            headline.case_count) == (3, 2, 2)
    assert headline.metrics['ranked_only.ber@5'].denominator == 3
    ambiguous = groups['track:original_ambiguous']
    assert ambiguous.query_count == 1 and ambiguous.eligible_query_count == 0
    assert all(metric.mean is None and metric.denominator == 0
               for metric in ambiguous.metrics.values())
    empty = groups['outcome:partial']
    assert empty.query_count == 0 and empty.excluded_queries == ()
    assert all(metric.mean is None and metric.denominator == 0
               for metric in empty.metrics.values())


def _isolated_anchor_inputs(inputs: PoolInputs) -> PoolInputs:
    """Put an isolated approved anchor outside all top-10/context sets."""
    _, corpus, _, _ = inputs
    books = []
    for book in corpus.books:
        records = book.records
        if book.book_path == 'books/gamma.pdf':
            extra = _records(book.book_path, tuple(
                (f'background-{n}', 'Synthetic background.', n + 2, 0)
                for n in range(6)
            ))
            records = (*records, *(
                replace(record, flow_index=n + 1)
                for n, record in enumerate(extra)
            ))
        books.append(PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, records, (),
        ))
    corpus = build_pdf_library_corpus(books, corpus.policy)
    golden, corpus, runs, pages = helpers.synthetic_inputs(corpus)
    ids = [record.record_id for record in corpus.records
           if record.record_id != 'a3'] + ['a3']
    changed = []
    for item in runs:
        kind = item.run.manifest.retrieval.retriever.value
        hits = () if kind == 'bm25' else tuple(PdfMultibookRankedHit(
            record_id=record, score=1 / (61 + index) if kind == 'hybrid'
            else 0.9 - index * 0.05,
            semantic_rank=index + 1 if kind == 'hybrid' else None,
        ) for index, record in enumerate(ids))
        results = tuple(result.model_copy(update={
            'ranking': hits, 'context': _neighbor_context(hits, corpus),
        }) for result in item.run.results)
        run = build_pdf_multibook_run(item.run.manifest, results)
        changed.append(PdfPoolingRun(
            run=run, artifact_sha256=sha256(
                run.model_dump_json().encode(),
            ).hexdigest(),
        ))
    return golden, corpus, tuple(changed), pages


def test_blank_anchor_only_rows_do_not_block_unrelated_retrieval_bindings(
    pool_inputs: PoolInputs,
) -> None:
    inputs = _isolated_anchor_inputs(pool_inputs)
    pool = build_pool(inputs)
    for item in inputs[2]:
        for result in item.run.results:
            assert 'a3' not in {hit.record_id for hit in result.ranking[:10]}
            assert all(hit.record_id != 'a3' for part in result.context
                       for hit in part.records)
    sheet = fill(worksheet(inputs, pool), lambda row: (
        ('', '') if row['record_id'] == 'a3' else ('irrelevant', '')
    ))
    judgments = collect(inputs, pool, sheet)
    assert any(not label.complete and label.task.record_id == 'a3'
               for label in judgments.judgments)
    rows = score(inputs, pool, judgments)
    assert next(row for row in rows if row.query.key == (
        'which', 'producers',
    )).ranked_only[1].ber == 0


def test_mrr_uses_earliest_required_hit_and_required_supply_hit_separately(
    pool_inputs: PoolInputs,
) -> None:
    pool = build_pool(pool_inputs)
    decisions = {
        ('queue', 'a1'): ('supporting', 'yes'),
        ('queue', 'a2'): ('required', 'no'),
        ('cache', 'b1'): ('required', 'yes'),
        ('queue', 'extra-0'): ('required', 'yes'),
    }
    sheet = fill(worksheet(pool_inputs, pool), lambda row: decisions.get(
        (row['unit_id'], row['record_id']), ('irrelevant', ''),
    ))
    labels = collect(pool_inputs, pool, sheet)
    assert any(label.relevance is not None
               and label.relevance.value == 'supporting'
               and label.supplies_evidence is True
               for label in labels.judgments)
    row = next(row for row in score(pool_inputs, pool, labels)
               if row.query.key == ('both', None)
               and row.retriever.value == 'minilm')
    assert row.mrr_at_10_localization == 1 / 2
    assert row.mrr_at_10_supply == 1 / 4
    assert row.ranked_only[0].ber == 0.5
    assert row.ranked_only[0].sufficient_set_success is False


def test_gain_subtraction_does_not_clamp_negative_values() -> None:
    from src.evaluation.pdf.multibook.scoring.metrics import _difference
    assert _difference(0.25, 0.75) == -0.5
    assert _difference(None, 0.0) is None
