"""Blind human-review worksheets preserve query and evidence-unit identity."""

import csv
import io

import pytest

from src.evaluation.pdf.multibook_pool_worksheet import (
    POOL_WORKSHEET_COLUMNS, render_retrieval_pool_worksheet,
)
from tests.evaluation.pdf import multibook_contract_helpers as contracts
from tests.evaluation.pdf import multibook_pool_helpers as helpers
from tests.evaluation.pdf import multibook_runner_helpers as runner
from tests.evaluation.pdf.multibook_pool_helpers import PoolInputs, build_pool


corpus = contracts.corpus
pool_inputs = helpers.pool_inputs
forbid_real_model_loading = runner.forbid_real_model_loading


def test_blind_sheet_has_blank_labels_and_exact_query_unit_record_tasks(
    pool_inputs: PoolInputs,
) -> None:
    golden, corpus, _, _ = pool_inputs
    pool = build_pool(pool_inputs)
    markdown, sheet = render_retrieval_pool_worksheet(golden, corpus, pool)
    reader = csv.DictReader(io.StringIO(sheet))
    assert tuple(reader.fieldnames or ()) == POOL_WORKSHEET_COLUMNS
    rows = list(reader)
    assert set((row['case_id'], row['interpretation_id'] or None,
                row['unit_id'], row['record_id']) for row in rows) == {
        task.key for task in pool.tasks
    }
    for row in rows:
        assert not any(row[key] for key in (
            'relevance', 'supplies_evidence', 'note',
        ))
    for hidden in ('bm25', 'minilm', 'hybrid', 'origins', 'score',
                   'anchor', 'expected_outcome', 'reference_notes'):
        assert hidden not in markdown.lower()
        assert hidden not in sheet.lower()
    assert 'Conditional meaning for this unit:' in markdown
    assert 'do not combine meanings' in markdown
    assert 'No evidence-unit tasks' in markdown
    for task in pool.tasks:
        text = next(record.text for record in corpus.source_records
                    if record.record_id == task.record_id)
        assert text in markdown
    assert render_retrieval_pool_worksheet(golden, corpus, pool) == (
        markdown, sheet,
    )


def test_prevalidated_copied_worksheet_inputs_are_rechecked(
    pool_inputs: PoolInputs,
) -> None:
    golden, corpus, _, _ = pool_inputs
    pool = build_pool(pool_inputs)
    malformed = pool.model_copy(update={'tasks': (
        pool.tasks[0].model_copy(update={'record_id': 'unknown'}),
    )})
    with pytest.raises(ValueError):
        render_retrieval_pool_worksheet(golden, corpus, malformed)
    with pytest.raises(ValueError):
        changed = pool.model_copy(update={
            'golden_content_fingerprint': '0' * 64,
        })
        render_retrieval_pool_worksheet(golden, corpus, changed)


def test_review_order_is_independent_of_origins_and_pool_fingerprint(
    pool_inputs: PoolInputs,
) -> None:
    from src.ingestion.pdf.fingerprints import fingerprint_canonical_json
    golden, corpus, _, _ = pool_inputs
    pool = build_pool(pool_inputs)
    before = render_retrieval_pool_worksheet(golden, corpus, pool)
    changed = pool.model_copy(update={'candidates': tuple(
        candidate.model_copy(update={'origins': tuple(reversed(
            candidate.origins,
        ))}) for candidate in pool.candidates
    )})
    assert changed.fingerprint != pool.fingerprint
    assert render_retrieval_pool_worksheet(golden, corpus, changed) == before
    rows = list(csv.DictReader(io.StringIO(before[1])))
    records = {record.record_id: record for record in corpus.source_records}
    for task in pool.tasks:
        group = [row['record_id'] for row in rows
                 if (row['case_id'], row['interpretation_id'] or None,
                     row['unit_id']) == (*task.query.key, task.unit_id)]
        expected = sorted(group, key=lambda record_id: (
            fingerprint_canonical_json({
                'schema': 'fredl.pdf.blind-review-order-v1',
                'query': task.query.model_dump(mode='json'),
                'unit_id': task.unit_id, 'record_id': record_id,
                'book_path': records[record_id].book_path,
            }), record_id,
        ))
        assert group == expected
    assert any([row['record_id'] for row in rows if row['unit_id'] == unit]
               != [task.record_id for task in pool.tasks
                   if task.unit_id == unit]
               for unit in golden.units)


def test_markdown_preserves_backticks_and_trailing_newlines(
    pool_inputs: PoolInputs,
) -> None:
    from dataclasses import replace
    from src.evaluation.pdf.multibook_pool_worksheet import _review_record
    _, corpus, _, _ = pool_inputs
    text = 'Example:\n```python\nprint(1)\n```\n\n'
    record = replace(corpus.source_records[0], text=text)
    markdown = '\n'.join(_review_record('W0001', record))
    assert f'Record: {record.record_id}' in markdown
    assert f'````text\n{text}````' in markdown
