"""Exact bindings, duplicate rejection and blank versus negative labels."""

import csv
from hashlib import sha256
import io

import pytest

from src.evaluation.pdf.multibook.scoring.collector import (
    collect_pdf_multibook_judgments, PdfMissingJudgments,
)
from tests.evaluation.pdf.multibook.retrieval import (
    contract_helpers as contracts,
)
from tests.evaluation.pdf.multibook.pooling import pool_helpers as helpers
from tests.evaluation.pdf.multibook.retrieval import runner_helpers as runner
from tests.evaluation.pdf.multibook.scoring.metric_helpers import (
    collect, fill, score, submission, worksheet,
)
from tests.evaluation.pdf.multibook.pooling.pool_helpers import (
    PoolInputs,
    build_pool,
)


corpus = contracts.corpus
pool_inputs = helpers.pool_inputs
forbid_real_model_loading = runner.forbid_real_model_loading


def test_blank_rows_are_preserved_and_block_final_metrics(
    pool_inputs: PoolInputs,
) -> None:
    pool = build_pool(pool_inputs)
    sheet = worksheet(pool_inputs, pool)
    labels = collect(pool_inputs, pool, sheet)
    assert len(labels.judgments) == len(pool.tasks)
    assert all(not label.complete and label.relevance is None
               for label in labels.judgments)
    with pytest.raises(PdfMissingJudgments) as error:
        score(pool_inputs, pool, labels)
    assert error.value.bindings
    assert any(key[:2] == ('which', 'queues') for key in error.value.bindings)
    negative = collect(pool_inputs, pool, fill(sheet, lambda row: (
        'irrelevant', '',
    )))
    assert all(label.complete for label in negative.judgments)
    result = score(pool_inputs, pool, negative)
    assert result[0].ranked_only[0].ber == 0


@pytest.mark.parametrize('change', [
    'duplicate', 'missing', 'unknown', 'interpretation', 'item', 'unit',
    'short_row', 'extra_cell', 'columns', 'invalid_label',
    'supply_without_label',
    'irrelevant_supply', 'invalid_supply',
])
def test_invalid_worksheet_is_rejected(
    pool_inputs: PoolInputs, change: str,
) -> None:
    pool = build_pool(pool_inputs)
    sheet = worksheet(pool_inputs, pool)
    rows = list(csv.reader(io.StringIO(sheet)))
    if change == 'duplicate':
        rows.append(rows[1])
    elif change == 'missing':
        rows.pop()
    elif change in ('unknown', 'interpretation', 'item', 'unit'):
        indices = {'unknown': 4, 'interpretation': 2, 'item': 0, 'unit': 3}
        index = indices[change]
        rows[1][index] = 'unknown'
    elif change == 'short_row':
        rows[1].pop()
    elif change == 'extra_cell':
        rows[1].append('extra')
    elif change == 'columns':
        rows[0][1] = 'other'
    elif change == 'invalid_label':
        rows[1][5] = 'helpful'
    elif change == 'invalid_supply':
        rows[1][5:7] = ['required', 'maybe']
    elif change == 'irrelevant_supply':
        rows[1][5:7] = ['irrelevant', 'no']
    else:
        rows[1][6] = 'yes'
    output = io.StringIO()
    csv.writer(output, lineterminator='\n').writerows(rows)
    with pytest.raises(ValueError):
        collect(pool_inputs, pool, output.getvalue())


def test_relevant_blank_supply_stays_incomplete_and_notes_are_verbatim(
    pool_inputs: PoolInputs,
) -> None:
    pool = build_pool(pool_inputs)
    sheet = fill(worksheet(pool_inputs, pool), lambda row: ('required', ''))
    sheet = sheet.replace('required,,', 'required,,  human note  ')
    labels = collect(pool_inputs, pool, sheet)
    assert all(label.note == '  human note  ' for label in labels.judgments)
    assert all(not label.complete for label in labels.judgments)
    with pytest.raises(PdfMissingJudgments):
        score(pool_inputs, pool, labels)


def test_hash_validation_and_cross_file_duplicates_fail(
    pool_inputs: PoolInputs,
) -> None:
    golden, corpus, runs, pages = pool_inputs
    pool = build_pool(pool_inputs)
    item = submission(worksheet(pool_inputs, pool))
    changed = item.model_copy(update={'sha256': '0' * 64})
    for items in [(changed,), (item, item)]:
        with pytest.raises(ValueError):
            collect_pdf_multibook_judgments(
                golden=golden, golden_sha256=contracts.HASH, corpus=corpus,
                runs=runs, reading_pages=pages, pool=pool, submissions=items,
            )
    assert item.sha256 == sha256(item.content).hexdigest()


def test_partitioned_files_and_row_order_preserve_exact_meanings(
    pool_inputs: PoolInputs,
) -> None:
    golden, corpus, runs, pages = pool_inputs
    pool = build_pool(pool_inputs)
    sheet = fill(worksheet(pool_inputs, pool), lambda row: (
        ('required', 'yes') if row['interpretation_id'] == 'queues'
        else ('supporting', 'no') if row['unit_id'] == 'cache'
        else ('irrelevant', '')
    ))
    rows = sheet.splitlines()
    parts = [submission('\n'.join([rows[0], *rows[1 + offset::2]]) + '\n')
             for offset in (0, 1)]
    labels = collect_pdf_multibook_judgments(
        golden=golden, golden_sha256=contracts.HASH, corpus=corpus, pool=pool,
        runs=runs, reading_pages=pages, submissions=parts,
    )
    reversed_labels = collect_pdf_multibook_judgments(
        golden=golden, golden_sha256=contracts.HASH, corpus=corpus, pool=pool,
        runs=runs, reading_pages=pages, submissions=tuple(reversed(parts)),
    )
    assert labels == reversed_labels
    a1 = [item for item in labels.judgments if item.task.record_id == 'a1']
    assert {item.relevance.value for item in a1
            if item.relevance is not None} == {
        'required', 'supporting', 'irrelevant',
    }
    assert all(item.relevance.value == 'irrelevant'
               for item in labels.judgments
               if item.task.query.key == ('which', 'producers')
               and item.relevance is not None)


def test_unclosed_csv_quotes_are_not_silently_repaired(
    pool_inputs: PoolInputs,
) -> None:
    pool = build_pool(pool_inputs)
    sheet = worksheet(pool_inputs, pool)
    broken = sheet.rstrip('\n') + '"unfinished note\n'
    with pytest.raises(ValueError, match='Malformed worksheet CSV'):
        collect(pool_inputs, pool, broken)
