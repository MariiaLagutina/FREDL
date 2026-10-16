"""Synthetic CLI chain: global runs, pooling, human labels and metrics.

Every expected value follows from the scripted semantic rankings and the
explicit synthetic labels below, never from observed output. BM25 and
Hybrid run through the same chain; only rules that hold for every engine
are asserted for them.
"""

from hashlib import sha256
import json
from pathlib import Path
import re
import socket
from typing import Any

import pytest
import torch
from torch import Tensor

from src.__main__ import main
from src.evaluation.pdf.multibook.curation.golden import PdfMultibookGolden
from src.evaluation.pdf.multibook.library import load_pdf_library_corpus
from src.evaluation.pdf.multibook.pooling.models import (
    PdfRetrievalJudgmentPool,
)
from src.evaluation.pdf.multibook.retrieval.runs import (
    PdfMultibookRetrievalRun,
)
from src.evaluation.pdf.multibook.scoring.judgment_models import (
    PdfCollectedJudgments,
)
from src.evaluation.pdf.multibook.scoring.metric_models import (
    PdfMultibookRetrievalEvaluation, PdfQueryRetrievalMetrics,
    PdfRetrievalAggregate,
)
from src.ingestion.pdf.reading_document import prepare_pdf_reading_document
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy
from src.retrieval.semantic.config import (
    DEFAULT_SEMANTIC_MODEL, DEFAULT_SEMANTIC_REVISION, SemanticEncoderConfig,
)
from tests.evaluation.pdf.multibook.curation.golden_helpers import _golden
from tests.evaluation.pdf.multibook.curation.package_helpers import _write_pdf
from tests.evaluation.pdf.multibook.pooling.pool_helpers import (
    pin_page_hashes,
)
from tests.evaluation.pdf.multibook.retrieval import runner_helpers as runner
from tests.evaluation.pdf.multibook.scoring.metric_helpers import fill


forbid_real_model_loading = runner.forbid_real_model_loading


def _page(lead: str, tag: str) -> str:
    """One long sentence per page, so each page becomes one record."""
    return lead + ' ' + ' '.join(f'{tag}{n}' for n in range(150)) + '.'


BOOKS = {
    'books/alpha.pdf': (_page('Queues hold work', 'qa'),
                        _page('Backpressure slows producers', 'qb'),
                        _page('Queues drain work', 'qc')),
    'books/beta.pdf': (_page('Caches keep answers', 'ca'),
                       _page('Eviction removes entries', 'cb')),
    'books/delta.pdf': tuple(_page('Delta filler', f'd{n}x')
                             for n in range(8)),
    'books/gamma.pdf': tuple(_page('Gamma filler', f'g{n}x')
                             for n in range(4)),
}
# Page names in library order; each page tag identifies its record text.
TAGS = {'qa': 'A1', 'qb': 'A2', 'qc': 'A3', 'ca': 'B1', 'cb': 'B2',
        **{f'd{n}x': f'D{n + 1}' for n in range(8)},
        **{f'g{n}x': f'G{n + 1}' for n in range(4)}}
PAGES = tuple(TAGS.values())
FILLERS = tuple(f'D{n}' for n in range(1, 9))
# Scripted MiniLM order per query text; unlisted pages follow in library
# order. Ranks are 1-based below.
SEMANTIC = {
    # B1 at rank 7: BER@5 < BER@10 for the cross-book question.
    'queues': ('A3', 'A1', 'D1', 'D2', 'D3', 'D4', 'B1', 'D5'),
    'queues or caches?': ('B1', 'A1'),
    # A2 at rank 11: outside top-10 and outside every context set.
    'backpressure': (*FILLERS, 'G1', 'G2', 'A2'),
    # A3 beyond top-10, but direct context of A2 at rank 1.
    'drain': ('A2', *FILLERS, 'G1'),
    'pick queues': ('A1',),
    'pick caches': ('B1',),
    'pick drain': ('A3',),
}
QUERIES = (*SEMANTIC, 'zxqmissing', 'pick one?')
# (case, interpretation, unit, page) -> (relevance, supplies_evidence);
# every other worksheet row is labelled irrelevant.
LABELS = {
    ('both', None, 'queue', 'A3'): ('required', 'no'),
    ('both', None, 'queue', 'A1'): ('required', 'yes'),
    ('both', None, 'cache', 'B1'): ('required', 'yes'),
    ('both', None, 'cache', 'B2'): ('supporting', 'yes'),
    ('which', None, 'queue', 'A1'): ('required', 'yes'),
    ('which', 'queues', 'queue', 'A1'): ('required', 'yes'),
    ('which', 'producers', 'backpressure', 'A2'): ('required', 'yes'),
    ('single', None, 'drain', 'A3'): ('required', 'yes'),
    ('pick', 'pick-queues', 'queue', 'A1'): ('required', 'yes'),
    ('pick', 'pick-caches', 'cache', 'B1'): ('required', 'yes'),
    ('pick', 'pick-drain', 'drain', 'A3'): ('required', 'yes'),
}
COMMANDS = ('pdf_collect_multibook_judgments',
            'pdf_evaluate_multibook_retrieval')


class ScriptedEncoder:
    """Give each query its own axis; record cosines follow SEMANTIC ranks."""

    def encode(self, texts: tuple[str, ...]) -> Tensor:
        vectors = []
        for text in texts:
            if text in QUERIES:
                vector = [0.0] * (len(QUERIES) + 1)
                vector[QUERIES.index(text)] = 1.0
            else:
                match = re.search(r'\b(q[abc]|c[ab]|[dg]\dx)0\b', text)
                assert match, 'Unknown synthetic record text'
                page = TAGS[match.group(1)]
                vector = []
                for query in QUERIES:
                    listed = SEMANTIC.get(query, ())
                    order = (*listed, *(p for p in PAGES if p not in listed))
                    vector.append(0.3 - 0.01 * order.index(page))
                vector.append(max(0.0, 1 - sum(v * v for v in vector)) ** 0.5)
            vectors.append(vector)
        return torch.tensor(vectors, dtype=torch.float32)


@pytest.fixture(autouse=True)
def offline_scripted_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace MiniLM with the scripted encoder and refuse any connection."""
    def factory(config: SemanticEncoderConfig) -> ScriptedEncoder:
        assert config.local_files_only is True
        assert (config.model_name, config.model_revision) == (
            DEFAULT_SEMANTIC_MODEL, DEFAULT_SEMANTIC_REVISION,
        )
        return ScriptedEncoder()

    def no_network(*args: Any, **kwargs: Any) -> None:
        pytest.fail('The synthetic CLI chain must stay offline')
    monkeypatch.setattr(
        'src.evaluation.pdf.multibook.retrieval.runner.MiniLMEncoder', factory,
    )
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(socket, 'create_connection', no_network)


@pytest.fixture()
def library(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """Write the PDFs and a Golden Set v2 adapted with two extra cases."""
    root = tmp_path / 'library'
    for path, pages in BOOKS.items():
        _write_pdf(root / path, pages)
    (root / 'books' / 'not-pinned.pdf').write_bytes(b'not a PDF')
    corpus = load_pdf_library_corpus(root, sorted(BOOKS),
                                     PdfRetrievalPolicy.ALL_RECORDS)
    by_page = {(book, page): name for book, names in (
        ('books/alpha.pdf', ('A1', 'A2', 'A3')),
        ('books/beta.pdf', ('B1', 'B2')),
        ('books/delta.pdf', FILLERS),
        ('books/gamma.pdf', ('G1', 'G2', 'G3', 'G4')),
    ) for page, name in enumerate(names, 1)}
    names: dict[str, str] = {}
    for record in corpus.records:
        assert len(record.page_spans) == 1
        names[record.record_id] = by_page[
            record.book_path, record.page_spans[0].page_number
        ]
    assert sorted(names.values()) == sorted(PAGES)
    data = _golden(corpus).model_dump(mode='json')
    data['evidence_units'].append({
        'unit_id': 'drain', 'required_content': 'Queues drain work.',
        'modality': 'text', 'locators': [{
            'book_path': 'books/alpha.pdf', 'physical_page': 3,
            'exact_anchor': 'Queues drain', 'page_text_start': 0,
            'page_text_end': 12, 'page_text_sha256': 'f' * 64,
        }],
    })
    both, which, absent = data['cases']
    both['question'] = 'queues'
    which['question'] = 'queues or caches?'
    which['interpretations'][0]['clarified_question'] = 'queues'
    which['interpretations'][1]['clarified_question'] = 'backpressure'
    absent['question'] = 'zxqmissing'

    def meaning(name: str, unit: str) -> dict[str, Any]:
        return {'interpretation_id': f'pick-{name}', 'label': name,
                'clarified_question': f'pick {name}',
                'expected_outcome': 'complete', 'source_support': 'complete',
                'text_context_support': 'complete',
                'sufficient_sets': [[unit]]}

    data['cases'] += [
        {'case_id': 'single', 'question': 'drain', 'category': 'single_book',
         'expected_outcome': 'complete', 'source_support': 'complete',
         'text_context_support': 'complete', 'expected_behavior': 'answer',
         'sufficient_sets': [['drain']]},
        {'case_id': 'pick', 'question': 'pick one?',
         'category': 'ambiguous_terminology', 'expected_outcome': 'ambiguous',
         'source_support': 'conditional',
         'text_context_support': 'conditional',
         'expected_behavior': 'clarification_required',
         'conditional_units': ['queue', 'cache', 'drain'],
         'interpretations': [meaning('queues', 'queue'),
                             meaning('caches', 'cache'),
                             meaning('drain', 'drain')]},
    ]
    golden = PdfMultibookGolden.model_validate_json(json.dumps(data))
    golden = pin_page_hashes(golden, {
        book: prepare_pdf_reading_document(root, book).reading_pages
        for book in golden.library.book_paths
    })
    golden_path = tmp_path / 'golden.json'
    golden_path.write_text(golden.model_dump_json(indent=1))
    return root, golden_path, names


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _cli(capsys: Any, *args: Path | str) -> str:
    main([str(arg) for arg in args])
    return str(capsys.readouterr().out)


def _cli_error(capsys: Any, *args: Path | str) -> str:
    with pytest.raises(SystemExit) as error:
        main([str(arg) for arg in args])
    assert error.value.code == 2
    return str(capsys.readouterr().err)


def _snapshot(*paths: Path) -> dict[Path, bytes]:
    files = [item for path in paths
             for item in ((path,) if path.is_file() else path.rglob('*'))]
    return {item: item.read_bytes() for item in files if item.is_file()}


def _retrieve_and_pool(
    root: Path, golden: Path, work: Path, capsys: Any,
) -> None:
    work.mkdir()
    entries = []
    for kind in ('bm25', 'minilm', 'hybrid'):
        output = work / f'{kind}.json'
        summary = _cli(capsys, 'pdf_run_multibook_retrieval', root, golden,
                       kind, output, '--policy=all_records')
        assert f'artifact SHA256: {_sha(output)}' in summary
        entries.append({'retriever': kind, 'path': output.name,
                        'sha256': _sha(output)})
    (work / 'runs.json').write_text(json.dumps({
        'schema_name': 'fredl.pdf.multibook-pooling-inputs', 'version': 1,
        'runs': entries,
    }))
    _cli(capsys, 'pdf_pool_multibook_judgments', golden, root,
         work / 'runs.json', work / 'review')


def _label(work: Path, names: dict[str, str]) -> None:
    """Fill the blind CSV from LABELS and pin it in the inputs manifest."""
    review = work / 'review'
    seen = []

    def decide(row: dict[str, str]) -> tuple[str, str]:
        key = (row['case_id'], row['interpretation_id'] or None,
               row['unit_id'], names[row['record_id']])
        seen.append(key)
        return LABELS.get(key, ('irrelevant', ''))
    sheet = work / 'human.csv'
    sheet.write_text(fill((review / 'worksheet.csv').read_text(), decide))
    # Every scripted label binds exactly one blind worksheet row.
    assert all(seen.count(key) == 1 for key in LABELS)
    package = json.loads((review / 'package-manifest.json').read_bytes())
    (work / 'inputs.json').write_text(json.dumps({
        'schema_name': 'fredl.pdf.multibook-judgment-inputs', 'version': 1,
        'pool': {'path': 'review/judgment-pool.json',
                 'sha256': _sha(review / 'judgment-pool.json')},
        'pool_fingerprint': package['pool_fingerprint'],
        'worksheets': [{'path': 'human.csv', 'sha256': _sha(sheet),
                        'reviewer': 'Synthetic reviewer'}],
    }))


def _chain(
    root: Path, golden: Path, names: dict[str, str], work: Path,
    capsys: Any,
) -> None:
    _retrieve_and_pool(root, golden, work, capsys)
    _label(work, names)
    for command, output in zip(COMMANDS, ('collected.json',
                                          'evaluation.json')):
        summary = _cli(capsys, command, golden, root, work / 'runs.json',
                       work / 'inputs.json', work / output)
        assert f'artifact SHA256 {_sha(work / output)}' in summary


def _evaluation(work: Path) -> PdfMultibookRetrievalEvaluation:
    return PdfMultibookRetrievalEvaluation.model_validate_json(
        (work / 'evaluation.json').read_bytes(),
    )


def _rows(
    evaluation: PdfMultibookRetrievalEvaluation, kind: str,
) -> dict[tuple[str, str | None], PdfQueryRetrievalMetrics]:
    return {row.query.key: row
            for row in (*evaluation.original_queries,
                        *evaluation.interpretations)
            if row.retriever.value == kind}


def _groups(
    evaluation: PdfMultibookRetrievalEvaluation, kind: str,
) -> dict[str, PdfRetrievalAggregate]:
    return {group.group: group for group in evaluation.aggregates
            if group.retriever.value == kind}


def _values(
    row: PdfQueryRetrievalMetrics, track: str, metric: str,
) -> list[Any]:
    return [getattr(item, metric) for item in getattr(row, track)]


def test_cli_chain_scores_controlled_evidence(
    library: tuple[Path, Path, dict[str, str]], tmp_path: Path, capsys: Any,
) -> None:
    root, golden, names = library
    work = tmp_path / 'work'
    _retrieve_and_pool(root, golden, work, capsys)
    _label(work, names)
    inputs = _snapshot(root, golden, work)
    for command, output in zip(COMMANDS, ('collected.json',
                                          'evaluation.json')):
        _cli(capsys, command, golden, root, work / 'runs.json',
             work / 'inputs.json', work / output)
    assert inputs == _snapshot(*inputs)
    evaluation = _evaluation(work)
    semantic = _rows(evaluation, 'minilm')

    # Cross-book set {queue, cache}: A3 locates queue without supplying it,
    # A1 (rank 2) supplies queue, B1 (rank 7) supplies cache.
    both = semantic['both', None]
    assert _values(both, 'ranked_only', 'ber') == [0.5, 1.0]
    assert _values(both, 'ranked_only', 'sufficient_set_success') == [
        False, True,
    ]
    assert _values(both, 'ranked_only', 'localization_coverage') == [
        0.5, 1.0,
    ]
    assert (both.mrr_at_10_localization, both.mrr_at_10_supply) == (1.0, 0.5)
    assert _values(both, 'context_only_gain', 'ber') == [0.0, 0.0]
    # A3 is reachable only as direct context of A2 at rank 1.
    single = semantic['single', None]
    assert _values(single, 'ranked_only', 'ber') == [0.0, 0.0]
    assert _values(single, 'with_context', 'ber') == [1.0, 1.0]
    assert _values(single, 'context_only_gain', 'ber') == [1.0, 1.0]
    assert _values(single, 'context_only_gain',
                   'sufficient_set_success') == [1.0, 1.0]
    assert {(names[hit.record_id], tuple(names[parent] for parent in
                                         hit.parent_record_ids))
            for hit in single.with_context[0].context} >= {('A3', ('A2',))}
    assert (single.mrr_at_10_localization, single.mrr_at_10_supply) == (0, 0)
    # Clarified variants keep their own labels: A1 is rank 2 for 'queues';
    # A2 at rank 11 is a measured miss (zero), not a null.
    assert _values(semantic['which', 'queues'], 'ranked_only', 'ber') == [
        1.0, 1.0,
    ]
    assert semantic['which', 'queues'].mrr_at_10_supply == 0.5
    producers = semantic['which', 'producers']
    assert _values(producers, 'with_context', 'ber') == [0.0, 0.0]
    assert producers.mrr_at_10_supply == 0.0
    assert all(_values(semantic['pick', name], 'ranked_only', 'ber')
               == [1.0, 1.0] for name in ('pick-queues', 'pick-caches',
                                          'pick-drain'))
    # The ambiguous original keeps diagnostics but no sufficient set.
    assert semantic['which', None].ranked_only[0].supplied_units == ('queue',)
    for kind in ('bm25', 'minilm', 'hybrid'):
        rows = _rows(evaluation, kind)
        assert len(rows) == 10
        for key in (('which', None), ('pick', None), ('absent', None)):
            row = rows[key]
            assert row.eligible_sufficient_sets == ()
            assert row.exclusion == 'no_declared_sufficient_set'
            assert all(value is None for track in (
                'ranked_only', 'with_context', 'context_only_gain',
            ) for metric in ('ber', 'sufficient_set_success',
                             'localization_coverage')
                for value in _values(row, track, metric))
            assert row.mrr_at_10_localization is None
            assert row.mrr_at_10_supply is None
        assert {key for key, row in rows.items()
                if row.track == 'clarified'} == {
            ('which', 'queues'), ('which', 'producers'),
            ('pick', 'pick-queues'), ('pick', 'pick-caches'),
            ('pick', 'pick-drain'),
        }
        assert {'headline_complete:original',
                'headline_complete:clarified'} <= set(_groups(evaluation,
                                                              kind))

    groups = _groups(evaluation, 'minilm')
    headline = groups['headline_complete:original']
    assert (headline.query_count, headline.interpretation_count,
            headline.case_count) == (2, 0, 2)
    assert headline.metrics['ranked_only.ber@5'].mean == 0.25
    assert headline.metrics['ranked_only.ber@10'].mean == 0.5
    assert headline.metrics['with_context.ber@5'].mean == 0.75
    assert headline.metrics['context_only_gain.ber@5'].mean == 0.5
    assert headline.metrics['mrr_at_10_supply'].mean == 0.25
    assert headline.metrics['mrr_at_10_localization'].mean == 0.5
    assert headline.metrics['ranked_only.ber@5'].denominator == 2
    # Case means: which (1 + 0) / 2 and pick (1 + 1 + 1) / 3.
    clarified = groups['headline_complete:clarified']
    ber = clarified.metrics['ranked_only.ber@5']
    assert (ber.mean, ber.denominator, ber.contributing_queries) == (
        0.75, 2, 5)
    assert (clarified.interpretation_count, clarified.case_count,
            clarified.eligible_case_count) == (5, 2, 2)
    combined = groups['diagnostic:complete_original_and_clarified']
    assert combined.metrics['ranked_only.ber@5'].mean == pytest.approx(
        4.5 / 7)
    assert {name for name in groups if name.startswith('category:')} == {
        'category:complementary_multi_book:original',
        'category:single_book:original',
        'category:ambiguous_terminology:original',
        'category:ambiguous_terminology:clarified',
        'category:hard_negative:original',
    }
    cross = groups['category:complementary_multi_book:original']
    assert (cross.query_count,
            cross.metrics['ranked_only.ber@5'].mean) == (1, 0.5)
    one_book = groups['category:single_book:original']
    assert one_book.metrics['with_context.ber@5'].mean == 1.0
    assert one_book.metrics['with_context.ber@5'].denominator == 1
    ambiguous = groups['category:ambiguous_terminology:original']
    assert (ambiguous.query_count, ambiguous.eligible_query_count) == (2, 0)
    assert ambiguous.metrics['ranked_only.ber@5'].denominator == 0
    meanings = groups['category:ambiguous_terminology:clarified']
    assert meanings.metrics['ranked_only.ber@5'] == ber
    negative = groups['category:hard_negative:original']
    assert negative.metrics['ranked_only.ber@5'].mean is None
    assert negative.metrics['ranked_only.ber@5'].denominator == 0

    # Provenance links each artifact to the exact bytes it consumed.
    collected = PdfCollectedJudgments.model_validate_json(
        (work / 'collected.json').read_bytes(),
    )
    pool = PdfRetrievalJudgmentPool.model_validate_json(
        (work / 'review' / 'judgment-pool.json').read_bytes(),
    )
    dataset = PdfMultibookGolden.model_validate_json(golden.read_bytes())
    assert evaluation.metric_policy == 'declared-sets-query-macro-v2'
    assert evaluation.golden_status == dataset.status == 'draft'
    assert evaluation.golden_sha256 == _sha(golden) == pool.golden_sha256
    assert evaluation.golden_content_fingerprint == dataset.content_fingerprint
    assert evaluation.pool_fingerprint == pool.fingerprint
    assert evaluation.judgments_fingerprint == collected.fingerprint
    assert evaluation.runs == pool.runs
    for run in evaluation.runs:
        path = work / f'{run.retriever.value}.json'
        stored = PdfMultibookRetrievalRun.model_validate_json(
            path.read_bytes(),
        )
        assert run.artifact_sha256 == _sha(path)
        assert run.results_fingerprint == stored.results_fingerprint
    assert evaluation.artifact_hashes == {
        'golden': _sha(golden),
        'pool': _sha(work / 'review' / 'judgment-pool.json'),
        'judgment_inputs': _sha(work / 'inputs.json'),
        'worksheet:0': _sha(work / 'human.csv'),
    }
    assert not list(tmp_path.rglob('.fredl-*'))


def test_isolated_cli_chains_are_deterministic(
    library: tuple[Path, Path, dict[str, str]], tmp_path: Path, capsys: Any,
) -> None:
    root, golden, names = library
    first, second = tmp_path / 'first', tmp_path / 'second'
    for work in (first, second):
        _chain(root, golden, names, work, capsys)
    files = sorted(path.relative_to(first) for path in first.rglob('*')
                   if path.is_file())
    assert len(files) == 13
    for name in files:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert _evaluation(first).fingerprint == _evaluation(second).fingerprint


@pytest.mark.parametrize('failure', ['incomplete', 'mismatched', 'run_drift'])
def test_cli_chain_rejects_incomplete_or_mismatched_judgments(
    library: tuple[Path, Path, dict[str, str]], tmp_path: Path, capsys: Any,
    failure: str,
) -> None:
    root, golden, names = library
    work = tmp_path / 'work'
    _retrieve_and_pool(root, golden, work, capsys)
    _label(work, names)
    inputs = json.loads((work / 'inputs.json').read_bytes())
    sheet = work / 'human.csv'
    record = next(key for key, name in names.items() if name == 'B1')
    if failure == 'run_drift':
        # Valid JSON with different bytes no longer matches the pool.
        run = work / 'minilm.json'
        run.write_text(json.dumps(json.loads(run.read_bytes())))
        runs = json.loads((work / 'runs.json').read_bytes())
        runs['runs'][1]['sha256'] = _sha(run)
        (work / 'runs.json').write_text(json.dumps(runs))
    else:
        lines = sheet.read_text().splitlines(keepends=True)
        index = next(index for index, line in enumerate(lines)
                     if ',both,,cache,' + record + ',' in line)
        lines[index] = (lines[index].replace(',required,yes,', ',,,')
                        if failure == 'incomplete'
                        else lines[index].replace(',both,,', ',which,,'))
        sheet.write_text(''.join(lines))
        inputs['worksheets'][0]['sha256'] = _sha(sheet)
        (work / 'inputs.json').write_text(json.dumps(inputs))
    output = work / 'evaluation.json'
    error = _cli_error(capsys, COMMANDS[1], golden, root, work / 'runs.json',
                       work / 'inputs.json', output)
    if failure == 'incomplete':
        # The exact unjudged contributing binding is reported.
        assert 'Incomplete judgments' in error and record in error
        _cli(capsys, COMMANDS[0], golden, root, work / 'runs.json',
             work / 'inputs.json', work / 'draft.json')
    else:
        assert 'no final artifact was published' in error
    assert not output.exists()
    assert not list(tmp_path.rglob('.fredl-*'))


def test_cli_chain_never_replaces_existing_outputs(
    library: tuple[Path, Path, dict[str, str]], tmp_path: Path, capsys: Any,
) -> None:
    root, golden, names = library
    work = tmp_path / 'work'
    _chain(root, golden, names, work, capsys)
    before = _snapshot(work)
    assert 'new file' in _cli_error(
        capsys, 'pdf_run_multibook_retrieval', root, golden, 'bm25',
        work / 'bm25.json', '--policy=all_records',
    )
    assert 'new output directory' in _cli_error(
        capsys, 'pdf_pool_multibook_judgments', golden, root,
        work / 'runs.json', work / 'review',
    )
    for command, output in zip(COMMANDS, ('collected.json',
                                          'evaluation.json')):
        assert 'new file' in _cli_error(
            capsys, command, golden, root, work / 'runs.json',
            work / 'inputs.json', work / output,
        )
    assert before == _snapshot(work)
