"""Historical v1 and current v2 metric policies, fields and fingerprints."""

import json
from typing import Any

from pydantic import ValidationError
import pytest

from src.evaluation.pdf.multibook.scoring.metric_models import (
    PdfMultibookRetrievalEvaluation,
)
from src.evaluation.pdf.multibook.scoring.report import (
    build_multibook_metric_report,
)
from src.ingestion.pdf.fingerprints import fingerprint_canonical_json
from tests.evaluation.pdf.multibook.pooling import pool_helpers as helpers
from tests.evaluation.pdf.multibook.pooling.pool_helpers import (
    PoolInputs,
    build_pool,
)
from tests.evaluation.pdf.multibook.retrieval import (
    contract_helpers as contracts,
)
from tests.evaluation.pdf.multibook.retrieval import runner_helpers as runner
from tests.evaluation.pdf.multibook.scoring.metric_helpers import (
    collect, fill, score, worksheet,
)


corpus = contracts.corpus
pool_inputs = helpers.pool_inputs
forbid_real_model_loading = runner.forbid_real_model_loading

# Computed by the producer before metric policy v2 existed. A change means
# that historical evaluation artifacts no longer keep their identity.
HISTORICAL_V1_FINGERPRINT = (
    'e2464fafbf9bbc52fc7da7b3404d7f6000487927c7a4df204cd10ba60b241b49'
)


def _historical_v1_artifact() -> dict[str, Any]:
    """Return a minimal artifact in the exact historical v1 field layout."""
    def coverage(cutoff: int) -> dict[str, Any]:
        return {'cutoff': cutoff, 'ber': 0.5, 'sufficient_set_success': False,
                'localization_coverage': 1.0, 'supplied_units': ['queue'],
                'localized_units': ['queue', 'cache'], 'context': []}

    def gain(cutoff: int) -> dict[str, Any]:
        return {'cutoff': cutoff, 'ber': 0.0, 'sufficient_set_success': 0.0,
                'localization_coverage': 0.0}

    row = {
        'retriever': 'bm25',
        'query': {'case_id': 'both', 'interpretation_id': None},
        'track': 'original', 'expected_behavior': 'answer',
        'outcome': 'complete',
        'eligible_sufficient_sets': [['queue', 'cache']], 'exclusion': None,
        'ranked_only': [coverage(5), coverage(10)],
        'with_context': [coverage(5), coverage(10)],
        'context_only_gain': [gain(5), gain(10)],
        'mrr_at_10_localization': 1.0, 'mrr_at_10_supply': 0.5,
    }
    aggregate = {
        'retriever': 'bm25', 'group': 'headline_complete', 'query_count': 1,
        'interpretation_count': 0, 'case_count': 1,
        'eligible_query_count': 1, 'excluded_queries': [],
        'metrics': {'ranked_only.ber@5': {'mean': 0.5, 'denominator': 1}},
    }
    return {
        'schema_name': 'fredl.pdf.multibook-retrieval-evaluation',
        'version': 1, 'metric_policy': 'declared-sets-query-macro-v1',
        'supply_policy': 'required-and-explicit-yes-v1',
        'golden_sha256': '1' * 64, 'golden_content_fingerprint': '2' * 64,
        'corpus_fingerprint': '3' * 64, 'library_fingerprint': '4' * 64,
        'query_plan_fingerprint': '5' * 64, 'comparison_identity': '6' * 64,
        'pool_fingerprint': '7' * 64, 'judgments_fingerprint': '8' * 64,
        'runs': [{'retriever': 'bm25', 'artifact_sha256': '9' * 64,
                  'manifest_identity': 'a' * 64,
                  'results_fingerprint': 'b' * 64}],
        'artifact_hashes': {'golden': '1' * 64, 'pool': 'c' * 64},
        'original_queries': [row], 'interpretations': [],
        'aggregates': [aggregate],
        'diagnostics': {'code_revision': 'synthetic', 'code_dirty': 'False'},
    }


def _load(data: dict[str, Any]) -> PdfMultibookRetrievalEvaluation:
    return PdfMultibookRetrievalEvaluation.model_validate_json(
        json.dumps(data),
    )


def _v2_report(inputs: PoolInputs) -> PdfMultibookRetrievalEvaluation:
    pool = build_pool(inputs)
    sheet = fill(worksheet(inputs, pool), lambda row: ('irrelevant', ''))
    labels = collect(inputs, pool, sheet)
    return build_multibook_metric_report(
        score(inputs, pool, labels), pool, labels, {},
        golden=inputs[0], golden_sha256=contracts.HASH,
    )


def test_historical_v1_artifact_validates_with_its_original_fingerprint(
) -> None:
    data = _historical_v1_artifact()
    artifact = _load(data)
    del data['diagnostics']
    assert artifact.fingerprint == HISTORICAL_V1_FINGERPRINT
    assert artifact.fingerprint == fingerprint_canonical_json(data)
    assert artifact.golden_status is None
    assert artifact.original_queries[0].category is None
    assert artifact.aggregates[0].weighting is None
    assert artifact.aggregates[0].group == 'headline_complete'


@pytest.mark.parametrize('change', [
    'golden_status', 'category', 'weighting', 'eligible_case_count',
    'contributing_queries', 'v2_group',
])
def test_historical_policy_rejects_any_v2_field_or_group(change: str) -> None:
    data = _historical_v1_artifact()
    aggregate = data['aggregates'][0]
    if change == 'golden_status':
        data['golden_status'] = 'draft'
    elif change == 'category':
        data['original_queries'][0]['category'] = 'complementary_multi_book'
    elif change == 'weighting':
        aggregate['weighting'] = 'query_variant'
    elif change == 'eligible_case_count':
        aggregate['eligible_case_count'] = 1
    elif change == 'contributing_queries':
        aggregate['metrics']['ranked_only.ber@5']['contributing_queries'] = 1
    else:
        aggregate['group'] = 'headline_complete:original'
    with pytest.raises(ValidationError, match='v1'):
        _load(data)


@pytest.mark.parametrize('policy', ['v1', 'v2'])
@pytest.mark.parametrize('copy', ['identical', 'different_mean'])
def test_duplicate_retriever_group_is_rejected_under_both_policies(
    pool_inputs: PoolInputs, policy: str, copy: str,
) -> None:
    if policy == 'v1':
        data = _historical_v1_artifact()
        name = 'headline_complete'
    else:
        data = json.loads(_v2_report(pool_inputs).model_dump_json())
        name = 'headline_complete:original'
    # The unchanged artifact is valid, so only the duplicate can fail.
    _load(data)
    group = next(group for group in data['aggregates']
                 if group['group'] == name and group['retriever'] == 'bm25')
    duplicate = json.loads(json.dumps(group))
    if copy == 'different_mean':
        duplicate['metrics']['ranked_only.ber@5']['mean'] = 0.25
    data['aggregates'].append(duplicate)
    with pytest.raises(ValidationError, match='Duplicate aggregate group'):
        _load(data)


def test_same_group_name_for_different_retrievers_is_valid(
    pool_inputs: PoolInputs,
) -> None:
    report = _v2_report(pool_inputs)
    retrievers = [group.retriever.value for group in report.aggregates
                  if group.group == 'headline_complete:original']
    assert sorted(retrievers) == ['bm25', 'hybrid', 'minilm']
    assert _load(json.loads(report.model_dump_json())) == report
    data = _historical_v1_artifact()
    other = json.loads(json.dumps(data['aggregates'][0]))
    other['retriever'] = 'minilm'
    data['aggregates'].append(other)
    assert [group.retriever.value for group in _load(data).aggregates] == [
        'bm25', 'minilm',
    ]
    # The unchanged historical artifact keeps its pinned fingerprint.
    assert _load(_historical_v1_artifact()).fingerprint == (
        HISTORICAL_V1_FINGERPRINT
    )


def test_v2_report_records_status_categories_and_separate_headlines(
    pool_inputs: PoolInputs,
) -> None:
    golden = pool_inputs[0]
    report = _v2_report(pool_inputs)
    assert report.metric_policy == 'declared-sets-query-macro-v2'
    assert report.golden_status == golden.status == 'draft'
    categories = {case.case_id: case.category for case in golden.cases}
    rows = (*report.original_queries, *report.interpretations)
    assert all(row.category is categories[row.query.case_id] for row in rows)
    groups = {(group.retriever.value, group.group)
              for group in report.aggregates}
    for kind in ('bm25', 'minilm', 'hybrid'):
        assert (kind, 'headline_complete:original') in groups
        assert (kind, 'headline_complete:clarified') in groups
        assert (kind, 'headline_complete') not in groups
    assert report == _load(json.loads(report.model_dump_json()))


def test_v2_fingerprint_hashes_status_and_category(
    pool_inputs: PoolInputs,
) -> None:
    report = _v2_report(pool_inputs)
    data = json.loads(report.model_dump_json())
    frozen = dict(data, golden_status='frozen')
    assert _load(frozen).fingerprint != report.fingerprint
    data['original_queries'][0]['category'] = 'single_book'
    assert _load(data).fingerprint != report.fingerprint
    payload = report.model_dump(mode='json', exclude={'diagnostics'})
    assert report.fingerprint == fingerprint_canonical_json(payload)


def _v2_change(data: dict[str, Any], change: str) -> None:
    clarified = [group for group in data['aggregates']
                 if group['group'] == 'headline_complete:clarified']
    if change == 'status':
        data['golden_status'] = None
    elif change == 'category':
        data['interpretations'][0]['category'] = None
    elif change == 'historical_headline':
        clarified[0]['group'] = 'headline_complete'
    elif change == 'unknown_group':
        clarified[0]['group'] = 'category:invented:clarified'
    elif change == 'weighting':
        clarified[0]['weighting'] = 'query_variant'
    elif change == 'missing_headline':
        data['aggregates'].remove(clarified[0])
    elif change == 'denominator':
        metric = clarified[0]['metrics']['ranked_only.ber@5']
        metric['contributing_queries'] = metric['denominator'] - 1
    elif change == 'case_count':
        clarified[0]['eligible_case_count'] = clarified[0]['case_count'] + 1
    else:
        data['metric_policy'] = 'declared-sets-query-macro-v1'


@pytest.mark.parametrize('change', [
    'status', 'category', 'historical_headline', 'unknown_group',
    'weighting', 'missing_headline', 'denominator', 'case_count',
    'historical_policy',
])
def test_v2_policy_rejects_missing_or_contradictory_provenance(
    pool_inputs: PoolInputs, change: str,
) -> None:
    data = json.loads(_v2_report(pool_inputs).model_dump_json())
    _v2_change(data, change)
    with pytest.raises(ValidationError):
        _load(data)


def test_report_records_the_validated_golden_status(
    pool_inputs: PoolInputs,
) -> None:
    golden, corpus, runs, pages = pool_inputs
    data = golden.model_dump(mode='json', by_alias=True)
    data.update(status='frozen', freeze_record={
        'approved_by': 'Synthetic reviewer', 'approval_note': 'Synthetic.',
    })
    frozen = type(golden).model_validate_json(json.dumps(data))
    # Freezing never changes content, so the same pool and labels apply.
    assert frozen.content_fingerprint == golden.content_fingerprint
    pool = build_pool(pool_inputs)
    labels = collect(pool_inputs, pool, fill(
        worksheet(pool_inputs, pool), lambda row: ('irrelevant', ''),
    ))
    rows = score(pool_inputs, pool, labels)
    report = build_multibook_metric_report(
        rows, pool, labels, {}, golden=frozen, golden_sha256=contracts.HASH,
    )
    assert report.golden_status == 'frozen'
    with pytest.raises(ValueError, match='Golden Set'):
        build_multibook_metric_report(
            rows, pool, labels, {}, golden=golden, golden_sha256='0' * 64,
        )
    data['cases'][0]['question'] = 'Changed?'
    changed = type(golden).model_validate_json(json.dumps(data))
    with pytest.raises(ValueError, match='Golden Set'):
        build_multibook_metric_report(
            rows, pool, labels, {}, golden=changed,
            golden_sha256=contracts.HASH,
        )
    wrong = (rows[0].model_copy(update={'category': None}), *rows[1:])
    with pytest.raises(ValueError, match='category'):
        build_multibook_metric_report(
            wrong, pool, labels, {}, golden=golden,
            golden_sha256=contracts.HASH,
        )
