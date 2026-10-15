"""Lossless provenance and complete queries across all selection paths."""

import json

import pytest

from tests.evaluation.pdf import multibook_contract_helpers as contracts
from tests.evaluation.pdf import multibook_pool_helpers as helpers
from tests.evaluation.pdf import multibook_runner_helpers as runner
from tests.evaluation.pdf.multibook_pool_helpers import PoolInputs, build_pool


corpus = contracts.corpus
pool_inputs = helpers.pool_inputs
forbid_real_model_loading = runner.forbid_real_model_loading


def test_one_record_preserves_rank_scores_context_and_both_anchor_kinds(
    pool_inputs: PoolInputs,
) -> None:
    golden, corpus, _, pages = pool_inputs
    data = golden.model_dump(mode='json')
    unit = data['evidence_units'][0]
    original = unit['locators'][0]
    unit['locators'].append(dict(original, exact_anchor='Workers',
                                 page_text_start=18, page_text_end=25))
    unit['supporting_locators'].append(dict(
        original, exact_anchor='poll', page_text_start=26, page_text_end=30,
    ))
    golden = type(golden).model_validate_json(json.dumps(data))
    runs = helpers.saved_runs(golden, corpus)
    pool = build_pool((golden, corpus, runs, pages), depth=5)
    candidate = next(item for item in pool.candidates
                     if item.key == ('both', None, 'a2'))
    assert candidate.book_path == 'books/alpha.pdf'
    assert {origin.kind for origin in candidate.origins} == {
        'ranked', 'context', 'anchor', 'supporting_anchor',
    }
    for item, identity in zip(runs, pool.runs):
        run = item.run
        kind = run.manifest.retrieval.retriever
        assert identity.artifact_sha256 == item.artifact_sha256
        assert identity.results_fingerprint == run.results_fingerprint
        assert identity.manifest_identity == run.manifest_identity
        result = run.results[0]
        ranked = [(rank, hit.score) for rank, hit in enumerate(
            result.ranking[:5], 1,
        ) if hit.record_id == 'a2']
        found = [origin for origin in candidate.origins
                 if origin.retriever == kind and origin.kind == 'ranked']
        assert [(origin.rank, origin.score) for origin in found] == ranked
        for origin in candidate.origins:
            if origin.retriever == kind and origin.kind == 'context':
                assert origin.parent_record_ids == ('a1',)
                assert origin.cutoff == 5
                assert origin.rank is origin.score is None
    anchor = next(origin for origin in candidate.origins
                  if origin.kind == 'anchor')
    assert (anchor.unit_id, anchor.locator_index) == ('queue', 1)
    assert len({origin.retriever for origin in candidate.origins
                if origin.retriever is not None}) == 3


@pytest.mark.parametrize('algorithm', [0, 1, 2])
@pytest.mark.parametrize('query_index', [0, 2])
def test_each_run_must_contain_every_original_and_clarified_query(
    pool_inputs: PoolInputs, algorithm: int, query_index: int,
) -> None:
    golden, corpus, runs, pages = pool_inputs
    item = runs[algorithm]
    results = tuple(result for index, result in enumerate(item.run.results)
                    if index != query_index)
    run = item.run.model_copy(update={'results': results})
    changed = list(runs)
    changed[algorithm] = item.model_copy(update={'run': run})
    with pytest.raises(ValueError):
        build_pool((golden, corpus, tuple(changed), pages))
