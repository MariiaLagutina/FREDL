"""Policy v2 headlines, case-weighted clarified groups and category slices."""

import pytest

from src.evaluation.pdf.multibook.scoring.metric_models import (
    PdfQueryRetrievalMetrics, PdfRetrievalAggregate,
)
from src.evaluation.pdf.multibook.scoring.report import (
    aggregate_multibook_metrics,
)
from tests.evaluation.pdf.multibook.scoring.metric_helpers import metric_row


AMBIGUOUS = 'ambiguous_terminology'


def _groups(
    rows: list[PdfQueryRetrievalMetrics],
) -> dict[str, PdfRetrievalAggregate]:
    return {group.group: group for group in aggregate_multibook_metrics(rows)
            if group.retriever.value == 'minilm'}


def _clear_and_clarified_rows() -> list[PdfQueryRetrievalMetrics]:
    """Two clear questions and two ambiguous cases with 1 and 3 meanings."""
    return [
        metric_row('single', category='single_book', ber=(1.0, 1.0)),
        metric_row('cross', category='complementary_multi_book',
                   ber=(0.5, 1.0)),
        metric_row('negative', category='hard_negative',
                   outcome='unsupported'),
        metric_row('one', category=AMBIGUOUS, outcome='ambiguous'),
        metric_row('one', 'a', category=AMBIGUOUS, ber=(1.0, 1.0)),
        metric_row('three', category=AMBIGUOUS, outcome='ambiguous'),
        metric_row('three', 'a', category=AMBIGUOUS, ber=(0.0, 0.0)),
        metric_row('three', 'b', category=AMBIGUOUS, ber=(0.0, 0.5)),
        metric_row('three', 'c', category=AMBIGUOUS, ber=(0.6, 1.0)),
    ]


def test_clarified_headline_gives_each_case_one_vote() -> None:
    groups = _groups(_clear_and_clarified_rows())
    clarified = groups['headline_complete:clarified']
    # Case means: one = 1.0 and three = (0 + 0 + 0.6) / 3 = 0.2.
    # Query-variant weighting would instead give 1.6 / 4 = 0.4.
    ber = clarified.metrics['ranked_only.ber@5']
    assert ber.mean == pytest.approx(0.6)
    assert (ber.denominator, ber.contributing_queries) == (2, 4)
    assert clarified.weighting == 'case'
    assert (clarified.query_count, clarified.interpretation_count,
            clarified.case_count, clarified.eligible_query_count,
            clarified.eligible_case_count) == (4, 4, 2, 4, 2)
    # BER@10 case means: 1.0 and (0 + 0.5 + 1) / 3 = 0.5.
    assert clarified.metrics['ranked_only.ber@10'].mean == pytest.approx(0.75)
    combined = groups['diagnostic:complete_original_and_clarified']
    assert combined.weighting == 'query_variant'
    # (1.0 + 0.5 + 1.0 + 0.0 + 0.0 + 0.6) / 6 query variants.
    assert combined.metrics['ranked_only.ber@5'].mean == pytest.approx(
        3.1 / 6)
    assert combined.metrics['ranked_only.ber@5'].denominator == 6


def test_original_headline_excludes_clarified_and_ambiguous_originals(
) -> None:
    groups = _groups(_clear_and_clarified_rows())
    headline = groups['headline_complete:original']
    assert headline.weighting == 'query_variant'
    assert (headline.query_count, headline.interpretation_count,
            headline.case_count) == (2, 0, 2)
    ber = headline.metrics['ranked_only.ber@5']
    assert (ber.mean, ber.denominator, ber.contributing_queries) == (
        0.75, 2, 2)
    ambiguous = groups['track:original_ambiguous']
    assert (ambiguous.query_count, ambiguous.eligible_query_count,
            ambiguous.eligible_case_count) == (2, 0, 0)
    assert all(metric.mean is None and metric.denominator == 0
               and metric.contributing_queries == 0
               for metric in ambiguous.metrics.values())


def test_category_groups_follow_actual_categories_and_tracks() -> None:
    groups = _groups(_clear_and_clarified_rows())
    categories = {name for name in groups if name.startswith('category:')}
    assert categories == {
        'category:single_book:original',
        'category:complementary_multi_book:original',
        'category:hard_negative:original',
        f'category:{AMBIGUOUS}:original',
        f'category:{AMBIGUOUS}:clarified',
    }
    single = groups['category:single_book:original']
    assert single.metrics['ranked_only.ber@5'].mean == 1.0
    assert single.metrics['ranked_only.ber@5'].denominator == 1
    cross = groups['category:complementary_multi_book:original']
    assert cross.metrics['ranked_only.ber@10'].mean == 1.0
    assert cross.metrics['ranked_only.ber@5'].mean == 0.5
    negative = groups['category:hard_negative:original']
    assert negative.query_count == 1 and negative.eligible_query_count == 0
    assert negative.metrics['ranked_only.ber@5'].mean is None
    assert negative.metrics['ranked_only.ber@5'].denominator == 0
    assert [query.case_id for query in negative.excluded_queries] == [
        'negative',
    ]
    # Ambiguous originals stay in their category's original track only.
    original = groups[f'category:{AMBIGUOUS}:original']
    assert {query.case_id for query in original.excluded_queries} == {
        'one', 'three',
    }
    assert original.interpretation_count == 0
    clarified = groups[f'category:{AMBIGUOUS}:clarified']
    assert clarified.weighting == 'case'
    assert clarified.metrics['ranked_only.ber@5'].mean == pytest.approx(0.6)
    assert clarified.metrics['ranked_only.ber@5'].denominator == 2


def test_null_interpretations_never_become_zero_or_extra_cases() -> None:
    rows = [
        metric_row('full', 'a', category=AMBIGUOUS, ber=(1.0, 1.0)),
        metric_row('mixed', 'a', category=AMBIGUOUS, outcome='unsupported'),
        metric_row('mixed', 'b', category=AMBIGUOUS, ber=(0.5, 0.5)),
        metric_row('empty', 'a', category=AMBIGUOUS, outcome='unsupported'),
        metric_row('empty', 'b', category=AMBIGUOUS,
                   outcome='source_incomplete'),
    ]
    group = _groups(rows)[f'category:{AMBIGUOUS}:clarified']
    ber = group.metrics['ranked_only.ber@5']
    # mixed averages only its non-null meaning; empty has no value at all.
    assert ber.mean == pytest.approx(0.75)
    assert (ber.denominator, ber.contributing_queries) == (2, 2)
    assert (group.case_count, group.eligible_case_count,
            group.eligible_query_count) == (3, 2, 2)
    assert len(group.excluded_queries) == 3
    assert _groups(rows)['headline_complete:clarified'].case_count == 2
    unsupported = _groups(rows)['outcome:unsupported:clarified']
    assert unsupported.metrics['ranked_only.ber@5'].mean is None
    assert unsupported.metrics['ranked_only.ber@5'].denominator == 0


def test_outcome_groups_keep_tracks_and_have_no_clarified_ambiguity() -> None:
    groups = _groups(_clear_and_clarified_rows())
    assert groups['outcome:complete:original'].query_count == 2
    assert groups['outcome:complete:clarified'].query_count == 4
    assert groups['outcome:ambiguous:original'].query_count == 2
    assert 'outcome:ambiguous:clarified' not in groups
    assert 'headline_complete' not in groups


def test_rows_without_category_cannot_be_aggregated() -> None:
    row = metric_row('single', category='single_book', ber=(1.0, 1.0))
    with pytest.raises(ValueError, match='category'):
        aggregate_multibook_metrics([row.model_copy(update={
            'category': None,
        })])


def test_case_macro_averages_several_values_beside_a_null() -> None:
    rows = [
        metric_row('A', 'first', category=AMBIGUOUS, ber=(1.0, 1.0)),
        metric_row('A', 'second', category=AMBIGUOUS, outcome='unsupported'),
        metric_row('A', 'third', category=AMBIGUOUS, ber=(0.5, 0.5)),
        metric_row('B', 'first', category=AMBIGUOUS, ber=(0.0, 0.0)),
    ]
    group = _groups(rows)[f'category:{AMBIGUOUS}:clarified']
    ber = group.metrics['ranked_only.ber@5']
    # A = (1.0 + 0.5) / 2 = 0.75 and B = 0.0; the null is not a zero.
    assert ber.mean == 0.375
    assert (ber.denominator, ber.contributing_queries) == (2, 3)
    assert (group.eligible_case_count, group.eligible_query_count) == (2, 3)
