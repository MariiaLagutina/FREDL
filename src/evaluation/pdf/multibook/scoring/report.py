"""Macro summaries and reproducible multi-book retrieval artifacts."""

from collections.abc import Sequence

from src.evaluation.pdf.manifest import (
    current_pdf_run_environment, PdfRetrieverKind,
)
from src.evaluation.pdf.multibook.curation.golden import (
    PdfExpectedOutcome, PdfMultibookCategory, PdfMultibookGolden,
)
from src.evaluation.pdf.multibook.scoring.judgment_models import (
    PdfCollectedJudgments,
)
from src.evaluation.pdf.multibook.scoring.metric_models import (
    METRIC_POLICY_V2, PdfAggregateWeighting, PdfMetricMean,
    PdfMultibookRetrievalEvaluation, PdfQueryRetrievalMetrics,
    PdfRetrievalAggregate, v2_group_weighting,
)
from src.evaluation.pdf.multibook.pooling.models import (
    PdfRetrievalJudgmentPool,
)


# (metric name, per-query track, cutoff index, field) of every averaged value.
_COVERAGE_METRICS = tuple(
    (f'{track}.{metric}@{cutoff}', track, index, metric)
    for track in ('ranked_only', 'with_context', 'context_only_gain')
    for index, cutoff in enumerate((5, 10))
    for metric in ('ber', 'sufficient_set_success', 'localization_coverage')
)
_RANK_METRICS = ('mrr_at_10_localization', 'mrr_at_10_supply')


def _metric_means(
    rows: Sequence[PdfQueryRetrievalMetrics],
    weighting: PdfAggregateWeighting,
) -> dict[str, PdfMetricMean]:
    means = {
        name: _mean([(row.query.case_id,
                      getattr(getattr(row, track)[index], field))
                     for row in rows], weighting)
        for name, track, index, field in _COVERAGE_METRICS
    }
    for name in _RANK_METRICS:
        means[name] = _mean([(row.query.case_id, getattr(row, name))
                             for row in rows], weighting)
    return means


def aggregate_multibook_metrics(
    rows: Sequence[PdfQueryRetrievalMetrics],
) -> tuple[PdfRetrievalAggregate, ...]:
    """Summarize v2-policy groups; always expose applicability counts.

    The primary headline is complete clear original questions. Complete
    clarified interpretations form a separate headline, averaged within each
    case first so that every case has equal weight. The combined group is
    diagnostic only. Original ambiguous rows never enter either headline.
    Category groups exist for each category and track that has queries.
    """
    if any(row.category is None for row in rows):
        raise ValueError('Every evaluated query needs its case category')
    output = []
    for kind in PdfRetrieverKind:
        selected = [row for row in rows if row.retriever == kind]
        original = [row for row in selected if row.track != 'clarified']
        tracks = {
            'original': original,
            'clarified': [row for row in selected if row.track == 'clarified'],
        }
        clear = [row for row in selected if row.track == 'original']
        complete = PdfExpectedOutcome.COMPLETE
        groups = {
            'headline_complete:original': [
                row for row in clear if row.outcome is complete
            ],
            'headline_complete:clarified': [
                row for row in tracks['clarified'] if row.outcome is complete
            ],
            'diagnostic:complete_original_and_clarified': [
                row for row in (*clear, *tracks['clarified'])
                if row.outcome is complete
            ],
            'track:original': clear,
            'track:original_ambiguous': [
                row for row in selected if row.track == 'original_ambiguous'
            ],
            'track:clarified': tracks['clarified'],
        }
        for track, items in tracks.items():
            for outcome in PdfExpectedOutcome:
                group = f'outcome:{outcome.value}:{track}'
                if v2_group_weighting(group) is not None:
                    groups[group] = [row for row in items
                                     if row.outcome is outcome]
        for track, items in tracks.items():
            for category in PdfMultibookCategory:
                members = [row for row in items if row.category is category]
                if members:
                    groups[f'category:{category.value}:{track}'] = members
        for group, items in groups.items():
            weighting = v2_group_weighting(group)
            if weighting is None:
                raise ValueError(f'Unknown aggregate group {group}')
            output.append(PdfRetrievalAggregate(
                retriever=kind, group=group, query_count=len(items),
                interpretation_count=sum(
                    row.query.interpretation_id is not None for row in items
                ),
                case_count=len({row.query.case_id for row in items}),
                eligible_query_count=sum(bool(row.eligible_sufficient_sets)
                                         for row in items),
                excluded_queries=tuple(row.query for row in items
                                       if row.exclusion is not None),
                metrics=_metric_means(items, weighting),
                weighting=weighting,
                eligible_case_count=len({
                    row.query.case_id for row in items
                    if row.eligible_sufficient_sets
                }),
            ))
    return tuple(output)


def _mean(
    values: Sequence[tuple[str, float | bool | None]],
    weighting: PdfAggregateWeighting,
) -> PdfMetricMean:
    """Average non-null values; case weighting averages each case first."""
    present = [(case_id, float(value)) for case_id, value in values
               if value is not None]
    if weighting == 'query_variant':
        numbers = [number for _, number in present]
    else:
        cases: dict[str, list[float]] = {}
        for case_id, number in present:
            cases.setdefault(case_id, []).append(number)
        numbers = [sum(items) / len(items) for items in cases.values()]
    return PdfMetricMean(mean=sum(numbers) / len(numbers) if numbers else None,
                         denominator=len(numbers),
                         contributing_queries=len(present))


def build_multibook_metric_report(
    rows: Sequence[PdfQueryRetrievalMetrics], pool: PdfRetrievalJudgmentPool,
    judgments: PdfCollectedJudgments, artifact_hashes: dict[str, str], *,
    golden: PdfMultibookGolden, golden_sha256: str,
) -> PdfMultibookRetrievalEvaluation:
    """Package already validated final scores; diagnostics are non-semantic.

    golden must be the validated input the pool was built from; its status
    and case categories are recorded, never inferred.
    """
    if (golden_sha256 != pool.golden_sha256
            or golden.content_fingerprint != pool.golden_content_fingerprint):
        raise ValueError('Golden Set differs from the pooled input')
    categories = {case.case_id: case.category for case in golden.cases}
    if any(categories.get(row.query.case_id) is not row.category
           for row in rows):
        raise ValueError('Query category differs from its Golden Set case')
    environment = current_pdf_run_environment()
    return PdfMultibookRetrievalEvaluation(
        metric_policy=METRIC_POLICY_V2,
        golden_sha256=pool.golden_sha256,
        golden_content_fingerprint=pool.golden_content_fingerprint,
        golden_status=golden.status,
        library_fingerprint=pool.library_fingerprint,
        corpus_fingerprint=pool.corpus_fingerprint,
        query_plan_fingerprint=pool.query_plan.fingerprint,
        comparison_identity=pool.comparison_identity,
        pool_fingerprint=pool.fingerprint,
        judgments_fingerprint=judgments.fingerprint, runs=pool.runs,
        artifact_hashes=artifact_hashes,
        original_queries=tuple(row for row in rows
                               if row.query.interpretation_id is None),
        interpretations=tuple(row for row in rows
                              if row.query.interpretation_id is not None),
        aggregates=aggregate_multibook_metrics(rows),
        diagnostics={'code_revision': environment.code_revision,
                     'code_dirty': str(environment.code_dirty),
                     **dict(environment.packages)},
    )
