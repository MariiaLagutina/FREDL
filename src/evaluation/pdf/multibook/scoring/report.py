"""Macro summaries and reproducible multi-book retrieval artifacts."""

from collections.abc import Sequence

from src.evaluation.pdf.manifest import (
    current_pdf_run_environment, PdfRetrieverKind,
)
from src.evaluation.pdf.multibook.scoring.judgment_models import (
    PdfCollectedJudgments,
)
from src.evaluation.pdf.multibook.scoring.metric_models import (
    PdfMetricMean, PdfMultibookRetrievalEvaluation, PdfQueryRetrievalMetrics,
    PdfRetrievalAggregate,
)
from src.evaluation.pdf.multibook_pool_models import PdfRetrievalJudgmentPool


def aggregate_multibook_metrics(
    rows: Sequence[PdfQueryRetrievalMetrics],
) -> tuple[PdfRetrievalAggregate, ...]:
    """Use one vote per query variant; always expose applicability counts.

    Headline is complete original/clarified queries only. Partial and
    extraction-limited rows are separate diagnostics, not headline failures.
    Original ambiguous rows never enter clarified/headline averages.
    """
    output = []
    for kind in PdfRetrieverKind:
        selected = [row for row in rows if row.retriever == kind]
        groups = {
            'headline_complete': [row for row in selected
                                  if row.outcome.value == 'complete'
                                  and row.track != 'original_ambiguous'],
            **{f'track:{track}': [
                row for row in selected if row.track == track
            ] for track in ('original', 'clarified', 'original_ambiguous')},
            **{f'outcome:{status}': [
                row for row in selected if row.outcome.value == status
            ] for status in (
                'complete', 'partial', 'ambiguous', 'unsupported',
                'extraction_limited', 'source_incomplete',
            )},
        }
        for group, items in groups.items():
            metrics = {}
            for track in ('ranked_only', 'with_context', 'context_only_gain'):
                for index, cutoff in enumerate((5, 10)):
                    for metric in ('ber', 'sufficient_set_success',
                                   'localization_coverage'):
                        values = [getattr(getattr(row, track)[index], metric)
                                  for row in items]
                        metrics[f'{track}.{metric}@{cutoff}'] = _mean(values)
            for metric in ('mrr_at_10_localization', 'mrr_at_10_supply'):
                metrics[metric] = _mean([
                    getattr(row, metric) for row in items
                ])
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
                metrics=metrics,
            ))
    return tuple(output)


def _mean(values: Sequence[float | None]) -> PdfMetricMean:
    numbers = [float(value) for value in values if value is not None]
    return PdfMetricMean(mean=sum(numbers) / len(numbers) if numbers else None,
                         denominator=len(numbers))


def build_multibook_metric_report(
    rows: Sequence[PdfQueryRetrievalMetrics], pool: PdfRetrievalJudgmentPool,
    judgments: PdfCollectedJudgments, artifact_hashes: dict[str, str],
) -> PdfMultibookRetrievalEvaluation:
    """Package already validated final scores; diagnostics are non-semantic."""
    environment = current_pdf_run_environment()
    return PdfMultibookRetrievalEvaluation(
        golden_sha256=pool.golden_sha256,
        golden_content_fingerprint=pool.golden_content_fingerprint,
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
