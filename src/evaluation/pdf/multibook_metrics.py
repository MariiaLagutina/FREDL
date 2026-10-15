"""Supply, localization, OR-of-AND success and truncated ranked MRR."""

from collections.abc import Mapping, Sequence

from src.evaluation.pdf.judgments import PdfRecordRelevance
from src.evaluation.pdf.library import PdfLibraryCorpus
from src.evaluation.pdf.manifest import PdfRetrieverKind
from src.evaluation.pdf.multibook_collector import (
    Binding, PdfMissingJudgments, require_collected_judgments_match,
)
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook_judgment_models import (
    PdfCollectedJudgment, PdfCollectedJudgments,
)
from src.evaluation.pdf.multibook_metric_models import (
    PdfContextGain, PdfEvidenceCoverage, PdfQueryRetrievalMetrics,
)
from src.evaluation.pdf.multibook_pool_models import (
    PdfPoolingRun, PdfRetrievalJudgmentPool,
)
from src.evaluation.pdf.multibook_pooling import (
    query_pool_units, require_pdf_retrieval_pool_matches,
)
from src.ingestion.pdf.reading_text import PdfReadingPage


def score_pdf_multibook_retrieval(
    *, golden: PdfMultibookGolden, golden_sha256: str,
    corpus: PdfLibraryCorpus, pool: PdfRetrievalJudgmentPool,
    runs: Sequence[PdfPoolingRun], judgments: PdfCollectedJudgments,
    reading_pages: Mapping[str, Sequence[PdfReadingPage]],
) -> tuple[PdfQueryRetrievalMetrics, ...]:
    """Refuse all final metrics until every contributing binding is complete.

    Declared sufficient sets are eligible, including partial/extraction-limited
    diagnostic sets. No modality or model-based eligibility inference applies.
    Localization requires REQUIRED; supply requires REQUIRED and explicit
    yes. Supporting labels never close a unit. Ambiguous originals have no
    sets: their conditional-unit observations remain diagnostic, not averages.
    """
    golden = PdfMultibookGolden.model_validate(golden.model_dump())
    pool = PdfRetrievalJudgmentPool.model_validate(pool.model_dump())
    require_pdf_retrieval_pool_matches(
        pool, golden=golden, golden_sha256=golden_sha256, corpus=corpus,
        runs=runs, reading_pages=reading_pages,
    )
    runs = tuple(PdfPoolingRun.model_validate(item.model_dump())
                 for item in runs)
    judgments = require_collected_judgments_match(judgments, pool)
    labels = {item.task.key: item for item in judgments.judgments}
    needed: set[Binding] = set()
    for item in runs:
        for result in item.run.results:
            records = {hit.record_id for hit in result.ranking[:10]}
            records.update(hit.record_id for part in result.context
                           for hit in part.records)
            needed.update((*result.query.key, unit, record)
                          for unit in query_pool_units(golden, result.query)
                          for record in records)
    # Order by query, unit and canonical record; avoid comparing None with str.
    query_order = {query.identity.key: index
                   for index, query in enumerate(pool.query_plan.queries)}
    record_order = {record.record_id: index
                    for index, record in enumerate(corpus.source_records)}
    missing = [key for key in needed
               if key not in labels or not labels[key].complete]
    if missing:
        raise PdfMissingJudgments(sorted(missing, key=lambda key: (
            query_order[key[:2]], key[2], record_order[key[3]],
        )))
    scores = []
    for item in sorted(runs, key=lambda item: list(PdfRetrieverKind).index(
        item.run.manifest.retrieval.retriever,
    )):
        for result, query in zip(item.run.results, pool.query_plan.queries):
            case = next(case for case in golden.cases
                        if case.case_id == result.query.case_id)
            subject = case if result.query.interpretation_id is None else next(
                meaning for meaning in case.interpretations
                if meaning.interpretation_id == result.query.interpretation_id
            )
            sets = subject.sufficient_sets
            units = query_pool_units(golden, result.query)
            by_unit = {unit: {record: labels[(*result.query.key, unit, record)]
                              for record in record_order
                              if (*result.query.key, unit, record) in labels}
                       for unit in units}
            ranked_tracks = []
            context_tracks = []
            gains = []
            for cutoff in (5, 10):
                ranked = tuple(hit.record_id
                               for hit in result.ranking[:cutoff])
                context = next(part for part in result.context
                               if part.cutoff == cutoff)
                ranked_score = _coverage(sets, by_unit, ranked, cutoff)
                expanded = _coverage(sets, by_unit, (
                    *ranked, *(hit.record_id for hit in context.records),
                ), cutoff).model_copy(update={'context': context.records})
                ranked_tracks.append(ranked_score)
                context_tracks.append(expanded)
                gains.append(PdfContextGain(
                    cutoff=5 if cutoff == 5 else 10,
                    ber=_difference(expanded.ber, ranked_score.ber),
                    sufficient_set_success=_difference(
                        expanded.sufficient_set_success,
                        ranked_score.sufficient_set_success,
                    ),
                    localization_coverage=_difference(
                        expanded.localization_coverage,
                        ranked_score.localization_coverage,
                    ),
                ))
            eligible = {unit for group in sets for unit in group}
            ranked10 = tuple(hit.record_id for hit in result.ranking[:10])
            scores.append(PdfQueryRetrievalMetrics(
                retriever=item.run.manifest.retrieval.retriever,
                query=result.query, track=query.track,
                expected_behavior=case.expected_behavior,
                outcome=subject.expected_outcome,
                eligible_sufficient_sets=sets,
                exclusion=None if sets else 'no_declared_sufficient_set',
                ranked_only=tuple(ranked_tracks),
                with_context=tuple(context_tracks),
                context_only_gain=tuple(gains),
                mrr_at_10_localization=_mrr(by_unit, eligible, ranked10, False)
                if sets else None,
                mrr_at_10_supply=_mrr(by_unit, eligible, ranked10, True)
                if sets else None,
            ))
    return tuple(scores)


def _coverage(
    sets: tuple[tuple[str, ...], ...],
    labels: dict[str, dict[str, PdfCollectedJudgment]],
    records: tuple[str, ...], cutoff: int,
) -> PdfEvidenceCoverage:
    localized = tuple(unit for unit, rows in labels.items() if any(
        rows[record].relevance is PdfRecordRelevance.REQUIRED
        for record in records
    ))
    supplied = tuple(unit for unit, rows in labels.items() if any(
        rows[record].relevance is PdfRecordRelevance.REQUIRED
        and rows[record].supplies_evidence is True for record in records
    ))
    return PdfEvidenceCoverage(
        cutoff=5 if cutoff == 5 else 10,
        ber=_best_coverage(sets, supplied),
        localization_coverage=_best_coverage(sets, localized),
        sufficient_set_success=any(set(group).issubset(supplied)
                                   for group in sets) if sets else None,
        localized_units=localized, supplied_units=supplied,
    )


def _best_coverage(
    sets: tuple[tuple[str, ...], ...], covered: tuple[str, ...],
) -> float | None:
    return max(len(set(group).intersection(covered)) / len(group)
               for group in sets) if sets else None


def _difference(first: float | None, second: float | None) -> float | None:
    return None if first is None or second is None else float(first - second)


def _mrr(
    labels: dict[str, dict[str, PdfCollectedJudgment]],
    units: set[str], ranking: tuple[str, ...], supply: bool,
) -> float:
    for rank, record in enumerate(ranking, 1):
        if any(
            labels[unit][record].relevance is PdfRecordRelevance.REQUIRED
            and (not supply or labels[unit][record].supplies_evidence is True)
            for unit in units
        ):
            return 1.0 / rank
    return 0.0
