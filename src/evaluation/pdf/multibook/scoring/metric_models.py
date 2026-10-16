"""Versioned retrieval metrics with explicit applicability and denominators."""

from typing import Annotated, Any, Final, Literal

from pydantic import Field, model_validator

from src.evaluation.pdf.manifest import PdfRetrieverKind
from src.evaluation.pdf.multibook.curation.golden import (
    PdfExpectedBehavior, PdfExpectedOutcome, PdfMultibookCategory,
)
from src.evaluation.pdf.multibook.pooling.models import PoolModel, SHA256
from src.evaluation.pdf.multibook.pooling.models import PdfPoolRunIdentity
from src.evaluation.pdf.multibook.retrieval.queries import PdfQueryIdentity
from src.evaluation.pdf.multibook.retrieval.runs import PdfMultibookContextHit
from src.ingestion.pdf.fingerprints import fingerprint_canonical_json


METRIC_POLICY_V1: Final = 'declared-sets-query-macro-v1'
METRIC_POLICY_V2: Final = 'declared-sets-query-macro-v2'
PdfMetricPolicy = Literal[
    'declared-sets-query-macro-v1', 'declared-sets-query-macro-v2',
]
PdfAggregateWeighting = Literal['query_variant', 'case']

# Groups written under the historical policy; its headline mixed tracks.
HISTORICAL_AGGREGATE_GROUPS = frozenset({
    'headline_complete', 'track:original', 'track:clarified',
    'track:original_ambiguous',
    *(f'outcome:{outcome.value}' for outcome in PdfExpectedOutcome),
})
_V2_FIXED_GROUPS: dict[str, PdfAggregateWeighting] = {
    'headline_complete:original': 'query_variant',
    'headline_complete:clarified': 'case',
    'diagnostic:complete_original_and_clarified': 'query_variant',
    'track:original': 'query_variant',
    'track:original_ambiguous': 'query_variant',
    'track:clarified': 'case',
}
_V2_TRACK_WEIGHTING: dict[str, PdfAggregateWeighting] = {
    'original': 'query_variant', 'clarified': 'case',
}


def v2_group_weighting(group: str) -> PdfAggregateWeighting | None:
    """Return the weighting a v2 group name requires, or None if invalid.

    Clarified groups average interpretations within each case first, so a
    case counts once however many meanings it has. Interpretations are never
    ambiguous, so no clarified ambiguous-outcome group exists.
    """
    if group in _V2_FIXED_GROUPS:
        return _V2_FIXED_GROUPS[group]
    kind, _, rest = group.partition(':')
    value, _, track = rest.rpartition(':')
    if track not in _V2_TRACK_WEIGHTING:
        return None
    if kind == 'outcome' and value in {
        outcome.value for outcome in PdfExpectedOutcome
    } and not (track == 'clarified'
               and value == PdfExpectedOutcome.AMBIGUOUS.value):
        return _V2_TRACK_WEIGHTING[track]
    if kind == 'category' and value in {
        category.value for category in PdfMultibookCategory
    }:
        return _V2_TRACK_WEIGHTING[track]
    return None


class PdfEvidenceCoverage(PoolModel):
    """Separate localized units from text supplied at each fixed cutoff."""

    cutoff: Literal[5, 10]
    ber: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    sufficient_set_success: bool | None
    localization_coverage: float | None = Field(
        ge=0, le=1, allow_inf_nan=False,
    )
    supplied_units: tuple[str, ...]
    localized_units: tuple[str, ...]
    context: tuple[PdfMultibookContextHit, ...] = ()


class PdfContextGain(PoolModel):
    """Report with-context minus ranked-only, preserving null applicability."""

    cutoff: Literal[5, 10]
    ber: float | None
    sufficient_set_success: float | None
    localization_coverage: float | None


class PdfQueryRetrievalMetrics(PoolModel):
    """One query variant and one engine; MRR always uses ranked top-10 only.

    category is the Golden Set category of the case; a clarified variant
    inherits it. Historical v1-policy rows have no category.
    """

    retriever: PdfRetrieverKind
    query: PdfQueryIdentity
    track: Literal['original', 'original_ambiguous', 'clarified']
    category: PdfMultibookCategory | None = None
    expected_behavior: PdfExpectedBehavior
    outcome: PdfExpectedOutcome
    eligible_sufficient_sets: tuple[tuple[str, ...], ...]
    exclusion: str | None
    ranked_only: tuple[PdfEvidenceCoverage, ...]
    with_context: tuple[PdfEvidenceCoverage, ...]
    context_only_gain: tuple[PdfContextGain, ...]
    mrr_at_10_localization: float | None = Field(ge=0, le=1)
    mrr_at_10_supply: float | None = Field(ge=0, le=1)


class PdfMetricMean(PoolModel):
    """Keep the metric's own denominator beside its value.

    denominator counts the averaged values: query variants or cases, as the
    aggregate's weighting says. contributing_queries (v2 policy) counts the
    non-null query-variant values behind them.
    """

    mean: float | None
    denominator: int = Field(ge=0)
    contributing_queries: int | None = Field(default=None, ge=0)


class PdfRetrievalAggregate(PoolModel):
    """Macro-average one group, never evidence pooled across meanings.

    weighting and eligible_case_count are recorded under the v2 policy.
    """

    retriever: PdfRetrieverKind
    group: str
    query_count: int = Field(ge=0)
    interpretation_count: int = Field(ge=0)
    case_count: int = Field(ge=0)
    eligible_query_count: int = Field(ge=0)
    excluded_queries: tuple[PdfQueryIdentity, ...]
    metrics: dict[str, PdfMetricMean]
    weighting: PdfAggregateWeighting | None = None
    eligible_case_count: int | None = Field(default=None, ge=0)


class PdfMultibookRetrievalEvaluation(PoolModel):
    """Bind final metrics to human inputs, runs and the metric policy.

    The v1 metric policy is historical: its artifacts stay readable with
    their original fingerprint and carry none of the v2 fields. The v2
    policy separates original and clarified headlines, weights clarified
    groups by case, adds category groups and records the Golden Set status.
    """

    schema_name: Literal['fredl.pdf.multibook-retrieval-evaluation'] = (
        'fredl.pdf.multibook-retrieval-evaluation'
    )
    version: Literal[1] = 1
    metric_policy: PdfMetricPolicy
    supply_policy: Literal['required-and-explicit-yes-v1'] = (
        'required-and-explicit-yes-v1'
    )
    golden_sha256: str = Field(pattern=SHA256)
    golden_content_fingerprint: str = Field(pattern=SHA256)
    golden_status: Literal['draft', 'frozen'] | None = None
    corpus_fingerprint: str = Field(pattern=SHA256)
    library_fingerprint: str = Field(pattern=SHA256)
    query_plan_fingerprint: str = Field(pattern=SHA256)
    comparison_identity: str = Field(pattern=SHA256)
    pool_fingerprint: str = Field(pattern=SHA256)
    judgments_fingerprint: str = Field(pattern=SHA256)
    runs: tuple[PdfPoolRunIdentity, ...]
    artifact_hashes: dict[str, Annotated[str, Field(pattern=SHA256)]]
    original_queries: tuple[PdfQueryRetrievalMetrics, ...]
    interpretations: tuple[PdfQueryRetrievalMetrics, ...]
    aggregates: tuple[PdfRetrievalAggregate, ...]
    diagnostics: dict[str, str]

    @model_validator(mode='after')
    def policy_fields(self) -> 'PdfMultibookRetrievalEvaluation':
        """Reject fields and groups that contradict the metric policy."""
        keys = [(group.retriever, group.group) for group in self.aggregates]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate aggregate group for one retriever')
        rows = (*self.original_queries, *self.interpretations)
        means = [mean for group in self.aggregates
                 for mean in group.metrics.values()]
        if self.metric_policy == METRIC_POLICY_V1:
            if (
                self.golden_status is not None
                or any(row.category is not None for row in rows)
                or any(group.weighting is not None
                       or group.eligible_case_count is not None
                       for group in self.aggregates)
                or any(mean.contributing_queries is not None
                       for mean in means)
            ):
                raise ValueError('Metric policy v1 has no v2 fields')
            if any(group.group not in HISTORICAL_AGGREGATE_GROUPS
                   for group in self.aggregates):
                raise ValueError('Metric policy v1 has only historical groups')
            return self
        if self.golden_status is None or any(row.category is None
                                             for row in rows):
            raise ValueError('Metric policy v2 needs status and categories')
        for group in self.aggregates:
            weighting = v2_group_weighting(group.group)
            if weighting is None or group.weighting != weighting or (
                group.eligible_case_count is None
                or group.eligible_case_count > group.case_count
            ):
                raise ValueError(f'Invalid v2 aggregate group {group.group}')
            for mean in group.metrics.values():
                if mean.contributing_queries is None or (
                    mean.contributing_queries != mean.denominator
                    if weighting == 'query_variant'
                    else mean.contributing_queries < mean.denominator
                ):
                    raise ValueError('Invalid v2 metric denominators')
        present = {(group.retriever, group.group) for group in self.aggregates}
        for retriever in {group.retriever for group in self.aggregates} | {
            run.retriever for run in self.runs
        }:
            for name in ('headline_complete:original',
                         'headline_complete:clarified'):
                if (retriever, name) not in present:
                    raise ValueError(f'Metric policy v2 needs group {name}')
        return self

    @property
    def fingerprint(self) -> str:
        """Environment/code diagnostics never change semantic identity.

        A v1-policy artifact hashes exactly its historical fields, so its
        fingerprint is unchanged; v2 fields are always hashed under v2.
        """
        payload = self.model_dump(mode='json', exclude={'diagnostics'})
        if self.metric_policy == METRIC_POLICY_V1:
            payload = _historical_payload(payload)
        return fingerprint_canonical_json(payload)


def _historical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop only v2 keys; v1 validation guarantees that they are null."""
    del payload['golden_status']
    for name in ('original_queries', 'interpretations'):
        for row in payload[name]:
            del row['category']
    for group in payload['aggregates']:
        del group['weighting'], group['eligible_case_count']
        for mean in group['metrics'].values():
            del mean['contributing_queries']
    return payload
