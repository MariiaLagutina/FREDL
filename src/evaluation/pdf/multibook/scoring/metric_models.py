"""Versioned retrieval metrics with explicit applicability and denominators."""

from typing import Annotated, Literal

from pydantic import Field

from src.evaluation.pdf.manifest import PdfRetrieverKind
from src.evaluation.pdf.multibook_golden import (
    PdfExpectedBehavior, PdfExpectedOutcome,
)
from src.evaluation.pdf.multibook.pooling.models import PoolModel, SHA256
from src.evaluation.pdf.multibook.pooling.models import PdfPoolRunIdentity
from src.evaluation.pdf.multibook_queries import PdfQueryIdentity
from src.evaluation.pdf.multibook_runs import PdfMultibookContextHit
from src.ingestion.pdf.fingerprints import fingerprint_canonical_json


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
    """One query variant and one engine; MRR always uses ranked top-10 only."""

    retriever: PdfRetrieverKind
    query: PdfQueryIdentity
    track: Literal['original', 'original_ambiguous', 'clarified']
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
    """Keep the metric's own denominator beside its value."""

    mean: float | None
    denominator: int = Field(ge=0)


class PdfRetrievalAggregate(PoolModel):
    """Macro-average query variants, never evidence pooled across meanings."""

    retriever: PdfRetrieverKind
    group: str
    query_count: int = Field(ge=0)
    interpretation_count: int = Field(ge=0)
    case_count: int = Field(ge=0)
    eligible_query_count: int = Field(ge=0)
    excluded_queries: tuple[PdfQueryIdentity, ...]
    metrics: dict[str, PdfMetricMean]


class PdfMultibookRetrievalEvaluation(PoolModel):
    """Bind final metrics to human inputs, runs and the metric policy."""

    schema_name: Literal['fredl.pdf.multibook-retrieval-evaluation'] = (
        'fredl.pdf.multibook-retrieval-evaluation'
    )
    version: Literal[1] = 1
    metric_policy: Literal['declared-sets-query-macro-v1'] = (
        'declared-sets-query-macro-v1'
    )
    supply_policy: Literal['required-and-explicit-yes-v1'] = (
        'required-and-explicit-yes-v1'
    )
    golden_sha256: str = Field(pattern=SHA256)
    golden_content_fingerprint: str = Field(pattern=SHA256)
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

    @property
    def fingerprint(self) -> str:
        """Environment/code diagnostics never change semantic identity."""
        return fingerprint_canonical_json(self.model_dump(
            mode='json', exclude={'diagnostics'},
        ))
