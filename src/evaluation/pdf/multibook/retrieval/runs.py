"""Validate future multi-book rankings; no retrieval or metric computation."""

from collections import defaultdict
import math
from typing import Literal

from pydantic import (
    BaseModel, ConfigDict, Field, field_validator, model_validator,
)

from src.evaluation.pdf.library import PdfLibraryCorpus
from src.evaluation.pdf.manifest import PdfRetrieverKind
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook.retrieval.manifest import (
    PdfMultibookRetrievalConfiguration,
    PdfMultibookRetrievalManifest,
    require_pdf_multibook_manifest_matches,
)
from src.evaluation.pdf.multibook.retrieval.queries import (
    PdfMultibookQueryPlan,
    PdfQueryIdentity,
)
from src.ingestion.pdf.fingerprints import fingerprint_canonical_json
from src.ingestion.pdf.index_records import (
    PdfIndexRecord,
    pdf_record_order_key,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False,
    )

    @field_validator("version", mode="before", check_fields=False)
    @classmethod
    def require_integer_version(cls, value: object) -> object:
        """Reject boolean or floating-point versions, even if equal to 1."""
        if type(value) is not int:
            raise ValueError("Schema version must be an integer")
        return value


class PdfMultibookRankedHit(_StrictModel):
    """Preserve a score and, for Hybrid, its contributing source ranks."""

    record_id: str = Field(min_length=1)
    score: float
    lexical_rank: int | None = Field(default=None, gt=0)
    semantic_rank: int | None = Field(default=None, gt=0)


class PdfMultibookContextHit(_StrictModel):
    """Name every direct ranked parent, without assigning context a rank."""

    record_id: str = Field(min_length=1)
    parent_record_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_parents(self) -> "PdfMultibookContextHit":
        """Reject duplicate provenance rather than silently deduplicate."""
        if len(set(self.parent_record_ids)) != len(self.parent_record_ids):
            raise ValueError("Duplicate context parents")
        return self


class PdfMultibookContextAtCutoff(_StrictModel):
    """Keep context for each evaluated top-k independently."""

    cutoff: Literal[5, 10]
    records: tuple[PdfMultibookContextHit, ...]


class PdfMultibookQueryResult(_StrictModel):
    """Store the full available ranking, followed by bounded context."""

    query: PdfQueryIdentity
    ranking: tuple[PdfMultibookRankedHit, ...]
    context: tuple[PdfMultibookContextAtCutoff, ...]

    @model_validator(mode="after")
    def require_unique_records(self) -> "PdfMultibookQueryResult":
        """Check reference multiplicity and the fixed cutoff order."""
        ids = [hit.record_id for hit in self.ranking]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate ranking record")
        if tuple(item.cutoff for item in self.context) != (5, 10):
            raise ValueError("Context cutoffs must be exactly (5, 10)")
        for item in self.context:
            refs = [record.record_id for record in item.records]
            if len(refs) != len(set(refs)):
                raise ValueError("Duplicate context record")
        return self


def _results_fingerprint(
    manifest: PdfMultibookRetrievalManifest,
    results: tuple[PdfMultibookQueryResult, ...],
) -> str:
    return fingerprint_canonical_json({
        "schema": "fredl.pdf.multibook-retrieval-results", "version": 1,
        "manifest_identity": manifest.identity,
        "results": [result.model_dump(mode="json") for result in results],
    })


class PdfMultibookRetrievalRun(_StrictModel):
    """Separate declared configuration identity from actual results."""

    schema_name: Literal["fredl.pdf.multibook-retrieval-run"] = (
        "fredl.pdf.multibook-retrieval-run"
    )
    version: Literal[1] = 1
    manifest: PdfMultibookRetrievalManifest
    manifest_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    results: tuple[PdfMultibookQueryResult, ...]
    results_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_identity(self) -> "PdfMultibookRetrievalRun":
        """Reject missing, reordered or altered query results and hashes."""
        keys = tuple(result.query.key for result in self.results)
        expected = tuple(query.identity.key
                         for query in self.manifest.query_plan.queries)
        if keys != expected:
            raise ValueError("Run query entries differ from query plan")
        if self.manifest_identity != self.manifest.identity:
            raise ValueError("Manifest identity differs")
        if self.results_fingerprint != _results_fingerprint(
            self.manifest, self.results,
        ):
            raise ValueError("Results fingerprint differs")
        return self


def build_pdf_multibook_run(
    manifest: PdfMultibookRetrievalManifest,
    results: tuple[PdfMultibookQueryResult, ...],
) -> PdfMultibookRetrievalRun:
    """Wrap supplied rankings without running or repairing them.

    Call require_pdf_multibook_run_matches before accepting an artifact:
    only that check can verify references against the current corpus.
    """
    return PdfMultibookRetrievalRun(
        manifest=manifest, manifest_identity=manifest.identity,
        results=results,
        results_fingerprint=_results_fingerprint(manifest, results),
    )


def require_pdf_multibook_run_matches(
    run: PdfMultibookRetrievalRun,
    *,
    golden: PdfMultibookGolden,
    golden_sha256: str,
    corpus: PdfLibraryCorpus,
    query_plan: PdfMultibookQueryPlan,
    retrieval: PdfMultibookRetrievalConfiguration,
) -> None:
    """Validate exact input bindings, rankings and one-hop provenance.

    This checks supplied data only; it cannot establish that scores were
    produced by the declared engine without rerunning that engine.
    """
    # Recheck the envelope and nested fields, even for copied model instances.
    run = PdfMultibookRetrievalRun.model_validate(run.model_dump())
    require_pdf_multibook_manifest_matches(
        run.manifest, golden=golden, golden_sha256=golden_sha256,
        corpus=corpus, query_plan=query_plan, retrieval=retrieval,
    )
    records = {record.record_id: record for record in corpus.records}
    for result in run.results:
        if any(hit.record_id not in records for hit in result.ranking):
            raise ValueError("Ranking names unknown or ineligible record")
        _require_ranking(result.ranking, retrieval, records)
        _require_context(result, records)


def _require_ranking(
    hits: tuple[PdfMultibookRankedHit, ...],
    config: PdfMultibookRetrievalConfiguration,
    records: dict[str, PdfIndexRecord],
) -> None:
    kind = config.retriever
    if kind is PdfRetrieverKind.HYBRID:
        _require_fusion(hits, config, len(records))
    else:
        if any(hit.lexical_rank is not None or hit.semantic_rank is not None
               for hit in hits):
            raise ValueError("Source ranks belong only to Hybrid")
        if kind is PdfRetrieverKind.MINILM:
            if len(hits) != len(records):
                raise ValueError("MiniLM ranking must cover eligible records")
            if any(not -1.000001 <= hit.score <= 1.000001 for hit in hits):
                raise ValueError("Invalid cosine score")
        elif any(hit.score <= 0 for hit in hits):
            raise ValueError("BM25 hits must have positive scores")

    def order(hit: PdfMultibookRankedHit) -> tuple[object, ...]:
        record = records[hit.record_id]
        if kind is PdfRetrieverKind.BM25:
            # The historical lexical tie rule uses document.key, including
            # a textual flow suffix; it is not the semantic row order.
            return (-hit.score, f"{record.book_path}#flow={record.flow_index}",
                    record.flow_start, record.flow_end)
        if kind is PdfRetrieverKind.HYBRID:
            return (-hit.score, hit.lexical_rank is None,
                    hit.lexical_rank or hit.semantic_rank)
        return (-hit.score, *pdf_record_order_key(record))

    if list(hits) != sorted(hits, key=order):
        raise ValueError("Invalid ranking order or tie order")


def _require_fusion(
    hits: tuple[PdfMultibookRankedHit, ...],
    config: PdfMultibookRetrievalConfiguration,
    count: int,
) -> None:
    rrf = config.rrf
    if rrf is None:
        raise ValueError("Hybrid needs RRF configuration")
    for field, weight in (("lexical_rank", rrf.lexical_weight),
                          ("semantic_rank", rrf.semantic_weight)):
        ranks = [rank for hit in hits
                 if (rank := getattr(hit, field)) is not None]
        if (weight == 0 and ranks) or sorted(ranks) != list(
            range(1, len(ranks) + 1)
        ) or len(ranks) > count:
            raise ValueError("Invalid Hybrid source ranks")
        if field == "semantic_rank" and weight > 0 and len(ranks) != count:
            raise ValueError("Hybrid ranks must cover eligible records")
    for hit in hits:
        if hit.lexical_rank is None and hit.semantic_rank is None:
            raise ValueError("Hybrid hit has no source rank")
        expected = sum(
            weight / (rrf.rank_constant + rank)
            for rank, weight in ((hit.lexical_rank, rrf.lexical_weight),
                                 (hit.semantic_rank, rrf.semantic_weight))
            if rank is not None
        )
        if not math.isclose(hit.score, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Hybrid score differs from RRF source ranks")


def _require_context(
    result: PdfMultibookQueryResult,
    records: dict[str, PdfIndexRecord],
) -> None:
    flows: dict[tuple[str, int], list[PdfIndexRecord]] = defaultdict(list)
    for record in sorted(records.values(), key=pdf_record_order_key):
        flows[record.book_path, record.flow_index].append(record)
    positions = {record.record_id: (flow, index)
                 for flow in flows.values()
                 for index, record in enumerate(flow)}
    for context in result.context:
        seeds = result.ranking[:context.cutoff]
        ranked = {hit.record_id for hit in seeds}
        expected: dict[str, list[str]] = {}
        for hit in seeds:
            flow, index = positions[hit.record_id]
            record = records[hit.record_id]
            for neighbor_index in (index - 1, index + 1):
                if not 0 <= neighbor_index < len(flow):
                    continue
                neighbor = flow[neighbor_index]
                if neighbor.record_id in ranked:
                    continue
                same = record.section_interval_id is not None and (
                    record.section_interval_id == neighbor.section_interval_id
                )
                touching = record.flow_end == neighbor.flow_start or (
                    neighbor.flow_end == record.flow_start
                )
                if same and touching:
                    expected.setdefault(neighbor.record_id, []).append(
                        record.record_id,
                    )
        actual = [(hit.record_id, hit.parent_record_ids)
                  for hit in context.records]
        if actual != [(key, tuple(parents))
                      for key, parents in expected.items()]:
            raise ValueError("Invalid or incomplete context provenance")
