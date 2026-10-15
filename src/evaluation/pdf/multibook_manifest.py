"""Multi-book retrieval identities, independent of single-book runs."""

import math
from typing import Literal

from pydantic import (
    BaseModel, ConfigDict, Field, field_validator, model_validator,
)

from src.evaluation.pdf.library import (
    PdfLibraryCorpus,
    PdfLibraryManifest,
    require_library_matches_manifest,
)
from src.evaluation.pdf.manifest import (
    PDF_SEMANTIC_RUNTIME,
    PDF_TOKENIZATION_VERSION,
    PdfBm25Configuration,
    PdfRetrieverKind,
    PdfRrfConfiguration,
    PdfRunEnvironment,
    PdfSemanticConfiguration,
)
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook_queries import (
    PdfMultibookQueryPlan,
    require_pdf_query_plan_matches,
)
from src.ingestion.pdf.fingerprints import fingerprint_canonical_json
from src.ingestion.pdf.retrieval_corpus import (
    PDF_RETRIEVAL_CORPUS_VERSION,
    PdfRetrievalPolicy,
)
from src.retrieval.bm25 import BM25Parameters
from src.retrieval.hybrid import RRFParameters


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


class PdfMultibookContextConfiguration(_StrictModel):
    """Declare one-hop eligible context, never a cross-book expansion."""

    rule: Literal["same-book-flow-interval-touching-neighbors-v1"] = (
        "same-book-flow-interval-touching-neighbors-v1"
    )
    cutoffs: tuple[Literal[5], Literal[10]] = (5, 10)
    passage_text: Literal["original_record_text"] = "original_record_text"


class PdfMultibookRetrievalConfiguration(_StrictModel):
    """Describe global search without adding a retrieval implementation."""

    retriever: PdfRetrieverKind
    candidate_scope: Literal["all_eligible_records"] = "all_eligible_records"
    ranking_depth: Literal["all_matches"] = "all_matches"
    tie_rule: Literal["existing-pdf-baselines-v1"] = (
        "existing-pdf-baselines-v1"
    )
    bm25: PdfBm25Configuration | None = None
    semantic: PdfSemanticConfiguration | None = None
    rrf: PdfRrfConfiguration | None = None

    @model_validator(mode="after")
    def require_algorithm(self) -> "PdfMultibookRetrievalConfiguration":
        """Require precisely the configurations used by the algorithm."""
        needs_bm25 = self.retriever is not PdfRetrieverKind.MINILM
        needs_semantic = self.retriever is not PdfRetrieverKind.BM25
        if (self.bm25 is not None) != needs_bm25 or (
            (self.semantic is not None) != needs_semantic
        ) or ((self.rrf is not None) != (
            self.retriever is PdfRetrieverKind.HYBRID
        )):
            raise ValueError("Incompatible retrieval configuration fields")
        if self.bm25 is not None:
            values = self.bm25.model_dump()
            if any(not math.isfinite(values[name]) for name in (
                "k1", "b", "metadata_weight", "identifier_weight",
            )):
                raise ValueError("BM25 configuration must be finite")
            if self.bm25.tokenization != PDF_TOKENIZATION_VERSION:
                raise ValueError("Unsupported PDF tokenization")
            BM25Parameters(**{
                name: values[name] for name in values if name != "tokenization"
            })
        if self.semantic is not None:
            if self.semantic.runtime != PDF_SEMANTIC_RUNTIME:
                raise ValueError("Unsupported semantic runtime")
            if not self.semantic.model_name.strip() or not (
                self.semantic.model_revision.strip()
            ):
                raise ValueError("Semantic model identity must not be blank")
        if self.rrf is not None:
            if not all(math.isfinite(weight) for weight in (
                self.rrf.lexical_weight, self.rrf.semantic_weight,
            )):
                raise ValueError("RRF configuration must be finite")
            RRFParameters(**self.rrf.model_dump())
        return self

    @property
    def fingerprint(self) -> str:
        """Hash algorithm-specific settings independently of results."""
        return fingerprint_canonical_json({
            "schema": "fredl.pdf.multibook-retrieval-configuration",
            "version": 1, "configuration": self.model_dump(mode="json"),
        })


class PdfMultibookCorpusBook(_StrictModel):
    """Keep per-book policy views and counts auditable."""

    book_path: str = Field(min_length=1)
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(gt=0)
    eligible_record_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_counts(self) -> "PdfMultibookCorpusBook":
        """Eligibility cannot add records to a book."""
        if self.eligible_record_count > self.record_count:
            raise ValueError("Eligible count exceeds source record count")
        return self


class PdfMultibookRetrievalManifest(_StrictModel):
    """Bind exact inputs and settings; environment is diagnostic only."""

    schema_name: Literal["fredl.pdf.multibook-retrieval-manifest"] = (
        "fredl.pdf.multibook-retrieval-manifest"
    )
    version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    golden_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    golden_content_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    library: PdfLibraryManifest
    library_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy: PdfRetrievalPolicy
    policy_version: Literal["pdf-retrieval-corpus-v1"] = (
        "pdf-retrieval-corpus-v1"
    )
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    books: tuple[PdfMultibookCorpusBook, ...]
    eligible_record_count: int = Field(gt=0)
    query_plan: PdfMultibookQueryPlan
    query_plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval: PdfMultibookRetrievalConfiguration
    context: PdfMultibookContextConfiguration = (
        PdfMultibookContextConfiguration()
    )
    code_revision: str = Field(min_length=1)
    code_dirty: bool
    environment: dict[str, str]

    @model_validator(mode="after")
    def require_bindings(self) -> "PdfMultibookRetrievalManifest":
        """Reject internally contradictory identity and count metadata."""
        if self.policy_version != PDF_RETRIEVAL_CORPUS_VERSION:
            raise ValueError("Unsupported corpus policy version")
        if self.library_fingerprint != self.library.fingerprint:
            raise ValueError("Library fingerprint differs from library pin")
        if tuple(book.book_path for book in self.books) != (
            self.library.book_paths
        ):
            raise ValueError("Corpus book order differs from library pin")
        if self.eligible_record_count != sum(
            book.eligible_record_count for book in self.books
        ):
            raise ValueError("Global eligible count differs from book counts")
        if self.query_plan_fingerprint != self.query_plan.fingerprint:
            raise ValueError("Query-plan fingerprint differs")
        if self.golden_content_fingerprint != (
            self.query_plan.golden_content_fingerprint
        ):
            raise ValueError("Query plan belongs to a different Golden Set")
        return self

    @property
    def comparison_identity(self) -> str:
        """Identify shared conditions, excluding the chosen algorithm."""
        return fingerprint_canonical_json({
            "schema": "fredl.pdf.multibook-comparison", "version": 1,
            "inputs": self.model_dump(mode="json", exclude={
                "retrieval", "code_revision", "code_dirty", "environment",
            }),
            "candidate_scope": self.retrieval.candidate_scope,
            "ranking_depth": self.retrieval.ranking_depth,
        })

    @property
    def identity(self) -> str:
        """Identify configuration, never assume equal rankings from it."""
        return fingerprint_canonical_json({
            "schema": "fredl.pdf.multibook-retrieval-identity", "version": 1,
            "comparison_identity": self.comparison_identity,
            "retrieval_fingerprint": self.retrieval.fingerprint,
        })


def build_pdf_multibook_manifest(
    *,
    golden: PdfMultibookGolden,
    golden_sha256: str,
    corpus: PdfLibraryCorpus,
    query_plan: PdfMultibookQueryPlan,
    retrieval: PdfMultibookRetrievalConfiguration,
    environment: PdfRunEnvironment,
) -> PdfMultibookRetrievalManifest:
    """Bind already-built synthetic or approved inputs, without I/O."""
    require_library_matches_manifest(corpus, golden.library)
    require_pdf_query_plan_matches(query_plan, golden)
    return PdfMultibookRetrievalManifest(
        dataset_id=golden.dataset_id, golden_sha256=golden_sha256,
        golden_content_fingerprint=golden.content_fingerprint,
        library=corpus.manifest,
        library_fingerprint=corpus.manifest.fingerprint,
        policy=corpus.policy, corpus_fingerprint=corpus.fingerprint,
        books=tuple(PdfMultibookCorpusBook(
            book_path=book.book_path,
            corpus_fingerprint=book.corpus.fingerprint,
            record_count=len(book.records),
            eligible_record_count=len(book.corpus.records),
        ) for book in corpus.books),
        eligible_record_count=len(corpus.records), query_plan=query_plan,
        query_plan_fingerprint=query_plan.fingerprint, retrieval=retrieval,
        code_revision=environment.code_revision,
        code_dirty=environment.code_dirty,
        environment=dict(environment.packages),
    )


def require_pdf_multibook_manifest_matches(
    manifest: PdfMultibookRetrievalManifest,
    *,
    golden: PdfMultibookGolden,
    golden_sha256: str,
    corpus: PdfLibraryCorpus,
    query_plan: PdfMultibookQueryPlan,
    retrieval: PdfMultibookRetrievalConfiguration,
) -> None:
    """Reject drift in exact bytes, content, corpus, queries or settings."""
    # model_copy/model_construct do not validate fields or nested instances.
    manifest = PdfMultibookRetrievalManifest.model_validate(
        manifest.model_dump(),
    )
    expected = build_pdf_multibook_manifest(
        golden=golden, golden_sha256=golden_sha256, corpus=corpus,
        query_plan=query_plan, retrieval=retrieval,
        environment=PdfRunEnvironment(
            manifest.code_revision, manifest.code_dirty, manifest.environment,
        ),
    )
    if manifest != expected:
        raise ValueError("Multi-book manifest differs from current inputs")
