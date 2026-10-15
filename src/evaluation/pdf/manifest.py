"""Describe reproducible PDF evaluation runs and reject incompatible ones."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
import platform
import subprocess
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.evaluation.pdf.models import PdfGoldenDataset
from src.ingestion.pdf.fingerprints import (
    fingerprint_canonical_json,
    fingerprint_pdf_records,
)
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.ingestion.pdf.retrieval_corpus import (
    PDF_RETRIEVAL_CORPUS_VERSION,
    PdfRetrievalCorpus,
    PdfRetrievalPolicy,
)
from src.retrieval.bm25 import BM25Parameters
from src.retrieval.hybrid import RRFParameters
from src.retrieval.semantic.config import SemanticEncoderConfig


PDF_EVALUATION_RUN_SCHEMA = "fredl.pdf.evaluation-run"
PDF_EVALUATION_RUN_VERSION = 1
# Bump when neighbour-context selection changes behaviour.
PDF_CONTEXT_RULE = "section-interval-touching-neighbors-v1"
# Bump when the PDF tokenizers or their field assignment change.
PDF_TOKENIZATION_VERSION = "text-content-code-metadata-v1"
# The MiniLM encoder runs on CPU in float32 with mean pooling and L2 norm.
PDF_SEMANTIC_RUNTIME = "cpu-float32-mean-pooling-l2-v1"
# Packages whose versions can change extraction, tokens, or embeddings.
PDF_ENVIRONMENT_PACKAGES = (
    "numpy", "pypdf", "tokenizers", "torch", "transformers",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class PdfRetrieverKind(str, Enum):
    """Name the retrievers whose rankings can be evaluated."""

    BM25 = "bm25"
    MINILM = "minilm"
    HYBRID = "hybrid"


class PdfBm25Configuration(_StrictModel):
    """Record BM25 parameters and tokenization that shape lexical scores."""

    k1: float
    b: float
    metadata_weight: float
    identifier_weight: float
    tokenization: str = PDF_TOKENIZATION_VERSION


class PdfSemanticConfiguration(_StrictModel):
    """Record the exact encoder inputs that shape embeddings."""

    model_name: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    max_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    runtime: str = PDF_SEMANTIC_RUNTIME


class PdfRrfConfiguration(_StrictModel):
    """Record reciprocal-rank fusion parameters."""

    rank_constant: int
    lexical_weight: float
    semantic_weight: float


class PdfRetrievalConfiguration(_StrictModel):
    """Collect every retrieval setting that can change a ranking."""

    retriever: PdfRetrieverKind
    top_k: int = Field(gt=0)
    candidate_k: Literal["all_records"] = "all_records"
    context_rule: str = PDF_CONTEXT_RULE
    bm25: PdfBm25Configuration
    semantic: PdfSemanticConfiguration | None = None
    rrf: PdfRrfConfiguration | None = None

    @property
    def fingerprint(self) -> str:
        """Hash the retrieval configuration, not corpus or model identity."""
        return fingerprint_canonical_json({
            "schema": "fredl.pdf.retrieval-configuration",
            "version": 1,
            "configuration": self.model_dump(mode="json"),
        })


class PdfAnswerabilityConfiguration(_StrictModel):
    """Identify the external model and protocol behind answerability."""

    model: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: str = Field(min_length=1)
    server_args: tuple[str, ...]
    temperature: float
    answerability_max_tokens: int = Field(gt=0)
    comparison_max_tokens: int = Field(gt=0)
    prompt_version: str = Field(min_length=1)
    guard_version: str = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        """Hash the model identity and decision protocol."""
        return fingerprint_canonical_json({
            "schema": "fredl.pdf.answerability-configuration",
            "version": 1,
            "configuration": self.model_dump(mode="json"),
        })


class PdfEvaluationManifest(_StrictModel):
    """Bind one evaluation run to its data, corpus, and configuration.

    Identity fields decide compatibility. Code revision and environment
    are recorded for audit only; a rerun proves their effect empirically.
    """

    schema_name: Literal["fredl.pdf.evaluation-run"] = Field(
        default="fredl.pdf.evaluation-run", alias="schema",
    )
    version: Literal[1] = 1
    book_path: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_set_fingerprint_version: Literal[2] = 2
    record_count: int = Field(ge=0)
    corpus_policy: PdfRetrievalPolicy
    corpus_policy_version: str = PDF_RETRIEVAL_CORPUS_VERSION
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    eligible_record_count: int = Field(ge=0)
    retrieval: PdfRetrievalConfiguration
    retrieval_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    answerability: PdfAnswerabilityConfiguration | None = None
    answerability_fingerprint: str | None = None
    code_revision: str = Field(min_length=1)
    code_dirty: bool
    environment: dict[str, str]

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", populate_by_name=True,
    )

    @property
    def identity(self) -> str:
        """Hash only the fields that define evaluation compatibility."""
        return fingerprint_canonical_json({
            "schema": PDF_EVALUATION_RUN_SCHEMA,
            "version": PDF_EVALUATION_RUN_VERSION,
            "identity": self.model_dump(
                mode="json",
                exclude={"code_revision", "code_dirty", "environment"},
            ),
        })


def bm25_configuration(parameters: BM25Parameters) -> PdfBm25Configuration:
    """Describe the BM25 parameters a baseline actually used."""
    return PdfBm25Configuration(
        k1=parameters.k1,
        b=parameters.b,
        metadata_weight=parameters.metadata_weight,
        identifier_weight=parameters.identifier_weight,
    )


def semantic_configuration(
    config: SemanticEncoderConfig,
) -> PdfSemanticConfiguration:
    """Describe the encoder configuration a baseline actually used."""
    return PdfSemanticConfiguration(
        model_name=config.model_name,
        model_revision=config.model_revision,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )


def rrf_configuration(parameters: RRFParameters) -> PdfRrfConfiguration:
    """Describe the fusion parameters a baseline actually used."""
    return PdfRrfConfiguration(
        rank_constant=parameters.rank_constant,
        lexical_weight=parameters.lexical_weight,
        semantic_weight=parameters.semantic_weight,
    )


def build_pdf_evaluation_manifest(
    *,
    dataset: PdfGoldenDataset,
    dataset_sha256: str,
    records: Sequence[PdfIndexRecord],
    corpus: PdfRetrievalCorpus,
    retrieval: PdfRetrievalConfiguration,
    code_revision: str,
    code_dirty: bool,
    environment: Mapping[str, str],
    answerability: PdfAnswerabilityConfiguration | None = None,
) -> PdfEvaluationManifest:
    """Compute every fingerprint from the objects used by the run."""
    record_set_fingerprint = fingerprint_pdf_records(records)
    if corpus.record_set_fingerprint != record_set_fingerprint:
        raise ValueError("PDF retrieval corpus was built from other records")
    if dataset.book_path != corpus_book_path(records):
        raise ValueError("PDF golden dataset belongs to another book")
    return PdfEvaluationManifest(
        book_path=dataset.book_path,
        dataset_id=dataset.dataset_id,
        dataset_sha256=dataset_sha256,
        record_set_fingerprint=record_set_fingerprint,
        record_count=len(records),
        corpus_policy=corpus.policy,
        corpus_fingerprint=corpus.fingerprint,
        eligible_record_count=len(corpus.records),
        retrieval=retrieval,
        retrieval_fingerprint=retrieval.fingerprint,
        answerability=answerability,
        answerability_fingerprint=(
            answerability.fingerprint if answerability else None
        ),
        code_revision=code_revision,
        code_dirty=code_dirty,
        environment=dict(environment),
    )


def require_pdf_manifest_compatibility(
    manifest: PdfEvaluationManifest,
    *,
    dataset: PdfGoldenDataset,
    dataset_sha256: str,
    records: Sequence[PdfIndexRecord],
    corpus: PdfRetrievalCorpus,
) -> None:
    """Reject a run whose data, corpus, or configuration no longer match."""
    checks = (
        ("dataset_id", manifest.dataset_id, dataset.dataset_id),
        ("dataset_sha256", manifest.dataset_sha256, dataset_sha256),
        ("book_path", manifest.book_path, dataset.book_path),
        (
            "record_set_fingerprint",
            manifest.record_set_fingerprint,
            fingerprint_pdf_records(records),
        ),
        ("corpus_policy", manifest.corpus_policy, corpus.policy),
        (
            "corpus_policy_version",
            manifest.corpus_policy_version,
            PDF_RETRIEVAL_CORPUS_VERSION,
        ),
        ("corpus_fingerprint", manifest.corpus_fingerprint,
         corpus.fingerprint),
        (
            "retrieval_fingerprint",
            manifest.retrieval_fingerprint,
            manifest.retrieval.fingerprint,
        ),
        (
            "answerability_fingerprint",
            manifest.answerability_fingerprint,
            manifest.answerability.fingerprint
            if manifest.answerability else None,
        ),
    )
    for name, recorded, actual in checks:
        if recorded != actual:
            raise ValueError(f"PDF evaluation run {name} differs")


def require_same_pdf_evaluation_identity(
    first: PdfEvaluationManifest,
    second: PdfEvaluationManifest,
) -> None:
    """Allow a direct comparison only between runs of one identity."""
    if first.identity != second.identity:
        raise ValueError("PDF evaluation runs are not directly comparable")


def corpus_book_path(records: Sequence[PdfIndexRecord]) -> str:
    """Return the single book path shared by evaluation records."""
    book_paths = {record.book_path for record in records}
    if len(book_paths) != 1:
        raise ValueError("PDF evaluation records must come from one book")
    return next(iter(book_paths))


@dataclass(frozen=True, slots=True)
class PdfRunEnvironment:
    """Describe the code and packages a run was produced with."""

    code_revision: str
    code_dirty: bool
    packages: Mapping[str, str]


def current_pdf_run_environment() -> PdfRunEnvironment:
    """Read the checked-out revision and relevant package versions."""
    revision, dirty = _code_revision()
    packages = {"python": platform.python_version()}
    for package in PDF_ENVIRONMENT_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "absent"
    return PdfRunEnvironment(revision, dirty, packages)


def manifest_environment(manifest: PdfEvaluationManifest) -> PdfRunEnvironment:
    """Return the environment recorded in a manifest."""
    return PdfRunEnvironment(
        manifest.code_revision, manifest.code_dirty, manifest.environment,
    )


def pdf_reproducibility_warnings(
    recorded: PdfRunEnvironment,
    other: PdfRunEnvironment,
) -> tuple[str, ...]:
    """Warn about code or package drift that fingerprints cannot see.

    Matching fingerprints prove equal inputs and settings, not equal code;
    a changed revision or package version can still change results.
    """
    warnings = []
    if recorded.code_revision != other.code_revision:
        warnings.append(
            f"code revision differs: {recorded.code_revision} vs "
            f"{other.code_revision}"
        )
    for name, environment in (("recorded", recorded), ("other", other)):
        if environment.code_dirty:
            warnings.append(f"{name} run used uncommitted code changes")
    for package in sorted(set(recorded.packages) | set(other.packages)):
        first = recorded.packages.get(package, "unrecorded")
        second = other.packages.get(package, "unrecorded")
        if first != second:
            warnings.append(f"{package} version differs: {first} vs {second}")
    return tuple(warnings)


def _code_revision() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=no"),
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return revision, bool(status.strip())
