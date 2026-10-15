"""Retrieval-pool contracts, separate from historical pilot labels."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import model_validator

from src.evaluation.pdf.manifest import PdfRetrieverKind
from src.evaluation.pdf.multibook_queries import PdfQueryIdentity
from src.evaluation.pdf.multibook_queries import PdfMultibookQueryPlan
from src.evaluation.pdf.multibook_runs import PdfMultibookRetrievalRun
from src.ingestion.pdf.fingerprints import fingerprint_canonical_json


SHA256 = r"^[0-9a-f]{64}$"


class PoolModel(BaseModel):
    """Reject unsupported fields, coercion and non-integer versions."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    @field_validator("version", mode="before", check_fields=False)
    @classmethod
    def integer_version(cls, value: object) -> object:
        """Do not accept booleans or floats as schema versions."""
        if type(value) is not int:
            raise ValueError("Version must be an integer")
        return value


class PdfPoolRunReference(PoolModel):
    """Pin run bytes; relative paths resolve beside the input manifest."""

    retriever: PdfRetrieverKind
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256)


class PdfPoolingInputs(PoolModel):
    """Require exactly one artifact of each global retriever."""

    schema_name: Literal["fredl.pdf.multibook-pooling-inputs"] = (
        "fredl.pdf.multibook-pooling-inputs"
    )
    version: Literal[1] = 1
    runs: tuple[PdfPoolRunReference, ...]

    @model_validator(mode="after")
    def all_retrievers(self) -> "PdfPoolingInputs":
        """Reject duplicates and omissions without repairing the manifest."""
        kinds = [ref.retriever for ref in self.runs]
        if len(kinds) != 3 or set(kinds) != set(PdfRetrieverKind):
            raise ValueError("Need exactly BM25, MiniLM and Hybrid")
        return self


class PdfPoolingRun(PoolModel):
    """Pair a structurally checked run with the hash of its input bytes."""

    artifact_sha256: str = Field(pattern=SHA256)
    run: PdfMultibookRetrievalRun


class PdfPoolRunIdentity(PoolModel):
    """Record exact bytes, configuration identity and results separately."""

    retriever: PdfRetrieverKind
    artifact_sha256: str = Field(pattern=SHA256)
    manifest_identity: str = Field(pattern=SHA256)
    results_fingerprint: str = Field(pattern=SHA256)


class PdfCandidateOrigin(PoolModel):
    """Explain inclusion, never relevance; context names its ranked parents."""

    kind: Literal["ranked", "context", "anchor", "supporting_anchor"]
    retriever: PdfRetrieverKind | None = None
    rank: int | None = Field(default=None, gt=0)
    score: float | None = Field(default=None, allow_inf_nan=False)
    cutoff: Literal[5, 10] | None = None
    parent_record_ids: tuple[str, ...] = ()
    unit_id: str | None = Field(default=None, min_length=1)
    locator_index: int | None = Field(default=None, ge=0)

    @field_validator("cutoff", mode="before")
    @classmethod
    def integer_cutoff(cls, value: object) -> object:
        """Reject float and bool cutoff aliases."""
        if value is not None and type(value) is not int:
            raise ValueError("Cutoff must be an integer")
        return value

    @model_validator(mode="after")
    def exact_fields(self) -> "PdfCandidateOrigin":
        """Require only the fields meaningful for this origin kind."""
        retrieval = self.kind in {"ranked", "context"}
        if (self.retriever is not None) != retrieval or (
            (self.rank is not None) != (self.kind == "ranked")
        ) or ((self.score is not None) != (self.kind == "ranked")) or (
            (self.cutoff is not None) != (self.kind == "context")
        ) or (bool(self.parent_record_ids) != (self.kind == "context")):
            raise ValueError("Invalid retrieval provenance")
        if len(set(self.parent_record_ids)) != len(self.parent_record_ids):
            raise ValueError("Duplicate context parents")
        if (self.unit_id is not None) != (not retrieval) or (
            (self.locator_index is not None) != (not retrieval)
        ):
            raise ValueError("Invalid anchor provenance")
        return self


class PdfPoolCandidate(PoolModel):
    """Deduplicate within one query while retaining all inclusion reasons."""

    query: PdfQueryIdentity
    book_path: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    origins: tuple[PdfCandidateOrigin, ...] = Field(min_length=1)

    @property
    def key(self) -> tuple[str, str | None, str]:
        """Return query plus canonical record identity."""
        return (*self.query.key, self.record_id)

    @model_validator(mode="after")
    def unique_origins(self) -> "PdfPoolCandidate":
        """Reject duplicated provenance rather than silently deduplicate."""
        values = [origin.model_dump_json() for origin in self.origins]
        if len(values) != len(set(values)):
            raise ValueError("Duplicate candidate origins")
        return self


class PdfRetrievalJudgmentTask(PoolModel):
    """Ask a human about one record for one unit of one query variant."""

    query: PdfQueryIdentity
    unit_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)

    @property
    def key(self) -> tuple[str, str | None, str, str]:
        """Keep original and clarified judgments separate."""
        return (*self.query.key, self.unit_id, self.record_id)


class PdfRetrievalJudgmentPool(PoolModel):
    """Bind the candidate inventory and blank tasks to reproducible inputs."""

    schema_name: Literal["fredl.pdf.multibook-retrieval-pool"] = (
        "fredl.pdf.multibook-retrieval-pool"
    )
    version: Literal[1] = 1
    golden_sha256: str = Field(pattern=SHA256)
    golden_content_fingerprint: str = Field(pattern=SHA256)
    library_fingerprint: str = Field(pattern=SHA256)
    corpus_fingerprint: str = Field(pattern=SHA256)
    comparison_identity: str = Field(pattern=SHA256)
    query_plan: PdfMultibookQueryPlan
    depth: Literal[5, 10] = 10
    runs: tuple[PdfPoolRunIdentity, ...]
    candidates: tuple[PdfPoolCandidate, ...]
    tasks: tuple[PdfRetrievalJudgmentTask, ...]

    @field_validator("depth", mode="before")
    @classmethod
    def integer_depth(cls, value: object) -> object:
        """Reject bool/float cutoff aliases."""
        if type(value) is not int:
            raise ValueError("Depth must be an integer")
        return value

    @model_validator(mode="after")
    def unique_bindings(self) -> "PdfRetrievalJudgmentPool":
        """Check internal uniqueness and task-to-candidate references."""
        if tuple(run.retriever for run in self.runs) != tuple(
            PdfRetrieverKind
        ):
            raise ValueError("Run identities must be BM25, MiniLM, Hybrid")
        keys = [candidate.key for candidate in self.candidates]
        tasks = [task.key for task in self.tasks]
        if len(keys) != len(set(keys)) or len(tasks) != len(set(tasks)):
            raise ValueError("Duplicate candidates or tasks")
        known = {query.identity.key for query in self.query_plan.queries}
        if any(candidate.query.key not in known
               for candidate in self.candidates):
            raise ValueError("Candidate names unknown query")
        known_candidates = set(keys)
        if any((*task.query.key, task.record_id) not in known_candidates
               for task in self.tasks):
            raise ValueError("Task names unknown candidate")
        if self.query_plan.golden_content_fingerprint != (
            self.golden_content_fingerprint
        ):
            raise ValueError("Query plan names different Golden Set")
        return self

    @property
    def fingerprint(self) -> str:
        """Hash the ordered semantic pool, independent of JSON formatting."""
        return fingerprint_canonical_json(self.model_dump(mode="json"))
