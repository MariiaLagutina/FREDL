"""Human relevance judgments for the multi-book golden set.

Judgments are made per evidence unit, not per question: one record can be
required for one unit and irrelevant to another. supplies_evidence keeps
a record that only points at evidence, such as a figure caption or a
flattened formula, apart from a record whose text actually provides it.
Only a human reviewer creates judgments; this module validates them.
"""

from collections import Counter
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
    ValidationError,
)

from src.evaluation.pdf.judgments import PdfRecordRelevance
from src.evaluation.pdf.multibook.library import PdfLibraryCorpus
from src.evaluation.pdf.multibook.curation.golden import (
    PdfMultibookCase,
    PdfMultibookGolden,
)
from src.ingestion.pdf.fingerprints import fingerprint_canonical_json


PDF_MULTIBOOK_POOL_SCHEMA = "fredl.pdf.multibook-judgment-pool"
PDF_MULTIBOOK_JUDGMENTS_SCHEMA = "fredl.pdf.multibook-judgments"
_SHA256 = r"^[0-9a-f]{64}$"


class PdfPoolSource(str, Enum):
    """Explain why a record was offered to the reviewer.

    Sources are kept private to the pool file and never shown on the
    worksheet, so the reviewer cannot tell which retriever found a record.
    """

    ANCHOR_OVERLAP = "anchor_overlap"
    LOCATOR_PAGE = "locator_page"
    BM25 = "bm25"
    MINILM = "minilm"
    HYBRID = "hybrid"


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PdfPoolEntry(_StrictModel):
    """Offer one record for judgment against one evidence unit."""

    case_id: str = Field(min_length=1)
    unit_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    sources: tuple[PdfPoolSource, ...] = Field(min_length=1)

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the identity a judgment must match."""
        return (self.case_id, self.unit_id, self.record_id)


class PdfJudgmentPool(_StrictModel):
    """Record which records a reviewer must judge, and why."""

    schema_name: Literal["fredl.pdf.multibook-judgment-pool"] = Field(
        default="fredl.pdf.multibook-judgment-pool", alias="schema",
    )
    version: Literal[1] = 1
    golden_content_fingerprint: str = Field(pattern=_SHA256)
    library_fingerprint: str = Field(pattern=_SHA256)
    retrieval_depth: int | None = Field(default=None, gt=0)
    entries: tuple[PdfPoolEntry, ...]

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True,
    )

    @model_validator(mode="after")
    def require_unique_entries(self) -> "PdfJudgmentPool":
        """Offer each record once per unit."""
        keys = [entry.key for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("Judgment pool entries must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        """Hash the pool so judgments can name exactly what was offered."""
        return fingerprint_canonical_json({
            "schema": PDF_MULTIBOOK_POOL_SCHEMA,
            "version": 1,
            "pool": self.model_dump(mode="json", by_alias=True),
        })


class PdfMultibookJudgment(_StrictModel):
    """Store one reviewer's label for a record and an evidence unit.

    relevance says how the record relates to the unit's content.
    supplies_evidence is required for a required or supporting record:
    true when its extracted text provides the content, false when it only
    points at content that the text does not carry.
    """

    case_id: str = Field(min_length=1)
    unit_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    relevance: PdfRecordRelevance
    supplies_evidence: bool | None = None
    reviewer: str = Field(min_length=1)
    note: str = ""

    @model_validator(mode="after")
    def require_supply_flag(self) -> "PdfMultibookJudgment":
        """Ask whether a relevant record supplies or only points."""
        relevant = self.relevance is not PdfRecordRelevance.IRRELEVANT
        if relevant and self.supplies_evidence is None:
            raise ValueError(
                f"{self.key}: relevant records need supplies_evidence"
            )
        if not relevant and self.supplies_evidence is not None:
            raise ValueError(
                f"{self.key}: irrelevant records take no supplies_evidence"
            )
        return self

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the pool identity this judgment answers."""
        return (self.case_id, self.unit_id, self.record_id)


class PdfMultibookJudgmentSet(_StrictModel):
    """Bind human judgments to one golden set, library, and pool."""

    schema_name: Literal["fredl.pdf.multibook-judgments"] = Field(
        default="fredl.pdf.multibook-judgments", alias="schema",
    )
    version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    golden_content_fingerprint: str = Field(pattern=_SHA256)
    library_fingerprint: str = Field(pattern=_SHA256)
    pool_fingerprint: str = Field(pattern=_SHA256)
    status: Literal["pilot", "draft", "frozen"]
    judgments: tuple[PdfMultibookJudgment, ...]

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True,
    )

    @model_validator(mode="after")
    def require_unique_judgments(self) -> "PdfMultibookJudgmentSet":
        """Allow one label per record and unit."""
        keys = [judgment.key for judgment in self.judgments]
        if len(keys) != len(set(keys)):
            raise ValueError("Multi-book judgments must be unique")
        return self


def pool_units(case: PdfMultibookCase) -> tuple[str, ...]:
    """Return the units a reviewer judges for one case.

    Units of sufficient sets and conditional units are judged. Related
    units of negative cases are context, not targets, so they are not.
    """
    return tuple(dict.fromkeys((
        *(unit for item in case.sufficient_sets for unit in item),
        *case.conditional_units,
    )))


def validate_multibook_judgments(
    judgments: PdfMultibookJudgmentSet,
    golden: PdfMultibookGolden,
    library: PdfLibraryCorpus,
    pool: PdfJudgmentPool,
) -> tuple[str, ...]:
    """Check bindings and completeness; return non-blocking warnings.

    Every pooled entry needs exactly one judgment, and no judgment may
    name an entry outside the pool. Unjudged records stay unjudged; they
    are never treated as irrelevant.
    """
    if judgments.golden_content_fingerprint != golden.content_fingerprint:
        raise ValueError("Judgments belong to a different golden set")
    if judgments.dataset_id != golden.dataset_id:
        raise ValueError("Judgments name a different dataset")
    if judgments.library_fingerprint != library.manifest.fingerprint:
        raise ValueError("Judgments belong to a different library")
    if judgments.pool_fingerprint != pool.fingerprint:
        raise ValueError("Judgments answer a different pool")
    require_pool_matches(pool, golden, library)
    offered = {entry.key for entry in pool.entries}
    judged = {judgment.key for judgment in judgments.judgments}
    extra = sorted(judged - offered)
    if extra:
        raise ValueError(f"Judgments outside the pool: {extra[:3]}")
    missing = sorted(offered - judged)
    if missing and judgments.status == "frozen":
        raise ValueError(f"Frozen judgments miss pool entries: {missing[:3]}")
    warnings = []
    if missing:
        warnings.append(f"{len(missing)} pooled entries are unjudged")
    books = {record.record_id: record.book_path
             for record in library.source_records}
    units = golden.units
    other_book = Counter(
        judgment.case_id
        for judgment in judgments.judgments
        if judgment.relevance is not PdfRecordRelevance.IRRELEVANT
        and books[judgment.record_id] != units[judgment.unit_id].book_path
    )
    for case_id, count in sorted(other_book.items()):
        warnings.append(
            f"{case_id}: {count} relevant record(s) come from a book other "
            "than their unit's; consider an alternative evidence unit"
        )
    return tuple(warnings)


def require_pool_matches(
    pool: PdfJudgmentPool,
    golden: PdfMultibookGolden,
    library: PdfLibraryCorpus,
) -> None:
    """Reject pools for another dataset, library, case, unit, or record."""
    if pool.golden_content_fingerprint != golden.content_fingerprint:
        raise ValueError("Judgment pool belongs to a different golden set")
    if pool.library_fingerprint != library.manifest.fingerprint:
        raise ValueError("Judgment pool belongs to a different library")
    cases = {case.case_id: case for case in golden.cases}
    records = {record.record_id for record in library.source_records}
    for entry in pool.entries:
        case = cases.get(entry.case_id)
        if case is None or entry.unit_id not in pool_units(case):
            raise ValueError(
                f"Pool entry names no judged unit: {entry.case_id}/"
                f"{entry.unit_id}"
            )
        if entry.record_id not in records:
            raise ValueError(
                f"Pool entry names unknown record {entry.record_id}",
            )


def load_pdf_judgment_pool(path: Path) -> PdfJudgmentPool:
    """Read one judgment pool."""
    try:
        return PdfJudgmentPool.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
    except (OSError, ValidationError) as error:
        raise ValueError("Judgment pool could not be read") from error


def load_pdf_multibook_judgments(path: Path) -> PdfMultibookJudgmentSet:
    """Read one multi-book judgment set."""
    try:
        return PdfMultibookJudgmentSet.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
    except OSError as error:
        raise ValueError("Multi-book judgments could not be read") from error
    except ValidationError as error:
        raise ValueError(
            f"Multi-book judgments are invalid: {error.errors()[0]['msg']}",
        ) from error


def judgment_counts(
    judgments: Sequence[PdfMultibookJudgment],
) -> dict[str, int]:
    """Count labels for a completeness summary."""
    counts = Counter(judgment.relevance.value for judgment in judgments)
    counts["points_only"] = sum(
        judgment.supplies_evidence is False for judgment in judgments
    )
    return dict(sorted(counts.items()))
