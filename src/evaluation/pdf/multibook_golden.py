"""Strict schema for a private, human-reviewed multi-book golden set.

The single-book golden format in models.py stays unchanged. This schema
adds what several books need: a pinned library, evidence units whose
locators point into one book each, and sufficient evidence sets with
OR-of-AND semantics. Original-source support and extracted-text support
are separate fields, so a lost formula or a clipped listing is never
recorded as a retrieval or answerability failure.
"""

from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    ValidationError,
)

from src.evaluation.pdf.library import PdfLibraryManifest
from src.evaluation.pdf.models import PdfGoldenModality
from src.ingestion.pdf.fingerprints import fingerprint_canonical_json


PDF_MULTIBOOK_GOLDEN_SCHEMA = "fredl.pdf.multibook-golden"
PDF_MULTIBOOK_GOLDEN_VERSION = 1
_SHA256 = r"^[0-9a-f]{64}$"


class PdfMultibookCategory(str, Enum):
    """Name the evidence pattern a question was chosen to test."""

    SINGLE_BOOK = "single_book"
    INDEPENDENT_MULTI_BOOK = "independent_multi_book"
    COMPLEMENTARY_MULTI_BOOK = "complementary_multi_book"
    MULTIPLE_SECTIONS_SINGLE_BOOK = "multiple_sections_single_book"
    AMBIGUOUS_TERMINOLOGY = "ambiguous_terminology"
    PARTIALLY_ANSWERABLE = "partially_answerable"
    UNSUPPORTED_LIBRARY = "unsupported_library"
    HARD_NEGATIVE = "hard_negative"
    RELIABILITY_LIMITED = "reliability_limited"


class PdfExpectedOutcome(str, Enum):
    """Describe the correct system behaviour for one question.

    unsupported: the books do not substantiate the requested answer.
    extraction_limited: the original PDF holds the answer, but the
    extracted text does not preserve it. source_incomplete: the original
    source itself lacks part of what is requested.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    EXTRACTION_LIMITED = "extraction_limited"
    SOURCE_INCOMPLETE = "source_incomplete"


class PdfSourceSupport(str, Enum):
    """Describe what the original PDF pages support."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    CONDITIONAL = "conditional"
    NOT_FOUND = "not_found"
    INCOMPLETE = "incomplete"


class PdfTextSupport(str, Enum):
    """Describe what the current extracted text supports."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    CONDITIONAL = "conditional"
    NOT_FOUND = "not_found"
    EXTRACTION_LIMITED = "extraction_limited"
    INCOMPLETE = "incomplete"


class PdfLimitationReason(str, Enum):
    """Explain why a correct answer cannot come from the extracted text."""

    VISUAL_EVIDENCE_UNAVAILABLE = "visual_evidence_unavailable"
    MATHEMATICAL_STRUCTURE_LOST = "mathematical_structure_lost"
    SOURCE_INCOMPLETE = "source_incomplete"


_EXTRACTION_REASONS = frozenset({
    PdfLimitationReason.VISUAL_EVIDENCE_UNAVAILABLE,
    PdfLimitationReason.MATHEMATICAL_STRUCTURE_LOST,
})

# Each outcome fixes both support fields; source_incomplete is never
# merged into unsupported or extraction_limited.
_OUTCOME_SUPPORT: dict[
    PdfExpectedOutcome, tuple[PdfSourceSupport, PdfTextSupport]
] = {
    PdfExpectedOutcome.COMPLETE: (
        PdfSourceSupport.COMPLETE, PdfTextSupport.COMPLETE,
    ),
    PdfExpectedOutcome.PARTIAL: (
        PdfSourceSupport.PARTIAL, PdfTextSupport.PARTIAL,
    ),
    PdfExpectedOutcome.AMBIGUOUS: (
        PdfSourceSupport.CONDITIONAL, PdfTextSupport.CONDITIONAL,
    ),
    PdfExpectedOutcome.UNSUPPORTED: (
        PdfSourceSupport.NOT_FOUND, PdfTextSupport.NOT_FOUND,
    ),
    PdfExpectedOutcome.EXTRACTION_LIMITED: (
        PdfSourceSupport.COMPLETE, PdfTextSupport.EXTRACTION_LIMITED,
    ),
    PdfExpectedOutcome.SOURCE_INCOMPLETE: (
        PdfSourceSupport.INCOMPLETE, PdfTextSupport.INCOMPLETE,
    ),
}

_CATEGORY_OUTCOMES: dict[PdfMultibookCategory, frozenset[PdfExpectedOutcome]]
_CATEGORY_OUTCOMES = {
    PdfMultibookCategory.SINGLE_BOOK: frozenset({PdfExpectedOutcome.COMPLETE}),
    PdfMultibookCategory.INDEPENDENT_MULTI_BOOK: frozenset({
        PdfExpectedOutcome.COMPLETE,
    }),
    PdfMultibookCategory.COMPLEMENTARY_MULTI_BOOK: frozenset({
        PdfExpectedOutcome.COMPLETE,
    }),
    PdfMultibookCategory.MULTIPLE_SECTIONS_SINGLE_BOOK: frozenset({
        PdfExpectedOutcome.COMPLETE,
    }),
    PdfMultibookCategory.AMBIGUOUS_TERMINOLOGY: frozenset({
        PdfExpectedOutcome.AMBIGUOUS,
    }),
    PdfMultibookCategory.PARTIALLY_ANSWERABLE: frozenset({
        PdfExpectedOutcome.PARTIAL,
    }),
    PdfMultibookCategory.UNSUPPORTED_LIBRARY: frozenset({
        PdfExpectedOutcome.UNSUPPORTED,
    }),
    PdfMultibookCategory.HARD_NEGATIVE: frozenset({
        PdfExpectedOutcome.UNSUPPORTED,
    }),
    PdfMultibookCategory.RELIABILITY_LIMITED: frozenset({
        PdfExpectedOutcome.COMPLETE,
        PdfExpectedOutcome.EXTRACTION_LIMITED,
        PdfExpectedOutcome.SOURCE_INCOMPLETE,
    }),
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PdfEvidenceLocator(_StrictModel):
    """Point at exact text on one physical page of one book.

    Offsets are half-open positions in the page's reading text, the same
    coordinates that record page spans use.
    """

    book_path: str = Field(min_length=1)
    physical_page: int = Field(gt=0)
    exact_anchor: str = Field(min_length=1)
    page_text_start: int = Field(ge=0)
    page_text_end: int = Field(gt=0)
    page_text_sha256: str = Field(pattern=_SHA256)
    verified_section_path: tuple[str, ...] = ()
    reliability_findings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_anchor_span(self) -> "PdfEvidenceLocator":
        """Keep the span exactly as long as its anchor text."""
        if self.page_text_end - self.page_text_start != len(
            self.exact_anchor
        ):
            raise ValueError("Evidence locator span must match its anchor")
        return self


class PdfEvidenceUnit(_StrictModel):
    """Name minimal content that must be found, and where it is.

    Every locator of one unit belongs to one book. Supporting locators
    corroborate the unit without being required for it.
    """

    unit_id: str = Field(min_length=1)
    required_content: str = Field(min_length=1)
    modality: PdfGoldenModality
    locators: tuple[PdfEvidenceLocator, ...] = Field(min_length=1)
    supporting_locators: tuple[PdfEvidenceLocator, ...] = ()
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_one_book(self) -> "PdfEvidenceUnit":
        """Reject units that mix books or repeat a locator."""
        locators = (*self.locators, *self.supporting_locators)
        if len({locator.book_path for locator in locators}) != 1:
            raise ValueError(
                f"Evidence unit {self.unit_id} must cite exactly one book"
            )
        keys = [
            (locator.physical_page, locator.page_text_start,
             locator.page_text_end)
            for locator in locators
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Evidence unit {self.unit_id} repeats a locator")
        return self

    @property
    def book_path(self) -> str:
        """Return the one book this unit cites."""
        return self.locators[0].book_path


class PdfMultibookCase(_StrictModel):
    """Store one approved question and its source-level expectations.

    sufficient_sets is OR across sets and AND within a set; each set is
    minimal. conditional_units serve only one reading of an ambiguous
    question; related_units are near but not sufficient evidence.
    reference_notes guide human answer review and never enter metrics.
    """

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: PdfMultibookCategory
    expected_outcome: PdfExpectedOutcome
    source_support: PdfSourceSupport
    text_context_support: PdfTextSupport
    limitation_reason: PdfLimitationReason | None = None
    sufficient_sets: tuple[tuple[str, ...], ...] = ()
    conditional_units: tuple[str, ...] = ()
    related_units: tuple[str, ...] = ()
    reference_notes: str = ""

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        """Reject a whitespace-only question."""
        if not value.strip():
            raise ValueError("Multi-book question must not be blank")
        return value

    @field_validator("sufficient_sets")
    @classmethod
    def require_minimal_sets(
        cls,
        sets: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[str, ...], ...]:
        """Reject empty, repeated, or redundant evidence sets."""
        members = [frozenset(item) for item in sets]
        for item, member in zip(sets, members):
            if not item or len(item) != len(member):
                raise ValueError(
                    "Sufficient sets must be non-empty without repeats"
                )
        for index, member in enumerate(members):
            for other_index, other in enumerate(members):
                if index != other_index and other <= member:
                    raise ValueError(
                        "Sufficient sets must be distinct and minimal"
                    )
        return sets

    @model_validator(mode="after")
    def validate_outcome(self) -> "PdfMultibookCase":
        """Keep outcome, support, limitation, and evidence consistent."""
        outcome = self.expected_outcome
        if self.expected_outcome not in _CATEGORY_OUTCOMES[self.category]:
            raise ValueError(
                f"{self.case_id}: {self.category.value} cannot expect "
                f"{outcome.value}"
            )
        if (self.source_support, self.text_context_support) != (
            _OUTCOME_SUPPORT[outcome]
        ):
            raise ValueError(
                f"{self.case_id}: support fields contradict {outcome.value}"
            )
        _require_limitation(self.case_id, outcome, self.limitation_reason)
        _require_evidence_shape(self)
        return self

    @property
    def set_units(self) -> frozenset[str]:
        """Return every unit named by a sufficient set."""
        return frozenset(
            unit for item in self.sufficient_sets for unit in item
        )

    @property
    def unit_ids(self) -> tuple[str, ...]:
        """Return every unit this case references, without repeats."""
        return tuple(dict.fromkeys((
            *(unit for item in self.sufficient_sets for unit in item),
            *self.conditional_units,
            *self.related_units,
        )))


class PdfGoldenProvenance(_StrictModel):
    """Bind a dataset to the exact review files it was imported from."""

    package_id: str = Field(min_length=1)
    candidates_sha256: str = Field(pattern=_SHA256)
    review_decisions_sha256: str = Field(pattern=_SHA256)
    source_manifest_sha256: str = Field(pattern=_SHA256)
    decisions: tuple[str, ...] = ()


class PdfMultibookGolden(_StrictModel):
    """Define one versioned multi-book evaluation set.

    A draft can be revised. A frozen dataset is never overwritten; any
    change requires a new dataset_id.
    """

    schema_name: Literal["fredl.pdf.multibook-golden"] = Field(
        default="fredl.pdf.multibook-golden", alias="schema",
    )
    version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    status: Literal["draft", "frozen"]
    page_numbering: Literal["physical_pdf_page"] = "physical_pdf_page"
    language: str = Field(min_length=1)
    library: PdfLibraryManifest
    provenance: PdfGoldenProvenance | None = None
    evidence_units: tuple[PdfEvidenceUnit, ...]
    cases: tuple[PdfMultibookCase, ...] = Field(min_length=1)

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_references(self) -> "PdfMultibookGolden":
        """Check identities, unit references, books, and categories."""
        _require_unique("case", (case.case_id for case in self.cases))
        _require_unique(
            "evidence unit", (unit.unit_id for unit in self.evidence_units),
        )
        units = self.units
        books = set(self.library.book_paths)
        for unit in self.evidence_units:
            if unit.book_path not in books:
                raise ValueError(
                    f"Evidence unit {unit.unit_id} cites a book outside "
                    "the library"
                )
        referenced: set[str] = set()
        for case in self.cases:
            missing = [unit for unit in case.unit_ids if unit not in units]
            if missing:
                raise ValueError(
                    f"{case.case_id} references unknown evidence units: "
                    f"{', '.join(missing)}"
                )
            referenced.update(case.unit_ids)
            _require_category_books(case, units)
        unused = sorted(set(units) - referenced)
        if unused:
            raise ValueError(
                f"Unreferenced evidence units: {', '.join(unused)}",
            )
        return self

    @property
    def units(self) -> dict[str, PdfEvidenceUnit]:
        """Return evidence units by identifier."""
        return {unit.unit_id: unit for unit in self.evidence_units}

    @property
    def content_fingerprint(self) -> str:
        """Hash the dataset content, ignoring only its draft/frozen status.

        A draft and the frozen dataset made from it share this value, which
        proves that freezing changed nothing else.
        """
        return fingerprint_canonical_json({
            "schema": PDF_MULTIBOOK_GOLDEN_SCHEMA,
            "version": PDF_MULTIBOOK_GOLDEN_VERSION,
            "content": self.model_dump(
                mode="json", by_alias=True, exclude={"status"},
            ),
        })


def load_pdf_multibook_golden(path: Path) -> PdfMultibookGolden:
    """Read and validate one multi-book golden file."""
    try:
        return PdfMultibookGolden.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
    except OSError as error:
        raise ValueError("Multi-book golden set could not be read") from error
    except ValidationError as error:
        raise ValueError(
            f"Multi-book golden set is invalid: {error.errors()[0]['msg']}",
        ) from error


def _require_limitation(
    case_id: str,
    outcome: PdfExpectedOutcome,
    reason: PdfLimitationReason | None,
) -> None:
    if outcome is PdfExpectedOutcome.EXTRACTION_LIMITED:
        if reason not in _EXTRACTION_REASONS:
            raise ValueError(
                f"{case_id}: extraction_limited needs an extraction reason"
            )
    elif outcome is PdfExpectedOutcome.SOURCE_INCOMPLETE:
        if reason is not PdfLimitationReason.SOURCE_INCOMPLETE:
            raise ValueError(
                f"{case_id}: source_incomplete needs reason source_incomplete"
            )
    elif reason is not None:
        raise ValueError(f"{case_id}: {outcome.value} takes no limitation")


def _require_evidence_shape(case: PdfMultibookCase) -> None:
    outcome = case.expected_outcome
    has_sets = bool(case.sufficient_sets)
    if outcome in {
        PdfExpectedOutcome.COMPLETE,
        PdfExpectedOutcome.PARTIAL,
        PdfExpectedOutcome.EXTRACTION_LIMITED,
    } and not has_sets:
        raise ValueError(
            f"{case.case_id}: {outcome.value} needs evidence sets",
        )
    if outcome in {
        PdfExpectedOutcome.AMBIGUOUS,
        PdfExpectedOutcome.UNSUPPORTED,
        PdfExpectedOutcome.SOURCE_INCOMPLETE,
    } and has_sets:
        raise ValueError(
            f"{case.case_id}: {outcome.value} has no sufficient evidence set"
        )
    if outcome is PdfExpectedOutcome.AMBIGUOUS:
        if len(set(case.conditional_units)) < 2:
            raise ValueError(
                f"{case.case_id}: ambiguous needs two conditional units"
            )
    elif case.conditional_units:
        raise ValueError(
            f"{case.case_id}: only ambiguous cases have conditional units"
        )
    groups = (
        case.set_units, frozenset(case.conditional_units),
        frozenset(case.related_units),
    )
    if len(case.related_units) != len(groups[2]) or len(
        case.conditional_units
    ) != len(groups[1]):
        raise ValueError(f"{case.case_id}: evidence units are repeated")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError(
            f"{case.case_id}: an evidence unit has more than one role"
        )


def _require_category_books(
    case: PdfMultibookCase,
    units: dict[str, PdfEvidenceUnit],
) -> None:
    set_books = [
        {units[unit].book_path for unit in item}
        for item in case.sufficient_sets
    ]
    all_books = set().union(*set_books) if set_books else set()
    category = case.category
    if category in {
        PdfMultibookCategory.SINGLE_BOOK,
        PdfMultibookCategory.MULTIPLE_SECTIONS_SINGLE_BOOK,
    } and len(all_books) != 1:
        raise ValueError(f"{case.case_id}: evidence must come from one book")
    if (
        category is PdfMultibookCategory.MULTIPLE_SECTIONS_SINGLE_BOOK
        and not any(len(item) >= 2 for item in case.sufficient_sets)
    ):
        raise ValueError(f"{case.case_id}: needs a set of two or more units")
    if category is PdfMultibookCategory.INDEPENDENT_MULTI_BOOK and (
        any(len(books) != 1 for books in set_books) or len(all_books) < 2
    ):
        raise ValueError(
            f"{case.case_id}: independent evidence needs one-book sets "
            "from two or more books"
        )
    if category is PdfMultibookCategory.COMPLEMENTARY_MULTI_BOOK and any(
        len(books) < 2 for books in set_books
    ):
        raise ValueError(
            f"{case.case_id}: every complementary set must span two books"
        )


def _require_unique(kind: str, values: Iterable[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"Duplicate {kind} identifier: {value}")
        seen.add(value)
