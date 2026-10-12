"""Strict models for private PDF retrieval golden datasets."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import model_validator


class PdfGoldenAnswerability(str, Enum):
    """Describe how completely the labelled book supports an answer."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"


class PdfGoldenEvidenceRole(str, Enum):
    """Describe how one source region contributes to the expected answer."""

    REQUIRED = "required"
    SUPPORTING = "supporting"
    IRRELEVANT = "irrelevant"


class PdfGoldenModality(str, Enum):
    """Identify the source representation needed for one evidence item."""

    TEXT = "text"
    STRUCTURED_TEXT = "structured_text"
    CAPTION = "caption"
    FIGURE = "figure"


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class PdfGoldenEvidence(_StrictModel):
    """Label physical PDF pages and their relevance to one question."""

    physical_pages: tuple[int, ...] = Field(min_length=1)
    role: PdfGoldenEvidenceRole
    modality: PdfGoldenModality
    review_status: Literal["pending_visual_review"] | None = None

    @field_validator("physical_pages")
    @classmethod
    def validate_physical_pages(
        cls,
        pages: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Require unique positive pages in reading order."""
        if any(page <= 0 for page in pages):
            raise ValueError("Physical PDF pages must be positive")
        if tuple(sorted(set(pages))) != pages:
            raise ValueError(
                "Physical PDF pages must be unique and increasing"
            )
        return pages


class PdfGoldenCase(_StrictModel):
    """Store one question, expected answer, and source evidence labels."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    why_now: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    answerability: PdfGoldenAnswerability
    evidence: tuple[PdfGoldenEvidence, ...]
    tags: tuple[str, ...] = Field(min_length=1)

    @field_validator("id", "question", "why_now", "expected_answer")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject whitespace-only labels and learning context."""
        if not value.strip():
            raise ValueError("PDF golden text fields must not be blank")
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: tuple[str, ...]) -> tuple[str, ...]:
        """Keep non-empty tags unique without changing their order."""
        if any(not tag.strip() for tag in tags):
            raise ValueError("PDF golden tags must not be blank")
        if len(tags) != len(set(tags)):
            raise ValueError("PDF golden tags must be unique")
        return tags

    @model_validator(mode="after")
    def validate_evidence_relationship(self) -> "PdfGoldenCase":
        """Keep answerability aligned with explicit source evidence."""
        if (
            self.answerability is PdfGoldenAnswerability.NOT_FOUND
            and self.evidence
        ):
            raise ValueError("Not-found PDF cases must not contain evidence")
        if (
            self.answerability is not PdfGoldenAnswerability.NOT_FOUND
            and not self.evidence
        ):
            raise ValueError("Answerable PDF cases must contain evidence")
        return self


class PdfGoldenDataset(_StrictModel):
    """Define one versioned private PDF retrieval evaluation set."""

    schema_version: Literal[1]
    dataset_id: str = Field(min_length=1)
    book_path: str = Field(min_length=1)
    page_numbering: Literal["physical_pdf_page"]
    language: str = Field(min_length=1)
    cases: tuple[PdfGoldenCase, ...] = Field(min_length=1)

    @field_validator("dataset_id", "book_path", "language")
    @classmethod
    def reject_blank_metadata(cls, value: str) -> str:
        """Reject ambiguous dataset identity and source metadata."""
        if not value.strip():
            raise ValueError("PDF golden dataset metadata must not be blank")
        return value

    @field_validator("cases")
    @classmethod
    def require_unique_case_ids(
        cls,
        cases: tuple[PdfGoldenCase, ...],
    ) -> tuple[PdfGoldenCase, ...]:
        """Prevent ambiguous result alignment during evaluation."""
        case_ids = [case.id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("PDF golden case IDs must be unique")
        return cases
