"""Plan original and human-clarified queries without performing retrieval."""

from enum import Enum
from typing import Literal

from pydantic import (
    BaseModel, ConfigDict, Field, field_validator, model_validator,
)

from src.evaluation.pdf.multibook_golden import (
    PdfExpectedBehavior,
    PdfMultibookGolden,
)
from src.ingestion.pdf.fingerprints import fingerprint_canonical_json


class PdfQueryMode(str, Enum):
    """Select original questions, clarified variants, or both."""

    ORIGINAL = "original"
    CLARIFIED = "clarified"
    ALL = "all"


class PdfQueryIdentity(BaseModel):
    """Keep a variant attached to its original case without encoding IDs."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    case_id: str = Field(min_length=1)
    interpretation_id: str | None = Field(default=None, min_length=1)

    @property
    def key(self) -> tuple[str, str | None]:
        """Return the structured identity, including original null."""
        return self.case_id, self.interpretation_id


class PdfMultibookQuery(BaseModel):
    """Keep only approved question text as retrieval input."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    identity: PdfQueryIdentity
    text: str = Field(min_length=1)
    track: Literal["original", "original_ambiguous", "clarified"]

    @model_validator(mode="after")
    def require_variant(self) -> "PdfMultibookQuery":
        """Reject blank text and a track inconsistent with its identity."""
        if not self.text.strip():
            raise ValueError("Query text must not be blank")
        if (self.identity.interpretation_id is not None) != (
            self.track == "clarified"
        ):
            raise ValueError("Query track differs from interpretation")
        return self


class PdfMultibookQueryPlan(BaseModel):
    """Bind an ordered query selection to Golden Set v2 content."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_name: Literal["fredl.pdf.multibook-query-plan"] = (
        "fredl.pdf.multibook-query-plan"
    )
    version: Literal[1] = 1
    golden_content_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: PdfQueryMode
    queries: tuple[PdfMultibookQuery, ...] = Field(min_length=1)

    @field_validator("version", mode="before")
    @classmethod
    def require_integer_version(cls, value: object) -> object:
        """JSON booleans and floats are not schema version integers."""
        if type(value) is not int:
            raise ValueError("Schema version must be an integer")
        return value

    @model_validator(mode="after")
    def require_selection(self) -> "PdfMultibookQueryPlan":
        """Reject duplicate identities and entries outside the mode."""
        keys = [query.identity.key for query in self.queries]
        if len(keys) != len(set(keys)):
            raise ValueError("Query plan has duplicate query identities")
        for query in self.queries:
            clarified = query.track == "clarified"
            if (self.mode is PdfQueryMode.ORIGINAL and clarified) or (
                self.mode is PdfQueryMode.CLARIFIED and not clarified
            ):
                raise ValueError("Query plan entry differs from query mode")
        return self

    @property
    def fingerprint(self) -> str:
        """Hash exact text, identities, mode and Golden Set order."""
        return fingerprint_canonical_json(self.model_dump(mode="json"))


def build_pdf_multibook_query_plan(
    golden: PdfMultibookGolden,
    mode: PdfQueryMode = PdfQueryMode.ALL,
) -> PdfMultibookQueryPlan:
    """Select cases in source order, originals before interpretations.

    Original-only plans do not require clarified text. Any selected
    clarified variant must have an approved, nonblank question; no text
    is inferred from its label, units, notes or expected outcome.
    """
    queries = []
    for case in golden.cases:
        if mode is not PdfQueryMode.CLARIFIED:
            ambiguous = (
                case.expected_behavior
                is PdfExpectedBehavior.CLARIFICATION_REQUIRED
            )
            queries.append(PdfMultibookQuery(
                identity=PdfQueryIdentity(case_id=case.case_id),
                text=case.question,
                track="original_ambiguous" if ambiguous else "original",
            ))
        if mode is not PdfQueryMode.ORIGINAL:
            for interpretation in case.interpretations:
                text = interpretation.clarified_question
                if text is None or not text.strip():
                    raise ValueError(
                        "Missing clarified question: "
                        f"{case.case_id}/{interpretation.interpretation_id}"
                    )
                queries.append(PdfMultibookQuery(
                    identity=PdfQueryIdentity(
                        case_id=case.case_id,
                        interpretation_id=interpretation.interpretation_id,
                    ),
                    text=text, track="clarified",
                ))
    return PdfMultibookQueryPlan(
        golden_content_fingerprint=golden.content_fingerprint,
        mode=mode, queries=tuple(queries),
    )


def require_pdf_query_plan_matches(
    plan: PdfMultibookQueryPlan,
    golden: PdfMultibookGolden,
) -> None:
    """Reject missing, reordered, extra or altered variants explicitly."""
    expected = build_pdf_multibook_query_plan(golden, plan.mode)
    if plan != expected:
        raise ValueError("Multi-book query plan differs from Golden Set")
