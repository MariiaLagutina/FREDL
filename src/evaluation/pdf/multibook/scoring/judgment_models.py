"""Interpretation-aware human labels and explicit read-only input pins."""

from typing import Literal

from pydantic import Field, model_validator

from src.evaluation.pdf.judgments import PdfRecordRelevance
from src.evaluation.pdf.multibook_pool_models import PoolModel, SHA256
from src.evaluation.pdf.multibook_pool_models import PdfRetrievalJudgmentTask
from src.ingestion.pdf.fingerprints import fingerprint_canonical_json


class PdfHashedInput(PoolModel):
    """Pin exact bytes; relative paths resolve beside the input manifest."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256)


class PdfWorksheetInput(PdfHashedInput):
    """Attribute a human worksheet without changing its contents."""

    reviewer: str = Field(min_length=1, pattern=r"\S")


class PdfJudgmentInputs(PoolModel):
    """Bind worksheet files to exactly one Slice 3 pool."""

    schema_name: Literal['fredl.pdf.multibook-judgment-inputs'] = (
        'fredl.pdf.multibook-judgment-inputs'
    )
    version: Literal[1] = 1
    pool: PdfHashedInput
    pool_fingerprint: str = Field(pattern=SHA256)
    worksheets: tuple[PdfWorksheetInput, ...] = Field(min_length=1)


class PdfWorksheetSubmission(PoolModel):
    """Keep exact bytes through validation, including newlines and notes."""

    content: bytes
    sha256: str = Field(pattern=SHA256)
    reviewer: str = Field(min_length=1, pattern=r"\S")


class PdfWorksheetSource(PoolModel):
    """Record human-input identity without publishing machine-local paths."""

    sha256: str = Field(pattern=SHA256)
    reviewer: str = Field(min_length=1, pattern=r"\S")


class PdfCollectedJudgment(PoolModel):
    """Blank decisions remain unjudged; relevant blank supply is incomplete."""

    task: PdfRetrievalJudgmentTask
    relevance: PdfRecordRelevance | None
    supplies_evidence: bool | None
    reviewer: str = Field(min_length=1, pattern=r"\S")
    worksheet_sha256: str = Field(pattern=SHA256)
    note: str = ''

    @model_validator(mode='after')
    def consistent_supply(self) -> 'PdfCollectedJudgment':
        """Never invent a label from a supply marker."""
        if self.relevance in (None, PdfRecordRelevance.IRRELEVANT) and (
            self.supplies_evidence is not None
        ):
            raise ValueError('Blank/irrelevant label requires blank supply')
        return self

    @property
    def complete(self) -> bool:
        """Require both dimensions when relevance is required/supporting."""
        return self.relevance is not None and (
            self.relevance is PdfRecordRelevance.IRRELEVANT
            or self.supplies_evidence is not None
        )


class PdfCollectedJudgments(PoolModel):
    """Preserve labels for the exact pool; no migration or freezing path."""

    schema_name: Literal['fredl.pdf.multibook-collected-judgments'] = (
        'fredl.pdf.multibook-collected-judgments'
    )
    version: Literal[1] = 1
    status: Literal['draft'] = 'draft'
    golden_sha256: str = Field(pattern=SHA256)
    golden_content_fingerprint: str = Field(pattern=SHA256)
    library_fingerprint: str = Field(pattern=SHA256)
    pool_fingerprint: str = Field(pattern=SHA256)
    sources: tuple[PdfWorksheetSource, ...] = Field(min_length=1)
    judgments: tuple[PdfCollectedJudgment, ...]

    @model_validator(mode='after')
    def unique_bindings(self) -> 'PdfCollectedJudgments':
        """Reject identical duplicates as well as conflicting labels."""
        keys = [label.task.key for label in self.judgments]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate judgment bindings')
        sources = [(source.sha256, source.reviewer) for source in self.sources]
        if len(set(sources)) != len(sources):
            raise ValueError('Duplicate worksheet sources')
        if any((item.worksheet_sha256, item.reviewer) not in sources
               for item in self.judgments):
            raise ValueError('Judgment names unknown worksheet source')
        return self

    @property
    def fingerprint(self) -> str:
        """Hash semantic labels and their exact human-input provenance."""
        return fingerprint_canonical_json(self.model_dump(mode='json'))
