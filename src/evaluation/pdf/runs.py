"""Store PDF retrieval runs and score them without mixing metric kinds."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.evaluation.pdf.manifest import PdfEvaluationManifest
from src.evaluation.pdf.models import PdfGoldenDataset
from src.evaluation.pdf.retrieval_metrics import PdfRetrievalResult
from src.ingestion.pdf.index_records import PdfIndexRecord


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class PdfStoredRetrievalResult(_StrictModel):
    """Keep one case's complete ranking and its neighbour context."""

    case_id: str = Field(min_length=1)
    ranked_record_ids: tuple[str, ...]
    context_record_ids: tuple[str, ...]
    abstained: bool

    def to_result(self) -> PdfRetrievalResult:
        """Return the metric input for this stored case."""
        return PdfRetrievalResult(
            case_id=self.case_id,
            ranked_record_ids=self.ranked_record_ids,
            context_record_ids=self.context_record_ids,
            abstained=self.abstained,
        )


class PdfRetrievalRun(_StrictModel):
    """Bind stored rankings to the manifest that produced them."""

    manifest: PdfEvaluationManifest
    results: tuple[PdfStoredRetrievalResult, ...]


class PdfPageOverlapCase(_StrictModel):
    """Report page overlap as a diagnostic, never as relevance."""

    case_id: str
    evidence_units: int
    ranked_units_with_page_overlap: int
    context_units_with_page_overlap: int


def write_pdf_retrieval_run(path: Path, run: PdfRetrievalRun) -> None:
    """Write one run without replacing an earlier artefact."""
    _write_new(path, run.model_dump_json(by_alias=True, indent=1))


def load_pdf_retrieval_run(path: Path) -> PdfRetrievalRun:
    """Read one stored run with strict validation."""
    try:
        return PdfRetrievalRun.model_validate_json(
            path.read_text(encoding="utf-8"), strict=False,
        )
    except OSError as error:
        raise ValueError("PDF retrieval run could not be read") from error
    except ValidationError as error:
        raise ValueError("PDF retrieval run JSON is invalid") from error


def page_overlap_diagnostics(
    dataset: PdfGoldenDataset,
    records: Sequence[PdfIndexRecord],
    run: PdfRetrievalRun,
) -> tuple[PdfPageOverlapCase, ...]:
    """Count evidence units touched by retrieved pages, per case."""
    pages = {
        record.record_id: {span.page_number for span in record.page_spans}
        for record in records
    }
    top_k = run.manifest.retrieval.top_k
    results = {result.case_id: result for result in run.results}
    diagnostics = []
    for case in dataset.cases:
        result = results[case.id]
        ranked = set().union(*(
            pages[record_id] for record_id in result.ranked_record_ids[:top_k]
        ))
        context = ranked.union(*(
            pages[record_id] for record_id in result.context_record_ids
        ))
        units = [set(item.physical_pages) for item in case.evidence]
        diagnostics.append(PdfPageOverlapCase(
            case_id=case.id,
            evidence_units=len(units),
            ranked_units_with_page_overlap=sum(
                bool(unit & ranked) for unit in units
            ),
            context_units_with_page_overlap=sum(
                bool(unit & context) for unit in units
            ),
        ))
    return tuple(diagnostics)


def _write_new(path: Path, content: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content + "\n")
    except FileExistsError as error:
        raise ValueError(f"Refusing to overwrite {path.name}") from error
    except OSError as error:
        raise ValueError(f"Could not write {path.name}") from error
