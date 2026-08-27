"""Run retrieved-source validation across complete search results."""

from collections.abc import Mapping

from src.models import RetrievalResults
from src.retrieval.validation.invariants import (
    MAX_SOURCE_LENGTH,
    validate_source,
)
from src.retrieval.validation.models import (
    SourceValidationIssue,
    SourceValidationReport,
)


def validate_retrieval_results(
    results: RetrievalResults,
    source_texts: Mapping[str, str],
    *,
    max_source_length: int = MAX_SOURCE_LENGTH,
) -> SourceValidationReport:
    """Validate every retrieved source without stopping at first failure."""
    issues: list[SourceValidationIssue] = []
    source_count = 0
    for result_index, result in enumerate(results.search_results):
        for source_index, source in enumerate(result.retrieved_sources):
            source_count += 1
            issues.extend(
                validate_source(
                    source,
                    source_texts,
                    result_index=result_index,
                    source_index=source_index,
                    question_id=result.question_id,
                    max_source_length=max_source_length,
                )
            )

    return SourceValidationReport(
        result_count=len(results.search_results),
        source_count=source_count,
        issues=tuple(issues),
    )
