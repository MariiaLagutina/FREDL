"""Render reviewable Markdown evidence for retrieval misses."""

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

from src.evaluation.retrieval.error_analysis import (
    classify_structural_miss,
    collect_top_five_misses,
)
from src.evaluation.retrieval.error_models import (
    RetrievalErrorCategory,
    RetrievalMissAnalysis,
    RetrievalMissAnnotation,
    RetrievalMissEvidence,
)
from src.evaluation.retrieval.metrics import sources_match
from src.evaluation.retrieval.models import (
    RetrievalDatasetKind,
    RetrievalEvaluationCase,
)
from src.models import MinimalSource


EvaluationDataset: TypeAlias = tuple[
    RetrievalDatasetKind,
    Sequence[RetrievalEvaluationCase],
]


def write_error_analysis_markdown(
    output_path: Path,
    datasets: Sequence[EvaluationDataset],
    project_root: Path,
    annotations: Mapping[str, RetrievalMissAnnotation] | None = None,
) -> None:
    """Write top-five miss evidence for human classification."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_error_analysis_markdown(datasets, project_root, annotations),
        encoding="utf-8",
    )


def render_error_analysis_markdown(
    datasets: Sequence[EvaluationDataset],
    project_root: Path,
    annotations: Mapping[str, RetrievalMissAnnotation] | None = None,
) -> str:
    """Render datasets and ranked sources in a stable review format."""
    lines = [
        "# Retrieval Error Analysis Evidence",
        "",
        (
            "This generated report lists questions with no relevant source "
            "in the top five. It combines deterministic structural evidence "
            "with separately stored human-reviewed classifications."
        ),
    ]
    reviewed = annotations or {}
    for dataset, cases in datasets:
        misses = collect_top_five_misses(cases)
        analyzed_misses = [
            (
                miss,
                classify_structural_miss(miss)
                or _manual_analysis(miss, reviewed.get(miss.question_id)),
            )
            for miss in misses
        ]
        lines.extend(
            [
                "",
                f"## {dataset.value}",
                "",
                f"- Evaluated questions: {len(cases)}",
                f"- Top-5 misses: {len(misses)}",
                *_category_summary(analyzed_misses),
            ]
        )
        for miss, analysis in analyzed_misses:
            relevant_rank = miss.relevant_rank or "not found"
            lines.extend(
                [
                    "",
                    f"### {miss.question_id}",
                    "",
                    f"**Question:** {miss.question}",
                    "",
                    f"**First relevant rank:** {relevant_rank}",
                    "",
                    "**Reference sources:**",
                    "",
                    *(
                        f"- `{_source_label(source)}`"
                        for source in miss.references
                    ),
                    "",
                    "| Rank | Retrieved source | Relevant |",
                    "| ---: | --- | :---: |",
                ]
            )
            lines.extend(
                f"| {rank} | `{_source_label(source)}` | "
                f"{'yes' if _is_relevant(source, miss.references) else 'no'} |"
                for rank, source in enumerate(miss.retrieved, start=1)
            )
            lines.extend(["", "**Reference excerpts:**"])
            for number, source in enumerate(miss.references, start=1):
                lines.extend(
                    _excerpt_section(
                        f"Reference {number}",
                        source,
                        project_root,
                    )
                )
            lines.extend(["", "**Top-3 retrieved excerpts:**"])
            for rank, source in enumerate(miss.retrieved[:3], start=1):
                lines.extend(
                    _excerpt_section(
                        f"Retrieved rank {rank}",
                        source,
                        project_root,
                    )
                )
            lines.extend(_analysis_lines(analysis))
    return "\n".join(lines) + "\n"


def _category_summary(
    analyzed_misses: Sequence[
        tuple[RetrievalMissEvidence, RetrievalMissAnalysis | None]
    ],
) -> list[str]:
    """Summarize classifications and the dominant reviewed cause."""
    counts = Counter(
        analysis.category
        for _, analysis in analyzed_misses
        if analysis is not None
    )
    pending = sum(analysis is None for _, analysis in analyzed_misses)
    dominant = (
        max(
            RetrievalErrorCategory,
            key=lambda category: counts[category],
        ).value
        if counts
        else "none"
    )
    lines = [
        f"- Classified misses: {sum(counts.values())}",
        f"- Pending review: {pending}",
        f"- Dominant category: `{dominant}`",
        "",
        "| Category | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{category.value}` | {counts[category]} |"
        for category in RetrievalErrorCategory
    )
    return lines


def _manual_analysis(
    evidence: RetrievalMissEvidence,
    annotation: RetrievalMissAnnotation | None,
) -> RetrievalMissAnalysis | None:
    """Combine one reviewed cause with the measured miss evidence."""
    if annotation is None:
        return None
    return RetrievalMissAnalysis(
        question_id=evidence.question_id,
        question=evidence.question,
        category=annotation.category,
        relevant_rank=evidence.relevant_rank,
        hypothesis=annotation.hypothesis,
        proposed_fix=annotation.proposed_fix,
        next_test=annotation.next_test,
    )


def _analysis_lines(
    analysis: RetrievalMissAnalysis | None,
) -> list[str]:
    """Render a confirmed classification or explicit review placeholders."""
    if analysis is None:
        values = ("_pending_",) * 4
    else:
        values = (
            f"`{analysis.category.value}`",
            analysis.hypothesis,
            analysis.proposed_fix,
            analysis.next_test,
        )
    category, hypothesis, proposed_fix, next_test = values
    return [
        "",
        f"**Category:** {category}",
        "",
        f"**Hypothesis:** {hypothesis}",
        "",
        f"**Proposed fix:** {proposed_fix}",
        "",
        f"**Next test:** {next_test}",
    ]


def _excerpt_section(
    label: str,
    source: MinimalSource,
    project_root: Path,
) -> list[str]:
    """Render an exact source slice as an indented Markdown code block."""
    excerpt = _read_source_excerpt(project_root, source)
    indented = [f"    {line}" for line in excerpt.splitlines()]
    if not indented:
        indented = ["    "]
    return ["", f"#### {label}: `{_source_label(source)}`", "", *indented]


def _read_source_excerpt(
    project_root: Path,
    source: MinimalSource,
) -> str:
    """Read one validated half-open source range below the project root."""
    root = project_root.resolve()
    source_path = (root / source.file_path).resolve()
    if not source_path.is_relative_to(root):
        raise ValueError("Retrieval source path must be inside project root")
    text = source_path.read_text(encoding="utf-8")
    start = source.first_character_index
    end = source.last_character_index
    if start < 0 or end <= start or end > len(text):
        raise ValueError("Retrieval source range must be inside its file")
    return text[start:end]


def _source_label(source: MinimalSource) -> str:
    """Format one exact source span compactly."""
    return (
        f"{source.file_path}:"
        f"{source.first_character_index}-{source.last_character_index}"
    )


def _is_relevant(
    retrieved: MinimalSource,
    references: Sequence[MinimalSource],
) -> bool:
    """Mark rankings with the same relevance rule as evaluation."""
    return any(sources_match(retrieved, reference) for reference in references)
