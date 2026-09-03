"""Tests for reviewable retrieval error-analysis reports."""

from pathlib import Path

import pytest

from src.evaluation.retrieval.error_report import (
    render_error_analysis_markdown,
)
from src.evaluation.retrieval.models import (
    RetrievalDatasetKind,
    RetrievalEvaluationCase,
)
from src.evaluation.retrieval.error_models import (
    RetrievalErrorCategory,
    RetrievalMissAnnotation,
)
from src.models import MinimalSource


def _source(path: str, end: int = 20) -> MinimalSource:
    return MinimalSource(
        file_path=path,
        first_character_index=0,
        last_character_index=end,
    )


def _write_file(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_report_shows_source_excerpts_for_comparison(tmp_path: Path) -> None:
    reference_text = "Use LLM.embed for vectors."
    reference = _source("docs/cache.md", len(reference_text))
    _write_file(tmp_path, reference.file_path, reference_text)
    retrieved_sources = []
    for rank in range(1, 6):
        path = f"docs/noise-{rank}.md"
        text = f"General prompt example {rank}."
        _write_file(tmp_path, path, text)
        retrieved_sources.append(_source(path, len(text)))
    retrieved = tuple(retrieved_sources)
    retrieved += (reference,)
    case = RetrievalEvaluationCase(
        question_id="docs-001",
        question="Where is the cache configured?",
        references=(reference,),
        retrieved=retrieved,
    )

    report = render_error_analysis_markdown(
        [(RetrievalDatasetKind.DOCS, (case,))],
        tmp_path,
    )

    assert "## Docs" in report
    assert "- Evaluated questions: 1" in report
    assert "- Top-5 misses: 1" in report
    assert "- Classified misses: 1" in report
    assert "- Pending review: 0" in report
    assert "- Dominant category: `relevant_below_top_5`" in report
    assert "### docs-001" in report
    assert "**Question:** Where is the cache configured?" in report
    assert "**First relevant rank:** 6" in report
    assert f"`docs/cache.md:0-{len(reference_text)}`" in report
    assert "Use LLM.embed for vectors." in report
    assert "#### Retrieved rank 1:" in report
    assert "General prompt example 1." in report
    assert "General prompt example 3." in report
    assert "General prompt example 4." not in report
    assert "**Category:** `relevant_below_top_5`" in report
    assert "**Hypothesis:** The lexical retriever found" in report


def test_report_excludes_questions_with_relevant_top_five_result(
    tmp_path: Path,
) -> None:
    reference = _source("src/cache.py")
    _write_file(tmp_path, reference.file_path, "x" * 20)
    case = RetrievalEvaluationCase(
        question_id="code-hit",
        question="Where is the cache configured?",
        references=(reference,),
        retrieved=(reference,),
    )

    report = render_error_analysis_markdown(
        [(RetrievalDatasetKind.CODE, (case,))],
        tmp_path,
    )

    assert "- Top-5 misses: 0" in report
    assert "### code-hit" not in report


def test_report_rejects_source_outside_project_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.md"
    outside.write_text("outside source text", encoding="utf-8")
    source = _source(str(outside), len("outside source text"))
    case = RetrievalEvaluationCase(
        question_id="unsafe",
        question="Where is the source?",
        references=(source,),
        retrieved=(),
    )

    with pytest.raises(ValueError, match="inside project root"):
        render_error_analysis_markdown(
            [(RetrievalDatasetKind.DOCS, (case,))],
            tmp_path,
        )


def test_report_applies_human_review_to_unclassified_miss(
    tmp_path: Path,
) -> None:
    reference = _source("docs/reference.md")
    retrieved = _source("src/implementation.py")
    _write_file(tmp_path, reference.file_path, "x" * 20)
    _write_file(tmp_path, retrieved.file_path, "y" * 20)
    case = RetrievalEvaluationCase(
        question_id="reviewed",
        question="Where is the feature described?",
        references=(reference,),
        retrieved=(retrieved,),
    )
    annotation = RetrievalMissAnnotation(
        question_id="reviewed",
        category=RetrievalErrorCategory.WRONG_FILE,
        hypothesis="The implementation answers a Docs-labelled question.",
        proposed_fix="Prefer documentation paths for Docs questions.",
        next_test="Test a Docs-only path preference.",
    )

    report = render_error_analysis_markdown(
        [(RetrievalDatasetKind.DOCS, (case,))],
        tmp_path,
        {"reviewed": annotation},
    )

    assert "**Category:** `wrong_file`" in report
    assert annotation.hypothesis in report
