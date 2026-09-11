"""Tests for the assignment-compatible Python Fire CLI."""

import json
from pathlib import Path
from unittest.mock import ANY, call, patch

import pytest

from src.__main__ import main
from src.cli import BATCH_MODEL_WAIT_MESSAGE
from src.evaluation.answer_quality import AnswerQualityDiagnosticReport
from src.generation import (
    DevicePreference,
    GenerationConfig,
    GroundedAnswerResult,
    LoadedGenerationBackend,
    QueryAnswerResult,
)
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.evaluation.retrieval import (
    RetrievalDatasetKind,
    RetrievalEvaluationReport,
    RetrievalMetrics,
)
from src.retrieval.bm25 import BM25Parameters
from src.retrieval.index_store import (
    PipelineConfig,
    SCHEMA_VERSION,
    fingerprint_pipeline,
)
from src.retrieval.validation import SourceValidationReport


FINGERPRINT = "a" * 64
PIPELINE_FINGERPRINT = "b" * 64


def test_answer_command_loads_backend_and_prints_trace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One command composes local RAG and exposes evidence used by Qwen."""
    retrieved = MinimalSource(
        file_path="data/raw/guide.md",
        first_character_index=0,
        last_character_index=20,
    )
    used = MinimalSource(
        file_path="data/raw/cache.py",
        first_character_index=10,
        last_character_index=40,
    )
    result = QueryAnswerResult(
        answer="The cache uses LRU. [Source 1]",
        retrieved_sources=(retrieved, used),
        context_sources=(used,),
        used_context_tokens=42,
        skipped_source_count=1,
        prompt_version="v1",
        generation_attempts=2,
    )
    backend = LoadedGenerationBackend(object(), object(), "cpu")

    with (
        patch("src.cli.delayed_status") as status,
        patch(
            "src.cli.load_generation_backend",
            return_value=backend,
        ) as load_backend,
        patch("src.cli.answer_query", return_value=result) as run_answer,
    ):
        main(
            [
                "answer",
                "Which cache policy is used?",
                "--k",
                "2",
                "--context_token_budget",
                "1000",
                "--offline",
            ]
        )

    config = load_backend.call_args.args[0]
    status.assert_called_once_with(
        "Please wait, the local RAG answer is still running..."
    )
    assert config == GenerationConfig(local_files_only=True)
    assert run_answer.call_args.args[:6] == (
        "Which cache policy is used?",
        Path("."),
        Path("data/raw"),
        Path("data/processed/bm25-index.json"),
        backend,
        config,
    )
    assert run_answer.call_args.kwargs == {
        "k": 2,
        "context_token_budget": 1000,
        "auxiliary_path_penalty": 0.5,
        "path_candidate_depth": 20,
    }
    output = capsys.readouterr().out
    assert "answer:               The cache uses LRU. [Source 1]" in output
    assert "data/raw/cache.py" in output
    assert "data/raw/guide.md" in output
    assert "used_context_tokens:  42" in output
    assert "skipped_source_count: 1" in output
    assert "prompt_version:       v1" in output
    assert "generation_attempts:  2" in output
    assert "device:               cpu" in output


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (ValueError("Question must not be empty"), "must not be empty"),
        (RuntimeError("CUDA was requested"), "CUDA was requested"),
        (OSError("Model cache unavailable"), "Model cache unavailable"),
    ],
)
def test_answer_command_reports_expected_failures_without_traceback(
    failure: Exception,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected RAG boundary failures remain concise for terminal users."""
    with patch("src.cli.load_generation_backend", side_effect=failure):
        with pytest.raises(SystemExit) as exit_info:
            main(["answer", "Where is the cache?"])

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["answer", "   "], "Question must not be empty"),
        (["answer", "cache", "--k", "0"], "Search k must be greater"),
        (
            ["answer", "cache", "--context_token_budget", "0"],
            "Context token budget must be greater",
        ),
    ],
)
def test_answer_rejects_degenerate_input_before_model_loading(
    arguments: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cheap CLI validation runs before loading the local answer model."""
    with patch("src.cli.load_generation_backend") as load_backend:
        with pytest.raises(SystemExit) as exit_info:
            main(arguments)

    load_backend.assert_not_called()
    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_answer_dataset_uses_assignment_paths_and_one_backend() -> None:
    """The required batch command loads once and preserves the input name."""
    source = MinimalSource(
        file_path="data/raw/guide.md",
        first_character_index=0,
        last_character_index=20,
    )
    search_results = StudentSearchResults(
        search_results=[],
        k=5,
    )
    answers = StudentSearchResultsAndAnswer(
        search_results=[
            MinimalAnswer(
                question_id="q-1",
                question="Where is the cache?",
                retrieved_sources=[source],
                answer="The cache is documented. [Source 1]",
            )
        ],
        k=5,
    )
    backend = LoadedGenerationBackend(object(), object(), "cpu")

    with (
        patch(
            "src.cli.load_student_search_results",
            return_value=search_results,
        ) as load_results,
        patch("src.cli.delayed_status") as status,
        patch(
            "src.cli.load_generation_backend",
            return_value=backend,
        ) as load_backend,
        patch(
            "src.cli.generate_dataset_answers",
            return_value=answers,
        ) as generate_answers,
        patch("src.cli.save_student_answers") as save_answers,
    ):
        main(
            [
                "answer_dataset",
                "--student_search_results_path",
                "data/output/search_results/Public/questions.json",
                "--save_directory",
                "data/output/search_results_and_answer/Public",
                "--context_token_budget",
                "1000",
                "--offline",
            ]
        )

    input_path = Path("data/output/search_results/Public/questions.json")
    output_path = Path(
        "data/output/search_results_and_answer/Public/questions.json"
    )
    load_results.assert_called_once_with(input_path)
    status.assert_called_once_with(BATCH_MODEL_WAIT_MESSAGE)
    config = load_backend.call_args.args[0]
    assert config == GenerationConfig(local_files_only=True)
    generate_answers.assert_called_once_with(
        Path("."),
        Path("data/raw"),
        search_results,
        backend,
        config,
        context_token_budget=1000,
        progress=ANY,
    )
    save_answers.assert_called_once_with(answers, output_path)


def test_answer_dataset_writes_valid_json_without_retrieval(
    tmp_path: Path,
) -> None:
    """The CLI composes real batch I/O and context without searching again."""
    source_text = "The cache stores validated answers."
    source_path = tmp_path / "data" / "raw" / "guide.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source_text, encoding="utf-8")
    search_results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q-1",
                question="What does the cache store?",
                retrieved_sources=[
                    MinimalSource(
                        file_path="data/raw/guide.md",
                        first_character_index=0,
                        last_character_index=len(source_text),
                    )
                ],
            )
        ],
        k=1,
    )
    input_path = tmp_path / "retrieval" / "questions.json"
    input_path.parent.mkdir()
    input_path.write_text(
        search_results.model_dump_json(indent=2),
        encoding="utf-8",
    )

    class FakeTokenizer:
        def encode(
            self,
            text: str,
            *,
            add_special_tokens: bool,
        ) -> list[int]:
            del add_special_tokens
            return list(range(len(text.split())))

    grounded = GroundedAnswerResult(
        answer="It stores validated answers. [Source 1]",
        sources=(search_results.search_results[0].retrieved_sources[0],),
        prompt_version="v1",
    )
    backend = LoadedGenerationBackend(FakeTokenizer(), object(), "cpu")
    output_directory = tmp_path / "answers"

    with (
        patch("src.cli.delayed_status"),
        patch("src.cli.load_generation_backend", return_value=backend),
        patch(
            "src.generation.batch.workflow.generate_grounded_answer",
            return_value=grounded,
        ) as generate_answer,
        patch("src.cli.run_stored_search") as search_one,
        patch("src.cli.run_stored_retrieval") as search_batch,
    ):
        main(
            [
                "answer_dataset",
                "--student_search_results_path",
                str(input_path),
                "--save_directory",
                str(output_directory),
                "--project_root",
                str(tmp_path),
                "--offline",
            ]
        )

    output_path = output_directory / input_path.name
    parsed = StudentSearchResultsAndAnswer.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert parsed.k == 1
    assert parsed.search_results[0].question_id == "q-1"
    assert parsed.search_results[0].retrieved_sources == (
        search_results.search_results[0].retrieved_sources
    )
    assert parsed.search_results[0].answer == grounded.answer
    generate_answer.assert_called_once()
    search_one.assert_not_called()
    search_batch.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FileNotFoundError("results.json"), "File not found"),
        (
            ValueError("Student search results JSON is invalid"),
            "JSON is invalid",
        ),
        (RuntimeError("CUDA was requested"), "CUDA was requested"),
    ],
)
def test_answer_dataset_reports_expected_failures_without_traceback(
    failure: Exception,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected batch boundaries remain concise for terminal users."""
    with patch("src.cli.load_student_search_results", side_effect=failure):
        with pytest.raises(SystemExit) as exit_info:
            main(
                [
                    "answer_dataset",
                    "--student_search_results_path",
                    "results.json",
                    "--save_directory",
                    "answers",
                ]
            )

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_diagnose_answer_quality_saves_failures_and_summary() -> None:
    """The diagnostic command exposes report counts without failing a batch."""
    search_results = StudentSearchResults(search_results=[], k=5)
    backend = LoadedGenerationBackend(object(), object(), "cuda")
    report = AnswerQualityDiagnosticReport(
        model_name="Qwen/test",
        device="cuda",
        prompt_version="grounded-v1",
        max_new_tokens=128,
        context_token_budget=1000,
        search_k=5,
        cases=(),
    )

    with (
        patch(
            "src.cli.load_student_search_results",
            return_value=search_results,
        ),
        patch("src.cli.delayed_status") as status,
        patch(
            "src.cli.load_generation_backend",
            return_value=backend,
        ) as load_backend,
        patch(
            "src.cli.diagnose_dataset_answers",
            return_value=report,
        ) as diagnose,
        patch("src.cli.write_diagnostic_report") as write_report,
    ):
        main(
            [
                "diagnose_answer_quality",
                "--student_search_results_path",
                "results/questions.json",
                "--output_path",
                "reports/quality.json",
                "--context_token_budget",
                "1000",
                "--model",
                "Qwen/test",
                "--device",
                "cuda",
                "--max_new_tokens",
                "128",
                "--offline",
            ]
        )

    config = load_backend.call_args.args[0]
    assert config == GenerationConfig(
        model_name="Qwen/test",
        device=DevicePreference.CUDA,
        max_new_tokens=128,
        local_files_only=True,
    )
    status.assert_called_once_with(BATCH_MODEL_WAIT_MESSAGE)
    diagnose.assert_called_once_with(
        Path("."),
        Path("data/raw"),
        search_results,
        backend,
        config,
        context_token_budget=1000,
        progress=ANY,
    )
    write_report.assert_called_once_with(
        report,
        Path("reports/quality.json"),
    )


def test_index_command_builds_current_schema_snapshot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public command runs production ingestion and persists its result."""
    corpus_root = tmp_path / "data" / "raw"
    corpus_root.mkdir(parents=True)
    (corpus_root / "guide.md").write_text(
        "# Cache\n\nThe cache stores chunks.\n",
        encoding="utf-8",
    )

    main(
        [
            "index",
            "--project_root",
            str(tmp_path),
            "--corpus_root",
            "data/raw",
            "--index_path",
            "data/processed/test-index.json",
        ]
    )

    index_path = tmp_path / "data" / "processed" / "test-index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert len(payload["corpus_fingerprint"]) == 64
    assert len(payload["pipeline_fingerprint"]) == 64
    captured = capsys.readouterr()
    assert "document_count:" in captured.out
    assert "schema_version:       3" in captured.out


def test_index_command_persists_requested_bm25_parameters(
    tmp_path: Path,
) -> None:
    """A controlled experiment records its BM25 parameters in the index."""
    corpus_root = tmp_path / "data" / "raw"
    corpus_root.mkdir(parents=True)
    (corpus_root / "guide.md").write_text("# Cache\n", encoding="utf-8")

    main(
        [
            "index",
            "--project_root",
            str(tmp_path),
            "--metadata_weight",
            "1.5",
        ]
    )

    payload = json.loads(
        (tmp_path / "data" / "processed" / "bm25-index.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["parameters"] == {
        "b": 0.65,
        "k1": 1.4,
        "metadata_weight": 1.5,
        "identifier_weight": 0.0,
    }


def test_index_command_accepts_assignment_max_chunk_size(
    tmp_path: Path,
) -> None:
    """The mandatory flag controls chunks and the stored pipeline identity."""
    corpus_root = tmp_path / "data" / "raw"
    corpus_root.mkdir(parents=True)
    (corpus_root / "guide.md").write_text(
        "# Cache\n\n" + "cache data " * 80,
        encoding="utf-8",
    )

    main(
        [
            "index",
            "--project_root",
            str(tmp_path),
            "--max_chunk_size",
            "200",
        ]
    )

    payload = json.loads(
        (tmp_path / "data" / "processed" / "bm25-index.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["pipeline_fingerprint"] == fingerprint_pipeline(
        PipelineConfig(max_chunk_size=200),
        index_schema_version=SCHEMA_VERSION,
    )
    assert all(
        document["chunk"]["end"] - document["chunk"]["start"] <= 200
        for document in payload["documents"]
    )


def test_search_command_routes_one_raw_query() -> None:
    """The Fire search command reaches the stored single-query workflow."""
    with (
        patch(
            "src.cli._current_corpus_fingerprint",
            return_value=FINGERPRINT,
        ),
        patch(
            "src.cli._current_pipeline_fingerprint",
            return_value=PIPELINE_FINGERPRINT,
        ),
        patch("src.cli.run_stored_search", return_value=[]) as run_search,
    ):
        main(["search", "Where is the cache?", "--k", "3"])

    run_search.assert_called_once_with(
        Path("data/processed/bm25-index.json"),
        FINGERPRINT,
        PIPELINE_FINGERPRINT,
        "Where is the cache?",
        3,
        identifier_match_weight=0.0,
        identifier_candidate_depth=0,
        auxiliary_path_penalty=0.5,
        path_candidate_depth=20,
    )


def test_search_uses_requested_bm25_pipeline_fingerprint() -> None:
    """Search rejects accidental reuse of an index from another experiment."""
    parameters = BM25Parameters(metadata_weight=1.5)
    expected = fingerprint_pipeline(
        PipelineConfig(parameters=parameters),
        index_schema_version=SCHEMA_VERSION,
    )
    with (
        patch(
            "src.cli._current_corpus_fingerprint",
            return_value=FINGERPRINT,
        ),
        patch("src.cli.run_stored_search", return_value=[]) as run_search,
    ):
        main(["search", "cache", "--metadata_weight", "1.5"])

    assert run_search.call_args.args[2] == expected


def test_search_uses_requested_chunk_size_pipeline_fingerprint() -> None:
    """Search can load an index built with the mandatory chunk-size flag."""
    expected = fingerprint_pipeline(
        PipelineConfig(max_chunk_size=1200),
        index_schema_version=SCHEMA_VERSION,
    )
    with (
        patch(
            "src.cli._current_corpus_fingerprint",
            return_value=FINGERPRINT,
        ),
        patch("src.cli.run_stored_search", return_value=[]) as run_search,
    ):
        main(["search", "cache", "--max_chunk_size", "1200"])

    assert run_search.call_args.args[2] == expected


def test_search_rejects_empty_query_before_corpus_scan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty query fails before fingerprint or index work begins."""
    with patch("src.cli._current_corpus_fingerprint") as fingerprint:
        with pytest.raises(SystemExit) as exit_info:
            main(["search", "   "])

    fingerprint.assert_not_called()
    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.err == "Error: Question must not be empty\n"
    assert "Traceback" not in captured.err


def test_search_dataset_uses_assignment_paths_and_output_name() -> None:
    """The Fire batch command preserves the dataset filename in its output."""
    with (
        patch(
            "src.cli._current_corpus_fingerprint",
            return_value=FINGERPRINT,
        ),
        patch(
            "src.cli._current_pipeline_fingerprint",
            return_value=PIPELINE_FINGERPRINT,
        ),
        patch("src.cli.run_stored_retrieval") as run_retrieval,
    ):
        main(
            [
                "search_dataset",
                "--dataset_path",
                "data/datasets/questions.json",
                "--save_directory",
                "data/output/search_results/Public",
                "--k",
                "3",
                "--identifier_match_weight",
                "0.1",
                "--identifier_candidate_depth",
                "50",
                "--auxiliary_path_penalty",
                "0",
                "--path_candidate_depth",
                "0",
            ]
        )

    run_retrieval.assert_called_once_with(
        Path("data/processed/bm25-index.json"),
        FINGERPRINT,
        PIPELINE_FINGERPRINT,
        Path("data/datasets/questions.json"),
        Path("data/output/search_results/Public/questions.json"),
        3,
        progress=ANY,
        identifier_match_weight=0.1,
        identifier_candidate_depth=50,
        auxiliary_path_penalty=0.0,
        path_candidate_depth=0,
    )


def test_search_rejects_non_positive_k_before_corpus_scan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An invalid limit fails before fingerprint or index work begins."""
    with patch("src.cli._current_corpus_fingerprint") as fingerprint:
        with pytest.raises(SystemExit) as exit_info:
            main(["search", "cache", "--k", "0"])

    fingerprint.assert_not_called()
    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.err == "Error: Search k must be greater than zero\n"
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FileNotFoundError("questions.json"), "File not found"),
        (ValueError("Question dataset JSON is invalid"), "JSON is invalid"),
        (
            ValueError(
                "Stored BM25 index schema is incompatible; reindex required"
            ),
            "reindex required",
        ),
    ],
)
def test_search_dataset_reports_expected_failures_without_traceback(
    failure: Exception,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected file and validation failures remain concise for users."""
    with (
        patch(
            "src.cli._current_corpus_fingerprint",
            return_value=FINGERPRINT,
        ),
        patch("src.cli.run_stored_retrieval", side_effect=failure),
    ):
        with pytest.raises(SystemExit) as exit_info:
            main(
                [
                    "search_dataset",
                    "--dataset_path",
                    "questions.json",
                    "--save_directory",
                    "results",
                ]
            )

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_validate_sources_command_returns_audit_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The Fire validation command exposes stable report counts."""
    report = SourceValidationReport(result_count=100, source_count=500)
    with patch(
        "src.cli.validate_retrieval_file",
        return_value=report,
    ) as validate_file:
        main(
            [
                "validate_sources",
                "--results_path",
                "results/docs.json",
            ]
        )

    validate_file.assert_called_once_with(
        Path("results/docs.json"),
        Path("."),
        Path("data/raw"),
        max_source_length=2000,
    )
    captured = capsys.readouterr()
    assert "result_count:         100" in captured.out
    assert "source_count:         500" in captured.out
    assert "invalid_source_count: 0" in captured.out
    assert "passed:               true" in captured.out


def test_evaluate_command_reports_docs_and_code_separately(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public evaluator loads and labels both required datasets."""
    metrics = RetrievalMetrics(2, 0.25, 1.0, 1.0, 1.0, 0.75)
    with (
        patch(
            "src.cli.load_evaluation_cases",
            side_effect=[(), ()],
        ) as load_cases,
        patch(
            "src.cli.evaluate_cases",
            side_effect=[
                RetrievalEvaluationReport(
                    RetrievalDatasetKind.DOCS,
                    metrics,
                ),
                RetrievalEvaluationReport(
                    RetrievalDatasetKind.CODE,
                    metrics,
                ),
            ],
        ) as evaluate_loaded_cases,
    ):
        main(
            [
                "evaluate",
                "--docs_ground_truth_path",
                "datasets/docs.json",
                "--docs_results_path",
                "results/docs.json",
                "--code_ground_truth_path",
                "datasets/code.json",
                "--code_results_path",
                "results/code.json",
                "--project_root",
                "/project",
            ]
        )

    assert load_cases.call_args_list == [
        call(
            Path("/project/datasets/docs.json"),
            Path("/project/results/docs.json"),
        ),
        call(
            Path("/project/datasets/code.json"),
            Path("/project/results/code.json"),
        ),
    ]
    assert evaluate_loaded_cases.call_args_list == [
        call(RetrievalDatasetKind.DOCS, ()),
        call(RetrievalDatasetKind.CODE, ()),
    ]
    output = capsys.readouterr().out
    assert "Docs:" in output
    assert "Code:" in output
    assert output.count("query_count:  2") == 2
    assert output.count("recall_at_1:  0.250000") == 2
    assert output.count("mrr:          0.750000") == 2


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FileNotFoundError("docs.json"), "File not found"),
        (ValueError("Retrieval results JSON is invalid"), "JSON is invalid"),
        (
            ValueError(
                "Ground truth and retrieval results must contain the same IDs"
            ),
            "same IDs",
        ),
    ],
)
def test_evaluate_command_reports_expected_failures_without_traceback(
    failure: Exception,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected evaluator input failures remain concise for users."""
    with patch(
        "src.cli.load_evaluation_cases",
        side_effect=failure,
    ):
        with pytest.raises(SystemExit) as exit_info:
            main(
                [
                    "evaluate",
                    "--docs_ground_truth_path",
                    "docs-ground-truth.json",
                    "--docs_results_path",
                    "docs-results.json",
                    "--code_ground_truth_path",
                    "code-ground-truth.json",
                    "--code_results_path",
                    "code-results.json",
                ]
            )

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_analyze_retrieval_errors_writes_docs_and_code_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI exposes the generated report path and miss counts."""
    docs_cases = ("docs-case",)
    code_cases = ("code-case",)
    with (
        patch(
            "src.cli.load_evaluation_cases",
            side_effect=[docs_cases, code_cases],
        ),
        patch("src.cli.write_error_analysis_markdown") as write_report,
        patch("src.cli._current_corpus_fingerprint", return_value=FINGERPRINT),
        patch(
            "src.cli._current_pipeline_fingerprint",
            return_value=PIPELINE_FINGERPRINT,
        ),
        patch("src.cli.IndexStore") as index_store,
        patch(
            "src.cli.collect_top_five_misses",
            side_effect=[("docs-miss",), ("code-1", "code-2")],
        ),
    ):
        index_store.return_value.load.return_value.documents = ()
        main(
            [
                "analyze_retrieval_errors",
                "--docs_ground_truth_path",
                "datasets/docs.json",
                "--docs_results_path",
                "results/docs.json",
                "--code_ground_truth_path",
                "datasets/code.json",
                "--code_results_path",
                "results/code.json",
                "--output_path",
                "reports/errors.md",
                "--project_root",
                "/project",
            ]
        )

    write_report.assert_called_once_with(
        Path("/project/reports/errors.md"),
        (
            (RetrievalDatasetKind.DOCS, docs_cases),
            (RetrievalDatasetKind.CODE, code_cases),
        ),
        Path("/project"),
        {},
        (),
    )
    index_store.assert_called_once_with(
        Path("/project/data/processed/bm25-index.json")
    )
    index_store.return_value.load.assert_called_once_with(
        FINGERPRINT,
        PIPELINE_FINGERPRINT,
    )
    output = capsys.readouterr().out
    assert "docs_top_5_misses: 1" in output
    assert "code_top_5_misses: 2" in output
    assert "/project/reports/errors.md" in output
