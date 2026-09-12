"""Assignment-compatible command functions for the public CLI."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from tqdm import tqdm

from src.evaluation.answer_quality import (
    diagnose_dataset_answers,
    write_diagnostic_report,
)
from src.evaluation.retrieval import (
    RetrievalDatasetKind,
    RetrievalEvaluationReport,
    RetrievalMetrics,
    collect_top_five_misses,
    evaluate_cases,
    load_evaluation_cases,
    write_error_analysis_markdown,
)
from src.evaluation.retrieval.error_annotations import load_error_annotations
from src.ingestion import discover_files
from src.cli_progress import delayed_status
from src.generation import (
    BatchAnswerProgress,
    DEFAULT_MODEL_NAME,
    DevicePreference,
    GenerationConfig,
    answer_query,
    generate_dataset_answers,
    load_generation_backend,
    load_student_search_results,
    save_student_answers,
)
from src.generation.cache import ValidatedAnswerCache
from src.models import UnansweredQuestion
from src.retrieval import (
    DEFAULT_AUXILIARY_PATH_PENALTY,
    DEFAULT_PATH_CANDIDATE_DEPTH,
    run_stored_retrieval,
    run_stored_search,
)
from src.retrieval.bm25 import BM25Parameters
from src.retrieval.index_store import (
    IndexStore,
    PipelineConfig,
    SCHEMA_VERSION,
    build_index,
    fingerprint_corpus,
    fingerprint_pipeline,
)
from src.retrieval.validation import (
    MAX_SOURCE_LENGTH,
    SourceValidationReport,
    validate_retrieval_file,
)
from src.retrieval.tokenization import require_searchable_query


DEFAULT_INDEX_PATH = Path("data/processed/bm25-index.json")
DEFAULT_CORPUS_ROOT = Path("data/raw")
DEFAULT_BM25_PARAMETERS = BM25Parameters()
DEFAULT_MAX_CHUNK_SIZE = PipelineConfig().max_chunk_size
DEFAULT_CONTEXT_TOKEN_BUDGET = 4096
DEFAULT_ANSWER_CACHE_PATH = Path(".local/cache/validated-answers.json")
ANSWER_WAIT_MESSAGE = "Please wait, the local RAG answer is still running..."
BATCH_MODEL_WAIT_MESSAGE = (
    "Please wait, the local answer model is still loading..."
)
DEFAULT_ERROR_ANALYSIS_PATH = Path(
    "data/output/evaluation/retrieval-error-analysis.md"
)
DEFAULT_ERROR_ANNOTATIONS_PATH = Path(
    "data/output/evaluation/retrieval-error-annotations.json"
)


class CliError(Exception):
    """Represent an expected user-facing command failure."""


def answer(
    question: str,
    k: int = 5,
    context_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    index_path: str = str(DEFAULT_INDEX_PATH),
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
    model: str = DEFAULT_MODEL_NAME,
    device: str = DevicePreference.AUTO.value,
    max_new_tokens: int = 256,
    offline: bool = False,
    k1: float = DEFAULT_BM25_PARAMETERS.k1,
    b: float = DEFAULT_BM25_PARAMETERS.b,
    metadata_weight: float = DEFAULT_BM25_PARAMETERS.metadata_weight,
    identifier_weight: float = DEFAULT_BM25_PARAMETERS.identifier_weight,
    auxiliary_path_penalty: float = DEFAULT_AUXILIARY_PATH_PENALTY,
    path_candidate_depth: int = DEFAULT_PATH_CANDIDATE_DEPTH,
    answer_cache_path: str = str(DEFAULT_ANSWER_CACHE_PATH),
) -> dict[str, object]:
    """Answer one user question with traceable local RAG evidence."""
    try:
        require_searchable_query(question)
        _require_positive_k(k)
        _require_positive_context_budget(context_token_budget)
        generation_config = GenerationConfig(
            model_name=model,
            device=DevicePreference(device),
            max_new_tokens=max_new_tokens,
            local_files_only=offline,
        )
        with delayed_status(ANSWER_WAIT_MESSAGE):
            result = answer_query(
                question,
                Path(project_root),
                Path(corpus_root),
                Path(index_path),
                None,
                generation_config,
                _pipeline_config(
                    max_chunk_size,
                    k1,
                    b,
                    metadata_weight,
                    identifier_weight,
                ),
                k=k,
                context_token_budget=context_token_budget,
                auxiliary_path_penalty=auxiliary_path_penalty,
                path_candidate_depth=path_candidate_depth,
                answer_cache=ValidatedAnswerCache(
                    _below_root(Path(project_root), Path(answer_cache_path))
                ),
            )
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        raise CliError(_error_message(error)) from None

    return {
        "answer": result.answer,
        "sources": [
            source.model_dump() for source in result.context_sources
        ],
        "retrieved_sources": [
            source.model_dump() for source in result.retrieved_sources
        ],
        "used_context_tokens": result.used_context_tokens,
        "skipped_source_count": result.skipped_source_count,
        "prompt_version": result.prompt_version,
        "model": generation_config.model_name,
        "device": result.generation_device,
        "cache_hit": result.cache_hit,
    }


def answer_dataset(
    student_search_results_path: str,
    save_directory: str,
    context_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
    model: str = DEFAULT_MODEL_NAME,
    device: str = DevicePreference.AUTO.value,
    max_new_tokens: int = 256,
    offline: bool = False,
) -> str:
    """Generate and save answers for one persisted retrieval dataset."""
    try:
        root = Path(project_root)
        input_path = _below_root(root, Path(student_search_results_path))
        output_path = _below_root(root, Path(save_directory)) / input_path.name
        search_results = load_student_search_results(input_path)
        generation_config = GenerationConfig(
            model_name=model,
            device=DevicePreference(device),
            max_new_tokens=max_new_tokens,
            local_files_only=offline,
        )
        with delayed_status(BATCH_MODEL_WAIT_MESSAGE):
            backend = load_generation_backend(generation_config)
        with tqdm(
            total=len(search_results.search_results),
            desc="Answering",
            unit="question",
        ) as progress_bar:
            progress: BatchAnswerProgress = (
                lambda _position, _total: progress_bar.update()
            )
            answers = generate_dataset_answers(
                root,
                _below_root(root, Path(corpus_root)),
                search_results,
                backend,
                generation_config,
                context_token_budget=context_token_budget,
                progress=progress,
            )
        save_student_answers(answers, output_path)
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        raise CliError(_error_message(error)) from None
    return str(output_path)


def diagnose_answer_quality(
    student_search_results_path: str,
    output_path: str,
    context_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
    model: str = DEFAULT_MODEL_NAME,
    device: str = DevicePreference.AUTO.value,
    max_new_tokens: int = 256,
    offline: bool = False,
) -> dict[str, object]:
    """Generate and save review evidence without hiding failed cases."""
    try:
        root = Path(project_root)
        input_path = _below_root(root, Path(student_search_results_path))
        resolved_output = _below_root(root, Path(output_path))
        search_results = load_student_search_results(input_path)
        generation_config = GenerationConfig(
            model_name=model,
            device=DevicePreference(device),
            max_new_tokens=max_new_tokens,
            local_files_only=offline,
        )
        with delayed_status(BATCH_MODEL_WAIT_MESSAGE):
            backend = load_generation_backend(generation_config)
        with tqdm(
            total=len(search_results.search_results),
            desc="Diagnosing",
            unit="question",
        ) as progress_bar:
            progress: BatchAnswerProgress = (
                lambda _position, _total: progress_bar.update()
            )
            report = diagnose_dataset_answers(
                root,
                _below_root(root, Path(corpus_root)),
                search_results,
                backend,
                generation_config,
                context_token_budget=context_token_budget,
                progress=progress,
            )
        write_diagnostic_report(report, resolved_output)
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        raise CliError(_error_message(error)) from None

    return {
        "output_path": str(resolved_output),
        "question_count": len(report.cases),
        "passed_count": report.passed_count,
        "failed_count": report.failed_count,
        "model": report.model_name,
        "device": report.device,
    }


def index(
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    index_path: str = str(DEFAULT_INDEX_PATH),
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
    k1: float = DEFAULT_BM25_PARAMETERS.k1,
    b: float = DEFAULT_BM25_PARAMETERS.b,
    metadata_weight: float = DEFAULT_BM25_PARAMETERS.metadata_weight,
    identifier_weight: float = DEFAULT_BM25_PARAMETERS.identifier_weight,
) -> dict[str, object]:
    """Build and save a production-compatible BM25 index."""
    try:
        root = Path(project_root)
        config = _pipeline_config(
            max_chunk_size,
            k1,
            b,
            metadata_weight,
            identifier_weight,
        )
        build = build_index(
            root,
            _below_root(root, Path(corpus_root)),
            config,
            index_schema_version=SCHEMA_VERSION,
        )
        output_path = _below_root(root, Path(index_path))
        IndexStore(output_path).save(
            build.index,
            build.corpus_fingerprint,
            build.pipeline_fingerprint,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CliError(_error_message(error)) from None
    return {
        "index_path": str(output_path),
        "schema_version": SCHEMA_VERSION,
        "document_count": len(build.index.documents),
        "corpus_fingerprint": build.corpus_fingerprint,
        "pipeline_fingerprint": build.pipeline_fingerprint,
    }


def search(
    query: str,
    k: int = 5,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    index_path: str = str(DEFAULT_INDEX_PATH),
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
    k1: float = DEFAULT_BM25_PARAMETERS.k1,
    b: float = DEFAULT_BM25_PARAMETERS.b,
    metadata_weight: float = DEFAULT_BM25_PARAMETERS.metadata_weight,
    identifier_weight: float = DEFAULT_BM25_PARAMETERS.identifier_weight,
    identifier_match_weight: float = 0.0,
    identifier_candidate_depth: int = 0,
    auxiliary_path_penalty: float = DEFAULT_AUXILIARY_PATH_PENALTY,
    path_candidate_depth: int = DEFAULT_PATH_CANDIDATE_DEPTH,
) -> list[dict[str, object]]:
    """Return the top-k exact source locations for one raw query."""
    try:
        require_searchable_query(query)
        _require_positive_k(k)
        root = Path(project_root)
        fingerprint = _current_corpus_fingerprint(root, Path(corpus_root))
        sources = run_stored_search(
            _below_root(root, Path(index_path)),
            fingerprint,
            _current_pipeline_fingerprint(
                _pipeline_config(
                    max_chunk_size,
                    k1,
                    b,
                    metadata_weight,
                    identifier_weight,
                )
            ),
            query,
            k,
            identifier_match_weight=identifier_match_weight,
            identifier_candidate_depth=identifier_candidate_depth,
            auxiliary_path_penalty=auxiliary_path_penalty,
            path_candidate_depth=path_candidate_depth,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CliError(_error_message(error)) from None
    return [source.model_dump() for source in sources]


def search_dataset(
    dataset_path: str,
    save_directory: str,
    k: int = 5,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    index_path: str = str(DEFAULT_INDEX_PATH),
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
    k1: float = DEFAULT_BM25_PARAMETERS.k1,
    b: float = DEFAULT_BM25_PARAMETERS.b,
    metadata_weight: float = DEFAULT_BM25_PARAMETERS.metadata_weight,
    identifier_weight: float = DEFAULT_BM25_PARAMETERS.identifier_weight,
    identifier_match_weight: float = 0.0,
    identifier_candidate_depth: int = 0,
    auxiliary_path_penalty: float = DEFAULT_AUXILIARY_PATH_PENALTY,
    path_candidate_depth: int = DEFAULT_PATH_CANDIDATE_DEPTH,
) -> str:
    """Search one question dataset and save its validated result JSON."""
    try:
        _require_positive_k(k)
        root = Path(project_root)
        dataset = _below_root(root, Path(dataset_path))
        output = _below_root(root, Path(save_directory)) / dataset.name
        fingerprint = _current_corpus_fingerprint(root, Path(corpus_root))
        run_stored_retrieval(
            _below_root(root, Path(index_path)),
            fingerprint,
            _current_pipeline_fingerprint(
                _pipeline_config(
                    max_chunk_size,
                    k1,
                    b,
                    metadata_weight,
                    identifier_weight,
                )
            ),
            dataset,
            output,
            k,
            progress=_progress_questions,
            identifier_match_weight=identifier_match_weight,
            identifier_candidate_depth=identifier_candidate_depth,
            auxiliary_path_penalty=auxiliary_path_penalty,
            path_candidate_depth=path_candidate_depth,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CliError(_error_message(error)) from None
    return str(output)


def _progress_questions(
    questions: Sequence[UnansweredQuestion],
) -> Iterable[UnansweredQuestion]:
    """Display CLI batch progress without coupling retrieval to tqdm."""
    yield from tqdm(questions, desc="Searching", unit="question")


def validate_sources(
    results_path: str,
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
    max_source_length: int = MAX_SOURCE_LENGTH,
) -> dict[str, object]:
    """Audit every source in one retrieval-results JSON file."""
    try:
        root = Path(project_root)
        report = validate_retrieval_file(
            _below_root(root, Path(results_path)),
            root,
            _below_root(root, Path(corpus_root)),
            max_source_length=max_source_length,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CliError(_error_message(error)) from None
    return _validation_report_dict(report)


def evaluate(
    student_search_results_path: str,
    dataset_path: str,
    project_root: str = ".",
) -> None:
    """Evaluate one assignment dataset against persisted search results."""
    try:
        root = Path(project_root)
        report = evaluate_cases(
            RetrievalDatasetKind.DATASET,
            load_evaluation_cases(
                _below_root(root, Path(dataset_path)),
                _below_root(root, Path(student_search_results_path)),
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CliError(_error_message(error)) from None
    _print_evaluation_report(report)


def evaluate_all(
    docs_ground_truth_path: str,
    docs_results_path: str,
    code_ground_truth_path: str,
    code_results_path: str,
    project_root: str = ".",
) -> None:
    """Evaluate persisted Docs and Code retrieval results in one run."""
    try:
        root = Path(project_root)
        docs_report = evaluate_cases(
            RetrievalDatasetKind.DOCS,
            load_evaluation_cases(
                _below_root(root, Path(docs_ground_truth_path)),
                _below_root(root, Path(docs_results_path)),
            ),
        )
        code_report = evaluate_cases(
            RetrievalDatasetKind.CODE,
            load_evaluation_cases(
                _below_root(root, Path(code_ground_truth_path)),
                _below_root(root, Path(code_results_path)),
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CliError(_error_message(error)) from None
    _print_evaluation_report(docs_report)
    _print_evaluation_report(code_report)


def analyze_retrieval_errors(
    docs_ground_truth_path: str,
    docs_results_path: str,
    code_ground_truth_path: str,
    code_results_path: str,
    output_path: str = str(DEFAULT_ERROR_ANALYSIS_PATH),
    annotations_path: str = str(DEFAULT_ERROR_ANNOTATIONS_PATH),
    index_path: str = str(DEFAULT_INDEX_PATH),
    corpus_root: str = str(DEFAULT_CORPUS_ROOT),
    project_root: str = ".",
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    k1: float = DEFAULT_BM25_PARAMETERS.k1,
    b: float = DEFAULT_BM25_PARAMETERS.b,
    metadata_weight: float = DEFAULT_BM25_PARAMETERS.metadata_weight,
    identifier_weight: float = DEFAULT_BM25_PARAMETERS.identifier_weight,
) -> dict[str, object]:
    """Write reviewable top-five miss evidence for Docs and Code."""
    try:
        root = Path(project_root)
        docs_cases = load_evaluation_cases(
            _below_root(root, Path(docs_ground_truth_path)),
            _below_root(root, Path(docs_results_path)),
        )
        code_cases = load_evaluation_cases(
            _below_root(root, Path(code_ground_truth_path)),
            _below_root(root, Path(code_results_path)),
        )
        resolved_output = _below_root(root, Path(output_path))
        resolved_annotations = _below_root(root, Path(annotations_path))
        annotations = (
            load_error_annotations(resolved_annotations)
            if resolved_annotations.exists()
            else {}
        )
        config = _pipeline_config(
            max_chunk_size,
            k1,
            b,
            metadata_weight,
            identifier_weight,
        )
        index = IndexStore(_below_root(root, Path(index_path))).load(
            _current_corpus_fingerprint(root, Path(corpus_root)),
            _current_pipeline_fingerprint(config),
        )
        write_error_analysis_markdown(
            resolved_output,
            (
                (RetrievalDatasetKind.DOCS, docs_cases),
                (RetrievalDatasetKind.CODE, code_cases),
            ),
            root,
            annotations,
            tuple(document.chunk for document in index.documents),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CliError(_error_message(error)) from None
    return {
        "output_path": str(resolved_output),
        "docs_top_5_misses": len(collect_top_five_misses(docs_cases)),
        "code_top_5_misses": len(collect_top_five_misses(code_cases)),
    }


def _current_corpus_fingerprint(
    project_root: Path,
    corpus_root: Path,
) -> str:
    """Identify the currently discovered assignment corpus."""
    resolved_corpus = _below_root(project_root, corpus_root)
    manifest = discover_files(project_root, resolved_corpus)
    return fingerprint_corpus(project_root, manifest)


def _pipeline_config(
    max_chunk_size: int,
    k1: float,
    b: float,
    metadata_weight: float,
    identifier_weight: float,
) -> PipelineConfig:
    """Build the shared CLI configuration for index compatibility checks."""
    return PipelineConfig(
        max_chunk_size=max_chunk_size,
        parameters=BM25Parameters(
            k1=k1,
            b=b,
            metadata_weight=metadata_weight,
            identifier_weight=identifier_weight,
        )
    )


def _require_positive_context_budget(context_token_budget: int) -> None:
    """Reject an unusable context budget before loading the model."""
    if context_token_budget <= 0:
        raise ValueError("Context token budget must be greater than zero")


def _current_pipeline_fingerprint(config: PipelineConfig) -> str:
    """Identify the requested production-compatible index pipeline."""
    return fingerprint_pipeline(
        config,
        index_schema_version=SCHEMA_VERSION,
    )


def _below_root(project_root: Path, path: Path) -> Path:
    """Resolve a relative CLI path from the configured project root."""
    if path.is_absolute():
        return path
    return project_root / path


def _require_positive_k(k: int) -> None:
    """Reject an invalid limit before corpus or index I/O begins."""
    if k <= 0:
        raise ValueError("Search k must be greater than zero")


def _error_message(error: Exception) -> str:
    """Format an expected boundary failure without implementation details."""
    if isinstance(error, FileNotFoundError):
        missing_path = error.filename or str(error)
        return f"File not found: {missing_path}"
    if isinstance(error, NotADirectoryError):
        invalid_path = error.filename or str(error)
        return f"Directory not found: {invalid_path}"
    return str(error) or error.__class__.__name__


def _validation_report_dict(
    report: SourceValidationReport,
) -> dict[str, object]:
    """Convert an internal audit report into terminal-friendly values."""
    return {
        "result_count": report.result_count,
        "source_count": report.source_count,
        "valid_source_count": report.valid_source_count,
        "invalid_source_count": report.invalid_source_count,
        "passed": report.passed,
        "issues": [
            {
                "kind": issue.kind.value,
                "result_index": issue.result_index,
                "source_index": issue.source_index,
                "question_id": issue.question_id,
                "file_path": issue.file_path,
                "detail": issue.detail,
            }
            for issue in report.issues
        ],
    }


def _print_evaluation_report(
    report: RetrievalEvaluationReport,
) -> None:
    """Print one labelled dataset with stable terminal field names."""
    metrics: RetrievalMetrics = report.metrics
    print(f"{report.dataset.value}:")
    print(f"  query_count:  {metrics.query_count}")
    print(f"  recall_at_1:  {metrics.recall_at_1:.6f}")
    print(f"  recall_at_3:  {metrics.recall_at_3:.6f}")
    print(f"  recall_at_5:  {metrics.recall_at_5:.6f}")
    print(f"  recall_at_10: {metrics.recall_at_10:.6f}")
    print(f"  mrr:          {metrics.mean_reciprocal_rank:.6f}")
