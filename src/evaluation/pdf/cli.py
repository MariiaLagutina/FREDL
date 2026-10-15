"""Reproducible PDF evaluation commands over private library inputs.

Every private input is a path argument. Outputs are created exclusively,
so no existing judgment, golden set, or report can be overwritten.
"""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from functools import wraps
from hashlib import sha256
import json
from pathlib import Path
from typing import cast, TypeVar

from src.cli import CliError

from src.evaluation.pdf.answerability import (
    PDF_ANSWERABILITY_PROMPT_VERSION,
    PdfAnswerabilityEvaluation,
)
from src.evaluation.pdf.answerability_runs import (
    build_pdf_answerability_manifest,
    evaluate_pdf_answerability_run,
    load_pdf_answerability_run,
    LocalChatCompletionClient,
    PDF_COMPARISON_GUARD_VERSION,
    PdfAnswerabilityRun,
    require_supported_passage_policy,
    run_pdf_answerability,
    write_pdf_answerability_run,
)
from src.evaluation.pdf.bm25_baseline import PdfBm25Baseline
from src.evaluation.pdf.context import find_pdf_neighbor_context
from src.evaluation.pdf.hybrid_baseline import (
    PDF_HYBRID_PARAMETERS,
    PdfHybridBaseline,
)
from src.evaluation.pdf.judgments import (
    load_pdf_judgment_set,
    PdfJudgmentSet,
    PdfRecordRelevance,
)
from src.evaluation.pdf.loader import load_pdf_golden_dataset
from src.evaluation.pdf.manifest import (
    bm25_configuration,
    build_pdf_evaluation_manifest,
    PdfRetrievalConfiguration,
    PdfRetrieverKind,
    current_pdf_run_environment,
    manifest_environment,
    PdfAnswerabilityConfiguration,
    PdfRunEnvironment,
    pdf_reproducibility_warnings,
    require_pdf_manifest_compatibility,
    rrf_configuration,
    semantic_configuration,
)
from src.evaluation.pdf.migration import (
    finalize_pdf_judgment_migration,
    migrate_pdf_judgments,
    PdfJudgmentMigration,
    PdfReviewDecision,
)
from src.evaluation.pdf.minilm_baseline import PdfMiniLmBaseline
from src.evaluation.pdf.models import PdfGoldenDataset
from src.evaluation.pdf.retrieval_metrics import evaluate_pdf_retrieval
from src.evaluation.pdf.runs import (
    load_pdf_retrieval_run,
    page_overlap_diagnostics,
    PdfRetrievalRun,
    PdfStoredRetrievalResult,
    write_pdf_retrieval_run,
)
from src.evaluation.pdf.snapshots import (
    load_pdf_record_snapshot,
    snapshot_pdf_records,
    write_pdf_record_snapshot,
)
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.ingestion.pdf.outline import PdfOutlineEntry
from src.ingestion.pdf.pipeline import build_pdf_book_records
from src.ingestion.pdf.retrieval_corpus import (
    build_pdf_retrieval_corpus,
    classify_pdf_records,
    PdfRetrievalCorpus,
    PdfRetrievalPolicy,
)
from src.retrieval.semantic import MiniLMEncoder
from src.retrieval.semantic.config import SemanticEncoderConfig


DEFAULT_PDF_POLICY = PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE
DEFAULT_ANSWERABILITY_SERVER = "http://127.0.0.1:18080/v1/chat/completions"
DEFAULT_ANSWERABILITY_RUNTIME = "llama.cpp llama-server b11574 (CUDA 12.8)"
DEFAULT_ANSWERABILITY_SERVER_ARGS = (
    "-ngl 99 -c 16384 --host 127.0.0.1 --reasoning off --reasoning-budget 0"
)
_Search = Callable[[str], tuple[str, ...]]
_Command = TypeVar("_Command", bound=Callable[..., str])


def _user_errors(command: _Command) -> _Command:
    """Report invalid inputs and refused overwrites as CLI errors."""
    @wraps(command)
    def run(*args: object, **kwargs: object) -> str:
        try:
            return command(*args, **kwargs)
        except (OSError, ValueError) as error:
            raise CliError(str(error)) from error

    return cast(_Command, run)


@_user_errors
def pdf_snapshot_records(
    library_root: str,
    book_path: str,
    output: str,
) -> str:
    """Write the current record set of one book as a verified snapshot."""
    book = build_pdf_book_records(Path(library_root), book_path)
    snapshot = snapshot_pdf_records(
        book.records,
        code_revision=current_pdf_run_environment().code_revision,
    )
    write_pdf_record_snapshot(Path(output), snapshot)
    return (
        f"{len(book.records)} records, "
        f"fingerprint {snapshot.record_set_fingerprint_v2}"
    )


@_user_errors
def pdf_migrate_judgments(
    source_judgments: str,
    source_snapshot: str,
    source_dataset: str,
    target_dataset: str,
    library_root: str,
    output_dir: str,
    review_decisions: str | None = None,
) -> str:
    """Migrate historical labels; write judgments only when all are decided.

    While any label is pending and no decisions are supplied, only the
    migration report and review queue are written.
    """
    source_path = Path(source_judgments)
    source_bytes = source_path.read_bytes()
    source_set = PdfJudgmentSet.model_validate_json(source_bytes)
    _, source_records = load_pdf_record_snapshot(Path(source_snapshot))
    old_dataset = load_pdf_golden_dataset(Path(source_dataset))
    new_dataset = load_pdf_golden_dataset(Path(target_dataset))
    target = build_pdf_book_records(
        Path(library_root), new_dataset.book_path,
    )
    migration = migrate_pdf_judgments(
        source_set=source_set,
        source_sha256=sha256(source_bytes).hexdigest(),
        source_dataset=old_dataset,
        source_records=source_records,
        target_dataset=new_dataset,
        target_records=target.records,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_new(out / "migration-report.json", json.dumps(
        _migration_report(migration), indent=1, ensure_ascii=False,
    ))
    _write_new(out / "review-queue.md", _review_queue(
        migration, new_dataset, source_records, target.records,
    ))
    if review_decisions is None and migration.pending:
        return _migration_summary(migration) + "; judgments not written"

    decisions = (
        _load_decisions(Path(review_decisions))
        if review_decisions is not None else ()
    )
    final = finalize_pdf_judgment_migration(
        migration, decisions, new_dataset, target.records,
    )
    _write_new(
        out / "record-judgments-v2.json",
        final.model_dump_json(indent=1, exclude_none=True),
    )
    return (
        f"{len(migration.migrated)} labels migrated automatically, "
        f"{len(migration.pending)} decided by review, "
        f"{len(final.judgments)} labels written"
    )


@_user_errors
def pdf_run_retrieval(
    library_root: str,
    dataset: str,
    retriever: str,
    output: str,
    policy: str = DEFAULT_PDF_POLICY.value,
    top_k: int = 5,
) -> str:
    """Rank every golden question and store the run with its manifest."""
    dataset_path = Path(dataset)
    golden = load_pdf_golden_dataset(dataset_path)
    book = build_pdf_book_records(Path(library_root), golden.book_path)
    corpus = _corpus(book.records, book.outline, PdfRetrievalPolicy(policy))
    kind = PdfRetrieverKind(retriever)
    semantic = SemanticEncoderConfig()
    configuration = _retrieval_configuration(kind, top_k, semantic)
    search = _searcher(kind, corpus, semantic)

    results = []
    for case in golden.cases:
        ranked = search(case.question)
        results.append(PdfStoredRetrievalResult(
            case_id=case.id,
            ranked_record_ids=ranked,
            context_record_ids=find_pdf_neighbor_context(
                ranked, corpus.records, top_k=top_k,
            ),
            abstained=kind is PdfRetrieverKind.BM25 and not ranked,
        ))
    environment = current_pdf_run_environment()
    manifest = build_pdf_evaluation_manifest(
        dataset=golden,
        dataset_sha256=_sha256(dataset_path),
        records=book.records,
        corpus=corpus,
        retrieval=configuration,
        code_revision=environment.code_revision,
        code_dirty=environment.code_dirty,
        environment=environment.packages,
    )
    write_pdf_retrieval_run(
        Path(output), PdfRetrievalRun(manifest=manifest,
                                      results=tuple(results)),
    )
    return f"{kind.value}: {len(results)} cases, run {manifest.identity}"


@_user_errors
def pdf_evaluate_retrieval(
    run: str,
    dataset: str,
    judgments: str,
    library_root: str,
    output: str,
) -> str:
    """Score a stored run after checking every compatibility fingerprint."""
    stored = load_pdf_retrieval_run(Path(run))
    dataset_path = Path(dataset)
    golden = load_pdf_golden_dataset(dataset_path)
    book = build_pdf_book_records(Path(library_root), golden.book_path)
    corpus = _corpus(book.records, book.outline,
                     stored.manifest.corpus_policy)
    require_pdf_manifest_compatibility(
        stored.manifest,
        dataset=golden,
        dataset_sha256=_sha256(dataset_path),
        records=book.records,
        corpus=corpus,
    )
    eligible = {record.record_id for record in corpus.records}
    for result in stored.results:
        if not set(result.ranked_record_ids) <= eligible:
            raise ValueError("PDF run ranks a record outside its corpus")
        if not set(result.context_record_ids) <= eligible:
            raise ValueError("PDF run context is outside its corpus")
    judgment_set = load_pdf_judgment_set(
        Path(judgments), golden, book.records,
    )
    evaluation = evaluate_pdf_retrieval(
        golden,
        judgment_set,
        book.records,
        tuple(result.to_result() for result in stored.results),
        top_k=stored.manifest.retrieval.top_k,
    )
    warnings = pdf_reproducibility_warnings(
        manifest_environment(stored.manifest),
        current_pdf_run_environment(),
    )
    report = {
        "run_identity": stored.manifest.identity,
        "reproducibility_warnings": list(warnings),
        "manifest": stored.manifest.model_dump(mode="json", by_alias=True),
        "judgments_sha256": _sha256(Path(judgments)),
        "human_labeled_relevance": {
            "mean_required_recall_at_k": evaluation.mean_required_recall_at_k,
            "all_required_at_k_rate": evaluation.all_required_at_k_rate,
            "mean_reciprocal_rank": evaluation.mean_reciprocal_rank,
            "negative_abstention_accuracy":
                evaluation.negative_abstention_accuracy,
        },
        "evidence_coverage": {
            "mean_ranked_required_evidence_coverage_at_k":
                evaluation.mean_ranked_required_evidence_coverage_at_k,
            "all_ranked_required_evidence_at_k_rate":
                evaluation.all_ranked_required_evidence_at_k_rate,
            "mean_context_required_evidence_coverage_at_k":
                evaluation.mean_context_required_evidence_coverage_at_k,
            "all_context_required_evidence_at_k_rate":
                evaluation.all_context_required_evidence_at_k_rate,
        },
        "page_overlap_diagnostic": [
            item.model_dump() for item in page_overlap_diagnostics(
                golden, book.records, stored,
            )
        ],
        "cases": [
            {
                key: (value.value if hasattr(value, "value") else value)
                for key, value in asdict(score).items()
            }
            for score in evaluation.case_scores
        ],
    }
    _write_new(Path(output), json.dumps(report, indent=1))
    return _with_warnings(
        f"evaluated {len(evaluation.case_scores)} cases", warnings,
    )


@_user_errors
def pdf_compare_runs(first: str, second: str) -> str:
    """Report whether two runs are directly comparable and why not."""
    left = load_pdf_retrieval_run(Path(first)).manifest
    right = load_pdf_retrieval_run(Path(second)).manifest
    lines = [
        "same evaluation identity"
        if left.identity == right.identity
        else "different evaluation identity: not directly comparable"
    ]
    for name in ("record_set_fingerprint", "corpus_fingerprint",
                 "retrieval_fingerprint", "dataset_sha256"):
        if getattr(left, name) != getattr(right, name):
            lines.append(f"{name} differs")
    warnings = pdf_reproducibility_warnings(
        manifest_environment(left), manifest_environment(right),
    )
    lines.extend(f"warning: {item}" for item in warnings)
    return "\n".join(lines)


@_user_errors
def pdf_run_answerability(
    retrieval_run: str,
    dataset: str,
    library_root: str,
    model_file: str,
    output: str,
    server_url: str = DEFAULT_ANSWERABILITY_SERVER,
    model_name: str = "Qwen3-4B-Q4_K_M",
    runtime: str = DEFAULT_ANSWERABILITY_RUNTIME,
    server_args: str = DEFAULT_ANSWERABILITY_SERVER_ARGS,
    answerability_max_tokens: int = 3,
    comparison_max_tokens: int = 1,
) -> str:
    """Classify answerability from a compatible stored retrieval run.

    The local server must already run the model in model_file; the file is
    hashed so the run records the exact weights.
    """
    stored = load_pdf_retrieval_run(Path(retrieval_run))
    dataset_path = Path(dataset)
    golden = load_pdf_golden_dataset(dataset_path)
    book = build_pdf_book_records(Path(library_root), golden.book_path)
    corpus = _corpus(book.records, book.outline,
                     stored.manifest.corpus_policy)
    require_pdf_manifest_compatibility(
        stored.manifest,
        dataset=golden,
        dataset_sha256=_sha256(dataset_path),
        records=book.records,
        corpus=corpus,
    )
    configuration = PdfAnswerabilityConfiguration(
        model=model_name,
        model_sha256=_sha256_streamed(Path(model_file)),
        runtime=runtime,
        server_args=tuple(server_args.split()),
        temperature=0.0,
        answerability_max_tokens=answerability_max_tokens,
        comparison_max_tokens=comparison_max_tokens,
        prompt_version=PDF_ANSWERABILITY_PROMPT_VERSION,
        guard_version=PDF_COMPARISON_GUARD_VERSION,
    )
    client = LocalChatCompletionClient(
        server_url, model_name, configuration.temperature,
    )
    environment = current_pdf_run_environment()
    cases = run_pdf_answerability(
        golden, stored, book.records, client, configuration,
    )
    run = PdfAnswerabilityRun(
        manifest=build_pdf_answerability_manifest(
            retrieval_run=stored,
            configuration=configuration,
            environment=environment,
        ),
        cases=cases,
    )
    write_pdf_answerability_run(Path(output), run)
    return _with_warnings(
        f"answerability: {len(cases)} cases, run {run.manifest.identity}",
        pdf_reproducibility_warnings(
            manifest_environment(stored.manifest), environment,
        ),
    )


@_user_errors
def pdf_evaluate_answerability(
    run: str,
    dataset: str,
    output: str,
) -> str:
    """Score model-only and pipeline decisions as separate metrics."""
    stored = load_pdf_answerability_run(Path(run))
    dataset_path = Path(dataset)
    golden = load_pdf_golden_dataset(dataset_path)
    if stored.manifest.dataset_sha256 != _sha256(dataset_path):
        raise ValueError("PDF answerability run dataset_sha256 differs")
    if stored.manifest.answerability_fingerprint != (
        stored.manifest.answerability.fingerprint
    ):
        raise ValueError("PDF answerability run fingerprint differs")
    require_supported_passage_policy(stored.manifest.passage_policy)
    model_only, pipeline = evaluate_pdf_answerability_run(golden, stored)
    current = current_pdf_run_environment()
    warnings = pdf_reproducibility_warnings(
        PdfRunEnvironment(
            stored.manifest.code_revision,
            stored.manifest.code_dirty,
            stored.manifest.environment,
        ),
        current,
    )
    report = {
        "run_identity": stored.manifest.identity,
        "reproducibility_warnings": list(warnings),
        "manifest": stored.manifest.model_dump(mode="json", by_alias=True),
        "model_only": _answerability_metrics(model_only),
        "pipeline": _answerability_metrics(pipeline),
        "cases": [case.model_dump(mode="json") for case in stored.cases],
    }
    _write_new(Path(output), json.dumps(report, indent=1))
    return _with_warnings(
        f"model-only accuracy {model_only.accuracy:.3f}, "
        f"pipeline accuracy {pipeline.accuracy:.3f}",
        warnings,
    )


def _answerability_metrics(
    evaluation: PdfAnswerabilityEvaluation,
) -> dict[str, object]:
    return {
        "accuracy": evaluation.accuracy,
        "answerable_acceptance_rate": evaluation.answerable_acceptance_rate,
        "not_found_rejection_rate": evaluation.not_found_rejection_rate,
        "balanced_accuracy": evaluation.balanced_accuracy,
        "false_refusals": [
            score.case_id for score in evaluation.case_scores
            if score.false_refusal
        ],
        "false_acceptances": [
            score.case_id for score in evaluation.case_scores
            if score.false_acceptance
        ],
    }


def _sha256_streamed(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _with_warnings(summary: str, warnings: Sequence[str]) -> str:
    if not warnings:
        return summary
    return "\n".join(
        [summary, *(f"warning: {item}" for item in warnings)]
    )


def _corpus(
    records: Sequence[PdfIndexRecord],
    outline: Sequence[PdfOutlineEntry],
    policy: PdfRetrievalPolicy,
) -> PdfRetrievalCorpus:
    return build_pdf_retrieval_corpus(
        records, classify_pdf_records(records, outline), policy,
    )


def _retrieval_configuration(
    kind: PdfRetrieverKind,
    top_k: int,
    semantic: SemanticEncoderConfig,
) -> PdfRetrievalConfiguration:
    uses_semantic = kind is not PdfRetrieverKind.BM25
    return PdfRetrievalConfiguration(
        retriever=kind,
        top_k=top_k,
        bm25=bm25_configuration(PdfBm25Baseline(()).parameters),
        semantic=semantic_configuration(semantic) if uses_semantic else None,
        rrf=(
            rrf_configuration(PDF_HYBRID_PARAMETERS)
            if kind is PdfRetrieverKind.HYBRID else None
        ),
    )


def _searcher(
    kind: PdfRetrieverKind,
    corpus: PdfRetrievalCorpus,
    semantic: SemanticEncoderConfig,
) -> _Search:
    texts = corpus.retrieval_texts
    if kind is PdfRetrieverKind.BM25:
        bm25 = PdfBm25Baseline(corpus.records, retrieval_texts=texts)
        return lambda query: tuple(
            hit.record.record_id for hit in bm25.search(query)
        )
    encoder = MiniLMEncoder(semantic)
    if kind is PdfRetrieverKind.MINILM:
        minilm = PdfMiniLmBaseline(
            corpus.records, encoder, retrieval_texts=texts,
        )
        return lambda query: tuple(
            hit.record.record_id for hit in minilm.search(query)
        )
    hybrid = PdfHybridBaseline(corpus.records, encoder, retrieval_texts=texts)
    return lambda query: tuple(
        hit.record.record_id for hit in hybrid.search(query)
    )


def _migration_summary(migration: PdfJudgmentMigration) -> str:
    return (
        f"{len(migration.migrated)} labels migrated, "
        f"{len(migration.pending)} pending review"
    )


def _migration_report(migration: PdfJudgmentMigration) -> dict[str, object]:
    return {
        "source": migration.source.model_dump(),
        "target_dataset_id": migration.target_dataset_id,
        "target_record_set_fingerprint":
            migration.target_record_set_fingerprint,
        "mappings": [
            {
                "source_record_id": item.source_record_id,
                "kind": item.kind.value,
                "target_record_id": item.target_record_id,
                "reason": item.reason,
                "candidates": [asdict(candidate) for candidate in
                               item.candidates],
            }
            for item in migration.mappings
        ],
        "migrated": [
            item.model_dump(mode="json") for item in migration.migrated
        ],
        "pending": [
            {
                "judgment": item.judgment.model_dump(mode="json"),
                "mapping_kind": item.mapping.kind.value,
                "candidate_record_id": item.mapping.target_record_id,
                "reason": item.reason,
            }
            for item in migration.pending
        ],
    }


def _review_queue(
    migration: PdfJudgmentMigration,
    dataset: PdfGoldenDataset,
    source_records: Sequence[PdfIndexRecord],
    target_records: Sequence[PdfIndexRecord],
) -> str:
    sources = {record.record_id: record for record in source_records}
    targets = {record.record_id: record for record in target_records}
    cases = {case.id: case for case in dataset.cases}
    lines = ["# PDF judgment review queue", ""]
    if not migration.pending:
        lines.append("No label requires review.")
    for item in migration.pending:
        judgment = item.judgment
        case = cases[judgment.case_id]
        evidence = case.evidence[judgment.evidence_index]
        source = sources[judgment.record_id]
        target = (
            targets.get(item.mapping.target_record_id)
            if item.mapping.target_record_id else None
        )
        lines.extend([
            f"## {judgment.case_id} / evidence {judgment.evidence_index} / "
            f"{judgment.record_id}",
            "",
            f"- Question: {case.question}",
            f"- Evidence requirement: `{evidence.role.value}`, "
            f"`{evidence.modality.value}`, physical pages "
            f"{', '.join(map(str, evidence.physical_pages))}",
            f"- Historical label: `{judgment.relevance.value}`",
            f"- Mapping: `{item.mapping.kind.value}` ({item.mapping.reason})",
            f"- Candidate record: `{item.mapping.target_record_id}`",
            "",
            "### Original text",
            "",
            *(f"    {line}" for line in source.text.splitlines()),
            "",
        ])
        if target is not None:
            lines.extend([
                "### New text",
                "",
                *(f"    {line}" for line in target.text.splitlines()),
                "",
                "### Difference",
                "",
                *_difference(source.text, target.text),
                "",
            ])
    return "\n".join(lines).rstrip() + "\n"


def _difference(source: str, target: str) -> list[str]:
    """Describe text added or removed around the historical text."""
    compact_source = "".join(source.split())
    compact_target = "".join(target.split())
    start = compact_target.find(compact_source)
    if start == -1:
        return ["- The historical text is not contained in the new text."]
    before = _expand(target, start, from_start=True)
    after = _expand(target, start + len(compact_source), from_start=False)
    return [
        f"- Added before: {before!r}" if before else "- Nothing added before",
        f"- Added after: {after!r}" if after else "- Nothing added after",
    ]


def _expand(text: str, compact_offset: int, *, from_start: bool) -> str:
    """Return the original text before or after a compact offset."""
    seen = 0
    for index, character in enumerate(text):
        if seen == compact_offset and not character.isspace():
            return text[:index].strip() if from_start else (
                text[index:].strip()
            )
        if not character.isspace():
            seen += 1
    return text.strip() if from_start else ""


def _load_decisions(path: Path) -> tuple[PdfReviewDecision, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        PdfReviewDecision(
            case_id=item["case_id"],
            evidence_index=int(item["evidence_index"]),
            source_record_id=item["source_record_id"],
            target_record_id=item.get("target_record_id"),
            relevance=(
                PdfRecordRelevance(item["relevance"])
                if item.get("relevance") else None
            ),
        )
        for item in data["decisions"]
    )


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, content: str) -> None:
    """Create a file; accept an existing one only if it is identical."""
    text = content.rstrip("\n") + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as error:
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Refusing to overwrite {path}") from error
