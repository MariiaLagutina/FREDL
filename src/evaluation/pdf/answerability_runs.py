"""Run the local PDF answerability protocol over a stored retrieval run.

Answerability has its own manifest and metrics. It reads passages from a
retrieval run but never changes retrieval results or retrieval metrics.
"""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse
import urllib.request

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.evaluation.pdf.answerability import (
    apply_pdf_comparison_guard,
    build_pdf_answerability_messages,
    build_pdf_comparison_messages,
    evaluate_pdf_answerability,
    find_pdf_comparison_subjects,
    parse_pdf_answerability_decision,
    parse_pdf_comparison_decision,
    PdfAnswerabilityDecision,
    PdfAnswerabilityEvaluation,
    PdfAnswerabilityResult,
    PdfComparisonDecision,
)
from src.evaluation.pdf.manifest import (
    PDF_CONTEXT_RULE,
    PdfAnswerabilityConfiguration,
    PdfEvaluationManifest,
    PdfRunEnvironment,
)
from src.evaluation.pdf.models import PdfGoldenDataset
from src.evaluation.pdf.runs import PdfRetrievalRun
from src.generation.backend import ChatMessage
from src.ingestion.pdf.fingerprints import fingerprint_canonical_json
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy


PDF_COMPARISON_GUARD_VERSION = "subject-guard-v1+model-verifier-v1"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class PdfChatClient(Protocol):
    """Complete one chat request with a bounded number of tokens."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> str:
        """Return the generated text."""


class LocalChatCompletionClient:
    """Call an OpenAI-compatible chat server on this machine only.

    A loopback address keeps library passages inside the user's machine.
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        temperature: float,
        timeout_seconds: float = 120,
    ) -> None:
        """Reject servers outside the local machine before any request."""
        host = urlparse(url).hostname
        if host not in _LOOPBACK_HOSTS:
            raise ValueError("PDF answerability server must be on loopback")
        self._url = url
        self._model_name = model_name
        self._temperature = temperature
        self._timeout = timeout_seconds

    def complete(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> str:
        """Send one deterministic request and return the message content."""
        payload = json.dumps({
            "model": self._model_name,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": self._temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as reply:
            result = json.load(reply)
        return str(result["choices"][0]["message"]["content"])


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PdfAnswerabilityPassagePolicy(_StrictModel):
    """Describe which records the model reads and which of their text.

    Retrieval searches the corpus policy's searchable text, but the model
    reads the original record text, boilerplate included. Bump the version
    when passage selection or passage text changes.
    """

    version: Literal[1] = 1
    passage_text: Literal["original_record_text"] = "original_record_text"
    selection: Literal["ranked_top_k_then_context_deduplicated"] = (
        "ranked_top_k_then_context_deduplicated"
    )
    context_rule: str = Field(min_length=1)
    searchable_text_policy: PdfRetrievalPolicy


def pdf_answerability_passage_policy(
    retrieval: PdfEvaluationManifest,
) -> PdfAnswerabilityPassagePolicy:
    """Derive the passage policy, refusing an unsupported context rule."""
    if retrieval.retrieval.context_rule != PDF_CONTEXT_RULE:
        raise ValueError(
            "PDF answerability does not support context rule "
            f"{retrieval.retrieval.context_rule}",
        )
    return PdfAnswerabilityPassagePolicy(
        context_rule=retrieval.retrieval.context_rule,
        searchable_text_policy=retrieval.corpus_policy,
    )


def require_supported_passage_policy(
    policy: PdfAnswerabilityPassagePolicy,
) -> None:
    """Reject a stored run whose passages this code would build otherwise."""
    if policy.context_rule != PDF_CONTEXT_RULE:
        raise ValueError("PDF answerability run passage policy differs")


class PdfAnswerabilityManifest(_StrictModel):
    """Bind answerability decisions to one retrieval run and one model."""

    schema_name: Literal["fredl.pdf.answerability-run"] = Field(
        default="fredl.pdf.answerability-run", alias="schema",
    )
    version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_run_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: int = Field(gt=0)
    passage_policy: PdfAnswerabilityPassagePolicy
    answerability: PdfAnswerabilityConfiguration
    answerability_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_revision: str = Field(min_length=1)
    code_dirty: bool
    environment: dict[str, str]

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True,
    )

    @property
    def identity(self) -> str:
        """Hash the fields that define answerability compatibility."""
        return fingerprint_canonical_json({
            "schema": "fredl.pdf.answerability-run",
            "version": 1,
            "identity": self.model_dump(
                mode="json",
                exclude={"code_revision", "code_dirty", "environment"},
            ),
        })


class PdfAnswerabilityCaseRun(_StrictModel):
    """Keep every stage of one case's decision for audit."""

    case_id: str = Field(min_length=1)
    selected_record_ids: tuple[str, ...]
    context_record_ids: tuple[str, ...]
    comparison_subjects: tuple[str, str] | None
    model_decision: PdfAnswerabilityDecision
    after_subject_guard: PdfAnswerabilityDecision
    comparison_verifier: PdfComparisonDecision | None
    final_decision: PdfAnswerabilityDecision


class PdfAnswerabilityRun(_StrictModel):
    """Store answerability decisions with their own manifest."""

    manifest: PdfAnswerabilityManifest
    cases: tuple[PdfAnswerabilityCaseRun, ...]


def build_pdf_answerability_manifest(
    *,
    retrieval_run: PdfRetrievalRun,
    configuration: PdfAnswerabilityConfiguration,
    environment: PdfRunEnvironment,
) -> PdfAnswerabilityManifest:
    """Derive the answerability manifest from its retrieval run."""
    retrieval = retrieval_run.manifest
    return PdfAnswerabilityManifest(
        dataset_id=retrieval.dataset_id,
        dataset_sha256=retrieval.dataset_sha256,
        retrieval_run_identity=retrieval.identity,
        record_set_fingerprint=retrieval.record_set_fingerprint,
        corpus_fingerprint=retrieval.corpus_fingerprint,
        top_k=retrieval.retrieval.top_k,
        passage_policy=pdf_answerability_passage_policy(retrieval),
        answerability=configuration,
        answerability_fingerprint=configuration.fingerprint,
        code_revision=environment.code_revision,
        code_dirty=environment.code_dirty,
        environment=dict(environment.packages),
    )


def run_pdf_answerability(
    dataset: PdfGoldenDataset,
    retrieval_run: PdfRetrievalRun,
    records: Sequence[PdfIndexRecord],
    client: PdfChatClient,
    configuration: PdfAnswerabilityConfiguration,
) -> tuple[PdfAnswerabilityCaseRun, ...]:
    """Classify each case from its top-k records and neighbour context."""
    pdf_answerability_passage_policy(retrieval_run.manifest)
    texts: Mapping[str, str] = {
        record.record_id: record.text for record in records
    }
    results = {result.case_id: result for result in retrieval_run.results}
    top_k = retrieval_run.manifest.retrieval.top_k
    cases = []
    for case in dataset.cases:
        result = results[case.id]
        selected = tuple(dict.fromkeys((
            *result.ranked_record_ids[:top_k], *result.context_record_ids,
        )))
        passages = tuple(texts[record_id] for record_id in selected)
        model = parse_pdf_answerability_decision(client.complete(
            build_pdf_answerability_messages(case.question, passages),
            configuration.answerability_max_tokens,
        ))
        guarded = apply_pdf_comparison_guard(case.question, passages, model)
        subjects = find_pdf_comparison_subjects(case.question)
        verifier = None
        final = guarded
        if guarded is PdfAnswerabilityDecision.ANSWERABLE and subjects:
            verifier = parse_pdf_comparison_decision(client.complete(
                build_pdf_comparison_messages(case.question, passages),
                configuration.comparison_max_tokens,
            ))
            if verifier is PdfComparisonDecision.NOT_SUPPORTED:
                final = PdfAnswerabilityDecision.NOT_FOUND
        cases.append(PdfAnswerabilityCaseRun(
            case_id=case.id,
            selected_record_ids=selected,
            context_record_ids=result.context_record_ids,
            comparison_subjects=subjects,
            model_decision=model,
            after_subject_guard=guarded,
            comparison_verifier=verifier,
            final_decision=final,
        ))
    return tuple(cases)


def evaluate_pdf_answerability_run(
    dataset: PdfGoldenDataset,
    run: PdfAnswerabilityRun,
) -> tuple[PdfAnswerabilityEvaluation, PdfAnswerabilityEvaluation]:
    """Score the raw model label and the full pipeline separately."""
    model_only = evaluate_pdf_answerability(dataset, tuple(
        PdfAnswerabilityResult(case.case_id, case.model_decision)
        for case in run.cases
    ))
    pipeline = evaluate_pdf_answerability(dataset, tuple(
        PdfAnswerabilityResult(case.case_id, case.final_decision)
        for case in run.cases
    ))
    return model_only, pipeline


def write_pdf_answerability_run(path: Path, run: PdfAnswerabilityRun) -> None:
    """Write one answerability run without replacing an earlier one."""
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(run.model_dump_json(by_alias=True, indent=1) + "\n")
    except FileExistsError as error:
        raise ValueError(f"Refusing to overwrite {path}") from error


def load_pdf_answerability_run(path: Path) -> PdfAnswerabilityRun:
    """Read one stored answerability run."""
    try:
        return PdfAnswerabilityRun.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
    except OSError as error:
        raise ValueError("PDF answerability run could not be read") from error
    except ValidationError as error:
        raise ValueError("PDF answerability run JSON is invalid") from error
