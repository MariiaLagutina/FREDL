"""Tests for the stored PDF answerability protocol and its manifest."""

from collections.abc import Sequence
import json

from pydantic import ValidationError
import pytest

from src.evaluation.pdf.answerability import (
    PdfAnswerabilityDecision,
    PdfComparisonDecision,
)
from src.evaluation.pdf.answerability_runs import (
    build_pdf_answerability_manifest,
    evaluate_pdf_answerability_run,
    LocalChatCompletionClient,
    PdfAnswerabilityPassagePolicy,
    PdfAnswerabilityRun,
    require_supported_passage_policy,
    run_pdf_answerability,
)
from src.evaluation.pdf.manifest import (
    bm25_configuration,
    PDF_CONTEXT_RULE,
    build_pdf_evaluation_manifest,
    PdfAnswerabilityConfiguration,
    PdfRetrievalConfiguration,
    PdfRetrieverKind,
    pdf_reproducibility_warnings,
    PdfRunEnvironment,
)
from src.evaluation.pdf.models import PdfGoldenDataset
from src.evaluation.pdf.runs import PdfRetrievalRun, PdfStoredRetrievalResult
from src.generation.backend import ChatMessage
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.ingestion.pdf.retrieval_corpus import (
    build_pdf_retrieval_corpus,
    classify_pdf_records,
    PdfRetrievalPolicy,
)
from src.retrieval.bm25 import BM25Parameters


BOOK = "books/example.pdf"
TEXTS = (
    "Queues hold work until a worker is free.",
    "Caches keep recent answers close to the reader.",
    "Retries repeat a failed request after a pause.",
)


class _ScriptedClient:
    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = list(labels)
        self.requests: list[tuple[str, int]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> str:
        self.requests.append((messages[-1].content, max_tokens))
        return self.labels.pop(0)


def _records() -> tuple[PdfIndexRecord, ...]:
    records = []
    start = 0
    for index, text in enumerate(TEXTS):
        records.append(PdfIndexRecord(
            record_id=f"pdf:{index}", text=text, book_path=BOOK,
            section_path=("Guide",), flow_index=0, flow_start=start,
            flow_end=start + len(text),
            page_spans=(PdfChunkPageSpan(index + 1, 0, len(text), 0,
                                         len(text)),),
            section_interval_id="pdf-section:a",
        ))
        start += len(text)
    return tuple(records)


def _dataset() -> PdfGoldenDataset:
    def case(identifier: str, question: str, answerability: str) -> object:
        return {
            "id": identifier, "question": question, "why_now": "Testing.",
            "expected_answer": "Synthetic.", "answerability": answerability,
            "evidence": [] if answerability == "not_found" else [
                {"physical_pages": [1], "role": "required",
                 "modality": "text"},
            ],
            "tags": ["synthetic"],
        }

    return PdfGoldenDataset.model_validate_json(json.dumps({
        "schema_version": 1, "dataset_id": "golden-v2", "book_path": BOOK,
        "page_numbering": "physical_pdf_page", "language": "en",
        "cases": [
            case("answerable", "What do queues hold?", "complete"),
            case("comparison",
                 "What advantages do caches have over retries?",
                 "not_found"),
            case("absent", "Which database is recommended?", "not_found"),
        ],
    }))


def _retrieval_run() -> PdfRetrievalRun:
    records = _records()
    corpus = build_pdf_retrieval_corpus(
        records, classify_pdf_records(records, ()),
        PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE,
    )
    manifest = build_pdf_evaluation_manifest(
        dataset=_dataset(), dataset_sha256="d" * 64, records=records,
        corpus=corpus,
        retrieval=PdfRetrievalConfiguration(
            retriever=PdfRetrieverKind.BM25, top_k=1,
            bm25=bm25_configuration(BM25Parameters()),
        ),
        code_revision="abc", code_dirty=False, environment={},
    )
    return PdfRetrievalRun(manifest=manifest, results=(
        PdfStoredRetrievalResult(case_id="answerable",
                                 ranked_record_ids=("pdf:0", "pdf:1"),
                                 context_record_ids=("pdf:1",),
                                 abstained=False),
        PdfStoredRetrievalResult(case_id="comparison",
                                 ranked_record_ids=("pdf:1", "pdf:0"),
                                 context_record_ids=("pdf:2",),
                                 abstained=False),
        PdfStoredRetrievalResult(case_id="absent",
                                 ranked_record_ids=("pdf:2",),
                                 context_record_ids=(), abstained=False),
    ))


def _configuration() -> PdfAnswerabilityConfiguration:
    return PdfAnswerabilityConfiguration(
        model="local-model", model_sha256="e" * 64, runtime="test",
        server_args=("--reasoning", "off"), temperature=0.0,
        answerability_max_tokens=3, comparison_max_tokens=1,
        prompt_version="pdf-answerability-v1", guard_version="v1",
    )


def test_protocol_keeps_model_guard_and_verifier_stages() -> None:
    client = _ScriptedClient(["ANSWERABLE", "ANSWERABLE", "NO", "ANSWERABLE"])

    cases = run_pdf_answerability(
        _dataset(), _retrieval_run(), _records(), client, _configuration(),
    )

    answerable, comparison, absent = cases
    assert answerable.selected_record_ids == ("pdf:0", "pdf:1")
    assert "Queues hold work" in client.requests[0][0]
    assert client.requests[0][1] == 3
    assert comparison.model_decision is PdfAnswerabilityDecision.ANSWERABLE
    assert comparison.comparison_verifier is (
        PdfComparisonDecision.NOT_SUPPORTED
    )
    assert comparison.final_decision is PdfAnswerabilityDecision.NOT_FOUND
    assert client.requests[2][1] == 1
    assert absent.final_decision is PdfAnswerabilityDecision.ANSWERABLE


def test_model_only_and_pipeline_metrics_are_separate() -> None:
    client = _ScriptedClient(["ANSWERABLE", "ANSWERABLE", "NO", "NOT_FOUND"])
    retrieval = _retrieval_run()
    run = PdfAnswerabilityRun(
        manifest=build_pdf_answerability_manifest(
            retrieval_run=retrieval,
            configuration=_configuration(),
            environment=PdfRunEnvironment("abc", False, {}),
        ),
        cases=run_pdf_answerability(
            _dataset(), retrieval, _records(), client, _configuration(),
        ),
    )

    model_only, pipeline = evaluate_pdf_answerability_run(_dataset(), run)

    assert model_only.not_found_rejection_rate == 0.5
    assert pipeline.not_found_rejection_rate == 1.0
    assert run.manifest.retrieval_run_identity == retrieval.manifest.identity
    assert run.manifest.identity != retrieval.manifest.identity


def test_manifest_versions_the_model_visible_passage_policy() -> None:
    retrieval = _retrieval_run()
    manifest = build_pdf_answerability_manifest(
        retrieval_run=retrieval,
        configuration=_configuration(),
        environment=PdfRunEnvironment("abc", False, {}),
    )

    assert manifest.model_dump(mode="json")["passage_policy"] == {
        "version": 1,
        "passage_text": "original_record_text",
        "selection": "ranked_top_k_then_context_deduplicated",
        "context_rule": "section-interval-touching-neighbors-v1",
        "searchable_text_policy": "exclude_navigation_and_boilerplate",
    }
    assert manifest.passage_policy.context_rule == PDF_CONTEXT_RULE
    changed = manifest.model_copy(update={
        "passage_policy": manifest.passage_policy.model_copy(
            update={"searchable_text_policy": PdfRetrievalPolicy.ALL_RECORDS},
        ),
    })
    assert changed.identity != manifest.identity
    with pytest.raises(ValidationError):
        PdfAnswerabilityPassagePolicy.model_validate({
            "passage_text": "searchable_text", "context_rule": "x",
            "searchable_text_policy": "all_records",
        })


def test_passages_use_original_record_text_not_searchable_text() -> None:
    records = _records()
    run = _retrieval_run()
    client = _ScriptedClient(["NOT_FOUND"] * 3)

    run_pdf_answerability(_dataset(), run, records, client, _configuration())

    prompt = client.requests[0][0]
    assert TEXTS[0] in prompt and TEXTS[1] in prompt
    assert prompt.index(TEXTS[0]) < prompt.index(TEXTS[1])


def test_unsupported_context_rule_is_refused() -> None:
    retrieval = _retrieval_run()
    manifest = retrieval.manifest
    other = retrieval.model_copy(update={
        "manifest": manifest.model_copy(update={
            "retrieval": manifest.retrieval.model_copy(
                update={"context_rule": "same-section-path-v0"},
            ),
        }),
    })

    with pytest.raises(ValueError, match="context rule"):
        run_pdf_answerability(
            _dataset(), other, _records(), _ScriptedClient([]),
            _configuration(),
        )
    with pytest.raises(ValueError, match="passage policy differs"):
        require_supported_passage_policy(PdfAnswerabilityPassagePolicy(
            context_rule="same-section-path-v0",
            searchable_text_policy=PdfRetrievalPolicy.ALL_RECORDS,
        ))


def test_answerability_server_must_be_on_loopback() -> None:
    with pytest.raises(ValueError, match="loopback"):
        LocalChatCompletionClient("http://192.0.2.1:8080/v1", "model", 0.0)


def test_warns_about_code_and_dependency_drift() -> None:
    recorded = PdfRunEnvironment("abc", False, {"torch": "2.4", "pypdf": "6"})
    same = PdfRunEnvironment("abc", False, {"torch": "2.4", "pypdf": "6"})
    drifted = PdfRunEnvironment("def", True, {"torch": "2.5", "pypdf": "6"})

    assert pdf_reproducibility_warnings(recorded, same) == ()
    assert pdf_reproducibility_warnings(recorded, drifted) == (
        "code revision differs: abc vs def",
        "other run used uncommitted code changes",
        "torch version differs: 2.4 vs 2.5",
    )
