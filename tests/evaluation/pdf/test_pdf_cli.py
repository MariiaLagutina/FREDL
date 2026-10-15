"""End-to-end tests for the PDF evaluation commands on a synthetic book."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import pytest

from src.cli import CliError
from src.evaluation.pdf.cli import (
    pdf_evaluate_answerability,
    pdf_evaluate_retrieval,
    pdf_migrate_judgments,
    pdf_run_answerability,
    pdf_run_retrieval,
    pdf_snapshot_records,
)
from src.evaluation.pdf.runs import load_pdf_retrieval_run
from src.evaluation.pdf.snapshots import load_pdf_record_snapshot
from src.ingestion.pdf.fingerprints import legacy_fingerprint_pdf_records_v1


BOOK = "books/example.pdf"
PAGES = (
    "Caching keeps recent answers close to the reader.",
    "Queues hold work until a worker is free to take it.",
    "Retries repeat a failed request after a short pause.",
)


def _write_book(path: Path) -> None:
    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    }))
    for text in PAGES:
        page = writer.add_blank_page(width=400, height=400)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        content = DecodedStreamObject()
        content.set_data(
            f"BT /F1 11 Tf 20 300 Td ({text}) Tj ET".encode("ascii")
        )
        page[NameObject("/Contents")] = writer._add_object(content)
    path.parent.mkdir(parents=True)
    with path.open("wb") as stream:
        writer.write(stream)


def _write_dataset(path: Path, dataset_id: str) -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "dataset_id": dataset_id,
        "book_path": BOOK,
        "page_numbering": "physical_pdf_page",
        "language": "en",
        "cases": [
            {
                "id": "case-1",
                "question": "What do queues hold?",
                "why_now": "I size workers.",
                "expected_answer": "Work until a worker is free.",
                "answerability": "complete",
                "evidence": [{"physical_pages": [2], "role": "required",
                              "modality": "text"}],
                "tags": ["queues"],
            },
            {
                "id": "case-2",
                "question": "Which database engine is recommended?",
                "why_now": "I choose storage.",
                "expected_answer": "The book does not say.",
                "answerability": "not_found",
                "evidence": [],
                "tags": ["negative"],
            },
        ],
    }), encoding="utf-8")


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    library = tmp_path / "library"
    _write_book(library / BOOK)
    private = tmp_path / "private"
    private.mkdir()
    _write_dataset(private / "golden-v1.json", "golden-v1")
    _write_dataset(private / "golden-v2.json", "golden-v2")
    pdf_snapshot_records(str(library), BOOK, str(private / "old.json"))
    _, records = load_pdf_record_snapshot(private / "old.json")
    (private / "judgments-v1.json").write_text(json.dumps({
        "schema_version": 1,
        "dataset_id": "golden-v1",
        "book_path": BOOK,
        "record_set_fingerprint": legacy_fingerprint_pdf_records_v1(records),
        "judgments": [{
            "case_id": "case-1",
            "evidence_index": 0,
            "record_id": records[0].record_id,
            "relevance": "required",
        }],
    }), encoding="utf-8")
    return tmp_path


def _migrate(workspace: Path) -> str:
    private = workspace / "private"
    return pdf_migrate_judgments(
        str(private / "judgments-v1.json"),
        str(private / "old.json"),
        str(private / "golden-v1.json"),
        str(private / "golden-v2.json"),
        str(workspace / "library"),
        str(private / "migration"),
    )


def test_migration_writes_new_judgments_without_touching_sources(
    workspace: Path,
) -> None:
    private = workspace / "private"
    before = (private / "judgments-v1.json").read_bytes()

    summary = _migrate(workspace)

    assert summary == (
        "1 labels migrated automatically, 0 decided by review, "
        "1 labels written"
    )
    assert (private / "judgments-v1.json").read_bytes() == before
    migrated = json.loads(
        (private / "migration" / "record-judgments-v2.json").read_text(),
    )
    assert migrated["schema_version"] == 2
    assert migrated["dataset_id"] == "golden-v2"
    assert migrated["judgments"][0]["provenance"]["mapping"] == (
        "byte_identical"
    )
    assert _migrate(workspace) == summary
    report = private / "migration" / "migration-report.json"
    report.write_text(report.read_text() + " ", encoding="utf-8")
    with pytest.raises(CliError, match="Refusing to overwrite"):
        _migrate(workspace)


def test_run_and_evaluation_reject_changed_inputs(workspace: Path) -> None:
    private = workspace / "private"
    library = str(workspace / "library")
    _migrate(workspace)
    run = private / "bm25-run.json"

    pdf_run_retrieval(library, str(private / "golden-v2.json"), "bm25",
                      str(run))
    pdf_evaluate_retrieval(
        str(run), str(private / "golden-v2.json"),
        str(private / "migration" / "record-judgments-v2.json"),
        library, str(private / "bm25-eval.json"),
    )

    report = json.loads((private / "bm25-eval.json").read_text())
    assert set(report) >= {
        "human_labeled_relevance", "evidence_coverage",
        "page_overlap_diagnostic", "manifest",
    }
    assert report["manifest"]["corpus_policy"] == (
        "exclude_navigation_and_boilerplate"
    )
    assert report["human_labeled_relevance"]["mean_required_recall_at_k"] == (
        1.0
    )
    with (private / "golden-v2.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(CliError, match="dataset_sha256"):
        pdf_evaluate_retrieval(
            str(run), str(private / "golden-v2.json"),
            str(private / "migration" / "record-judgments-v2.json"),
            library, str(private / "bm25-eval-2.json"),
        )


class _LabelHandler(BaseHTTPRequestHandler):
    """Answer like a local classifier: only the queue question is found."""

    def do_POST(self) -> None:  # noqa: N802 - required handler name
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        content = request["messages"][-1]["content"]
        question = content.split("<question>")[-1]
        label = "ANSWERABLE" if "queues" in question else "NOT_FOUND"
        body = json.dumps(
            {"choices": [{"message": {"content": label}}]},
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Keep test output quiet."""


def test_answerability_run_and_metrics_stay_separate(workspace: Path) -> None:
    private = workspace / "private"
    library = str(workspace / "library")
    dataset = str(private / "golden-v2.json")
    retrieval = private / "bm25-run.json"
    pdf_run_retrieval(library, dataset, "bm25", str(retrieval))
    model = private / "model.gguf"
    model.write_bytes(b"synthetic weights")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LabelHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        pdf_run_answerability(
            str(retrieval), dataset, library, str(model),
            str(private / "answerability-run.json"),
            server_url=(
                f"http://127.0.0.1:{server.server_address[1]}"
                "/v1/chat/completions"
            ),
        )
    finally:
        server.shutdown()
        server.server_close()

    summary = pdf_evaluate_answerability(
        str(private / "answerability-run.json"), dataset,
        str(private / "answerability-eval.json"),
    )

    report = json.loads((private / "answerability-eval.json").read_text())
    assert summary.startswith(
        "model-only accuracy 1.000, pipeline accuracy 1.000"
    )
    assert set(report) >= {"model_only", "pipeline", "manifest"}
    assert "human_labeled_relevance" not in report
    assert report["manifest"]["retrieval_run_identity"] == (
        load_pdf_retrieval_run(retrieval).manifest.identity
    )
    assert report["manifest"]["passage_policy"]["passage_text"] == (
        "original_record_text"
    )
    assert report["manifest"]["passage_policy"]["context_rule"] == (
        "section-interval-touching-neighbors-v1"
    )
    assert report["manifest"]["answerability"]["model_sha256"] == (
        hashlib.sha256(b"synthetic weights").hexdigest()
    )
    with pytest.raises(CliError, match="loopback"):
        pdf_run_answerability(
            str(retrieval), dataset, library, str(model),
            str(private / "remote.json"),
            server_url="http://192.0.2.1:8080/v1/chat/completions",
        )
