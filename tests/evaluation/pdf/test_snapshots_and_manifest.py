"""Tests for record snapshots and evaluation run compatibility."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.evaluation.pdf.manifest import (
    bm25_configuration,
    build_pdf_evaluation_manifest,
    PdfEvaluationManifest,
    PdfRetrievalConfiguration,
    PdfRetrieverKind,
    require_pdf_manifest_compatibility,
    require_same_pdf_evaluation_identity,
    rrf_configuration,
    semantic_configuration,
)
from src.evaluation.pdf.models import PdfGoldenDataset
from src.evaluation.pdf.snapshots import (
    load_pdf_record_snapshot,
    snapshot_pdf_records,
    write_pdf_record_snapshot,
)
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord
from src.ingestion.pdf.retrieval_corpus import (
    build_pdf_retrieval_corpus,
    classify_pdf_records,
    PdfRetrievalCorpus,
    PdfRetrievalPolicy,
)
from src.retrieval.bm25 import BM25Parameters
from src.retrieval.hybrid import RRFParameters
from src.retrieval.semantic.config import SemanticEncoderConfig


BOOK = "books/example.pdf"


def _records() -> tuple[PdfIndexRecord, ...]:
    texts = ("Caches expire after an interval.", "Queues hold work.")
    records = []
    start = 0
    for index, text in enumerate(texts):
        records.append(PdfIndexRecord(
            record_id=f"pdf:{index}",
            text=text,
            book_path=BOOK,
            section_path=("Guide",),
            flow_index=0,
            flow_start=start,
            flow_end=start + len(text),
            page_spans=(PdfChunkPageSpan(index + 1, 0, len(text), 0,
                                         len(text)),),
            section_interval_id="pdf-section:a",
        ))
        start += len(text)
    return tuple(records)


def test_snapshot_round_trip_keeps_records_and_fingerprints(
    tmp_path: Path,
) -> None:
    records = _records()
    path = tmp_path / "snapshot.json"
    snapshot = snapshot_pdf_records(tuple(reversed(records)),
                                    code_revision="abc")

    write_pdf_record_snapshot(path, snapshot)
    loaded, restored = load_pdf_record_snapshot(path)

    assert restored == records
    assert loaded.record_set_fingerprint_v2 == (
        snapshot.record_set_fingerprint_v2
    )
    with pytest.raises(ValueError, match="already exists"):
        write_pdf_record_snapshot(path, snapshot)


def test_tampered_snapshot_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    write_pdf_record_snapshot(
        path, snapshot_pdf_records(_records(), code_revision="abc"),
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["records"][0]["text"] = "Caches never expire."
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint differs"):
        load_pdf_record_snapshot(path)


def _dataset() -> PdfGoldenDataset:
    return PdfGoldenDataset.model_validate_json(json.dumps({
        "schema_version": 1, "dataset_id": "golden-v2", "book_path": BOOK,
        "page_numbering": "physical_pdf_page", "language": "en",
        "cases": [{
            "id": "case-1", "question": "When do caches expire?",
            "why_now": "Tuning.", "expected_answer": "After an interval.",
            "answerability": "complete",
            "evidence": [{"physical_pages": [1], "role": "required",
                          "modality": "text"}],
            "tags": ["cache"],
        }],
    }))


def _corpus(
    records: tuple[PdfIndexRecord, ...],
    policy: PdfRetrievalPolicy = PdfRetrievalPolicy.EXCLUDE_NAVIGATION,
) -> PdfRetrievalCorpus:
    return build_pdf_retrieval_corpus(
        records, classify_pdf_records(records, ()), policy,
    )


def _configuration(
    bm25: BM25Parameters | None = None,
) -> PdfRetrievalConfiguration:
    return PdfRetrievalConfiguration(
        retriever=PdfRetrieverKind.HYBRID,
        top_k=5,
        bm25=bm25_configuration(bm25 or BM25Parameters()),
        semantic=semantic_configuration(SemanticEncoderConfig()),
        rrf=rrf_configuration(RRFParameters(lexical_weight=1.0)),
    )


def _manifest(
    environment: dict[str, str] | None = None,
    configuration: PdfRetrievalConfiguration | None = None,
) -> PdfEvaluationManifest:
    records = _records()
    return build_pdf_evaluation_manifest(
        dataset=_dataset(),
        dataset_sha256="d" * 64,
        records=records,
        corpus=_corpus(records),
        retrieval=configuration or _configuration(),
        code_revision="abc",
        code_dirty=False,
        environment=environment or {"python": "3.13"},
    )


def test_manifest_accepts_its_own_inputs() -> None:
    records = _records()
    manifest = build_pdf_evaluation_manifest(
        dataset=_dataset(), dataset_sha256="d" * 64, records=records,
        corpus=_corpus(records), retrieval=_configuration(),
        code_revision="abc", code_dirty=False, environment={},
    )

    require_pdf_manifest_compatibility(
        manifest, dataset=_dataset(), dataset_sha256="d" * 64,
        records=tuple(reversed(records)), corpus=_corpus(records),
    )
    assert manifest.corpus_policy is PdfRetrievalPolicy.EXCLUDE_NAVIGATION


@pytest.mark.parametrize(
    "field",
    ["dataset_sha256", "record_set_fingerprint", "corpus_fingerprint"],
)
def test_manifest_rejects_changed_inputs(field: str) -> None:
    records = _records()
    manifest = build_pdf_evaluation_manifest(
        dataset=_dataset(), dataset_sha256="d" * 64, records=records,
        corpus=_corpus(records), retrieval=_configuration(),
        code_revision="abc", code_dirty=False, environment={},
    )
    changed = (records[0], replace(records[1], text="Queues hold jobs."))
    arguments = {
        "dataset_sha256": ("e" * 64, records, _corpus(records)),
        "record_set_fingerprint": ("d" * 64, changed, _corpus(records)),
        "corpus_fingerprint": (
            "d" * 64, records,
            replace(_corpus(records), retrieval_texts=(
                "Caches expire after an interval.", "                 ",
            )),
        ),
    }[field]

    with pytest.raises(ValueError, match=field):
        require_pdf_manifest_compatibility(
            manifest, dataset=_dataset(), dataset_sha256=arguments[0],
            records=arguments[1], corpus=arguments[2],
        )


def test_identity_ignores_environment_but_not_configuration() -> None:
    base = _manifest()
    other_environment = _manifest(environment={"python": "3.14"})
    other_parameters = _manifest(
        configuration=_configuration(BM25Parameters(k1=1.2)),
    )

    require_same_pdf_evaluation_identity(base, other_environment)
    with pytest.raises(ValueError, match="not directly comparable"):
        require_same_pdf_evaluation_identity(base, other_parameters)
