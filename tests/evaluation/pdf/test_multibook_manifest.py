"""Multi-book configuration, fingerprints and manifest compatibility."""

import json
from typing import Any

from pydantic import ValidationError
import pytest

from src.evaluation.pdf.library import (
    PdfLibraryBookInput,
    PdfLibraryCorpus,
    build_pdf_library_corpus,
)
from src.evaluation.pdf.multibook_manifest import (
    PdfMultibookRetrievalConfiguration,
    PdfMultibookRetrievalManifest,
    require_pdf_multibook_manifest_matches,
)
from src.evaluation.pdf.multibook_runs import (
    PdfMultibookRankedHit,
    PdfMultibookRetrievalRun,
    build_pdf_multibook_run,
)
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy

from tests.evaluation.pdf import (
    multibook_contract_helpers as contract_fixtures,
)
from tests.evaluation.pdf.multibook_contract_helpers import (
    HASH,
    _check,
    _config,
    _manifest,
    _run,
)
from tests.evaluation.pdf.multibook_golden_helpers import _golden


corpus = contract_fixtures.corpus
no_retrieval_or_model_loading = contract_fixtures.no_retrieval_or_model_loading


def test_algorithm_identity_is_distinct_but_comparison_is_shared(
    corpus: PdfLibraryCorpus,
) -> None:
    manifests = [_manifest(corpus, kind)
                 for kind in ("bm25", "minilm", "hybrid")]
    assert len({manifest.identity for manifest in manifests}) == 3
    assert len({manifest.comparison_identity for manifest in manifests}) == 1
    assert len(corpus.books) == 4


def test_canonical_hashes_ignore_json_layout_and_environment(
    corpus: PdfLibraryCorpus,
) -> None:
    run = _run(corpus)
    data = run.model_dump(mode="json")
    loaded = PdfMultibookRetrievalRun.model_validate_json(
        json.dumps(data, sort_keys=True, indent=4),
    )
    assert loaded.results_fingerprint == run.results_fingerprint
    assert loaded.manifest.query_plan.fingerprint == (
        run.manifest.query_plan.fingerprint
    )
    data["manifest"]["code_revision"] = "different-revision"
    data["manifest"]["code_dirty"] = True
    data["manifest"]["environment"] = {"python": "different"}
    changed = PdfMultibookRetrievalRun.model_validate_json(json.dumps(data))
    assert changed.manifest.identity == run.manifest.identity
    assert changed.results_fingerprint == run.results_fingerprint
    _check(changed, corpus)


def test_configuration_does_not_determine_results(
    corpus: PdfLibraryCorpus,
) -> None:
    first = _run(corpus)
    changed = tuple(result.model_copy(update={"ranking": (
        PdfMultibookRankedHit(record_id="a1", score=3.0),
    )}) for result in first.results)
    second = build_pdf_multibook_run(first.manifest, changed)
    assert first.manifest_identity == second.manifest_identity
    assert first.results_fingerprint != second.results_fingerprint
    _check(first, corpus)
    _check(second, corpus)


@pytest.mark.parametrize("field", [
    "golden_sha256", "golden_content_fingerprint", "corpus_fingerprint",
    "library_fingerprint", "query_plan_fingerprint", "dataset_id",
    "eligible_record_count", "books",
])
def test_manifest_rejects_drift_and_inconsistent_metadata(
    corpus: PdfLibraryCorpus, field: str,
) -> None:
    manifest = _manifest(corpus)
    data = manifest.model_dump(mode="json")
    if field == "eligible_record_count":
        data[field] += 1
    elif field == "books":
        data[field][0]["corpus_fingerprint"] = "0" * 64
    else:
        data[field] = "different" if field == "dataset_id" else "0" * 64
    with pytest.raises(ValueError):
        changed = PdfMultibookRetrievalManifest.model_validate_json(
            json.dumps(data),
        )
        require_pdf_multibook_manifest_matches(
            changed, golden=_golden(corpus), golden_sha256=HASH,
            corpus=corpus, query_plan=manifest.query_plan,
            retrieval=manifest.retrieval,
        )


@pytest.mark.parametrize("field,value", [
    ("bm25", None), ("semantic", {}),
    ("rrf", {"rank_constant": 60, "lexical_weight": 1.0,
             "semantic_weight": 1.0}),
])
def test_incompatible_algorithm_fields_are_rejected(
    field: str, value: Any,
) -> None:
    data = _config().model_dump(mode="json")
    data[field] = value
    with pytest.raises(ValidationError):
        PdfMultibookRetrievalConfiguration.model_validate_json(
            json.dumps(data),
        )


def test_expected_retrieval_configuration_cannot_silently_change(
    corpus: PdfLibraryCorpus,
) -> None:
    manifest = _manifest(corpus)
    with pytest.raises(ValueError, match="manifest differs"):
        require_pdf_multibook_manifest_matches(
            manifest, golden=_golden(corpus), golden_sha256=HASH,
            corpus=corpus, query_plan=manifest.query_plan,
            retrieval=_config("hybrid"),
        )


def test_library_order_is_canonical_but_policy_drift_is_incompatible(
    corpus: PdfLibraryCorpus,
) -> None:
    inputs = [PdfLibraryBookInput(
        book.book_path, book.pdf_sha256, tuple(reversed(book.records)), (),
    ) for book in reversed(corpus.books)]
    reordered = build_pdf_library_corpus(inputs, corpus.policy)
    assert _manifest(reordered).identity == _manifest(corpus).identity
    changed = build_pdf_library_corpus(
        inputs, PdfRetrievalPolicy.EXCLUDE_NAVIGATION,
    )
    assert changed.manifest == corpus.manifest
    assert _manifest(changed).comparison_identity != (
        _manifest(corpus).comparison_identity
    )
    with pytest.raises(ValueError, match="manifest differs"):
        _check(_run(corpus), changed)


@pytest.mark.parametrize("field,value", [
    ("k1", 0.0), ("b", 1.1), ("metadata_weight", -1.0),
    ("identifier_weight", float("nan")), ("tokenization", "unknown"),
])
def test_invalid_lexical_configuration_is_rejected(
    field: str, value: Any,
) -> None:
    data = _config().model_dump(mode="json")
    data["bm25"][field] = value
    with pytest.raises(ValidationError):
        PdfMultibookRetrievalConfiguration.model_validate_json(
            json.dumps(data),
        )


@pytest.mark.parametrize("field,value", [
    ("runtime", "unknown"), ("model_revision", "  "), ("max_length", 0),
])
def test_invalid_semantic_configuration_is_rejected(
    field: str, value: Any,
) -> None:
    data = _config("minilm").model_dump(mode="json")
    data["semantic"][field] = value
    with pytest.raises(ValidationError):
        PdfMultibookRetrievalConfiguration.model_validate_json(
            json.dumps(data),
        )


@pytest.mark.parametrize("field,value", [
    ("rank_constant", 0), ("lexical_weight", -1.0),
    ("semantic_weight", float("inf")),
])
def test_invalid_fusion_configuration_is_rejected(
    field: str, value: Any,
) -> None:
    data = _config("hybrid").model_dump(mode="json")
    data["rrf"][field] = value
    with pytest.raises(ValidationError):
        PdfMultibookRetrievalConfiguration.model_validate_json(
            json.dumps(data),
        )


def test_compatibility_revalidates_copied_nested_configuration(
    corpus: PdfLibraryCorpus,
) -> None:
    manifest = _manifest(corpus, "minilm")
    assert manifest.retrieval.semantic is not None
    semantic = manifest.retrieval.semantic.model_copy(update={"max_length": 0})
    retrieval = manifest.retrieval.model_copy(update={"semantic": semantic})
    copied = manifest.model_copy(update={"retrieval": retrieval})
    with pytest.raises(ValidationError):
        require_pdf_multibook_manifest_matches(
            copied, golden=_golden(corpus), golden_sha256=HASH,
            corpus=corpus, query_plan=manifest.query_plan, retrieval=retrieval,
        )
