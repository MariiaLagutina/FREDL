"""Fail-closed input validation and offline encoder safeguards."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.cli import CliError
from src.evaluation.pdf.library import (
    PdfLibraryBookInput,
    PdfLibraryCorpus,
    build_pdf_library_corpus,
)
from src.evaluation.pdf.manifest import PdfRetrieverKind
from src.evaluation.pdf.multibook_retrieval_cli import (
    pdf_run_multibook_retrieval,
)
from src.evaluation.pdf.multibook_runner import run_pdf_multibook_retrieval
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy
from src.retrieval.semantic.config import SemanticEncoderConfig
from src.retrieval.semantic.runtime import SemanticBackendLoadError

from tests.evaluation.pdf import (
    multibook_contract_helpers as contract_fixtures,
    multibook_runner_helpers as runner_fixtures,
)
from tests.evaluation.pdf.multibook_contract_helpers import ENV, HASH
from tests.evaluation.pdf.multibook_runner_helpers import FakeEncoder, _queries


corpus = contract_fixtures.corpus
forbid_real_model_loading = runner_fixtures.forbid_real_model_loading
pdf_inputs = runner_fixtures.pdf_inputs


@pytest.mark.parametrize("change", [
    "duplicate", "texts", "eligible", "policy",
])
def test_bad_corpus_fails_before_encoder_loading(
    corpus: PdfLibraryCorpus, change: str,
) -> None:
    book = corpus.books[0]
    if change == "duplicate":
        book = replace(book, records=(*book.records, book.records[0]))
    elif change == "texts":
        book = replace(book, corpus=replace(book.corpus, retrieval_texts=()))
    elif change == "eligible":
        book = replace(book, corpus=replace(
            book.corpus, records=(replace(
                book.records[0], record_id="unknown",
            ),),
        ))
    else:
        book = replace(book, corpus=replace(
            book.corpus, policy=PdfRetrievalPolicy.EXCLUDE_NAVIGATION,
        ))
    changed = replace(corpus, books=(book, *corpus.books[1:]))

    def forbidden(config: SemanticEncoderConfig) -> FakeEncoder:
        pytest.fail("Invalid input reached encoder loading")

    with pytest.raises(ValueError):
        run_pdf_multibook_retrieval(
            golden=_queries(corpus), golden_sha256=HASH, corpus=changed,
            retriever=PdfRetrieverKind.MINILM, encoder_factory=forbidden,
        )


@pytest.mark.parametrize("change", ["span", "section", "index_key"])
def test_invalid_source_metadata_fails_before_model_loading(
    corpus: PdfLibraryCorpus, change: str,
) -> None:
    inputs = []
    for book in corpus.books:
        records = book.records
        if book.book_path == "books/alpha.pdf":
            first = records[0]
            if change == "span":
                first = replace(first, flow_end=first.flow_end + 1)
            elif change == "section":
                first = replace(first, section_path=("",))
            else:
                records = (*records, replace(first, record_id="same-span"))
            records = (first, *records[1:])
        inputs.append(PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, records, (),
        ))
    changed = build_pdf_library_corpus(inputs, corpus.policy)

    def forbidden(config: SemanticEncoderConfig) -> FakeEncoder:
        pytest.fail("Invalid source metadata reached model loading")

    with pytest.raises(ValueError):
        run_pdf_multibook_retrieval(
            golden=_queries(changed), golden_sha256=HASH, corpus=changed,
            retriever=PdfRetrieverKind.MINILM, encoder_factory=forbidden,
        )


@pytest.mark.parametrize("kind", ["minilm", "hybrid"])
def test_missing_local_model_fails_safely_without_output(
    pdf_inputs: tuple[Path, Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    root, golden = pdf_inputs

    def unavailable(config: SemanticEncoderConfig) -> FakeEncoder:
        assert config.local_files_only is True
        raise SemanticBackendLoadError("SYNTHETIC-SENSITIVE-MARKER")

    monkeypatch.setattr(
        "src.evaluation.pdf.multibook_runner.MiniLMEncoder", unavailable,
    )
    output = tmp_path / "run.json"
    with pytest.raises(CliError, match="downloads are disabled") as error:
        pdf_run_multibook_retrieval(
            str(root), str(golden), kind, str(output),
        )
    assert "SYNTHETIC-SENSITIVE-MARKER" not in str(error.value)
    assert not output.exists()
    assert not list(tmp_path.glob(".fredl-run-*"))


@pytest.mark.parametrize("kind", ["minilm", "hybrid"])
def test_falsey_encoder_cannot_trigger_network_enabled_legacy_fallback(
    corpus: PdfLibraryCorpus, kind: str,
) -> None:
    class FalseyEncoder(FakeEncoder):
        def __bool__(self) -> bool:
            return False

    with pytest.raises(ValueError, match="falsey"):
        run_pdf_multibook_retrieval(
            golden=_queries(corpus), golden_sha256=HASH, corpus=corpus,
            retriever=PdfRetrieverKind(kind), environment=ENV,
            encoder_factory=lambda config: FalseyEncoder(),
        )


@pytest.mark.parametrize("kind", ["minilm", "hybrid"])
def test_falsey_factory_is_used_without_loading_a_real_encoder(
    corpus: PdfLibraryCorpus, kind: str,
) -> None:
    class FalseyFactory:
        def __bool__(self) -> bool:
            return False

        def __call__(self, config: SemanticEncoderConfig) -> FakeEncoder:
            assert config.local_files_only is True
            return FakeEncoder()

    run = run_pdf_multibook_retrieval(
        golden=_queries(corpus), golden_sha256=HASH, corpus=corpus,
        retriever=PdfRetrieverKind(kind), environment=ENV,
        encoder_factory=FalseyFactory(),
    )
    assert len(run.results[0].ranking) == len(corpus.records)
