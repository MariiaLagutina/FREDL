"""Global retrieval and atomic CLI tests using synthetic books and encoders."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import errno
from hashlib import sha256
import json
import math
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest
import torch
from torch import Tensor

from src.__main__ import main
from src.cli import CliError
from src.evaluation.pdf.bm25_baseline import PdfBm25Baseline
from src.evaluation.pdf.library import (
    build_pdf_library_corpus,
    load_pdf_library_corpus,
    PdfLibraryBookInput,
    PdfLibraryCorpus,
)
from src.evaluation.pdf.manifest import PdfRetrieverKind
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook_queries import PdfQueryMode
from src.evaluation.pdf.multibook_retrieval_cli import (
    pdf_run_multibook_retrieval,
    write_pdf_multibook_run,
)
from src.evaluation.pdf.multibook_runner import run_pdf_multibook_retrieval
from src.evaluation.pdf.multibook_runs import PdfMultibookRetrievalRun
from src.ingestion.pdf.retrieval_corpus import (
    build_pdf_retrieval_corpus,
    PdfRecordClassification,
    PdfRecordRole,
    PdfRetrievalPolicy,
)
from src.retrieval.semantic.config import (
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_SEMANTIC_REVISION,
    SemanticEncoderConfig,
)
from src.retrieval.semantic.runtime import SemanticBackendLoadError
from src.retrieval.tokenization import TextTokenizer
from tests.evaluation.pdf.test_multibook_import import _write_pdf
from tests.evaluation.pdf.test_multibook_judgments import (
    _golden, _library, _records,
)
from tests.evaluation.pdf.test_multibook_retrieval_contracts import (
    ENV,
    HASH,
)


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: tuple[str, ...]) -> Tensor:
        self.calls.append(texts)
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "caches" in lowered:
                vectors.append((0.0, 1.0, 0.0))
            elif "queues" in lowered:
                vectors.append((1.0, 0.0, 0.0))
            else:
                vectors.append((0.0, 0.0, 1.0))
        return torch.tensor(vectors, dtype=torch.float32)


@pytest.fixture(autouse=True)
def forbid_real_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Real model loading is forbidden in runner tests")

    monkeypatch.setattr(
        "src.retrieval.semantic.encoder.load_semantic_backend", forbidden,
    )


@pytest.fixture()
def corpus() -> PdfLibraryCorpus:
    inputs = []
    for book in _library().books:
        records = book.records
        if book.book_path == "books/alpha.pdf":
            records = (records[0], replace(
                records[1], flow_start=records[0].flow_end,
                flow_end=records[0].flow_end + len(records[1].text),
            ), records[2])
        inputs.append(PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, records, (),
        ))
    for index, path in enumerate(("books/delta.pdf", "books/gamma.pdf")):
        inputs.append(PdfLibraryBookInput(
            path, str(index) * 64,
            _records(path, (
                (f"extra-{index}", "Synthetic background.", 1, 0),
            )), (),
        ))
    return build_pdf_library_corpus(inputs, PdfRetrievalPolicy.ALL_RECORDS)


def _queries(
    corpus: PdfLibraryCorpus, first: str = "queues",
) -> PdfMultibookGolden:
    data = _golden(corpus).model_dump(mode="json")
    data["cases"][0]["question"] = first
    data["cases"][1]["question"] = "queues or caches?"
    for item, text in zip(data["cases"][1]["interpretations"],
                          ("queues", "caches")):
        item["clarified_question"] = text
    data["cases"][2]["question"] = "zxqmissing"
    return PdfMultibookGolden.model_validate_json(json.dumps(data))


def _run(
    corpus: PdfLibraryCorpus,
    kind: str = "bm25",
    first: str = "queues",
    mode: PdfQueryMode = PdfQueryMode.ALL,
    encoder: FakeEncoder | None = None,
) -> PdfMultibookRetrievalRun:
    return run_pdf_multibook_retrieval(
        golden=_queries(corpus, first), golden_sha256=HASH, corpus=corpus,
        retriever=PdfRetrieverKind(kind), query_mode=mode, environment=ENV,
        encoder_factory=lambda config: encoder or FakeEncoder(),
    )


def test_bm25_statistics_and_scores_are_global(
    corpus: PdfLibraryCorpus,
) -> None:
    result = _run(corpus, first="work").results[0]
    assert [hit.record_id for hit in result.ranking] == ["a1"]
    tokenize = TextTokenizer().tokenize
    tokens = [tokenize(text) for text in corpus.retrieval_texts]
    length = len(tokens[0])
    average = sum(map(len, tokens)) / len(tokens)
    idf = math.log(1 + (len(tokens) - 1 + 0.5) / 1.5)
    expected = idf * 2.4 / (1 + 1.4 * (0.35 + 0.65 * length / average))
    assert result.ranking[0].score == pytest.approx(expected)
    single = PdfBm25Baseline(corpus.books[0].corpus.records).search("work")[0]
    assert result.ranking[0].score != pytest.approx(single.score)


def test_other_book_can_outrank_the_first_evidence_book(
    corpus: PdfLibraryCorpus,
) -> None:
    result = _run(corpus, first="work caches keep").results[0]
    assert result.ranking[0].record_id == "b1"
    assert {hit.record_id for hit in result.ranking} == {"a1", "b1"}


@pytest.mark.parametrize("kind", ["minilm", "hybrid"])
def test_global_semantic_inputs_are_exact_corpus_and_query_texts(
    corpus: PdfLibraryCorpus, kind: str,
) -> None:
    encoder = FakeEncoder()
    run = _run(corpus, kind, first="caches", encoder=encoder)
    assert encoder.calls[0] == corpus.retrieval_texts
    assert encoder.calls[1:] == [
        (query.text,) for query in run.manifest.query_plan.queries
    ]
    assert run.results[0].ranking[0].record_id == "b1"
    assert len(run.results[0].ranking) == len(corpus.records)
    assert len({record.book_path for record in corpus.records}) == 4


def test_hybrid_uses_global_source_ranks(corpus: PdfLibraryCorpus) -> None:
    lexical = _run(corpus, "bm25", first="caches").results[0].ranking
    semantic = _run(corpus, "minilm", first="caches").results[0].ranking
    fused = _run(corpus, "hybrid", first="caches").results[0].ranking
    lex = {hit.record_id: rank for rank, hit in enumerate(lexical, 1)}
    sem = {hit.record_id: rank for rank, hit in enumerate(semantic, 1)}
    for hit in fused:
        assert hit.lexical_rank == lex.get(hit.record_id)
        assert hit.semantic_rank == sem[hit.record_id]
        expected = 1 / (60 + sem[hit.record_id])
        if hit.record_id in lex:
            expected += 1 / (60 + lex[hit.record_id])
        assert hit.score == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["bm25", "minilm", "hybrid"])
def test_repeated_global_runs_preserve_order_and_fingerprints(
    corpus: PdfLibraryCorpus, kind: str,
) -> None:
    first = _run(corpus, kind, first="background")
    inputs = [PdfLibraryBookInput(
        book.book_path, book.pdf_sha256, tuple(reversed(book.records)), (),
    ) for book in reversed(corpus.books)]
    reordered = build_pdf_library_corpus(inputs, corpus.policy)
    second = _run(reordered, kind, first="background")
    assert first == second
    assert first.results_fingerprint == second.results_fingerprint
    if kind == "bm25":
        assert [hit.record_id for hit in first.results[0].ranking] == [
            "extra-0", "extra-1",
        ]


@pytest.mark.parametrize("mode,count", [
    (PdfQueryMode.ORIGINAL, 3), (PdfQueryMode.CLARIFIED, 2),
    (PdfQueryMode.ALL, 5),
])
def test_query_modes_and_meanings_remain_independent(
    corpus: PdfLibraryCorpus, mode: PdfQueryMode, count: int,
) -> None:
    run = _run(corpus, "minilm", mode=mode)
    assert len(run.results) == count
    variants = {result.query.interpretation_id: result
                for result in run.results
                if result.query.interpretation_id is not None}
    if mode is not PdfQueryMode.ORIGINAL:
        assert variants["queues"].ranking[0].record_id == "a1"
        assert variants["producers"].ranking[0].record_id == "b1"


def test_empty_bm25_ranking_is_not_an_answerability_label(
    corpus: PdfLibraryCorpus,
) -> None:
    result = _run(corpus).results[-1]
    assert result.ranking == ()
    assert all(not context.records for context in result.context)
    assert "answerability" not in result.model_dump()


def _policy_corpus(corpus: PdfLibraryCorpus, mask: bool) -> PdfLibraryCorpus:
    policy = (PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE if mask
              else PdfRetrievalPolicy.EXCLUDE_NAVIGATION)
    books = tuple(replace(book, corpus=build_pdf_retrieval_corpus(
        book.records,
        tuple(PdfRecordClassification(
            record.record_id,
            PdfRecordRole.INDEX if record.record_id == "a2"
            else PdfRecordRole.CONTENT,
            ((0, 6),) if mask and record.record_id == "a1" else (),
        ) for record in book.records), policy,
    )) for book in corpus.books)
    return replace(corpus, policy=policy, books=books)


def test_policy_excludes_navigation_and_masks_only_searchable_text(
    corpus: PdfLibraryCorpus,
) -> None:
    filtered = _policy_corpus(corpus, False)
    assert _run(filtered).results[0].context[0].records == ()
    assert _run(filtered, first="workers").results[0].ranking == ()
    masked = _policy_corpus(corpus, True)
    assert _run(masked).results[0].ranking == ()
    assert masked.source_records[0].text == corpus.source_records[0].text


@pytest.mark.parametrize("change", ["gap", "flow", "interval", "unknown"])
def test_context_stays_within_book_flow_interval_and_touching_ranges(
    corpus: PdfLibraryCorpus, change: str,
) -> None:
    inputs = []
    for book in corpus.books:
        records = book.records
        if book.book_path == "books/alpha.pdf":
            fields: dict[str, Any] = {}
            if change == "gap":
                fields = {"flow_start": records[1].flow_start + 1,
                          "flow_end": records[1].flow_end + 1}
            elif change == "flow":
                fields = {"flow_index": 1}
            else:
                fields = {"section_interval_id": None if change == "unknown"
                          else "different-interval"}
            records = (records[0], replace(records[1], **fields), records[2])
        inputs.append(PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, records, (),
        ))
    changed = build_pdf_library_corpus(inputs, corpus.policy)
    assert _run(changed).results[0].context[0].records == ()


def test_direct_context_parents_do_not_cross_books_or_expand_recursively(
    corpus: PdfLibraryCorpus,
) -> None:
    result = _run(corpus).results[0]
    for context in result.context:
        assert [(hit.record_id, hit.parent_record_ids)
                for hit in context.records] == [
            ("a2", ("a1",)),
        ]


def test_top5_and_top10_have_separate_context(
    corpus: PdfLibraryCorpus,
) -> None:
    inputs = []
    for book in corpus.books:
        records = book.records
        if book.book_path == "books/alpha.pdf":
            chain = []
            cursor = 0
            for index in range(11):
                text = ("marker " * (12 - index)).strip()
                record = _records(book.book_path, (
                    (f"chain-{index}", text, 1, cursor),
                ))[0]
                chain.append(replace(record, flow_start=cursor,
                                     flow_end=cursor + len(text)))
                cursor += len(text)
            records = tuple(chain)
        inputs.append(PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, records, (),
        ))
    changed = build_pdf_library_corpus(inputs, corpus.policy)
    result = _run(changed, first="marker").results[0]
    assert [hit.record_id for hit in result.ranking] == [
        f"chain-{index}" for index in range(11)
    ]
    assert result.context[0].records[0].record_id == "chain-5"
    assert result.context[0].records[0].parent_record_ids == ("chain-4",)
    assert result.context[1].records[0].record_id == "chain-10"
    assert result.context[1].records[0].parent_record_ids == ("chain-9",)


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


@pytest.fixture()
def pdf_inputs(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "library"
    books = {
        "books/alpha.pdf": (
            "Queues hold work.", "Backpressure slows producers.",
        ),
        "books/beta.pdf": (
            "Caches keep answers.", "Eviction removes entries.",
        ),
        "books/delta.pdf": ("Synthetic delta background.",),
        "books/gamma.pdf": ("Synthetic gamma background.",),
    }
    for path, pages in books.items():
        _write_pdf(root / path, pages)
    corpus = load_pdf_library_corpus(root, sorted(books),
                                     PdfRetrievalPolicy.ALL_RECORDS)
    golden = tmp_path / "golden.json"
    golden.write_text(_queries(corpus).model_dump_json(indent=1))
    # Discovery of this unpinned, invalid file would break the workflow.
    (root / "books" / "not-pinned.pdf").write_bytes(b"not a PDF")
    return root, golden


@pytest.mark.parametrize("kind", ["bm25", "minilm", "hybrid"])
def test_four_pdf_cli_workflow_is_offline_and_writes_valid_run(
    pdf_inputs: tuple[Path, Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    root, golden = pdf_inputs
    configurations = []

    def fake(config: SemanticEncoderConfig) -> FakeEncoder:
        assert config.local_files_only is True
        assert config.model_name == DEFAULT_SEMANTIC_MODEL
        assert config.model_revision == DEFAULT_SEMANTIC_REVISION
        configurations.append(config)
        return FakeEncoder()

    monkeypatch.setattr(
        "src.evaluation.pdf.multibook_runner.MiniLMEncoder", fake,
    )
    output = tmp_path / "run.json"
    summary = pdf_run_multibook_retrieval(
        str(root), str(golden), kind, str(output),
    )
    raw = output.read_bytes()
    run = PdfMultibookRetrievalRun.model_validate_json(raw)
    assert len(run.manifest.library.books) == 4
    assert run.manifest.golden_sha256 == sha256(
        golden.read_bytes(),
    ).hexdigest()
    assert f"artifact SHA256: {sha256(raw).hexdigest()}" in summary
    assert run.results_fingerprint in summary
    assert len(configurations) == (0 if kind == "bm25" else 1)
    assert not list(tmp_path.glob(".fredl-run-*"))
    before = raw
    with pytest.raises(CliError, match="new file"):
        pdf_run_multibook_retrieval(str(root), str(golden), kind, str(output))
    assert output.read_bytes() == before


@pytest.mark.parametrize("change", ["hash", "records", "path", "clarified"])
def test_invalid_inputs_fail_before_loading_models_and_leave_no_output(
    pdf_inputs: tuple[Path, Path], tmp_path: Path, change: str,
) -> None:
    root, golden = pdf_inputs
    data = json.loads(golden.read_bytes())
    if change == "hash":
        data["library"]["books"][0]["pdf_sha256"] = "0" * 64
    elif change == "records":
        data["library"]["books"][0]["record_set_fingerprint"] = "0" * 64
    elif change == "path":
        data["library"]["books"][0]["book_path"] = "../outside.pdf"
    else:
        data["cases"][1]["interpretations"][0]["clarified_question"] = None
    golden.write_text(json.dumps(data))
    output = tmp_path / "run.json"
    with pytest.raises(CliError) as error:
        pdf_run_multibook_retrieval(
            str(root), str(golden), "minilm", str(output),
        )
    assert str(root) not in str(error.value)
    assert "exact_anchor" not in str(error.value)
    assert not output.exists()
    assert not list(tmp_path.glob(".fredl-run-*"))


@pytest.mark.parametrize("flag,value", [
    ("retriever", "unknown"), ("query_mode", "unknown"), ("policy", "unknown"),
])
def test_invalid_cli_options_fail_without_reading_inputs(
    flag: str, value: str,
) -> None:
    args = dict(library_root="missing", golden="missing", retriever="bm25",
                output="missing/run.json")
    args[flag] = value
    with pytest.raises(CliError, match="Invalid"):
        pdf_run_multibook_retrieval(**args)


@pytest.mark.parametrize("failure", ["write", "link", "race"])
def test_atomic_publication_never_leaves_partial_or_overwrites_racing_output(
    corpus: PdfLibraryCorpus, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    run = _run(corpus)
    output = tmp_path / "run.json"
    if failure == "write":
        def fail_sync(fd: int) -> None:
            raise OSError("Synthetic flush failure")
        monkeypatch.setattr(
            "src.evaluation.pdf.multibook_retrieval_cli.os.fsync", fail_sync,
        )
    else:
        def fail_link(source: Path, target: Path) -> None:
            if failure == "race":
                target.write_bytes(b"concurrent output")
            raise FileExistsError("Synthetic publish failure")
        monkeypatch.setattr(
            "src.evaluation.pdf.multibook_retrieval_cli.os.link", fail_link,
        )
    with pytest.raises(OSError):
        write_pdf_multibook_run(output, run)
    if failure == "race":
        assert output.read_bytes() == b"concurrent output"
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".fredl-run-*"))


def test_cli_is_registered_without_running_retrieval(capsys: Any) -> None:
    with pytest.raises(SystemExit) as result:
        main(["pdf_run_multibook_retrieval", "--help"])
    assert result.value.code == 0
    captured = capsys.readouterr()
    assert "query_mode" in captured.out + captured.err


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


def test_public_cli_named_arguments_execute_synthetic_runner(
    pdf_inputs: tuple[Path, Path], tmp_path: Path, capsys: Any,
) -> None:
    root, golden = pdf_inputs
    output = tmp_path / "named-run.json"
    main([
        "pdf_run_multibook_retrieval", f"--library_root={root}",
        f"--golden={golden}", "--retriever=bm25", f"--output={output}",
        "--query_mode=clarified", "--policy=all_records",
    ])
    run = PdfMultibookRetrievalRun.model_validate_json(output.read_bytes())
    assert len(run.results) == 2
    assert all(result.query.interpretation_id is not None
               for result in run.results)
    assert "artifact SHA256:" in capsys.readouterr().out


@pytest.mark.parametrize("shape", ["directory", "missing_parent", "symlink"])
def test_bad_output_is_rejected_before_reading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str,
) -> None:
    output = tmp_path
    if shape == "missing_parent":
        output = tmp_path / "missing" / "run.json"
    elif shape == "symlink":
        output = tmp_path / "run.json"
        output.symlink_to(tmp_path / "absent-target.json")

    def forbidden(path: Path) -> bytes:
        pytest.fail("Output preflight should reject before reading inputs")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(CliError, match="new file"):
        pdf_run_multibook_retrieval(
            "unused", "unused", "bm25", str(output),
        )


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


def test_hybrid_does_not_truncate_global_candidates_to_twenty() -> None:
    inputs = []
    for book_index, path in enumerate((
        "books/alpha.pdf", "books/beta.pdf",
        "books/delta.pdf", "books/gamma.pdf",
    )):
        records = []
        for index in range(book_index, 41, 4):
            text = ("marker " * (42 - index)).strip()
            record = _records(path, ((f"doc-{index}", text, 1, 0),))[0]
            records.append(replace(record, flow_index=index))
        inputs.append(PdfLibraryBookInput(
            path, "abcd"[book_index] * 64, tuple(records), (),
        ))
    corpus = build_pdf_library_corpus(inputs, PdfRetrievalPolicy.ALL_RECORDS)

    class ReverseEncoder(FakeEncoder):
        def encode(self, texts: tuple[str, ...]) -> Tensor:
            vectors = []
            for text in texts:
                count = text.split().count("marker")
                x = 1.0 if count <= 1 else (43 - count) / 43
                vectors.append((x, math.sqrt(1 - x * x)))
            return torch.tensor(vectors, dtype=torch.float32)

    run = _run(corpus, "hybrid", first="marker", encoder=ReverseEncoder())
    ranking = run.results[0].ranking
    assert len(ranking) == 41
    # This record is outside both top-20 lists, yet contributes to full RRF.
    middle = next(hit for hit in ranking if hit.record_id == "doc-20")
    assert middle.lexical_rank == middle.semantic_rank == 21
    assert middle.score == pytest.approx(2 / 81)


def test_unsupported_hard_links_report_safe_cli_error_and_clean_up(
    pdf_inputs: tuple[Path, Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, golden = pdf_inputs
    output = tmp_path / "run.json"

    def unsupported(source: Path, target: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, "SYNTHETIC-SENSITIVE-MARKER")

    monkeypatch.setattr(
        "src.evaluation.pdf.multibook_retrieval_cli.os.link", unsupported,
    )
    with pytest.raises(CliError, match="hard-link support") as error:
        pdf_run_multibook_retrieval(
            str(root), str(golden), "bm25", str(output),
        )
    assert "SYNTHETIC-SENSITIVE-MARKER" not in str(error.value)
    assert not output.exists()
    assert not list(tmp_path.glob(".fredl-run-*"))


def test_two_concurrent_writers_publish_exactly_one_complete_artifact(
    corpus: PdfLibraryCorpus, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    run = _run(corpus)
    output = tmp_path / "run.json"
    barrier = Barrier(2)
    real_link = os.link

    def synchronized_link(source: Path, target: Path) -> None:
        barrier.wait(timeout=5)
        real_link(source, target)

    monkeypatch.setattr(
        "src.evaluation.pdf.multibook_retrieval_cli.os.link",
        synchronized_link,
    )

    def publish() -> str | None:
        try:
            return write_pdf_multibook_run(output, run)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        receipts = list(workers.map(lambda _: publish(), range(2)))
    hashes = [receipt for receipt in receipts if receipt is not None]
    assert hashes == [sha256(output.read_bytes()).hexdigest()]
    assert PdfMultibookRetrievalRun.model_validate_json(
        output.read_bytes(),
    ) == run
    assert not list(tmp_path.glob(".fredl-run-*"))
