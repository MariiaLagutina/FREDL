"""Synthetic CLI integration and exclusive atomic artifact publication."""

from concurrent.futures import ThreadPoolExecutor
import errno
from hashlib import sha256
import json
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from src.__main__ import main
from src.cli import CliError
from src.evaluation.pdf.library import PdfLibraryCorpus
from src.evaluation.pdf.multibook.retrieval.cli import (
    pdf_run_multibook_retrieval,
    write_pdf_multibook_run,
)
from src.evaluation.pdf.multibook.retrieval.runs import (
    PdfMultibookRetrievalRun,
)
from src.retrieval.semantic.config import (
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_SEMANTIC_REVISION,
    SemanticEncoderConfig,
)

from tests.evaluation.pdf.multibook.retrieval import (
    contract_helpers as contract_fixtures,
    runner_helpers as runner_fixtures,
)
from tests.evaluation.pdf.multibook.retrieval.runner_helpers import (
    FakeEncoder,
    _run,
)


corpus = contract_fixtures.corpus
forbid_real_model_loading = runner_fixtures.forbid_real_model_loading
pdf_inputs = runner_fixtures.pdf_inputs


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
        "src.evaluation.pdf.multibook.retrieval.runner.MiniLMEncoder", fake,
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
            "src.evaluation.pdf.multibook.retrieval.cli.os.fsync", fail_sync,
        )
    else:
        def fail_link(source: Path, target: Path) -> None:
            if failure == "race":
                target.write_bytes(b"concurrent output")
            raise FileExistsError("Synthetic publish failure")
        monkeypatch.setattr(
            "src.evaluation.pdf.multibook.retrieval.cli.os.link", fail_link,
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


def test_unsupported_hard_links_report_safe_cli_error_and_clean_up(
    pdf_inputs: tuple[Path, Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, golden = pdf_inputs
    output = tmp_path / "run.json"

    def unsupported(source: Path, target: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, "SYNTHETIC-SENSITIVE-MARKER")

    monkeypatch.setattr(
        "src.evaluation.pdf.multibook.retrieval.cli.os.link", unsupported,
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
        "src.evaluation.pdf.multibook.retrieval.cli.os.link",
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
