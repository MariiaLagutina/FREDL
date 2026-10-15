"""Synthetic packages, artifact hashes and exclusive output writes."""

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from src.__main__ import main
from src.cli import CliError
from src.evaluation.pdf.multibook_pool_cli import (
    load_pdf_pooling_runs, pdf_pool_multibook_judgments,
)
from src.evaluation.pdf.multibook_pool_models import PdfRetrievalJudgmentPool
from tests.evaluation.pdf import multibook_pool_helpers as pool_helpers
from tests.evaluation.pdf import multibook_runner_helpers as runner


pdf_inputs = runner.pdf_inputs
package_inputs = pool_helpers.package_inputs
forbid_real_model_loading = runner.forbid_real_model_loading


def test_public_command_writes_complete_package_with_blank_tasks(
    package_inputs: tuple[Path, Path, Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: Any,
) -> None:
    root, golden, runs = package_inputs
    before = {path: path.read_bytes() for path in (golden, runs)}

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('Pooling may not execute retrieval')
    monkeypatch.setattr(
        'src.evaluation.pdf.bm25_baseline.PdfBm25Baseline.search', forbidden,
    )
    output = tmp_path / 'review'
    main(['pdf_pool_multibook_judgments', str(golden), str(root), str(runs),
          str(output), '--depth=10'])
    assert 'blank tasks' in capsys.readouterr().out
    completion = json.loads((output / 'package-manifest.json').read_bytes())
    assert set(completion['files']) == {
        'judgment-pool.json', 'worksheet.md', 'worksheet.csv',
        'pool-summary.md',
    }
    for name, digest in completion['files'].items():
        assert sha256((output / name).read_bytes()).hexdigest() == digest
        assert (output / name).stat().st_mode & 0o777 == 0o600
    pool = PdfRetrievalJudgmentPool.model_validate_json(
        (output / 'judgment-pool.json').read_bytes(),
    )
    assert completion['pool_fingerprint'] == pool.fingerprint
    assert all(path.read_bytes() == data for path, data in before.items())
    assert output.stat().st_mode & 0o777 == 0o700
    snapshot = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(CliError, match='never reused'):
        pdf_pool_multibook_judgments(str(golden), str(root), str(runs),
                                     str(output))
    assert snapshot == {p.name: p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize('change', [
    'missing', 'duplicate', 'bad_hash', 'bytes', 'missing_file', 'alias',
    'wrong_kind', 'bad_version', 'extra', 'golden_bytes', 'pdf_bytes',
])
def test_bad_artifacts_fail_before_output_creation(
    package_inputs: tuple[Path, Path, Path], tmp_path: Path, change: str,
) -> None:
    root, golden, manifest = package_inputs
    data = json.loads(manifest.read_bytes())
    if change == 'missing':
        data['runs'].pop()
    elif change == 'duplicate':
        data['runs'][1] = data['runs'][0]
    elif change == 'bad_hash':
        data['runs'][0]['sha256'] = '0' * 64
    elif change == 'bytes':
        with (manifest.parent / data['runs'][0]['path']).open('a') as handle:
            handle.write('\n')
    elif change == 'missing_file':
        data['runs'][0]['path'] = 'absent.json'
    elif change == 'alias':
        data['runs'][1]['path'] = data['runs'][0]['path']
    elif change == 'wrong_kind':
        a, b = data['runs'][:2]
        a['path'], b['path'] = b['path'], a['path']
        a['sha256'], b['sha256'] = b['sha256'], a['sha256']
    elif change == 'bad_version':
        data['version'] = True
    elif change == 'extra':
        data['recover'] = True
    elif change == 'golden_bytes':
        with golden.open('a') as handle:
            handle.write('\n')
    else:
        (root / 'books' / 'alpha.pdf').write_bytes(b'changed')
    manifest.write_text(json.dumps(data))
    output = tmp_path / 'review'
    with pytest.raises(CliError) as error:
        pdf_pool_multibook_judgments(str(golden), str(root), str(manifest),
                                     str(output))
    assert str(tmp_path) not in str(error.value)
    assert not output.exists()


@pytest.mark.parametrize('shape', ['exists', 'symlink', 'parent', 'depth'])
def test_output_preflight_does_not_read_private_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str,
) -> None:
    output = tmp_path / 'review'
    if shape == 'exists':
        output.mkdir()
    elif shape == 'symlink':
        output.symlink_to(tmp_path / 'absent')
    elif shape == 'parent':
        output = tmp_path / 'absent' / 'review'

    def forbidden(path: Path) -> bytes:
        pytest.fail('Invalid output must fail before reading inputs')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    with pytest.raises(CliError):
        pdf_pool_multibook_judgments('unused', 'unused', 'unused', str(output),
                                     depth=6 if shape == 'depth' else 10)


def test_run_paths_resolve_relative_to_manifest_and_order_is_irrelevant(
    package_inputs: tuple[Path, Path, Path], tmp_path: Path,
) -> None:
    _, _, path = package_inputs
    original = load_pdf_pooling_runs(path)
    data = json.loads(path.read_bytes())
    data['runs'].reverse()
    for ref in data['runs']:
        ref['path'] = str(path.parent / ref['path'])
    other = tmp_path / 'absolute.json'
    other.write_text(json.dumps(data))
    assert load_pdf_pooling_runs(other) == tuple(reversed(original))
