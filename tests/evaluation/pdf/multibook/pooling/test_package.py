"""Atomic package visibility, no-replace publication and staging cleanup."""

from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from src.evaluation.pdf.multibook.pooling import package as publication
from src.evaluation.pdf.multibook.pooling.package import write_review_package


def test_write_failure_cleans_only_own_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / 'review'
    real_open = os.open

    def fail(
        path: Any, flags: int, mode: int = 0o777, **kwargs: Any,
    ) -> int:
        if Path(path).name == 'second.md':
            raise OSError('Synthetic failure')
        return real_open(path, flags, mode, **kwargs)
    monkeypatch.setattr(
        'src.evaluation.pdf.multibook.pooling.package.os.open', fail,
    )
    with pytest.raises(OSError):
        write_review_package(output, {'first.md': 'one', 'second.md': 'two'},
                             'a' * 64)
    assert not output.exists()


def test_concurrent_packages_never_overwrite_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / 'review'
    barrier = Barrier(2)
    real_rename = publication._rename_new_directory

    def synchronized(staging: Path, target: Path) -> None:
        barrier.wait(timeout=5)
        real_rename(staging, target)
    monkeypatch.setattr(publication, '_rename_new_directory', synchronized)

    def write(text: str) -> bool:
        try:
            write_review_package(
                output, {'worksheet.csv': text, 'worksheet.md': text},
                'a' * 64,
            )
            return True
        except FileExistsError:
            return False
    with ThreadPoolExecutor(max_workers=2) as workers:
        result = list(workers.map(write, ['first', 'second']))
    assert sorted(result) == [False, True]
    assert (output / 'worksheet.csv').read_text() in {'first', 'second'}
    assert (output / 'worksheet.csv').read_bytes() == (
        output / 'worksheet.md'
    ).read_bytes()
    assert (output / 'package-manifest.json').exists()
    assert not list(tmp_path.glob('.fredl-pool-*'))


@pytest.mark.parametrize('failure', ['rename', 'corruption', 'unsupported'])
def test_final_path_is_absent_until_complete_verified_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    output = tmp_path / 'review'
    real_rename = publication._rename_new_directory

    def inspect(staging: Path, target: Path) -> None:
        assert not target.exists()
        completion = json.loads(
            (staging / 'package-manifest.json').read_bytes(),
        )
        assert set(completion['files']) == {'worksheet.csv'}
        assert (staging / 'worksheet.csv').read_text() == 'complete'
        if failure in {'rename', 'unsupported'}:
            code = errno.EIO if failure == 'rename' else errno.ENOTSUP
            raise OSError(code, 'Synthetic publication failure')
        real_rename(staging, target)
    monkeypatch.setattr(publication, '_rename_new_directory', inspect)
    if failure == 'corruption':
        real_read = Path.read_bytes

        def corrupt(path: Path) -> bytes:
            data = real_read(path)
            return data + b'bad' if path.name == 'worksheet.csv' else data
        monkeypatch.setattr(Path, 'read_bytes', corrupt)
    with pytest.raises(OSError):
        write_review_package(output, {'worksheet.csv': 'complete'}, 'a' * 64)
    assert not output.exists()
    assert not list(tmp_path.glob('.fredl-pool-*'))


def test_atomic_publish_never_replaces_even_empty_concurrent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / 'review'
    real_rename = publication._rename_new_directory

    def race(staging: Path, target: Path) -> None:
        target.mkdir()
        real_rename(staging, target)
    monkeypatch.setattr(publication, '_rename_new_directory', race)
    with pytest.raises(FileExistsError):
        write_review_package(output, {'worksheet.csv': 'ours'}, 'a' * 64)
    assert output.is_dir() and not list(output.iterdir())
    assert not list(tmp_path.glob('.fredl-pool-*'))


def test_success_is_a_single_complete_directory_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / 'review'
    real_rename = publication._rename_new_directory
    calls = []

    def inspect(staging: Path, target: Path) -> None:
        assert staging.parent == target.parent
        assert not target.exists()
        assert {path.name for path in staging.iterdir()} == {
            'worksheet.csv', 'worksheet.md', 'package-manifest.json',
        }
        calls.append(True)
        real_rename(staging, target)
    monkeypatch.setattr(publication, '_rename_new_directory', inspect)
    write_review_package(output, {'worksheet.csv': 'one',
                                  'worksheet.md': 'two'}, 'a' * 64)
    assert calls == [True]
    assert not list(tmp_path.glob('.fredl-pool-*'))
