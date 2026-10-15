"""Publish a complete review directory with Linux atomic no-replace rename."""

import ctypes
import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from src.evaluation.pdf.multibook.pooling.models import (
    PdfRetrievalJudgmentPool,
)


def _rename_new_directory(source: Path, target: Path) -> None:
    """Fail closed unless renameat2(RENAME_NOREPLACE) is available.

    Ordinary rename can replace an empty destination directory. There is no
    check-then-rename fallback, because that would race another writer.
    """
    if sys.platform != 'linux':
        raise OSError(errno.ENOTSUP, 'Atomic no-replace rename needed')
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, 'renameat2', None)
    if rename is None:
        raise OSError(errno.ENOTSUP, 'renameat2 is unavailable')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                       ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    # AT_FDCWD=-100; RENAME_NOREPLACE=1. Staging is beside the destination.
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def write_review_package(
    target: Path, files: dict[str, str], pool_fingerprint: str,
) -> None:
    """Validate staged bytes and publish the whole package without replacement.

    Requires Linux and filesystem support for RENAME_NOREPLACE, on the same
    filesystem. No fsync is performed; crash durability is not guaranteed.
    Abrupt process death can leave a hidden staging directory, never a
    partially written final directory.
    """
    if os.path.lexists(target):
        raise FileExistsError(errno.EEXIST, 'Output already exists')
    if any(Path(name).name != name or name == 'package-manifest.json'
           for name in files):
        raise ValueError('Invalid package filename')
    payloads = {name: text.encode('utf-8') for name, text in files.items()}
    if 'judgment-pool.json' in payloads:
        pool = PdfRetrievalJudgmentPool.model_validate_json(
            payloads['judgment-pool.json'],
        )
        if pool.fingerprint != pool_fingerprint:
            raise ValueError('Package pool fingerprint differs')
    hashes = {name: sha256(data).hexdigest()
              for name, data in payloads.items()}
    payloads['package-manifest.json'] = (json.dumps({
        'schema_name': 'fredl.pdf.multibook-pool-package', 'version': 1,
        'pool_fingerprint': pool_fingerprint, 'files': hashes,
    }, sort_keys=True, indent=1) + '\n').encode('utf-8')
    staging = Path(tempfile.mkdtemp(prefix='.fredl-pool-', dir=target.parent))
    try:
        for name, data in payloads.items():
            descriptor = os.open(
                staging / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
            )
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(data)
        # Check serialized bytes, including completion, before publication.
        for name, expected in payloads.items():
            if (staging / name).read_bytes() != expected:
                raise OSError(errno.EIO, 'Staged package bytes differ')
        _rename_new_directory(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
