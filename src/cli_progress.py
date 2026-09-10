"""Provide delayed terminal status without coupling it to RAG logic."""

from collections.abc import Iterator
from contextlib import contextmanager
import sys
from threading import Timer
from typing import TextIO


DEFAULT_STATUS_DELAY_SECONDS = 5.0


@contextmanager
def delayed_status(
    message: str,
    *,
    delay_seconds: float = DEFAULT_STATUS_DELAY_SECONDS,
    stream: TextIO | None = None,
) -> Iterator[None]:
    """Print a status only when the wrapped operation remains unfinished."""
    if not message.strip():
        raise ValueError("Delayed status message must not be empty")
    if delay_seconds < 0:
        raise ValueError("Delayed status delay must not be negative")

    output = stream or sys.stderr
    timer = Timer(
        delay_seconds,
        _print_status,
        args=(message, output),
    )
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def _print_status(message: str, stream: TextIO) -> None:
    """Write one flushable line from the timer callback."""
    print(message, file=stream, flush=True)
