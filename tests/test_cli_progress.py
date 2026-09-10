"""Tests for delayed terminal progress feedback."""

from io import StringIO
from unittest.mock import patch

import pytest

from src.cli_progress import DEFAULT_STATUS_DELAY_SECONDS, delayed_status


def test_default_status_delay_avoids_noise_for_fast_operations() -> None:
    """The user sees progress only after five seconds without a result."""
    assert DEFAULT_STATUS_DELAY_SECONDS == 5.0


def test_delayed_status_starts_daemon_timer_and_always_cancels() -> None:
    """The status lifecycle cannot keep the process alive after completion."""
    output = StringIO()

    with patch("src.cli_progress.Timer") as timer_class:
        timer = timer_class.return_value
        with delayed_status(
            "Please wait...",
            delay_seconds=2.0,
            stream=output,
        ):
            timer.start.assert_called_once_with()

    timer_class.assert_called_once()
    assert timer_class.call_args.args[0] == 2.0
    assert timer.daemon is True
    timer.cancel.assert_called_once_with()
    assert output.getvalue() == ""


def test_delayed_status_callback_prints_to_selected_stream() -> None:
    """A slow operation emits one flushed status line."""
    output = StringIO()

    with patch("src.cli_progress.Timer") as timer_class:
        with delayed_status("Please wait...", stream=output):
            callback = timer_class.call_args.args[1]
            callback_arguments = timer_class.call_args.kwargs["args"]
            callback(*callback_arguments)

    assert output.getvalue() == "Please wait...\n"


@pytest.mark.parametrize(
    ("message", "delay", "error"),
    [
        ("  ", 1.0, "message must not be empty"),
        ("Please wait", -1.0, "delay must not be negative"),
    ],
)
def test_delayed_status_rejects_invalid_configuration(
    message: str,
    delay: float,
    error: str,
) -> None:
    """Invalid UI configuration fails before creating a timer."""
    with pytest.raises(ValueError, match=error):
        with delayed_status(message, delay_seconds=delay):
            pass
