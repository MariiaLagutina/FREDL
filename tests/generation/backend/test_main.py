"""Tests for the real-model smoke command boundary."""

from unittest.mock import patch

import pytest

from src.generation import GenerationBackendLoadError
from src.generation.backend.__main__ import main
from src.generation.backend.runtime import LoadedGenerationBackend


def test_smoke_command_reports_configuration_and_timings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI controls reach the backend and produce an inspectable report."""
    backend = LoadedGenerationBackend(object(), object(), "cpu")
    with (
        patch(
            "src.generation.backend.__main__.load_generation_backend",
            return_value=backend,
        ) as load_backend,
        patch(
            "src.generation.backend.__main__.generate_answer",
            return_value="Local answer",
        ) as generate,
        patch(
            "src.generation.backend.__main__.perf_counter",
            side_effect=(10.0, 11.25, 20.0, 22.5),
        ),
    ):
        main(
            [
                "--device",
                "cpu",
                "--offline",
                "--max-new-tokens",
                "17",
                "--prompt",
                "Test question",
            ]
        )

    config = load_backend.call_args.args[0]
    assert config.device.value == "cpu"
    assert config.local_files_only
    assert config.max_new_tokens == 17
    assert generate.call_args.args[0][0].content == "Test question"
    assert generate.call_args.args[2] is config
    assert capsys.readouterr().out == (
        "model:              Qwen/Qwen3-0.6B\n"
        "device:             cpu\n"
        "offline_cache_only: True\n"
        "load_seconds:       1.250\n"
        "generation_seconds: 2.500\n"
        "answer:             Local answer\n"
    )


def test_smoke_command_reports_expected_load_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An expected cache miss remains concise at the CLI boundary."""
    with patch(
        "src.generation.backend.__main__.load_generation_backend",
        side_effect=GenerationBackendLoadError("checkpoint unavailable"),
    ):
        with pytest.raises(SystemExit) as exit_info:
            main(["--offline"])

    assert exit_info.value.code == 2
    assert capsys.readouterr().err == (
        "Error: checkpoint unavailable\n"
    )
