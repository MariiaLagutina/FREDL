"""Run a real local-model smoke check from the terminal."""

import argparse
from collections.abc import Sequence
import sys
from time import perf_counter

from src.generation.backend.config import (
    DEFAULT_MODEL_NAME,
    DevicePreference,
    GenerationConfig,
)
from src.generation.backend.generator import (
    ChatMessage,
    GenerationError,
    generate_answer,
)
from src.generation.backend.runtime import (
    GenerationBackendLoadError,
    load_generation_backend,
)


DEFAULT_SMOKE_PROMPT = "What is retrieval-augmented generation?"


def main(argv: Sequence[str] | None = None) -> None:
    """Load the configured model once and print one generated answer."""
    arguments = _parse_arguments(argv)
    config = GenerationConfig(
        model_name=arguments.model,
        device=DevicePreference(arguments.device),
        max_new_tokens=arguments.max_new_tokens,
        local_files_only=arguments.offline,
    )

    try:
        load_started = perf_counter()
        backend = load_generation_backend(config)
        load_seconds = perf_counter() - load_started

        generation_started = perf_counter()
        answer = generate_answer(
            [ChatMessage("user", arguments.prompt)],
            backend,
            config,
        )
        generation_seconds = perf_counter() - generation_started
    except (GenerationBackendLoadError, GenerationError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from None

    print(f"model:              {config.model_name}")
    print(f"device:             {backend.device}")
    print(f"offline_cache_only: {config.local_files_only}")
    print(f"load_seconds:       {load_seconds:.3f}")
    print(f"generation_seconds: {generation_seconds:.3f}")
    print(f"answer:             {answer}")


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse portable device, cache, and prompt controls."""
    parser = argparse.ArgumentParser(
        description="Smoke-test the local Qwen generation backend",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device",
        choices=tuple(preference.value for preference in DevicePreference),
        default=DevicePreference.AUTO.value,
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_SMOKE_PROMPT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
