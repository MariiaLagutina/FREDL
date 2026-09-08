"""Load the configured local Qwen model through a testable runtime boundary."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

from src.generation.backend.config import GenerationConfig, select_device


class GenerationBackendLoadError(RuntimeError):
    """Report an expected model or tokenizer loading failure."""


class GenerationRuntime(Protocol):
    """Describe framework operations required to load a generation model."""

    def cuda_is_available(self) -> bool:
        """Return whether the runtime can use a CUDA device."""

    def dtype_for(self, device: str) -> Any:
        """Return a safe model dtype for the selected device."""

    def load_tokenizer(
        self,
        model_name: str,
        *,
        local_files_only: bool,
    ) -> Any:
        """Load the tokenizer for one model identifier."""

    def load_model(
        self,
        model_name: str,
        *,
        local_files_only: bool,
        dtype: Any,
    ) -> Any:
        """Load the causal language model for one model identifier."""


@dataclass(frozen=True, slots=True)
class LoadedGenerationBackend:
    """Store the loaded model pair and its actual compute device."""

    tokenizer: Any
    model: Any
    device: str


class TransformersRuntime:
    """Provide lazy PyTorch and Transformers operations."""

    def __init__(self) -> None:
        """Import heavyweight frameworks only when loading is requested."""
        try:
            self._torch = import_module("torch")
            self._transformers = import_module("transformers")
        except ImportError as error:
            message = (
                "Qwen backend dependencies are unavailable; run `uv sync`"
            )
            raise GenerationBackendLoadError(message) from error

    def cuda_is_available(self) -> bool:
        """Return PyTorch's CUDA availability result."""
        return bool(self._torch.cuda.is_available())

    def dtype_for(self, device: str) -> Any:
        """Use float16 on CUDA and portable float32 on CPU."""
        return self._torch.float16 if device == "cuda" else self._torch.float32

    def load_tokenizer(
        self,
        model_name: str,
        *,
        local_files_only: bool,
    ) -> Any:
        """Load a native tokenizer without executing repository code."""
        return self._transformers.AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )

    def load_model(
        self,
        model_name: str,
        *,
        local_files_only: bool,
        dtype: Any,
    ) -> Any:
        """Load a native causal model without executing repository code."""
        return self._transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=False,
            dtype=dtype,
        )


def load_generation_backend(
    config: GenerationConfig,
    runtime: GenerationRuntime | None = None,
) -> LoadedGenerationBackend:
    """Load and place the model pair while reporting expected errors."""
    selected_runtime = runtime or TransformersRuntime()
    device = select_device(
        config.device,
        cuda_available=selected_runtime.cuda_is_available(),
    )
    dtype = selected_runtime.dtype_for(device)

    try:
        tokenizer = selected_runtime.load_tokenizer(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        model = selected_runtime.load_model(
            config.model_name,
            local_files_only=config.local_files_only,
            dtype=dtype,
        )
        model.to(device)
        model.eval()
    except (OSError, ValueError) as error:
        raise GenerationBackendLoadError(
            _load_error_message(config)
        ) from error

    return LoadedGenerationBackend(
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def _load_error_message(config: GenerationConfig) -> str:
    """Explain whether the caller should check cache or network access."""
    if config.local_files_only:
        return (
            f"Generation model '{config.model_name}' is unavailable in the "
            "local Hugging Face cache; run once with network access first"
        )
    return (
        f"Unable to load generation model '{config.model_name}'; check the "
        "model identifier and network access"
    )
