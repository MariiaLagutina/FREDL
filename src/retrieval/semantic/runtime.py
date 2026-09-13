"""Load the optional MiniLM encoder behind a testable runtime boundary."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

from src.retrieval.semantic.config import SemanticEncoderConfig


class SemanticBackendLoadError(RuntimeError):
    """Report an expected semantic model or tokenizer loading failure."""


class SemanticRuntime(Protocol):
    """Describe framework operations required by semantic model loading."""

    def load_tokenizer(
        self,
        model_name: str,
        *,
        local_files_only: bool,
    ) -> Any:
        """Load one tokenizer from its model identifier."""

    def load_model(
        self,
        model_name: str,
        *,
        local_files_only: bool,
    ) -> Any:
        """Load one base transformer model from its identifier."""


@dataclass(frozen=True, slots=True)
class LoadedSemanticBackend:
    """Store one tokenizer/model pair fixed to CPU inference."""

    tokenizer: Any
    model: Any
    dimension: int


class TransformersSemanticRuntime:
    """Provide lazy Hugging Face model-loading operations."""

    def __init__(self) -> None:
        """Import Transformers only when semantic loading is requested."""
        try:
            self._transformers = import_module("transformers")
        except ImportError as error:
            raise SemanticBackendLoadError(
                "Semantic encoder dependencies are unavailable; run `uv sync`"
            ) from error

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
    ) -> Any:
        """Load a portable float32 encoder without repository code."""
        return self._transformers.AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )


def load_semantic_backend(
    config: SemanticEncoderConfig,
    runtime: SemanticRuntime | None = None,
) -> LoadedSemanticBackend:
    """Load MiniLM once, place it on CPU, and activate evaluation mode."""
    selected_runtime = runtime or TransformersSemanticRuntime()
    try:
        tokenizer = selected_runtime.load_tokenizer(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        model = selected_runtime.load_model(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        model.to("cpu")
        model.eval()
        dimension = int(model.config.hidden_size)
        if dimension <= 0:
            raise ValueError("model hidden size must be positive")
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise SemanticBackendLoadError(_load_error_message(config)) from error
    return LoadedSemanticBackend(
        tokenizer=tokenizer,
        model=model,
        dimension=dimension,
    )


def _load_error_message(config: SemanticEncoderConfig) -> str:
    """Explain whether the caller should check cache or network access."""
    if config.local_files_only:
        return (
            f"Semantic model '{config.model_name}' is unavailable in the "
            "local Hugging Face cache; run once with network access first"
        )
    return (
        f"Unable to load semantic model '{config.model_name}'; check the "
        "model identifier and network access"
    )
