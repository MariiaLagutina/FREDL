"""Configuration for the optional local semantic encoder."""

from dataclasses import dataclass

DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True, slots=True)
class SemanticEncoderConfig:
    """Declare reproducible inputs for CPU embedding inference."""

    model_name: str = DEFAULT_SEMANTIC_MODEL
    batch_size: int = 32
    max_length: int = 256
    local_files_only: bool = False

    def __post_init__(self) -> None:
        """Reject unusable model and batching values before model loading."""
        if not self.model_name.strip():
            raise ValueError("Semantic model name must not be empty")
        if self.batch_size <= 0:
            raise ValueError("Semantic batch size must be greater than zero")
        if self.max_length <= 0:
            raise ValueError(
                "Semantic maximum length must be greater than zero"
            )
