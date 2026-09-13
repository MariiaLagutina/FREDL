"""Public semantic retrieval contracts."""

from src.retrieval.semantic.builder import (
    SemanticTextEncoder,
    build_semantic_index,
)
from src.retrieval.semantic.config import (
    DEFAULT_SEMANTIC_MODEL,
    SemanticEncoderConfig,
)
from src.retrieval.semantic.encoder import MiniLMEncoder, SemanticEncodingError
from src.retrieval.semantic.index import SemanticIndex
from src.retrieval.semantic.models import SemanticDocument, SemanticHit
from src.retrieval.semantic.runtime import (
    LoadedSemanticBackend,
    SemanticBackendLoadError,
    load_semantic_backend,
)

__all__ = [
    "DEFAULT_SEMANTIC_MODEL",
    "LoadedSemanticBackend",
    "MiniLMEncoder",
    "SemanticBackendLoadError",
    "SemanticDocument",
    "SemanticEncoderConfig",
    "SemanticEncodingError",
    "SemanticHit",
    "SemanticIndex",
    "SemanticTextEncoder",
    "build_semantic_index",
    "load_semantic_backend",
]
