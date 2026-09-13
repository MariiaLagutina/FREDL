"""Public semantic retrieval contracts."""

from src.retrieval.semantic.builder import (
    SemanticTextEncoder,
    build_semantic_index,
)
from src.retrieval.semantic.config import (
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_SEMANTIC_REVISION,
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
from src.retrieval.semantic.store import (
    IncompatibleSemanticIndexError,
    SEMANTIC_SCHEMA_VERSION,
    SemanticIndexStore,
)
from src.retrieval.semantic.workflow import (
    SemanticIndexBuildReport,
    build_and_store_semantic_index,
)

__all__ = [
    "DEFAULT_SEMANTIC_MODEL",
    "DEFAULT_SEMANTIC_REVISION",
    "IncompatibleSemanticIndexError",
    "LoadedSemanticBackend",
    "MiniLMEncoder",
    "SemanticBackendLoadError",
    "SemanticDocument",
    "SemanticEncoderConfig",
    "SemanticEncodingError",
    "SemanticHit",
    "SemanticIndex",
    "SemanticIndexBuildReport",
    "SemanticIndexStore",
    "SemanticTextEncoder",
    "SEMANTIC_SCHEMA_VERSION",
    "build_semantic_index",
    "build_and_store_semantic_index",
    "load_semantic_backend",
]
