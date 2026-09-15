"""Local HTTP API exposing search and answer over the RAG pipeline."""

from src.api.app import create_app

__all__ = ["create_app"]
