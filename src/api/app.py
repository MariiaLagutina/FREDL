"""Build the local HTTP API application."""

from fastapi import FastAPI

from src.api.schemas import HealthResponse


def create_app() -> FastAPI:
    """Construct the FastAPI application without loading any model state."""
    app = FastAPI(title="RAG against the machine")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report that the server process is reachable."""
        return HealthResponse()

    return app
