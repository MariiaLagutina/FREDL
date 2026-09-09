"""Single-query retrieval-augmented generation workflow."""

from src.generation.query.models import QueryAnswerResult
from src.generation.query.workflow import answer_query

__all__ = ["QueryAnswerResult", "answer_query"]
