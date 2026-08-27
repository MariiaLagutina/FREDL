"""Pydantic models shared by the RAG pipeline stages."""

import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represent one source location in the indexed corpus."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question that has not been answered yet."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with its reference answer and sources."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a collection of answered or unanswered RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class QuerySearchResult(BaseModel):
    """Represent retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class QueryAnswer(QuerySearchResult):
    """Represent retrieved sources and a generated answer for one question."""

    answer: str


class RetrievalResults(BaseModel):
    """Represent batch retrieval output produced by the search pipeline."""

    search_results: list[QuerySearchResult]
    k: int


class RetrievalResultsWithAnswers(BaseModel):
    """Represent batch retrieval and answer-generation output."""

    search_results: list[QueryAnswer]
    k: int


class MinimalSearchResults(QuerySearchResult):
    """Provide the assignment name for one query search result."""


class StudentSearchResults(BaseModel):
    """Provide the assignment name for batch retrieval output."""

    search_results: list[MinimalSearchResults]
    k: int


class MinimalAnswer(QueryAnswer):
    """Provide the assignment name for one answer-bearing result."""


class StudentSearchResultsAndAnswer(BaseModel):
    """Provide the assignment name for batch answer output."""

    search_results: list[MinimalAnswer]
    k: int
