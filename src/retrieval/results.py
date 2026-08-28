"""Convert ranked retrieval hits into the public source contract."""

from collections.abc import Callable, Iterable, Sequence

from src.models import (
    MinimalSource,
    QuerySearchResult,
    RagDataset,
    RetrievalResults,
    UnansweredQuestion,
)
from src.retrieval.bm25 import BM25Hit, BM25Index, BM25Retriever


QuestionProgress = Callable[
    [Sequence[UnansweredQuestion]],
    Iterable[UnansweredQuestion],
]


def search_dataset(
    index: BM25Index,
    dataset: RagDataset,
    k: int = 5,
    progress: QuestionProgress | None = None,
) -> RetrievalResults:
    """Search every dataset question in its original order."""
    if k <= 0:
        raise ValueError("Search k must be greater than zero")

    questions: Iterable[UnansweredQuestion] = dataset.rag_questions
    if progress is not None:
        questions = progress(dataset.rag_questions)

    return RetrievalResults(
        search_results=[
            search_question(index, question, k)
            for question in questions
        ],
        k=k,
    )


def search_question(
    index: BM25Index,
    question: UnansweredQuestion,
    k: int = 5,
) -> QuerySearchResult:
    """Search one dataset question and preserve its public identity."""
    return QuerySearchResult(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=search_sources(index, question.question, k),
    )


def search_sources(
    index: BM25Index,
    query: str,
    k: int = 5,
) -> list[MinimalSource]:
    """Search one raw query against a prebuilt index."""
    if k <= 0:
        raise ValueError("Search k must be greater than zero")

    ranked_hits = BM25Retriever(index).search(query, top_k=k)
    return select_sources(ranked_hits, k)


def select_sources(
    ranked_hits: Sequence[BM25Hit],
    k: int,
) -> list[MinimalSource]:
    """Return the first unique exact source locations in ranking order."""
    if k <= 0:
        raise ValueError("Search k must be greater than zero")

    sources: list[MinimalSource] = []
    seen: set[tuple[str, int, int]] = set()
    for hit in ranked_hits:
        key = hit.document.key
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            MinimalSource(
                file_path=key[0],
                first_character_index=key[1],
                last_character_index=key[2],
            )
        )
        if len(sources) == k:
            break
    return sources
