"""Global BM25, MiniLM and Hybrid rankings and query execution."""

from dataclasses import replace
import math

import pytest
import torch
from torch import Tensor

from src.evaluation.pdf.bm25_baseline import PdfBm25Baseline
from src.evaluation.pdf.library import (
    PdfLibraryBookInput,
    PdfLibraryCorpus,
    build_pdf_library_corpus,
)
from src.evaluation.pdf.multibook.retrieval.queries import PdfQueryMode
from src.ingestion.pdf.retrieval_corpus import PdfRetrievalPolicy
from src.retrieval.tokenization import TextTokenizer

from tests.evaluation.pdf.multibook.retrieval import (
    contract_helpers as contract_fixtures,
    runner_helpers as runner_fixtures,
)
from tests.evaluation.pdf.multibook.retrieval.runner_helpers import (
    FakeEncoder,
    _run,
)
from tests.evaluation.pdf.multibook_golden_helpers import _records


corpus = contract_fixtures.corpus
forbid_real_model_loading = runner_fixtures.forbid_real_model_loading


def test_bm25_statistics_and_scores_are_global(
    corpus: PdfLibraryCorpus,
) -> None:
    result = _run(corpus, first="work").results[0]
    assert [hit.record_id for hit in result.ranking] == ["a1"]
    tokenize = TextTokenizer().tokenize
    tokens = [tokenize(text) for text in corpus.retrieval_texts]
    length = len(tokens[0])
    average = sum(map(len, tokens)) / len(tokens)
    idf = math.log(1 + (len(tokens) - 1 + 0.5) / 1.5)
    expected = idf * 2.4 / (1 + 1.4 * (0.35 + 0.65 * length / average))
    assert result.ranking[0].score == pytest.approx(expected)
    single = PdfBm25Baseline(corpus.books[0].corpus.records).search("work")[0]
    assert result.ranking[0].score != pytest.approx(single.score)


def test_other_book_can_outrank_the_first_evidence_book(
    corpus: PdfLibraryCorpus,
) -> None:
    result = _run(corpus, first="work caches keep").results[0]
    assert result.ranking[0].record_id == "b1"
    assert {hit.record_id for hit in result.ranking} == {"a1", "b1"}


@pytest.mark.parametrize("kind", ["minilm", "hybrid"])
def test_global_semantic_inputs_are_exact_corpus_and_query_texts(
    corpus: PdfLibraryCorpus, kind: str,
) -> None:
    encoder = FakeEncoder()
    run = _run(corpus, kind, first="caches", encoder=encoder)
    assert encoder.calls[0] == corpus.retrieval_texts
    assert encoder.calls[1:] == [
        (query.text,) for query in run.manifest.query_plan.queries
    ]
    assert run.results[0].ranking[0].record_id == "b1"
    assert len(run.results[0].ranking) == len(corpus.records)
    assert len({record.book_path for record in corpus.records}) == 4


def test_hybrid_uses_global_source_ranks(corpus: PdfLibraryCorpus) -> None:
    lexical = _run(corpus, "bm25", first="caches").results[0].ranking
    semantic = _run(corpus, "minilm", first="caches").results[0].ranking
    fused = _run(corpus, "hybrid", first="caches").results[0].ranking
    lex = {hit.record_id: rank for rank, hit in enumerate(lexical, 1)}
    sem = {hit.record_id: rank for rank, hit in enumerate(semantic, 1)}
    for hit in fused:
        assert hit.lexical_rank == lex.get(hit.record_id)
        assert hit.semantic_rank == sem[hit.record_id]
        expected = 1 / (60 + sem[hit.record_id])
        if hit.record_id in lex:
            expected += 1 / (60 + lex[hit.record_id])
        assert hit.score == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["bm25", "minilm", "hybrid"])
def test_repeated_global_runs_preserve_order_and_fingerprints(
    corpus: PdfLibraryCorpus, kind: str,
) -> None:
    first = _run(corpus, kind, first="background")
    inputs = [PdfLibraryBookInput(
        book.book_path, book.pdf_sha256, tuple(reversed(book.records)), (),
    ) for book in reversed(corpus.books)]
    reordered = build_pdf_library_corpus(inputs, corpus.policy)
    second = _run(reordered, kind, first="background")
    assert first == second
    assert first.results_fingerprint == second.results_fingerprint
    if kind == "bm25":
        assert [hit.record_id for hit in first.results[0].ranking] == [
            "extra-0", "extra-1",
        ]


@pytest.mark.parametrize("mode,count", [
    (PdfQueryMode.ORIGINAL, 3), (PdfQueryMode.CLARIFIED, 2),
    (PdfQueryMode.ALL, 5),
])
def test_query_modes_and_meanings_remain_independent(
    corpus: PdfLibraryCorpus, mode: PdfQueryMode, count: int,
) -> None:
    run = _run(corpus, "minilm", mode=mode)
    assert len(run.results) == count
    variants = {result.query.interpretation_id: result
                for result in run.results
                if result.query.interpretation_id is not None}
    if mode is not PdfQueryMode.ORIGINAL:
        assert variants["queues"].ranking[0].record_id == "a1"
        assert variants["producers"].ranking[0].record_id == "b1"


def test_empty_bm25_ranking_is_not_an_answerability_label(
    corpus: PdfLibraryCorpus,
) -> None:
    result = _run(corpus).results[-1]
    assert result.ranking == ()
    assert all(not context.records for context in result.context)
    assert "answerability" not in result.model_dump()


def test_hybrid_does_not_truncate_global_candidates_to_twenty() -> None:
    inputs = []
    for book_index, path in enumerate((
        "books/alpha.pdf", "books/beta.pdf",
        "books/delta.pdf", "books/gamma.pdf",
    )):
        records = []
        for index in range(book_index, 41, 4):
            text = ("marker " * (42 - index)).strip()
            record = _records(path, ((f"doc-{index}", text, 1, 0),))[0]
            records.append(replace(record, flow_index=index))
        inputs.append(PdfLibraryBookInput(
            path, "abcd"[book_index] * 64, tuple(records), (),
        ))
    corpus = build_pdf_library_corpus(inputs, PdfRetrievalPolicy.ALL_RECORDS)

    class ReverseEncoder(FakeEncoder):
        def encode(self, texts: tuple[str, ...]) -> Tensor:
            vectors = []
            for text in texts:
                count = text.split().count("marker")
                x = 1.0 if count <= 1 else (43 - count) / 43
                vectors.append((x, math.sqrt(1 - x * x)))
            return torch.tensor(vectors, dtype=torch.float32)

    run = _run(corpus, "hybrid", first="marker", encoder=ReverseEncoder())
    ranking = run.results[0].ranking
    assert len(ranking) == 41
    # This record is outside both top-20 lists, yet contributes to full RRF.
    middle = next(hit for hit in ranking if hit.record_id == "doc-20")
    assert middle.lexical_rank == middle.semantic_rank == 21
    assert middle.score == pytest.approx(2 / 81)
