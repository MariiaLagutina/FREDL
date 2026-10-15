"""Eligible one-hop context within book, flow and section boundaries."""

from dataclasses import replace
from typing import Any

import pytest

from src.evaluation.pdf.library import (
    PdfLibraryBookInput,
    PdfLibraryCorpus,
    build_pdf_library_corpus,
)
from src.ingestion.pdf.retrieval_corpus import (
    PdfRecordClassification,
    PdfRecordRole,
    PdfRetrievalPolicy,
    build_pdf_retrieval_corpus,
)

from tests.evaluation.pdf import (
    multibook_contract_helpers as contract_fixtures,
    multibook_runner_helpers as runner_fixtures,
)
from tests.evaluation.pdf.multibook_runner_helpers import _run
from tests.evaluation.pdf.multibook_golden_helpers import _records


corpus = contract_fixtures.corpus
forbid_real_model_loading = runner_fixtures.forbid_real_model_loading


def _policy_corpus(corpus: PdfLibraryCorpus, mask: bool) -> PdfLibraryCorpus:
    policy = (PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE if mask
              else PdfRetrievalPolicy.EXCLUDE_NAVIGATION)
    books = tuple(replace(book, corpus=build_pdf_retrieval_corpus(
        book.records,
        tuple(PdfRecordClassification(
            record.record_id,
            PdfRecordRole.INDEX if record.record_id == "a2"
            else PdfRecordRole.CONTENT,
            ((0, 6),) if mask and record.record_id == "a1" else (),
        ) for record in book.records), policy,
    )) for book in corpus.books)
    return replace(corpus, policy=policy, books=books)


def test_policy_excludes_navigation_and_masks_only_searchable_text(
    corpus: PdfLibraryCorpus,
) -> None:
    filtered = _policy_corpus(corpus, False)
    assert _run(filtered).results[0].context[0].records == ()
    assert _run(filtered, first="workers").results[0].ranking == ()
    masked = _policy_corpus(corpus, True)
    assert _run(masked).results[0].ranking == ()
    assert masked.source_records[0].text == corpus.source_records[0].text


@pytest.mark.parametrize("change", ["gap", "flow", "interval", "unknown"])
def test_context_stays_within_book_flow_interval_and_touching_ranges(
    corpus: PdfLibraryCorpus, change: str,
) -> None:
    inputs = []
    for book in corpus.books:
        records = book.records
        if book.book_path == "books/alpha.pdf":
            fields: dict[str, Any] = {}
            if change == "gap":
                fields = {"flow_start": records[1].flow_start + 1,
                          "flow_end": records[1].flow_end + 1}
            elif change == "flow":
                fields = {"flow_index": 1}
            else:
                fields = {"section_interval_id": None if change == "unknown"
                          else "different-interval"}
            records = (records[0], replace(records[1], **fields), records[2])
        inputs.append(PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, records, (),
        ))
    changed = build_pdf_library_corpus(inputs, corpus.policy)
    assert _run(changed).results[0].context[0].records == ()


def test_direct_context_parents_do_not_cross_books_or_expand_recursively(
    corpus: PdfLibraryCorpus,
) -> None:
    result = _run(corpus).results[0]
    for context in result.context:
        assert [(hit.record_id, hit.parent_record_ids)
                for hit in context.records] == [
            ("a2", ("a1",)),
        ]


def test_top5_and_top10_have_separate_context(
    corpus: PdfLibraryCorpus,
) -> None:
    inputs = []
    for book in corpus.books:
        records = book.records
        if book.book_path == "books/alpha.pdf":
            chain = []
            cursor = 0
            for index in range(11):
                text = ("marker " * (12 - index)).strip()
                record = _records(book.book_path, (
                    (f"chain-{index}", text, 1, cursor),
                ))[0]
                chain.append(replace(record, flow_start=cursor,
                                     flow_end=cursor + len(text)))
                cursor += len(text)
            records = tuple(chain)
        inputs.append(PdfLibraryBookInput(
            book.book_path, book.pdf_sha256, records, (),
        ))
    changed = build_pdf_library_corpus(inputs, corpus.policy)
    result = _run(changed, first="marker").results[0]
    assert [hit.record_id for hit in result.ranking] == [
        f"chain-{index}" for index in range(11)
    ]
    assert result.context[0].records[0].record_id == "chain-5"
    assert result.context[0].records[0].parent_record_ids == ("chain-4",)
    assert result.context[1].records[0].record_id == "chain-10"
    assert result.context[1].records[0].parent_record_ids == ("chain-9",)
