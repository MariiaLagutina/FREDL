"""Search the complete pinned eligible corpus with existing PDF baselines."""

from collections.abc import Callable

from src.evaluation.pdf.bm25_baseline import PdfBm25Baseline
from src.evaluation.pdf.context import find_pdf_neighbor_context
from src.evaluation.pdf.hybrid_baseline import (
    PDF_HYBRID_PARAMETERS,
    PdfHybridBaseline,
)
from src.evaluation.pdf.library import PdfLibraryCorpus
from src.evaluation.pdf.manifest import (
    bm25_configuration,
    current_pdf_run_environment,
    PdfRetrieverKind,
    PdfRunEnvironment,
    rrf_configuration,
    semantic_configuration,
)
from src.evaluation.pdf.minilm_baseline import PdfMiniLmBaseline
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook.retrieval.manifest import (
    build_pdf_multibook_manifest,
    PdfMultibookRetrievalConfiguration,
)
from src.evaluation.pdf.multibook.retrieval.queries import (
    build_pdf_multibook_query_plan,
    PdfQueryMode,
)
from src.evaluation.pdf.multibook.retrieval.runs import (
    build_pdf_multibook_run,
    PdfMultibookContextAtCutoff,
    PdfMultibookContextHit,
    PdfMultibookQueryResult,
    PdfMultibookRankedHit,
    PdfMultibookRetrievalRun,
    require_pdf_multibook_run_matches,
)
from src.ingestion.documents import Chunk
from src.ingestion.pdf.fingerprints import fingerprint_pdf_records
from src.ingestion.pdf.index_records import canonical_pdf_record_order
from src.retrieval.bm25 import BM25Parameters
from src.retrieval.semantic import (
    MiniLMEncoder,
    SemanticEncoderConfig,
    SemanticTextEncoder,
)
from src.retrieval.tokenization import QueryTokenizer


EncoderFactory = Callable[[SemanticEncoderConfig], SemanticTextEncoder]
_Search = Callable[[str], tuple[PdfMultibookRankedHit, ...]]


def run_pdf_multibook_retrieval(
    *,
    golden: PdfMultibookGolden,
    golden_sha256: str,
    corpus: PdfLibraryCorpus,
    retriever: PdfRetrieverKind,
    query_mode: PdfQueryMode = PdfQueryMode.ALL,
    environment: PdfRunEnvironment | None = None,
    encoder_factory: EncoderFactory | None = None,
) -> PdfMultibookRetrievalRun:
    """Build one global index, execute the plan and validate all results.

    encoder_factory is a Python test seam for synthetic encoders, never a
    CLI option. Normal semantic execution always uses the pinned MiniLM
    configuration with local_files_only=True. Synthetic runs do not
    constitute model quality measurements.
    """
    plan = build_pdf_multibook_query_plan(golden, query_mode)
    _require_consistent_corpus(corpus)
    semantic = SemanticEncoderConfig(local_files_only=True)
    configuration = PdfMultibookRetrievalConfiguration(
        retriever=retriever,
        bm25=bm25_configuration(BM25Parameters())
        if retriever is not PdfRetrieverKind.MINILM else None,
        semantic=semantic_configuration(semantic)
        if retriever is not PdfRetrieverKind.BM25 else None,
        rrf=rrf_configuration(PDF_HYBRID_PARAMETERS)
        if retriever is PdfRetrieverKind.HYBRID else None,
    )
    manifest = build_pdf_multibook_manifest(
        golden=golden, golden_sha256=golden_sha256, corpus=corpus,
        query_plan=plan, retrieval=configuration,
        environment=environment or current_pdf_run_environment(),
    )
    # Hybrid also invokes the lexical tokenizer. Fail before model loading
    # when the query has no searchable lexical terms.
    if retriever is not PdfRetrieverKind.MINILM:
        tokenizer = QueryTokenizer()
        if any(not tokenizer.tokenize(query.text) for query in plan.queries):
            raise ValueError("Selected query has no searchable lexical terms")
    search = _global_searcher(retriever, corpus, semantic, encoder_factory)
    results = []
    for query in plan.queries:
        ranking = search(query.text)
        results.append(PdfMultibookQueryResult(
            query=query.identity, ranking=ranking,
            context=_neighbor_context(ranking, corpus),
        ))
    run = build_pdf_multibook_run(manifest, tuple(results))
    require_pdf_multibook_run_matches(
        run, golden=golden, golden_sha256=golden_sha256, corpus=corpus,
        query_plan=plan, retrieval=configuration,
    )
    return run


def _require_consistent_corpus(corpus: PdfLibraryCorpus) -> None:
    """Reject inconsistent in-memory corpora before constructing an index."""
    source_ids = [record.record_id for record in corpus.source_records]
    keys = [(record.book_path, record.flow_index, record.flow_start,
             record.flow_end) for record in corpus.source_records]
    if len(set(source_ids)) != len(source_ids) or len(set(keys)) != len(keys):
        raise ValueError("Duplicate library record identities or index keys")
    for book in corpus.books:
        for record in book.records:
            if record.flow_index < 0:
                raise ValueError("Reading flow index must not be negative")
            Chunk(record.book_path, record.flow_start, record.flow_end,
                  record.text, record.section_path)
        source = {record.record_id: record for record in book.records}
        eligible = {record.record_id for record in book.corpus.records}
        if book.records != canonical_pdf_record_order(book.records) or (
            book.corpus.records != canonical_pdf_record_order(
                book.corpus.records,
            )
        ):
            raise ValueError("Library records must be in canonical order")
        if any(record.book_path != book.book_path for record in book.records):
            raise ValueError("Record belongs to a different library book")
        if book.corpus.record_set_fingerprint != fingerprint_pdf_records(
            book.records,
        ) or book.corpus.policy is not corpus.policy:
            raise ValueError("Per-book corpus identity or policy differs")
        if len(eligible) != len(book.corpus.records) or any(
            source.get(record.record_id) != record
            for record in book.corpus.records
        ):
            raise ValueError("Unknown, duplicate or changed corpus records")
        excluded = tuple(record.record_id for record in book.records
                         if record.record_id not in eligible)
        if excluded != book.corpus.excluded_record_ids:
            raise ValueError("Excluded records differ from corpus eligibility")
        if len(book.corpus.retrieval_texts) != len(book.corpus.records) or any(
            len(text) != len(record.text) or not text.strip()
            for record, text in zip(book.corpus.records,
                                    book.corpus.retrieval_texts)
        ):
            raise ValueError("Searchable text is not aligned with records")


def _global_searcher(
    kind: PdfRetrieverKind,
    corpus: PdfLibraryCorpus,
    semantic: SemanticEncoderConfig,
    encoder_factory: EncoderFactory | None,
) -> _Search:
    records, texts = corpus.records, corpus.retrieval_texts
    if kind is PdfRetrieverKind.BM25:
        lexical = PdfBm25Baseline(records, retrieval_texts=texts)
        return lambda query: tuple(PdfMultibookRankedHit(
            record_id=hit.record.record_id, score=hit.score,
        ) for hit in lexical.search(query))
    factory = MiniLMEncoder if encoder_factory is None else encoder_factory
    encoder = factory(semantic)
    # Legacy adapters replace falsey encoders with a network-enabled default.
    if not encoder:
        raise ValueError("Semantic encoder must not be missing or falsey")
    if kind is PdfRetrieverKind.MINILM:
        minilm = PdfMiniLmBaseline(records, encoder, retrieval_texts=texts)
        return lambda query: tuple(PdfMultibookRankedHit(
            record_id=hit.record.record_id, score=hit.score,
        ) for hit in minilm.search(query))
    hybrid = PdfHybridBaseline(
        records, encoder, PDF_HYBRID_PARAMETERS, retrieval_texts=texts,
    )
    return lambda query: tuple(PdfMultibookRankedHit(
        record_id=hit.record.record_id, score=hit.score,
        lexical_rank=hit.lexical_rank, semantic_rank=hit.semantic_rank,
    ) for hit in hybrid.search(query))


def _neighbor_context(
    ranking: tuple[PdfMultibookRankedHit, ...],
    corpus: PdfLibraryCorpus,
) -> tuple[PdfMultibookContextAtCutoff, ...]:
    """Reuse one-hop selection per seed inside its eligible source book."""
    by_book = {book.book_path: book.corpus.records for book in corpus.books}
    records = {record.record_id: record for record in corpus.records}
    contexts = []
    for cutoff in (5, 10):
        seeds = ranking[:cutoff]
        ranked = {hit.record_id for hit in seeds}
        parents: dict[str, list[str]] = {}
        for hit in seeds:
            book = records[hit.record_id].book_path
            neighbors = find_pdf_neighbor_context(
                (hit.record_id,), by_book[book], top_k=1,
            )
            for neighbor in neighbors:
                if neighbor not in ranked:
                    parents.setdefault(neighbor, []).append(hit.record_id)
        contexts.append(PdfMultibookContextAtCutoff(
            cutoff=cutoff, records=tuple(PdfMultibookContextHit(
                record_id=record_id, parent_record_ids=tuple(origins),
            ) for record_id, origins in parents.items()),
        ))
    return tuple(contexts)
