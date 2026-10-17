"""Combine several PDF books into one library-wide retrieval corpus.

Each book keeps its own record set and corpus policy, because navigation
and boilerplate are classified from one book's structure. The library
corpus concatenates the eligible records in canonical order so one index
can search every book at once.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.ingestion.pdf.fingerprints import fingerprint_canonical_json
from src.ingestion.pdf.index_records import (
    canonical_pdf_record_order,
    PdfIndexRecord,
)
from src.ingestion.pdf.outline import PdfOutlineEntry
from src.ingestion.pdf.pipeline import build_pdf_book_records
from src.ingestion.pdf.retrieval_corpus import (
    build_pdf_retrieval_corpus,
    classify_pdf_records,
    PdfRetrievalCorpus,
    PdfRetrievalPolicy,
)


PDF_LIBRARY_RECORDS_SCHEMA = "fredl.pdf.library-records"
PDF_LIBRARY_CORPUS_SCHEMA = "fredl.pdf.library-corpus"
PDF_LIBRARY_SCHEMA_VERSION = 1
_HASH_CHUNK_BYTES = 1 << 20


class PdfLibraryBookIdentity(BaseModel):
    """Pin one book by its file hash and its record-set fingerprint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    book_path: str = Field(min_length=1)
    pdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class PdfLibraryManifest(BaseModel):
    """Pin the exact books a multi-book dataset or run belongs to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    books: tuple[PdfLibraryBookIdentity, ...] = Field(min_length=1)

    @field_validator("books")
    @classmethod
    def require_sorted_unique_books(
        cls,
        books: tuple[PdfLibraryBookIdentity, ...],
    ) -> tuple[PdfLibraryBookIdentity, ...]:
        """Keep one canonical order so equal libraries hash equally."""
        paths = [book.book_path for book in books]
        if paths != sorted(set(paths)):
            raise ValueError(
                "PDF library books must be unique and sorted by book_path"
            )
        return books

    @property
    def book_paths(self) -> tuple[str, ...]:
        """Return the library-relative paths of every pinned book."""
        return tuple(book.book_path for book in self.books)

    @property
    def fingerprint(self) -> str:
        """Hash the record identity of the whole library."""
        return fingerprint_canonical_json({
            "schema": PDF_LIBRARY_RECORDS_SCHEMA,
            "version": PDF_LIBRARY_SCHEMA_VERSION,
            "books": [book.model_dump(mode="json") for book in self.books],
        })


@dataclass(frozen=True, slots=True)
class PdfLibraryBookInput:
    """Hold one book's source records before a policy is applied."""

    book_path: str
    pdf_sha256: str
    records: tuple[PdfIndexRecord, ...]
    outline: tuple[PdfOutlineEntry, ...]


@dataclass(frozen=True, slots=True)
class PdfLibraryBook:
    """Hold one book's source records and its policy-filtered corpus."""

    book_path: str
    pdf_sha256: str
    records: tuple[PdfIndexRecord, ...]
    corpus: PdfRetrievalCorpus

    @property
    def identity(self) -> PdfLibraryBookIdentity:
        """Return the pin that datasets and runs record for this book."""
        return PdfLibraryBookIdentity(
            book_path=self.book_path,
            pdf_sha256=self.pdf_sha256,
            record_set_fingerprint=self.corpus.record_set_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class PdfLibraryCorpus:
    """Search every book together while keeping per-book provenance.

    records and retrieval_texts list the eligible records of all books in
    canonical order; source_records keeps every original record, including
    excluded ones, for context and answerability passages.
    """

    policy: PdfRetrievalPolicy
    books: tuple[PdfLibraryBook, ...]

    @property
    def manifest(self) -> PdfLibraryManifest:
        """Return the record identity of the library."""
        return PdfLibraryManifest(
            books=tuple(book.identity for book in self.books),
        )

    @property
    def records(self) -> tuple[PdfIndexRecord, ...]:
        """Return eligible records of every book in canonical order."""
        return tuple(
            record for book in self.books for record in book.corpus.records
        )

    @property
    def retrieval_texts(self) -> tuple[str, ...]:
        """Return searchable text aligned with records."""
        return tuple(
            text for book in self.books for text in book.corpus.retrieval_texts
        )

    @property
    def source_records(self) -> tuple[PdfIndexRecord, ...]:
        """Return every original record, eligible or not."""
        return tuple(record for book in self.books for record in book.records)

    @property
    def excluded_record_ids(self) -> tuple[str, ...]:
        """Return records the policy kept out of search, per book in order."""
        return tuple(
            record_id
            for book in self.books
            for record_id in book.corpus.excluded_record_ids
        )

    def book(self, book_path: str) -> PdfLibraryBook:
        """Return one book of the library."""
        for book in self.books:
            if book.book_path == book_path:
                return book
        raise ValueError(f"PDF library has no book {book_path}")

    @property
    def fingerprint(self) -> str:
        """Hash everything that can change a ranking over the library."""
        return fingerprint_canonical_json({
            "schema": PDF_LIBRARY_CORPUS_SCHEMA,
            "version": PDF_LIBRARY_SCHEMA_VERSION,
            "policy": self.policy.value,
            "records_fingerprint": self.manifest.fingerprint,
            "books": [
                [book.book_path, book.corpus.fingerprint]
                for book in self.books
            ],
        })


def build_pdf_library_corpus(
    books: Sequence[PdfLibraryBookInput],
    policy: PdfRetrievalPolicy,
) -> PdfLibraryCorpus:
    """Apply one policy to each book separately and combine the results."""
    if not books:
        raise ValueError("PDF library needs at least one book")
    paths = [book.book_path for book in books]
    if len(paths) != len(set(paths)):
        raise ValueError("PDF library book paths must be unique")
    record_ids = [
        record.record_id for book in books for record in book.records
    ]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("PDF library record IDs must be unique across books")
    built = []
    for book in sorted(books, key=lambda item: item.book_path):
        if not book.records:
            raise ValueError(f"PDF library book {book.book_path} is empty")
        if any(record.book_path != book.book_path for record in book.records):
            raise ValueError(
                f"PDF library records must belong to {book.book_path}"
            )
        records = canonical_pdf_record_order(book.records)
        built.append(PdfLibraryBook(
            book_path=book.book_path,
            pdf_sha256=book.pdf_sha256,
            records=records,
            corpus=build_pdf_retrieval_corpus(
                records, classify_pdf_records(records, book.outline), policy,
            ),
        ))
    return PdfLibraryCorpus(policy=policy, books=tuple(built))


def load_pdf_library_corpus(
    library_root: Path,
    book_paths: Sequence[str],
    policy: PdfRetrievalPolicy,
) -> PdfLibraryCorpus:
    """Run the verified pipeline on each book and build the library."""
    inputs = []
    for book_path in book_paths:
        book = build_pdf_book_records(library_root, book_path)
        inputs.append(PdfLibraryBookInput(
            book_path=book_path,
            pdf_sha256=sha256_file(library_root / book_path),
            records=book.records,
            outline=book.outline,
        ))
    return build_pdf_library_corpus(inputs, policy)


def require_library_matches_manifest(
    corpus: PdfLibraryCorpus,
    manifest: PdfLibraryManifest,
) -> None:
    """Reject a library whose files or records differ from the pin."""
    current = corpus.manifest
    if current.book_paths != manifest.book_paths:
        raise ValueError("PDF library books differ from the manifest")
    for actual, pinned in zip(current.books, manifest.books):
        if actual.pdf_sha256 != pinned.pdf_sha256:
            raise ValueError(f"PDF file differs: {pinned.book_path}")
        if actual.record_set_fingerprint != pinned.record_set_fingerprint:
            raise ValueError(f"PDF records differ: {pinned.book_path}")


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory at once."""
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
