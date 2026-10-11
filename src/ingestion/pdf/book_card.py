"""Build neutral FREDL book cards from PDF identity metadata."""

from dataclasses import dataclass
from hashlib import sha256

from src.ingestion.pdf.metadata import PdfBookIdentity


@dataclass(frozen=True, slots=True)
class BookCard:
    """Describe a book without reusing its published cover artwork."""

    title: str
    author: str | None
    page_count: int
    visual_seed: str


def build_book_card(identity: PdfBookIdentity) -> BookCard:
    """Create stable display data for a FREDL-generated book cover."""
    title = identity.title or "Untitled book"
    seed_source = "\0".join(
        (
            identity.title or "",
            identity.author or "",
            str(identity.page_count),
        )
    )
    visual_seed = sha256(seed_source.encode("utf-8")).hexdigest()[:16]

    return BookCard(
        title=title,
        author=identity.author,
        page_count=identity.page_count,
        visual_seed=visual_seed,
    )
