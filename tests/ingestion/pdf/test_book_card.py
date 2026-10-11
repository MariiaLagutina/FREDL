"""Tests for neutral FREDL book-card data."""

from src.ingestion.pdf import PdfBookIdentity, build_book_card


def test_build_book_card_preserves_display_metadata() -> None:
    """A generated card keeps the identity fields a reader needs."""
    identity = PdfBookIdentity(
        title="A Synthetic Book",
        author="Example Author",
        page_count=120,
    )

    card = build_book_card(identity)

    assert card.title == "A Synthetic Book"
    assert card.author == "Example Author"
    assert card.page_count == 120


def test_build_book_card_generates_a_stable_visual_seed() -> None:
    """The same identity receives the same FREDL visual between runs."""
    identity = PdfBookIdentity(
        title="A Synthetic Book",
        author="Example Author",
        page_count=120,
    )

    first_card = build_book_card(identity)
    second_card = build_book_card(identity)

    assert first_card.visual_seed == second_card.visual_seed
    assert len(first_card.visual_seed) == 16


def test_build_book_card_distinguishes_different_books() -> None:
    """Different book identities do not share the same visual seed."""
    first_card = build_book_card(
        PdfBookIdentity("First Book", "Example Author", 120)
    )
    second_card = build_book_card(
        PdfBookIdentity("Second Book", "Example Author", 120)
    )

    assert first_card.visual_seed != second_card.visual_seed


def test_build_book_card_handles_missing_metadata() -> None:
    """Missing optional metadata has a clear, minimal presentation."""
    card = build_book_card(
        PdfBookIdentity(title=None, author=None, page_count=12)
    )

    assert card.title == "Untitled book"
    assert card.author is None
