"""Page-span extraction and offset -> page mapping.

`extract_pages` (which needs a real PDF) is monkeypatched; what's under test is
the offset bookkeeping that lets a chunk be traced back to a page.
"""
import pytest

from app.services import pdf_extractor
from app.services.pdf_extractor import (
    PageSpan,
    extract_document,
    extract_text,
    page_for_offset,
)


@pytest.fixture
def pdf(monkeypatch):
    """Install a fake PDF as a list of page strings."""

    def install(pages: list[str]):
        monkeypatch.setattr(pdf_extractor, "extract_pages", lambda _: pages)
        return b"fake-pdf-bytes"

    return install


def test_spans_point_at_each_pages_text(pdf):
    text, spans = extract_document(pdf(["alpha", "beta", "gamma"]))
    assert [text[s.start:s.end] for s in spans] == ["alpha", "beta", "gamma"]


def test_page_numbers_are_one_based(pdf):
    _, spans = extract_document(pdf(["alpha", "beta"]))
    assert [s.page for s in spans] == [1, 2]


def test_empty_pages_are_skipped_but_still_consume_a_page_number(pdf):
    """A scanned/blank page 2 must not renumber the pages that follow it."""
    text, spans = extract_document(pdf(["alpha", "", "gamma"]))
    assert text == "alpha\n\ngamma"
    assert [s.page for s in spans] == [1, 3]
    assert [text[s.start:s.end] for s in spans] == ["alpha", "gamma"]


def test_joined_text_needs_no_further_stripping(pdf):
    """The invariant chunk offsets depend on.

    chunk_document() strips its input; if extract_document returned text with
    leading whitespace, every chunk offset would shift and pages would be
    misattributed. Per-page stripping plus empty-page filtering prevents that.
    """
    text, _ = extract_document(pdf(["  alpha  ", "", "  gamma  "]))
    assert text == text.strip()
    assert text == "alpha\n\ngamma"


def test_extract_text_matches_the_joined_document(pdf):
    """extract_text must stay a pure view of extract_document, never diverge."""
    data = pdf(["alpha", "beta"])
    assert extract_text(data) == extract_document(data)[0]


def test_no_pages_yields_empty_text_and_no_spans(pdf):
    assert extract_document(pdf([])) == ("", [])
    assert extract_document(pdf(["", ""])) == ("", [])


def test_offset_maps_to_its_containing_page():
    spans = [PageSpan(1, 0, 5), PageSpan(2, 7, 11)]
    assert page_for_offset(spans, 0) == 1
    assert page_for_offset(spans, 4) == 1
    assert page_for_offset(spans, 7) == 2
    assert page_for_offset(spans, 10) == 2


def test_offset_inside_a_separator_belongs_to_the_following_page():
    """Offsets 5-6 are the "\\n\\n" between pages; text there begins on page 2."""
    spans = [PageSpan(1, 0, 5), PageSpan(2, 7, 11)]
    assert page_for_offset(spans, 5) == 2


def test_offset_past_the_end_clamps_to_the_last_page():
    spans = [PageSpan(1, 0, 5), PageSpan(2, 7, 11)]
    assert page_for_offset(spans, 9999) == 2


def test_offset_with_no_spans_is_unknown():
    assert page_for_offset([], 0) is None
