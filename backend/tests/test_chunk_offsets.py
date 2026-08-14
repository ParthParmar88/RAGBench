"""Chunk offsets, and the page attribution built on top of them.

The critical property: adding offset tracking must not have moved a single
chunk boundary. If it did, the frozen baseline's retrieval results would no
longer be reproducible and every comparison against it would be invalid.
"""
from app.services.chunker import chunk_document, chunk_text
from app.services.pdf_extractor import PageSpan, page_for_offset

TEXT = "abcdefghij" * 3


def test_offsets_do_not_change_chunk_boundaries():
    """chunk_text must remain byte-identical to the baseline implementation."""
    assert chunk_text(TEXT, 10, 3) == [c.text for c in chunk_document(TEXT, 10, 3)]


def test_offset_locates_the_chunk_in_the_source():
    for chunk in chunk_document(TEXT, 10, 3):
        assert TEXT[chunk.start:chunk.start + len(chunk.text)] == chunk.text


def test_first_chunk_starts_at_zero():
    assert chunk_document(TEXT, 10, 3)[0].start == 0


def test_offsets_increase_by_the_step():
    chunks = chunk_document(TEXT, 10, 3)
    assert [c.start for c in chunks] == [0, 7, 14, 21]


def test_leading_whitespace_is_excluded_from_the_offset():
    """A window starting mid-gap is stripped, so its offset must skip the gap.

    Without this correction a chunk would be attributed to the page the
    whitespace belongs to rather than the page its text actually starts on.

    (overlap=1 rather than 0 — `chunk_overlap=0` is silently replaced by the
    configured default; see test_chunker.test_zero_overlap_is_silently_ignored.)
    """
    text = "aaaa" + " " * 6 + "bbbb"
    chunks = chunk_document(text, chunk_size=6, chunk_overlap=1)
    second = chunks[1]
    assert second.text.startswith("b")
    assert text[second.start] == "b"


def test_empty_text_yields_no_chunks():
    assert chunk_document("") == []
    assert chunk_document("   \n  ") == []


def test_chunks_map_to_the_page_they_start_on():
    """End-to-end: a two-page document with a chunk landing on each page."""
    page_one, page_two = "a" * 100, "b" * 100
    text = page_one + "\n\n" + page_two
    spans = [PageSpan(1, 0, 100), PageSpan(2, 102, 202)]

    pages = [
        page_for_offset(spans, c.start)
        for c in chunk_document(text, chunk_size=60, chunk_overlap=1)
    ]
    # Windows start at 0, 59, 118, 177; page 2 begins at offset 102.
    assert pages == [1, 1, 2, 2]


def test_every_chunk_resolves_to_a_page():
    """No chunk may end up with unknown provenance in a normal document."""
    text = "x" * 500 + "\n\n" + "y" * 500
    spans = [PageSpan(1, 0, 500), PageSpan(2, 502, 1002)]
    for chunk in chunk_document(text, chunk_size=120, chunk_overlap=20):
        assert page_for_offset(spans, chunk.start) in (1, 2)
