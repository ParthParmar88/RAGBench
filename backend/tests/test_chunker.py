"""Baseline chunker behaviour.

These tests pin V1's fixed-width sliding window. Experiment 002 replaces this
with structure-aware chunking; these assertions are what prove the new
behaviour differs *deliberately* rather than by accident.
"""
from app.services.chunker import chunk_text

# 30 chars, no whitespace — keeps .strip() from perturbing window boundaries so
# offsets can be asserted exactly.
TEXT = "abcdefghij" * 3


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_text_shorter_than_window_is_one_chunk():
    assert chunk_text("short document", chunk_size=100, chunk_overlap=10) == [
        "short document"
    ]


def test_text_exactly_window_size_is_one_chunk():
    assert chunk_text(TEXT, chunk_size=30, chunk_overlap=5) == [TEXT]


def test_consecutive_chunks_share_the_overlap():
    chunks = chunk_text(TEXT, chunk_size=10, chunk_overlap=3)
    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier[-3:] == later[:3]


def test_chunks_cover_the_whole_text_without_loss():
    """Stitching the chunks back together must reproduce the input exactly.

    Guards the property that actually matters for retrieval: no character of a
    source document may be dropped during ingestion.
    """
    chunks = chunk_text(TEXT, chunk_size=10, chunk_overlap=3)
    stitched = chunks[0] + "".join(c[3:] for c in chunks[1:])
    assert stitched == TEXT


def test_window_advances_by_size_minus_overlap():
    chunks = chunk_text(TEXT, chunk_size=10, chunk_overlap=3)
    assert chunks[0] == TEXT[0:10]
    assert chunks[1] == TEXT[7:17]  # step == 10 - 3
    assert chunks[-1] == TEXT[21:30]


def test_zero_overlap_is_silently_ignored():
    """KNOWN BUG, pinned deliberately — do not 'fix' this assertion.

        overlap = chunk_overlap or settings.CHUNK_OVERLAP

    `0` is falsy, so an explicit `chunk_overlap=0` is replaced by the configured
    default (150). With a 10-char window that makes step = max(1, 10-150) = 1,
    yielding a chunk per character instead of 3 disjoint chunks.

    No caller passes 0 today (both routes call `chunk_text(text)` bare), so
    baseline results are unaffected. But it means "no overlap" is unreachable
    through the public API — a control condition Experiment 002 needs. Fix it
    there, with `if chunk_overlap is None`, and flip this test then.
    """
    chunks = chunk_text(TEXT, chunk_size=10, chunk_overlap=0)
    assert chunks != [TEXT[0:10], TEXT[10:20], TEXT[20:30]]
    assert chunks[1] == TEXT[1:11]  # stride collapsed to 1


def test_overlap_not_smaller_than_size_still_terminates():
    """Documents a sharp edge rather than asserting it is desirable.

    step = max(1, size - overlap), so a misconfigured overlap >= size degrades
    to a one-character stride. It terminates — the loop breaks once the window
    reaches the end — but emits ~len(text) near-identical chunks, which would
    quietly explode embedding cost on ingest.
    """
    chunks = chunk_text(TEXT, chunk_size=5, chunk_overlap=5)
    assert len(chunks) == len(TEXT) - 5 + 1  # starts 0..25 inclusive
    assert all(len(c) == 5 for c in chunks)


def test_uses_configured_defaults_when_args_omitted():
    from app.config import settings

    chunks = chunk_text("x" * (settings.CHUNK_SIZE * 2))
    assert len(chunks[0]) == settings.CHUNK_SIZE
