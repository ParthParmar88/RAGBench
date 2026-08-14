"""Split text into fixed-size, overlapping chunks (simple sliding window).

`chunk_document` is the primitive; `chunk_text` is the original string-only API
kept intact. The window arithmetic is unchanged from the baseline — the only
addition is reporting where each chunk started, which ingestion needs to
recover page numbers.

Experiment 002 replaces this strategy. Until then it stays exactly as measured.
"""
from dataclasses import dataclass

from app.config import settings


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int  # offset of `text` within the (stripped) source document


def chunk_document(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Slide a window of `chunk_size` chars across the text, stepping by
    `chunk_size - chunk_overlap`, reporting each chunk's start offset."""
    size = chunk_size or settings.CHUNK_SIZE
    overlap = chunk_overlap or settings.CHUNK_OVERLAP

    text = text.strip()
    if not text:
        return []

    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    for start in range(0, len(text), step):
        window = text[start:start + size]
        stripped = window.strip()
        if stripped:
            # Leading whitespace removed by .strip() shifts the true start.
            lead = len(window) - len(window.lstrip())
            chunks.append(Chunk(text=stripped, start=start + lead))
        if start + size >= len(text):
            break
    return chunks


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Chunk text, discarding offsets. Behaviourally identical to the baseline."""
    return [c.text for c in chunk_document(text, chunk_size, chunk_overlap)]
