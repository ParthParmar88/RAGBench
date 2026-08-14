"""Split text into fixed-size, overlapping chunks (simple sliding window)."""
from app.config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Slide a window of `chunk_size` chars across the text, stepping by
    `chunk_size - chunk_overlap`."""
    size = chunk_size or settings.CHUNK_SIZE
    overlap = chunk_overlap or settings.CHUNK_OVERLAP

    text = text.strip()
    if not text:
        return []

    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks
