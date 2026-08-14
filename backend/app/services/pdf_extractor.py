"""Extract text from a PDF, page by page.

Chunking runs over the whole document as one string (baseline behaviour), which
destroys page boundaries. To recover them without changing where chunks split,
extraction also reports the character range each page occupies in that joined
string — so a chunk's start offset can be mapped back to its page afterwards.

Chunking per page instead would be simpler, but it would move every chunk
boundary and silently invalidate comparisons against the frozen baseline.
"""
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

PAGE_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class PageSpan:
    """Where one page's text lives inside the joined document string."""

    page: int  # 1-based page number in the source PDF
    start: int  # inclusive character offset
    end: int  # exclusive character offset


def extract_pages(file_bytes: bytes) -> list[str]:
    """Return a list of page texts (index 0 == page 1).

    Pages that yield no extractable text (e.g. scanned images) come back as "".
    """
    reader = PdfReader(BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text.strip())
    return pages


def extract_document(file_bytes: bytes) -> tuple[str, list[PageSpan]]:
    """Return (joined document text, page spans into that text).

    Empty pages are dropped from the text but still consume a page number, so
    spans always carry the true page number from the original PDF.

    Every page is stripped and empties are filtered, so the joined result never
    has leading or trailing whitespace — meaning the `.strip()` inside
    `chunk_document` is a no-op and offsets stay valid. Stripping here rather
    than relying on `extract_pages` keeps that invariant local: it holds no
    matter how page text is sourced.
    """
    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0

    for page_number, page_text in enumerate(extract_pages(file_bytes), start=1):
        page_text = page_text.strip()
        if not page_text:
            continue
        if parts:
            cursor += len(PAGE_SEPARATOR)
        parts.append(page_text)
        spans.append(PageSpan(page_number, cursor, cursor + len(page_text)))
        cursor += len(page_text)

    return PAGE_SEPARATOR.join(parts), spans


def extract_text(file_bytes: bytes) -> str:
    """Whole document as one string with page breaks."""
    return extract_document(file_bytes)[0]


def page_for_offset(spans: list[PageSpan], offset: int) -> int | None:
    """Map a character offset in the joined text back to its page number.

    An offset landing inside a page separator is attributed to the *following*
    page, since that's where the text at that position actually begins.
    """
    for span in spans:
        if offset < span.end:
            return span.page
    return spans[-1].page if spans else None
