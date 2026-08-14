"""Extract text from a PDF, page by page."""
from io import BytesIO

from pypdf import PdfReader


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


def extract_text(file_bytes: bytes) -> str:
    """Convenience: whole document as one string with page breaks."""
    return "\n\n".join(p for p in extract_pages(file_bytes) if p)
