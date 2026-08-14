"""Ingest a PDF: extract -> chunk -> resolve provenance -> embed -> store.

Ingestion is idempotent, keyed on the file's SHA-256:

    same name, same bytes      -> unchanged (no re-embedding, no duplicates)
    same name, different bytes -> replaced  (old chunks dropped first)
    new name                   -> created

The baseline had no such guard: every upload appended a fresh copy of every
chunk. Two uploads of the same PDF meant the retriever could return the same
passage twice under different ids, inflating apparent recall and quietly
corrupting any evaluation run that re-ingested its corpus.
"""
import hashlib
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import Session

from app.models import Document
from app.services import embeddings, pdf_extractor, vector_store
from app.services.chunker import chunk_document
from app.services.vector_store import PendingChunk

IngestStatus = Literal["created", "replaced", "unchanged"]


@dataclass
class IngestResult:
    title: str
    status: IngestStatus
    chunks_added: int
    page_count: int


class EmptyDocumentError(ValueError):
    """The PDF yielded no extractable text (likely scanned images)."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ingest_pdf(db: Session, title: str, file_bytes: bytes) -> IngestResult:
    checksum = _sha256(file_bytes)
    existing = vector_store.get_document(db, title)

    if existing is not None and existing.content_sha256 == checksum:
        return IngestResult(
            title=title,
            status="unchanged",
            chunks_added=existing.chunk_count,
            page_count=existing.page_count,
        )

    text, page_spans = pdf_extractor.extract_document(file_bytes)
    if not text.strip():
        raise EmptyDocumentError(title)

    chunks = chunk_document(text)
    if not chunks:
        raise EmptyDocumentError(title)

    # Map each chunk back to the page its text starts on.
    pending = [
        PendingChunk(
            content=chunk.text,
            page=pdf_extractor.page_for_offset(page_spans, chunk.start),
            chunk_index=index,
        )
        for index, chunk in enumerate(chunks)
    ]

    status: IngestStatus = "created"
    if existing is not None:
        # Replace wholesale rather than diff: chunk boundaries shift when a
        # document changes, so old and new chunks aren't comparable anyway.
        vector_store.delete_source(db, title)
        status = "replaced"

    document = Document(
        title=title,
        content_sha256=checksum,
        page_count=len(page_spans),
        chunk_count=0,
    )
    db.add(document)
    db.flush()  # assign document.id without ending the transaction

    vectors = embeddings.embed_texts([c.content for c in pending])
    added = vector_store.add_knowledge_units(db, document, pending, vectors)

    return IngestResult(
        title=title,
        status=status,
        chunks_added=added,
        page_count=len(page_spans),
    )
