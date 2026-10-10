"""Insert and retrieve knowledge units from pgvector."""
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Document, KnowledgeUnit


@dataclass
class RetrievedChunk:
    id: int
    title: str
    content: str
    score: float  # cosine similarity in [0, 1], higher = more relevant
    page: int | None = None  # None for chunks ingested before provenance existed
    chunk_index: int | None = None
    rerank_score: float | None = None  # cross-encoder logit; None when not reranked


@dataclass
class PendingChunk:
    """A chunk ready to persist, with its provenance resolved."""

    content: str
    page: int | None
    chunk_index: int


def get_document(db: Session, title: str) -> Document | None:
    return db.execute(
        select(Document).where(Document.title == title)
    ).scalar_one_or_none()


def add_knowledge_units(
    db: Session,
    document: Document,
    chunks: list[PendingChunk],
    embeddings: list[list[float]],
) -> int:
    """Persist a batch of chunks for one document. Returns the count added."""
    units = [
        KnowledgeUnit(
            document_id=document.id,
            title=document.title,
            content=chunk.content,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    db.add_all(units)
    document.chunk_count = len(units)
    db.commit()
    return len(units)


def search(db: Session, query_embedding: list[float], top_k: int | None = None) -> list[RetrievedChunk]:
    """Return the most similar knowledge units to the query embedding."""
    k = top_k or settings.TOP_K

    # cosine_distance is in [0, 2]; similarity = 1 - distance.
    distance = KnowledgeUnit.embedding.cosine_distance(query_embedding)
    stmt = (
        select(KnowledgeUnit, distance.label("distance"))
        .order_by(distance)
        .limit(k)
    )
    rows = db.execute(stmt).all()
    return [
        RetrievedChunk(
            id=ku.id,
            title=ku.title,
            content=ku.content,
            score=1.0 - float(dist),
            page=ku.page,
            chunk_index=ku.chunk_index,
        )
        for ku, dist in rows
    ]


def list_sources(db: Session) -> list[dict]:
    """Distinct source titles with their chunk counts.

    Counts come from `knowledge_units` rather than `Document.chunk_count` so the
    listing reflects what is actually retrievable, not what ingestion recorded.
    """
    stmt = (
        select(KnowledgeUnit.title, func.count(KnowledgeUnit.id))
        .group_by(KnowledgeUnit.title)
        .order_by(KnowledgeUnit.title)
    )
    return [{"title": title, "chunks": count} for title, count in db.execute(stmt).all()]


def delete_source(db: Session, title: str) -> int:
    """Delete a source and all its chunks. Returns chunk rows deleted."""
    deleted = (
        db.query(KnowledgeUnit)
        .filter(KnowledgeUnit.title == title)
        .delete(synchronize_session=False)
    )
    db.query(Document).filter(Document.title == title).delete(synchronize_session=False)
    db.commit()
    return deleted
