"""Insert and retrieve knowledge units from pgvector."""
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import KnowledgeUnit


@dataclass
class RetrievedChunk:
    id: int
    title: str
    content: str
    score: float  # cosine similarity in [0, 1], higher = more relevant


def add_knowledge_units(
    db: Session,
    title: str,
    contents: list[str],
    embeddings: list[list[float]],
) -> int:
    """Persist a batch of chunks for one source `title`. Returns the count added."""
    units = [
        KnowledgeUnit(title=title, content=content, embedding=embedding)
        for content, embedding in zip(contents, embeddings)
    ]
    db.add_all(units)
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
        RetrievedChunk(id=ku.id, title=ku.title, content=ku.content, score=1.0 - float(dist))
        for ku, dist in rows
    ]


def list_sources(db: Session) -> list[dict]:
    """Distinct source titles with their chunk counts."""
    stmt = (
        select(KnowledgeUnit.title, func.count(KnowledgeUnit.id))
        .group_by(KnowledgeUnit.title)
        .order_by(KnowledgeUnit.title)
    )
    return [{"title": title, "chunks": count} for title, count in db.execute(stmt).all()]


def delete_source(db: Session, title: str) -> int:
    """Delete all chunks for a given source title. Returns rows deleted."""
    deleted = (
        db.query(KnowledgeUnit)
        .filter(KnowledgeUnit.title == title)
        .delete(synchronize_session=False)
    )
    db.commit()
    return deleted
