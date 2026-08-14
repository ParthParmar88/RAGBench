"""SQLAlchemy ORM models.

Tables:
  - `knowledge_units` : retrievable chunks (title, content, embedding)
  - `model_pricing`   : time-versioned per-model rates (USD per 1M tokens)
  - `llm_usage`       : one row per LLM call (tokens + computed cost)
"""
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.config import settings
from app.database import Base


class KnowledgeUnit(Base):
    __tablename__ = "knowledge_units"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    title = Column(Text, nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.OPENAI_EMBEDDING_DIM), nullable=False)


class ModelPricing(Base):
    """Editable, time-versioned rate book.

    A price change = close the current row (`effective_to = now`) and insert a
    new one. The row with `effective_to IS NULL` is the current price.
    """

    __tablename__ = "model_pricing"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    provider = Column(String(50), nullable=False)
    model = Column(String(120), nullable=False, index=True)
    input_per_1m = Column(Numeric(12, 4), nullable=False)
    output_per_1m = Column(Numeric(12, 4), nullable=False)
    cached_input_per_1m = Column(Numeric(12, 4), nullable=True)
    currency = Column(String(10), nullable=False, default="USD")
    effective_from = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    effective_to = Column(DateTime(timezone=True), nullable=True)


class LlmUsage(Base):
    """One row per LLM call. Cost is computed at write-time and frozen here."""

    __tablename__ = "llm_usage"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    request_id = Column(String(64), nullable=True, index=True)

    provider = Column(String(50), nullable=False)
    model_requested = Column(String(120), nullable=False)  # the alias we asked for
    model = Column(String(120), nullable=False, index=True)  # resolved response.model
    operation = Column(String(40), nullable=False, index=True)

    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    cached_tokens = Column(Integer, nullable=False, default=0)

    input_cost = Column(Numeric(12, 6), nullable=True)
    output_cost = Column(Numeric(12, 6), nullable=True)
    total_cost = Column(Numeric(12, 6), nullable=True, index=True)
    currency = Column(String(10), nullable=False, default="USD")
    pricing_id = Column(BigInteger, ForeignKey("model_pricing.id"), nullable=True)

    latency_ms = Column(Integer, nullable=True)
    raw_usage = Column(JSONB, nullable=True)
