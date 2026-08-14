"""Database engine, session factory, and pgvector bootstrap."""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    future=True,
    # Resolve unqualified table names against the configured schema.
    connect_args={"options": f"-csearch_path={settings.PG_SCHEMA_NAME}"},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a scoped session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Enable pgvector, create tables, and build the ANN index.

    Models must be registered on `Base` before `create_all` runs, so we import
    them here (inside the function) to avoid circular imports at module load.
    """
    from app import models  # noqa: F401  (registers ORM classes on Base.metadata)

    with engine.connect() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{settings.PG_SCHEMA_NAME}"'))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(bind=engine)

    # HNSW index for fast cosine similarity search. Created after the table exists.
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS knowledge_units_embedding_hnsw_idx "
                "ON knowledge_units USING hnsw (embedding vector_cosine_ops)"
            )
        )
        conn.commit()

    # Lightweight, idempotent migration: add the compare-mode `pipeline` tag to
    # existing llm_usage tables (create_all only adds it for fresh databases).
    with engine.connect() as conn:
        conn.execute(
            text("ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS pipeline VARCHAR(40)")
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_llm_usage_pipeline "
                "ON llm_usage (pipeline)"
            )
        )
        conn.commit()

    # Seed initial model pricing (idempotent).
    from app.services.pricing_seed import seed_pricing

    db = SessionLocal()
    try:
        seed_pricing(db)
    finally:
        db.close()
