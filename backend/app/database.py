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

    _migrate_chunk_provenance()

    # Seed initial model pricing (idempotent).
    from app.services.pricing_seed import seed_pricing

    db = SessionLocal()
    try:
        seed_pricing(db)
    finally:
        db.close()


# Hand-rolled idempotent DDL, deliberately chosen over Alembic: this project has
# four tables and one developer, and a migration toolchain would cost more than
# it returns. Every statement below is safe to re-run. If the schema grows or a
# second environment appears, that trade-off flips — adopt Alembic then.
_CHUNK_PROVENANCE_DDL = (
    # 1. Provenance columns. Nullable, because pre-existing chunks genuinely
    #    have no recoverable page number.
    "ALTER TABLE knowledge_units ADD COLUMN IF NOT EXISTS document_id BIGINT",
    "ALTER TABLE knowledge_units ADD COLUMN IF NOT EXISTS page INTEGER",
    "ALTER TABLE knowledge_units ADD COLUMN IF NOT EXISTS chunk_index INTEGER",
    "CREATE INDEX IF NOT EXISTS ix_knowledge_units_document_id "
    "ON knowledge_units (document_id)",
    # 2. The FK can't use IF NOT EXISTS, so swallow the duplicate. Without this
    #    an upgraded database would lack the constraint a fresh one gets from
    #    create_all — and the two would drift apart.
    """
    DO $$ BEGIN
        ALTER TABLE knowledge_units
            ADD CONSTRAINT knowledge_units_document_id_fkey
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE;
    EXCEPTION WHEN duplicate_object THEN NULL; END $$
    """,
    # 3. Adopt orphaned chunks from before the `documents` table existed, so
    #    previously uploaded files keep working and stay listed. The empty
    #    checksum marks them as "provenance unknown": the next upload of that
    #    title sees a checksum mismatch and re-ingests it properly.
    """
    INSERT INTO documents (title, content_sha256, page_count, chunk_count)
    SELECT title, '', 0, COUNT(*)
      FROM knowledge_units
     WHERE document_id IS NULL
     GROUP BY title
    ON CONFLICT (title) DO NOTHING
    """,
    """
    UPDATE knowledge_units ku
       SET document_id = d.id
      FROM documents d
     WHERE ku.document_id IS NULL AND ku.title = d.title
    """,
)


def _migrate_chunk_provenance() -> None:
    """Add chunk provenance to an existing database without losing data."""
    with engine.connect() as conn:
        for statement in _CHUNK_PROVENANCE_DDL:
            conn.execute(text(statement))
        conn.commit()
