"""Application settings, loaded from environment / .env file."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIM: int = 1536

    # Database (discrete connection components)
    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DATABASE: str = "ragbench"
    PG_CONNECTION_USER: str = "postgres"
    PG_CONNECTION_PASSWORD: str = "root"
    PG_SCHEMA_NAME: str = "public"

    @property
    def DATABASE_URL(self) -> URL:
        """SQLAlchemy URL assembled from the discrete PG_* settings."""
        return URL.create(
            "postgresql+psycopg",
            username=self.PG_CONNECTION_USER,
            password=self.PG_CONNECTION_PASSWORD,
            host=self.PG_HOST,
            port=self.PG_PORT,
            database=self.PG_DATABASE,
        )

    # Retrieval / chunking
    TOP_K: int = 5
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 150

    # Reranking: fetch RERANK_CANDIDATES by vector search, keep the TOP_K the
    # cross-encoder scores highest. Off by default so the baseline stays
    # reproducible by configuration alone.
    RERANK_ENABLED: bool = False
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L12-v2"
    RERANK_CANDIDATES: int = 20

    # Relevance gate: when the best reranked chunk scores below this, answer
    # "not found" without calling the LLM. Cross-encoder logits, so it only
    # applies with reranking on. None disables it, and None is the recommended
    # setting: a threshold of -2.5 looked useful on the tuning benchmark and
    # then refused 3 of 34 answerable held-out questions (exp-005).
    MIN_RERANK_SCORE: float | None = None

    # CORS (comma-separated origins)
    CORS_ORIGINS: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
