"""Generate embeddings via the OpenAI embeddings API."""
from app.config import settings
from app.services.openai_client import client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input, in order."""
    if not texts:
        return []
    resp = client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=texts,
        dimensions=settings.OPENAI_EMBEDDING_DIM,
    )
    return [item.embedding for item in resp.data]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
