"""Cross-encoder reranking: re-score vector candidates against the question.

Vector search embeds the question and each chunk *independently*, then compares
them — fast, but blind to how the two actually relate. A cross-encoder reads the
question and a chunk *together* and scores the pair, which is far more precise
and far too slow to run over the whole corpus. So the two are chained: vector
search fetches a wide candidate set for recall, the cross-encoder picks the few
the LLM will see.

Runs locally on CPU. No API call, so it adds latency but no per-query cost.
"""
from functools import lru_cache

from app.config import settings
from app.services.vector_store import RetrievedChunk


@lru_cache(maxsize=1)
def _model():
    # Imported lazily: torch takes seconds to import, and a server running with
    # reranking disabled should never pay for it.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.RERANK_MODEL)


def warm_up() -> None:
    """Load the model now rather than inside the first user's request."""
    _model().predict([("warm", "up")])


def rerank(question: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    """Return the `top_n` chunks the cross-encoder scores highest, best first.

    Each returned chunk carries its `rerank_score`; its vector `score` is kept,
    so a citation can still report the similarity it was retrieved at.
    """
    if not chunks:
        return []
    scores = _model().predict([(question, c.content) for c in chunks])
    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = float(score)
    return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_n]
