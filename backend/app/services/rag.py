"""The RAG query path: embed -> search -> synthesize, with measurement.

This is the same flow the baseline route ran inline, lifted into a service so
it can be called without HTTP. Two things drove the move:

  - Evaluation needs the *full retrieved context set*, not just the passages the
    model chose to cite. The HTTP response deliberately doesn't carry that (it
    would bloat every chat reply), so the Phase 3 runner calls this directly.
  - Baseline metrics need a latency breakdown. `llm_usage.latency_ms` times the
    LLM call only, which says nothing about retrieval cost.

Two optional stages sit between retrieval and generation, both config-driven so
the baseline stays reproducible (`RERANK_ENABLED=false`, `MIN_RERANK_SCORE`
unset):

  - reranking: fetch `RERANK_CANDIDATES` by vector search, keep the `top_k` a
    cross-encoder scores highest
  - a relevance gate: if even the best reranked chunk scores below
    `MIN_RERANK_SCORE`, answer "not found" and skip the LLM entirely
"""
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy.orm import Session

from app.config import settings
from app.core.prompts import NOT_FOUND_ANSWER
from app.services import embeddings, reranker, usage as usage_service, vector_store
from app.services.llm import UsageContext
from app.services.synthesizer import SynthesizedAnswer, synthesize_answer
from app.services.vector_store import RetrievedChunk

if TYPE_CHECKING:
    from app.schemas import TurnUsage


@dataclass
class RetrievalStats:
    """Retrieval quality summary, from similarity scores alone.

    Cheap to compute and useful for triage: a bad answer whose top_score is 0.31
    is a retrieval failure, while one whose top_score is 0.89 is a generation
    failure. Different fixes.
    """

    chunk_count: int
    top_score: float | None
    avg_score: float | None
    top_rerank_score: float | None = None  # None when reranking is off


@dataclass
class Timings:
    """Wall-clock milliseconds for each stage of one query."""

    embedding_ms: int
    search_ms: int
    generation_ms: int
    total_ms: int
    rerank_ms: int = 0

    @property
    def retrieval_ms(self) -> int:
        """Everything before generation — embedding, ANN scan, and reranking."""
        return self.embedding_ms + self.search_ms + self.rerank_ms


@dataclass
class QueryResult:
    question: str
    direct_answer: str
    sources: list[dict]  # resolved citations (only what the model cited)
    contexts: list[RetrievedChunk]  # everything retrieved (what evaluation needs)
    retrieval: RetrievalStats
    timings: Timings
    usage: "TurnUsage | None" = None
    gated: bool = False  # True if the relevance gate answered without the LLM


def retrieval_stats(chunks: list[RetrievedChunk]) -> RetrievalStats:
    if not chunks:
        return RetrievalStats(chunk_count=0, top_score=None, avg_score=None)
    scores = [c.score for c in chunks]
    reranked = [c.rerank_score for c in chunks if c.rerank_score is not None]
    return RetrievalStats(
        chunk_count=len(chunks),
        top_score=round(max(scores), 4),
        avg_score=round(sum(scores) / len(scores), 4),
        top_rerank_score=round(max(reranked), 4) if reranked else None,
    )


def below_relevance_gate(chunks: list[RetrievedChunk]) -> bool:
    """True if nothing retrieved is relevant enough to answer from.

    Gates on the cross-encoder score, not vector similarity: on the benchmark,
    vector scores for answerable and unanswerable questions overlap almost
    completely (0.29–0.84 vs 0.37–0.70), so no vector threshold separates them.
    """
    threshold = settings.MIN_RERANK_SCORE
    if threshold is None or not chunks:
        return False
    scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    return bool(scores) and max(scores) < threshold


def _elapsed_ms(since: float) -> int:
    return int((perf_counter() - since) * 1000)


def answer_question(
    db: Session,
    question: str,
    *,
    request_id: str | None = None,
    top_k: int | None = None,
) -> QueryResult:
    """Answer one question against the ingested corpus.

    `top_k` is how many chunks the LLM sees. With reranking on, vector search
    fetches `RERANK_CANDIDATES` and the cross-encoder narrows them to `top_k`.
    """
    usage = UsageContext(request_id=request_id or str(uuid4()))
    started = perf_counter()
    k = top_k or settings.TOP_K

    stage = perf_counter()
    query_vector = embeddings.embed_query(question)
    embedding_ms = _elapsed_ms(stage)

    stage = perf_counter()
    fetch = max(settings.RERANK_CANDIDATES, k) if settings.RERANK_ENABLED else k
    chunks = vector_store.search(db, query_vector, top_k=fetch)
    search_ms = _elapsed_ms(stage)

    rerank_ms = 0
    if settings.RERANK_ENABLED:
        stage = perf_counter()
        chunks = reranker.rerank(question, chunks, top_n=k)
        rerank_ms = _elapsed_ms(stage)

    gated = below_relevance_gate(chunks)
    stage = perf_counter()
    if gated:
        answer, sources = SynthesizedAnswer(direct_answer=NOT_FOUND_ANSWER), []
    else:
        answer, sources = synthesize_answer(question, chunks, usage=usage)
    generation_ms = _elapsed_ms(stage)

    # Measured before persistence: writing usage rows is bookkeeping, not part
    # of the latency a user experiences.
    total_ms = _elapsed_ms(started)

    return QueryResult(
        question=question,
        direct_answer=answer.direct_answer,
        sources=sources,
        contexts=chunks,
        retrieval=retrieval_stats(chunks),
        timings=Timings(
            embedding_ms=embedding_ms,
            search_ms=search_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            rerank_ms=rerank_ms,
        ),
        usage=usage_service.persist(db, usage),
        gated=gated,
    )
