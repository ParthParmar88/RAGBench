"""The RAG query path: embed -> search -> synthesize, with measurement.

This is the same flow the baseline route ran inline, lifted into a service so
it can be called without HTTP. Two things drove the move:

  - Evaluation needs the *full retrieved context set*, not just the passages the
    model chose to cite. The HTTP response deliberately doesn't carry that (it
    would bloat every chat reply), so the Phase 3 runner calls this directly.
  - Baseline metrics need a latency breakdown. `llm_usage.latency_ms` times the
    LLM call only, which says nothing about retrieval cost.

Retrieval strategy is still hard-wired to vector search. The `Retriever`
abstraction arrives in Experiment 003, when a second strategy exists to justify
it — an interface with one implementation is just indirection.
"""
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy.orm import Session

from app.services import embeddings, usage as usage_service, vector_store
from app.services.llm import UsageContext
from app.services.synthesizer import synthesize_answer
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


@dataclass
class Timings:
    """Wall-clock milliseconds for each stage of one query."""

    embedding_ms: int
    search_ms: int
    generation_ms: int
    total_ms: int

    @property
    def retrieval_ms(self) -> int:
        """Everything before generation — the embedding call plus the ANN scan."""
        return self.embedding_ms + self.search_ms


@dataclass
class QueryResult:
    question: str
    direct_answer: str
    sources: list[dict]  # resolved citations (only what the model cited)
    contexts: list[RetrievedChunk]  # everything retrieved (what evaluation needs)
    retrieval: RetrievalStats
    timings: Timings
    usage: "TurnUsage | None" = None


def retrieval_stats(chunks: list[RetrievedChunk]) -> RetrievalStats:
    if not chunks:
        return RetrievalStats(chunk_count=0, top_score=None, avg_score=None)
    scores = [c.score for c in chunks]
    return RetrievalStats(
        chunk_count=len(chunks),
        top_score=round(max(scores), 4),
        avg_score=round(sum(scores) / len(scores), 4),
    )


def _elapsed_ms(since: float) -> int:
    return int((perf_counter() - since) * 1000)


def answer_question(
    db: Session,
    question: str,
    *,
    request_id: str | None = None,
    top_k: int | None = None,
) -> QueryResult:
    """Answer one question against the ingested corpus."""
    usage = UsageContext(request_id=request_id or str(uuid4()))
    started = perf_counter()

    stage = perf_counter()
    query_vector = embeddings.embed_query(question)
    embedding_ms = _elapsed_ms(stage)

    stage = perf_counter()
    chunks = vector_store.search(db, query_vector, top_k=top_k)
    search_ms = _elapsed_ms(stage)

    stage = perf_counter()
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
        ),
        usage=usage_service.persist(db, usage),
    )
