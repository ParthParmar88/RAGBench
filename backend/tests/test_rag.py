"""Query orchestration: retrieval stats and the latency breakdown."""
import pytest

from app.services import embeddings, rag, usage as usage_service, vector_store
from app.services.rag import answer_question, retrieval_stats
from app.services.synthesizer import SynthesizedAnswer
from app.services.vector_store import RetrievedChunk


def chunk(score, page=1, cid=1):
    return RetrievedChunk(
        id=cid, title="paper.pdf", content="text", score=score, page=page
    )


# --- retrieval stats -------------------------------------------------------


def test_stats_summarise_similarity_scores():
    stats = retrieval_stats([chunk(0.9), chunk(0.7), chunk(0.5)])
    assert stats.chunk_count == 3
    assert stats.top_score == 0.9
    assert stats.avg_score == 0.7


def test_empty_retrieval_reports_no_scores():
    """An empty corpus must yield nulls, not a misleading 0.0 similarity."""
    stats = retrieval_stats([])
    assert stats.chunk_count == 0
    assert stats.top_score is None
    assert stats.avg_score is None


def test_scores_are_rounded_for_reporting():
    stats = retrieval_stats([chunk(0.123456789)])
    assert stats.top_score == 0.1235


# --- query path ------------------------------------------------------------


@pytest.fixture
def query(monkeypatch):
    """Fake every I/O boundary so the orchestration can be tested offline."""
    calls = {}

    def install(chunks):
        monkeypatch.setattr(embeddings, "embed_query", lambda q: [0.0] * 4)

        def fake_search(db, vec, top_k=None):
            calls["top_k"] = top_k
            return chunks

        monkeypatch.setattr(vector_store, "search", fake_search)
        monkeypatch.setattr(
            rag, "synthesize_answer",
            lambda q, c, usage=None: (
                SynthesizedAnswer(direct_answer="answer", source_numbers=[1]),
                [{"number": 1, "title": "paper.pdf", "score": 0.9,
                  "snippet": "text", "page": 1}],
            ),
        )
        monkeypatch.setattr(usage_service, "persist", lambda db, ctx: None)
        return calls

    return install


def test_all_retrieved_contexts_are_returned(query):
    """Evaluation needs everything retrieved, not just the cited subset.

    Here 3 chunks are retrieved but only 1 is cited; context precision/recall
    must be computed from the 3.
    """
    query([chunk(0.9, cid=1), chunk(0.8, cid=2), chunk(0.7, cid=3)])
    result = answer_question(None, "why?")
    assert len(result.contexts) == 3
    assert len(result.sources) == 1


def test_timings_are_recorded_for_every_stage(query):
    query([chunk(0.9)])
    t = answer_question(None, "why?").timings
    for value in (t.embedding_ms, t.search_ms, t.generation_ms, t.total_ms):
        assert isinstance(value, int) and value >= 0


def test_retrieval_ms_is_embedding_plus_search(query):
    query([chunk(0.9)])
    t = answer_question(None, "why?").timings
    assert t.retrieval_ms == t.embedding_ms + t.search_ms


def test_total_covers_the_whole_query(query):
    """Total must not be less than the stages it contains."""
    query([chunk(0.9)])
    t = answer_question(None, "why?").timings
    assert t.total_ms >= t.embedding_ms + t.search_ms + t.generation_ms - 1


def test_top_k_override_reaches_the_retriever(query):
    """Experiment 004 retrieves a wider candidate set; the hook must work."""
    calls = query([chunk(0.9)])
    answer_question(None, "why?", top_k=20)
    assert calls["top_k"] == 20


def test_empty_corpus_still_produces_a_result(query):
    query([])
    result = answer_question(None, "why?")
    assert result.retrieval.chunk_count == 0
    assert result.contexts == []


# --- reranking and the relevance gate ---------------------------------------


@pytest.fixture
def pipeline(query, monkeypatch):
    """`query`, plus a fake cross-encoder and per-test settings overrides.

    The fake scores each chunk by its id, so tests can predict the reranked
    order without loading a model.
    """
    from app.config import settings
    from app.services import reranker

    def fake_rerank(question, chunks, top_n):
        for c in chunks:
            c.rerank_score = float(c.id)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_n]

    monkeypatch.setattr(reranker, "rerank", fake_rerank)

    def install(chunks, **overrides):
        for name, value in overrides.items():
            monkeypatch.setattr(settings, name, value)
        return query(chunks)

    return install


def many(n):
    return [chunk(0.9 - i * 0.01, cid=i) for i in range(n)]


def test_reranking_fetches_candidates_then_keeps_top_k(pipeline):
    calls = pipeline(many(20), RERANK_ENABLED=True, RERANK_CANDIDATES=20, TOP_K=5)
    result = answer_question(None, "why?")
    assert calls["top_k"] == 20
    assert [c.id for c in result.contexts] == [19, 18, 17, 16, 15]


def test_reranking_never_fetches_fewer_than_top_k(pipeline):
    calls = pipeline(many(10), RERANK_ENABLED=True, RERANK_CANDIDATES=3)
    answer_question(None, "why?", top_k=10)
    assert calls["top_k"] == 10


def test_reranking_off_is_the_baseline(pipeline):
    """Disabled, the vector order reaches the LLM untouched."""
    calls = pipeline(many(5), RERANK_ENABLED=False, TOP_K=5)
    result = answer_question(None, "why?")
    assert calls["top_k"] == 5
    assert [c.id for c in result.contexts] == [0, 1, 2, 3, 4]
    assert result.retrieval.top_rerank_score is None
    assert result.timings.rerank_ms == 0


def test_rerank_time_counts_as_retrieval(pipeline):
    pipeline(many(20), RERANK_ENABLED=True)
    t = answer_question(None, "why?").timings
    assert t.retrieval_ms == t.embedding_ms + t.search_ms + t.rerank_ms


def test_gate_refuses_without_calling_the_llm(pipeline, monkeypatch):
    from app.core.prompts import NOT_FOUND_ANSWER

    pipeline(many(20), RERANK_ENABLED=True, MIN_RERANK_SCORE=100.0)
    monkeypatch.setattr(rag, "synthesize_answer", lambda *a, **k: pytest.fail("LLM called"))
    result = answer_question(None, "why?")
    assert result.gated
    assert result.direct_answer == NOT_FOUND_ANSWER
    assert result.sources == []
    # Evaluation still needs what was retrieved, to see why it refused.
    assert len(result.contexts) == 5


def test_gate_passes_relevant_retrieval_through(pipeline):
    pipeline(many(20), RERANK_ENABLED=True, MIN_RERANK_SCORE=10.0)  # top score is 19
    result = answer_question(None, "why?")
    assert not result.gated
    assert result.direct_answer == "answer"


def test_gate_is_inert_without_rerank_scores(pipeline):
    """Vector similarity is on a different scale; the gate must not misread it."""
    pipeline(many(5), RERANK_ENABLED=False, MIN_RERANK_SCORE=100.0)
    assert not answer_question(None, "why?").gated


def test_gate_refusal_is_detected_as_a_refusal():
    """The gate's answer must count as a refusal in evaluation, or the gate's
    effect on the unanswerable category would be invisible."""
    from app.core.prompts import NOT_FOUND_ANSWER, SYNTHESIZER_SYSTEM
    from evaluation.metrics import is_refusal

    assert is_refusal(NOT_FOUND_ANSWER)
    assert NOT_FOUND_ANSWER in SYNTHESIZER_SYSTEM
