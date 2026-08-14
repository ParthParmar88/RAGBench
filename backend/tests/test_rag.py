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
