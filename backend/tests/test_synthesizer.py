"""Context assembly and citation resolution.

The LLM call is monkeypatched — what's under test is the code around it: how
retrieved chunks become a numbered context block, and how the model's claimed
[n] citations are mapped back to real chunks.

This path is a known measurement confound (the model self-reports which
passages it used), so its edges need to be understood rather than trusted.
"""
from types import SimpleNamespace

import pytest

from app.services import llm, synthesizer
from app.services.synthesizer import SynthesizedAnswer, synthesize_answer
from app.services.vector_store import RetrievedChunk

CHUNKS = [
    RetrievedChunk(id=1, title="dpr.pdf", content="Dense Passage Retrieval", score=0.912345),
    RetrievedChunk(id=2, title="bm25.pdf", content="Okapi BM25 ranking", score=0.734567),
    RetrievedChunk(id=3, title="rrf.pdf", content="Reciprocal Rank Fusion", score=0.512345),
]


@pytest.fixture
def captured(monkeypatch):
    """Replace the LLM call; record its kwargs and return a scripted answer."""
    calls: dict = {}

    def fake_chat_parse(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=calls["_answer"]))]
        )

    monkeypatch.setattr(llm, "chat_parse", fake_chat_parse)

    def run(answer: SynthesizedAnswer | None, chunks=CHUNKS, question="what is RRF?"):
        calls["_answer"] = answer
        return synthesize_answer(question, chunks, **{}), calls

    return run


def test_cited_passages_resolve_to_their_chunks(captured):
    (_, sources), _ = captured(
        SynthesizedAnswer(direct_answer="RRF fuses ranks.", source_numbers=[1, 3])
    )
    assert [s["number"] for s in sources] == [1, 3]
    assert [s["title"] for s in sources] == ["dpr.pdf", "rrf.pdf"]


def test_out_of_range_citations_are_dropped(captured):
    """A hallucinated [7] against 3 passages must not raise or mis-attribute."""
    (_, sources), _ = captured(
        SynthesizedAnswer(direct_answer="...", source_numbers=[0, 2, 7, -1, 99])
    )
    assert [s["number"] for s in sources] == [2]


def test_uncited_chunks_are_absent_from_sources(captured):
    """Documents the confound: `sources` reflects what the model claimed to use,
    not everything retrieved. Retrieval metrics must not be computed from it."""
    (_, sources), _ = captured(
        SynthesizedAnswer(direct_answer="...", source_numbers=[2])
    )
    assert len(sources) == 1 and len(CHUNKS) == 3


def test_scores_are_rounded_for_display(captured):
    (_, sources), _ = captured(
        SynthesizedAnswer(direct_answer="...", source_numbers=[1])
    )
    assert sources[0]["score"] == 0.9123


def test_snippet_is_truncated(captured):
    long_chunk = [RetrievedChunk(id=9, title="big.pdf", content="x" * 1000, score=0.5)]
    (_, sources), _ = captured(
        SynthesizedAnswer(direct_answer="...", source_numbers=[1]), chunks=long_chunk
    )
    assert len(sources[0]["snippet"]) == 280


def test_unparseable_response_degrades_to_a_safe_answer(captured):
    """A refusal or schema failure must not surface as a crash."""
    (answer, sources), _ = captured(None)
    assert "couldn't produce an answer" in answer.direct_answer
    assert sources == []


def test_context_block_numbers_and_titles_every_passage(captured):
    _, calls = captured(SynthesizedAnswer(direct_answer="...", source_numbers=[]))
    user_content = calls["messages"][1]["content"]
    assert "[1] (title: dpr.pdf)" in user_content
    assert "[3] (title: rrf.pdf)" in user_content
    assert "what is RRF?" in user_content


def test_empty_retrieval_still_sends_a_well_formed_prompt(captured):
    """Zero chunks must reach the model as an explicit 'no context' marker, so
    it answers with the grounded refusal instead of free-associating."""
    _, calls = captured(
        SynthesizedAnswer(direct_answer="...", source_numbers=[]), chunks=[]
    )
    assert "(no relevant context found)" in calls["messages"][1]["content"]


def test_operation_is_tagged_for_usage_accounting(captured):
    """usage.persist() picks the primary model by operation == 'synthesize'."""
    _, calls = captured(SynthesizedAnswer(direct_answer="...", source_numbers=[]))
    assert calls["operation"] == "synthesize"


def test_system_prompt_is_the_grounded_synthesizer_prompt(captured):
    _, calls = captured(SynthesizedAnswer(direct_answer="...", source_numbers=[]))
    assert calls["messages"][0]["content"] == synthesizer.SYNTHESIZER_SYSTEM
