"""Deterministic metrics — no LLM judge involved.

These exist because RAGAS is LLM-graded and therefore noisy: the same system
evaluated twice will not produce identical scores. That noise floor has to be
measured before any improvement can be believed, and it makes small effects
unresolvable.

The metrics here are exact set operations over text. They return the same number
every time, so a change in `quote_recall` between two experiments is a real
change in retrieval, full stop — no significance hand-waving required. When
RAGAS and these disagree, these are the ones to trust about *retrieval*; RAGAS
is still the authority on *answer quality*, which text matching cannot judge.

Ground truth is anchored to verbatim quotes rather than chunk ids, so every
metric here survives a change of chunking strategy — the point of Experiment 002.
"""
import math
from dataclasses import dataclass

from evaluation.dataset.schema import TestCase, normalize, quote_found_in

# Phrases that count as declining to answer. The synthesizer prompt instructs
# the exact first form; the rest catch paraphrases of it.
REFUSAL_MARKERS = (
    "couldn't find that information",
    "could not find that information",
    "couldn't find any information",
    "could not find any information",
    "not found in the provided documents",
    "isn't in the provided documents",
    "is not in the provided documents",
    "don't have information",
    "do not have information",
    "no information about",
    "not mentioned in the provided",
    "not addressed in the provided",
    "doesn't appear in the provided",
    "does not appear in the provided",
)


@dataclass
class RetrievalScore:
    """Per-case retrieval outcome, computed without an LLM."""

    quote_hit: bool  # at least one supporting quote was retrieved
    quote_recall: float  # fraction of supporting quotes retrieved
    doc_precision: float  # fraction of retrieved chunks from a relevant document
    doc_hit: bool  # at least one retrieved chunk from a relevant document


def score_retrieval(case: TestCase, contexts: list[str], titles: list[str]) -> RetrievalScore:
    """Score one case's retrieval against its quote-anchored ground truth.

    `contexts` and `titles` are parallel: the text and source document of each
    retrieved chunk.
    """
    if not case.supporting_quotes:
        # `ambiguous` and `unanswerable` cases have nothing to retrieve. Scoring
        # them as 0.0 would drag the corpus average down for no reason; they are
        # excluded from retrieval aggregates instead.
        return RetrievalScore(False, 0.0, 0.0, False)

    found = sum(1 for q in case.supporting_quotes if quote_found_in(q, contexts))
    recall = found / len(case.supporting_quotes)

    relevant = set(case.relevant_docs)
    from_relevant = sum(1 for t in titles if t in relevant)
    precision = from_relevant / len(titles) if titles else 0.0

    return RetrievalScore(
        quote_hit=found > 0,
        quote_recall=recall,
        doc_precision=precision,
        doc_hit=from_relevant > 0,
    )


def is_refusal(answer: str) -> bool:
    """True if the answer declines to answer rather than asserting something.

    Used two ways, in opposite directions:
      - on `unanswerable` cases, refusing is CORRECT (higher is better)
      - on answerable cases, refusing is a miss (lower is better)

    A system that refuses everything scores perfectly on the first and terribly
    on the second, so the pair can't be gamed.
    """
    text = normalize(answer)
    return any(marker in text for marker in REFUSAL_MARKERS)


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile. Reported alongside the mean because latency is
    tail-heavy — a good average hides the queries users actually complain about.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(pct / 100 * len(ordered)) - 1
    return round(ordered[min(max(index, 0), len(ordered) - 1)], 4)
