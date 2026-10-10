"""Deterministic evaluation metrics.

These carry more weight than usual: RAGAS is LLM-graded and noisy, so these are
the numbers that decide whether a retrieval change was real. A bug here would
silently make every experiment conclusion wrong in a way no later check catches.
"""
from evaluation.dataset.schema import TestCase
from evaluation.metrics import is_refusal, mean, percentile, score_retrieval

QUOTE_A = "Dense Passage Retrieval outperforms BM25"
QUOTE_B = "late interaction over BERT embeddings"


def case(quotes=(QUOTE_A,), docs=("dpr.pdf",), category="factual") -> TestCase:
    return TestCase(
        id="q001", question="?", ground_truth="!", category=category,
        relevant_docs=list(docs), supporting_quotes=list(quotes),
    )


# --- quote recall ----------------------------------------------------------


def test_retrieved_quote_counts_as_a_hit():
    score = score_retrieval(
        case(), ["... Dense Passage Retrieval outperforms BM25 by 9% ..."], ["dpr.pdf"]
    )
    assert score.quote_hit and score.quote_recall == 1.0


def test_missing_quote_is_a_miss():
    score = score_retrieval(case(), ["something entirely unrelated"], ["dpr.pdf"])
    assert not score.quote_hit and score.quote_recall == 0.0


def test_partial_recall_on_multi_quote_cases():
    """Multi-chunk cases need both passages; finding one is half credit."""
    score = score_retrieval(
        case(quotes=(QUOTE_A, QUOTE_B)),
        ["Dense Passage Retrieval outperforms BM25 substantially"],
        ["dpr.pdf"],
    )
    assert score.quote_recall == 0.5
    assert score.quote_hit  # found at least one


def test_quote_matching_survives_pdf_whitespace():
    """The property that makes this metric chunking-invariant."""
    score = score_retrieval(
        case(), ["Dense Passage\n  Retrieval   outperforms\nBM25"], ["dpr.pdf"]
    )
    assert score.quote_hit


# --- document precision ----------------------------------------------------


def test_doc_precision_counts_chunks_from_relevant_documents():
    score = score_retrieval(
        case(), ["x", "y", "z", "w"], ["dpr.pdf", "dpr.pdf", "bert.pdf", "rag.pdf"]
    )
    assert score.doc_precision == 0.5
    assert score.doc_hit


def test_doc_precision_is_zero_when_nothing_relevant_retrieved():
    score = score_retrieval(case(), ["x"], ["bert.pdf"])
    assert score.doc_precision == 0.0
    assert not score.doc_hit


def test_multi_document_case_counts_both_documents():
    score = score_retrieval(
        case(docs=("dpr.pdf", "colbert.pdf")), ["x", "y"], ["dpr.pdf", "colbert.pdf"]
    )
    assert score.doc_precision == 1.0


def test_empty_retrieval_scores_zero_without_dividing_by_zero():
    score = score_retrieval(case(), [], [])
    assert score.doc_precision == 0.0 and score.quote_recall == 0.0


def test_cases_without_quotes_are_neutral():
    """Ambiguous/unanswerable cases have nothing to retrieve; they must not be
    scored as retrieval failures and drag the average down."""
    score = score_retrieval(case(quotes=(), docs=()), ["x"], ["dpr.pdf"])
    assert not score.quote_hit and score.quote_recall == 0.0


# --- refusal detection -----------------------------------------------------


def test_the_prompted_refusal_is_detected():
    assert is_refusal(
        "I couldn't find that information in the provided documents."
    )


def test_paraphrased_refusals_are_detected():
    assert is_refusal("The documents do not have information on Mamba.")
    assert is_refusal("This is not mentioned in the provided sources.")


def test_a_real_answer_is_not_a_refusal():
    assert not is_refusal("The big Transformer achieved a BLEU score of 28.4.")


def test_an_answer_mentioning_documents_is_not_a_refusal():
    """Guards against matching on the word 'documents' alone."""
    assert not is_refusal("DPR retrieves 100 documents per query from the index.")


# --- aggregates ------------------------------------------------------------


def test_mean_of_empty_is_none_not_zero():
    """None means 'not measured'; 0.0 would be a claim about performance."""
    assert mean([]) is None
    assert mean([1.0, 2.0]) == 1.5


def test_percentile_picks_the_tail():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0


def test_percentile_of_single_value():
    assert percentile([7.0], 95) == 7.0


def test_percentile_of_empty_is_none():
    assert percentile([], 95) is None


# --- rescoring stored experiments -------------------------------------------


def test_rescore_recomputes_from_stored_contexts():
    """A stored row carries a stale score; rescoring must replace it using only
    the saved contexts, and report which cases changed."""
    from evaluation.dataset.schema import TestCase
    from evaluation.rescore import rescore_rows

    cases = {
        "q1": TestCase("q1", "?", "a", "factual", ["a.pdf"], ["the real quote"]),
        "q2": TestCase("q2", "?", "a", "factual", ["a.pdf"], ["another quote"]),
        "q3": TestCase("q3", "?", "refuse", "unanswerable"),
    }
    rows = [
        # Stored as a hit, but the quote is not in the contexts: stale.
        {"id": "q1", "category": "factual", "contexts": ["unrelated"], "context_titles": ["a.pdf"],
         "retrieval": {"quote_hit": True, "quote_recall": 1.0}},
        {"id": "q2", "category": "factual", "contexts": ["has another quote here"],
         "context_titles": ["a.pdf"], "retrieval": {"quote_hit": True, "quote_recall": 1.0}},
        # Nothing to retrieve: excluded from retrieval aggregates.
        {"id": "q3", "category": "unanswerable", "contexts": ["x"], "context_titles": ["a.pdf"],
         "retrieval": {"quote_hit": False, "quote_recall": 0.0}},
    ]

    result = rescore_rows(rows, cases)

    assert result["retrieval"]["n"] == 2
    assert result["retrieval"]["quote_hit_rate"] == 0.5
    assert result["changed_cases"] == ["q1"]
    assert result["per_case"]["q1"]["quote_hit"] is False
    assert result["retrieval"]["by_category"]["factual"] == {
        "n": 2, "quote_hit_rate": 0.5, "quote_recall": 0.5,
    }
