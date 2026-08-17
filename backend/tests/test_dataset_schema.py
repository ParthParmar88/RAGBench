"""Quote matching for benchmark ground truth.

Two failure modes, in tension:

  too strict  -> real quotes rejected over an en-dash, shrinking the benchmark
  too lenient -> paraphrases accepted, so retrieval that missed the passage
                 still scores as a hit and every experiment reads as better
                 than it is

The second is far more dangerous, because it fails silently and in the flattering
direction. These tests pin both edges.
"""
from evaluation.dataset.schema import normalize, quote_found_in

SOURCE = (
    "Dense Passage Retrieval  (DPR)   outperforms\nBM25 by 9%-19% absolute "
    "in top-20 passage retrieval accuracy."
)


# --- artefacts that must still match --------------------------------------


def test_line_breaks_and_runs_of_spaces_are_folded():
    assert quote_found_in("Dense Passage Retrieval (DPR) outperforms BM25", [SOURCE])


def test_case_is_ignored():
    assert quote_found_in("DENSE PASSAGE RETRIEVAL", [SOURCE])


def test_typographic_dashes_match_ascii_hyphens():
    """PDF extraction emits en-dashes and U+2212; models emit ASCII hyphens."""
    assert quote_found_in("9%–19% absolute", [SOURCE])
    assert quote_found_in("top‑20 passage retrieval", [SOURCE])


def test_unicode_minus_matches():
    assert quote_found_in("9%−19% absolute", [SOURCE])


def test_curly_quotes_match_straight_quotes():
    source = ["the model’s output was called “dense” retrieval"]
    assert quote_found_in("the model's output was called \"dense\" retrieval", source)


def test_ligatures_are_resolved():
    """PDF text often carries U+FB01 for 'fi'."""
    assert quote_found_in("fine-tuning the classifier", ["ﬁne-tuning the classiﬁer"])


def test_leading_and_trailing_ellipses_are_trimmed():
    """Models mark elision with '...'; the enclosed text is still a real quote."""
    assert quote_found_in("...outperforms BM25 by 9%-19%...", [SOURCE])


def test_quote_may_match_any_of_several_texts():
    assert quote_found_in("outperforms BM25", ["unrelated text", SOURCE])


# --- paraphrases that must NOT match ---------------------------------------


def test_paraphrase_is_rejected():
    """The load-bearing check: reworded content must fail."""
    assert not quote_found_in("DPR performs better than BM25", [SOURCE])


def test_reordered_words_are_rejected():
    assert not quote_found_in("BM25 outperforms Dense Passage Retrieval", [SOURCE])


def test_altered_numbers_are_rejected():
    """A hallucinated figure must not slip through on surrounding wording."""
    assert not quote_found_in("outperforms BM25 by 9%-91% absolute", [SOURCE])


def test_dropped_interior_words_are_rejected():
    """Trimming is edge-only; elision inside a quote is still a paraphrase."""
    assert not quote_found_in("Dense Passage Retrieval outperforms by 9%", [SOURCE])


def test_text_absent_from_the_source_is_rejected():
    assert not quote_found_in("outperforms ColBERT", [SOURCE])


def test_empty_source_list_matches_nothing():
    assert not quote_found_in("anything", [])


# --- normalize itself ------------------------------------------------------


def test_normalize_is_idempotent():
    once = normalize(SOURCE)
    assert normalize(once) == once


def test_normalize_preserves_word_order_and_count():
    assert normalize("A  b\nc").split() == ["a", "b", "c"]
