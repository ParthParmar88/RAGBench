"""Passage filtering and question self-containment.

Both guard benchmark quality rather than correctness, and both were added after
reading the first generated batch: bibliography chunks produced questions about
arXiv identifiers, and pair prompts leaked "Passage A" into the question text.
Neither defect is detectable by the schema validator — the cases were
structurally perfect and semantically worthless.
"""
from evaluation.generate_dataset import is_reference_section, leaks_scaffold

# Shortened from colbert.pdf chunk 55 — a real bibliography.
REFERENCES = """
Fernando Diaz, and Emine Yilmaz. 2019. Incorporating query term independence.
arXiv preprint arXiv:1904.09171 (2019).
Matthew Peters, Mark Neumann, and Luke Zettlemoyer. 2018. Deep contextualized
word representations. arXiv preprint arXiv:1802.05365 (2018).
Po-Sen Huang, Xiaodong He, and Larry Heck. 2013. Learning deep structured
semantic models. In Proceedings of the 22nd ACM International Conference.
"""

# Shortened from bert.pdf chunk 5 — citation-dense prose that must be KEPT.
RELATED_WORK = """
2 Related Work
There is a long history of pre-training general language representations, and
we briefly review the most widely-used approaches in this section. Learning
widely applicable representations of words has been an active area of research
for decades (Mikolov et al., 2013; Pennington et al., 2014), including
non-neural (Brown et al., 1992) and neural (Turian et al., 2010) methods.
"""

BODY = """
The Transformer follows this overall architecture using stacked self-attention
and point-wise, fully connected layers for both the encoder and decoder. The
encoder is composed of a stack of N = 6 identical layers.
"""


# --- reference filtering ---------------------------------------------------


def test_bibliography_is_excluded():
    assert is_reference_section(REFERENCES)


def test_citation_dense_prose_is_kept():
    """The discriminating case.

    Related-work sections are year-dense like bibliographies but carry no venue
    or arXiv markers. A year-only threshold would discard them, costing real
    technical content — measured at 26% of this corpus.
    """
    assert not is_reference_section(RELATED_WORK)


def test_ordinary_body_text_is_kept():
    assert not is_reference_section(BODY)


def test_empty_text_is_not_a_reference_section():
    assert not is_reference_section("")


# --- scaffold leakage ------------------------------------------------------


def test_passage_letter_references_are_rejected():
    assert leaks_scaffold(
        "How do the capabilities in Passage A compare with the metrics in Passage B?"
    )


def test_bare_passage_references_are_rejected():
    assert leaks_scaffold("What does the passage say about dropout?")
    assert leaks_scaffold("According to this excerpt, what is the BLEU score?")
    assert leaks_scaffold("What are both passages describing?")


def test_self_contained_questions_are_accepted():
    """Questions that name their subject must survive the filter."""
    assert not leaks_scaffold(
        "How does ColBERT's late interaction differ from DPR's dual encoder?"
    )
    assert not leaks_scaffold("What BLEU score did the big Transformer achieve?")


def test_the_word_document_is_not_treated_as_leakage():
    """'Document' is ordinary IR vocabulary; filtering it would reject valid
    retrieval questions."""
    assert not leaks_scaffold("How many documents does DPR retrieve per query?")
    assert not leaks_scaffold("What is the second stage of the ranking pipeline?")


# --- held-out split ----------------------------------------------------------


def test_holdout_excludes_passages_the_tuning_split_used():
    """A held-out question written from a passage v1 already used would test
    memorised territory. The passage must leave the sampling pool."""
    from evaluation.dataset.schema import TestCase
    from evaluation.generate_dataset import Passage, exclude_used

    used = Passage("bert.pdf", "BERT is pre-trained on two tasks. Masked LM is one of them.")
    fresh = Passage("bert.pdf", "Fine-tuning is straightforward for sequence tasks.")
    other = Passage("dpr.pdf", "DPR uses a dual-encoder trained with in-batch negatives.")
    corpus = {"bert.pdf": [used, fresh], "dpr.pdf": [other]}
    v1 = [TestCase("q001", "?", "a", "factual", ["bert.pdf"], ["Masked LM is one of them."])]

    remaining = exclude_used(corpus, v1)

    assert remaining["bert.pdf"] == [fresh]
    assert remaining["dpr.pdf"] == [other]


def test_exclusion_survives_extraction_artefacts():
    """Quotes are matched with the same normalization as everywhere else, so a
    line break inside the passage must not let a used passage slip through."""
    from evaluation.dataset.schema import TestCase
    from evaluation.generate_dataset import Passage, exclude_used

    passage = Passage("bert.pdf", "Masked LM is one\nof   them.")
    v1 = [TestCase("q001", "?", "a", "factual", ["bert.pdf"], ["Masked LM is one of them."])]

    assert exclude_used({"bert.pdf": [passage]}, v1) == {"bert.pdf": []}


def test_holdout_split_is_independent_of_the_tuning_split():
    from evaluation.generate_dataset import SPLITS

    v1, holdout = SPLITS["v1"], SPLITS["holdout"]
    assert holdout.exclude == "v1"
    assert holdout.seed != v1.seed
    assert holdout.id_prefix != v1.id_prefix
    # No hand-written adversarial question may appear in both.
    assert not {c.question for c in v1.handwritten} & {c.question for c in holdout.handwritten}
