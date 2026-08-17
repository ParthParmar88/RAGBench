"""Adversarial cases, written by hand.

These cannot be generated from the corpus, because their whole point is to sit
*outside* it. A model asked to invent questions from a passage will produce
questions that passage answers — the opposite of what's needed here.

Two kinds:

`unanswerable` — plausible for a corpus of IR/NLP papers, but genuinely absent
from these eight. Mostly things that postdate the papers or belong to another
field. A system scoring well everywhere else and confidently answering these is
not trustworthy: it is guessing fluently. This is the category most likely to
expose a regression that average scores hide.

`ambiguous` — underspecified, with several defensible readings. What's measured
is whether the answer acknowledges the ambiguity rather than silently picking
one interpretation and asserting it.

Neither category carries supporting quotes. For `unanswerable` there is nothing
to quote. For `ambiguous` the thing under test is answer behaviour, not
retrieval, and inventing quotes from memory would risk asserting text that
isn't in the corpus — the exact failure the quote check exists to prevent.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Handwritten:
    question: str
    ground_truth: str
    category: str
    notes: str = ""
    relevant_docs: tuple[str, ...] = field(default_factory=tuple)
    supporting_quotes: tuple[str, ...] = field(default_factory=tuple)


REFUSAL = (
    "This is not answerable from the provided documents. The correct behaviour "
    "is to say so rather than to answer."
)


UNANSWERABLE = [
    Handwritten(
        "How does Mamba's selective state-space mechanism compare to attention "
        "for long-sequence modelling?",
        REFUSAL,
        "unanswerable",
        "Mamba (2023) postdates every paper here. Tests whether a question "
        "phrased in fluent in-domain vocabulary is answered anyway.",
    ),
    Handwritten(
        "What was the total compute cost, in FLOPs, of training GPT-4?",
        REFUSAL,
        "unanswerable",
        "No paper here discusses GPT-4. Numeric questions are the riskiest to "
        "hallucinate because a plausible number looks like a real citation.",
    ),
    Handwritten(
        "What learning rate and batch size did LLaMA 3 use during pre-training?",
        REFUSAL,
        "unanswerable",
        "Hyperparameter questions retrieve hyperparameter tables from the wrong "
        "paper. Tests whether the model notices the model name doesn't match.",
    ),
    Handwritten(
        "What top-1 accuracy does ColBERT achieve on ImageNet?",
        REFUSAL,
        "unanswerable",
        "Category error: ColBERT is passage retrieval, not image classification. "
        "'ColBERT' will retrieve strongly, so the retrieved context looks "
        "relevant while the question is nonsense.",
    ),
    Handwritten(
        "How many researchers are employed at Google DeepMind?",
        REFUSAL,
        "unanswerable",
        "Out of domain entirely. The easy case — a system failing this fails "
        "badly.",
    ),
    Handwritten(
        "How much does it cost in US dollars to train BERT-large on AWS?",
        REFUSAL,
        "unanswerable",
        "BERT training is described; its dollar cost is not. Tests the boundary "
        "between what a paper says and what can be inferred from it.",
    ),
    Handwritten(
        "What does the BEIR benchmark report for GPT-4 embeddings in zero-shot "
        "retrieval?",
        REFUSAL,
        "unanswerable",
        "BEIR predates GPT-4. Both 'BEIR' and 'zero-shot retrieval' retrieve "
        "strongly, so retrieval scores will look healthy on a bad question.",
    ),
    Handwritten(
        "Which Python version does the reference DPR implementation require?",
        REFUSAL,
        "unanswerable",
        "Implementation detail absent from the paper. Tests confusion between "
        "the paper and its codebase.",
    ),
]


AMBIGUOUS = [
    Handwritten(
        "Which model performs best?",
        "Underspecified: 'best' depends on the task, benchmark and metric, and "
        "the corpus covers several models evaluated on different benchmarks. A "
        "good answer notes the ambiguity and asks which task or benchmark is "
        "meant, or answers conditionally.",
        "ambiguous",
        "No referent for 'best'. The failure mode is confidently naming one.",
    ),
    Handwritten(
        "How many documents should be retrieved?",
        "Depends on the setup: the corpus discusses different retrieval depths "
        "for different systems, and one paper shows that more context is not "
        "uniformly better because of position effects. A good answer gives the "
        "trade-off rather than one number.",
        "ambiguous",
        "Invites a spuriously specific number. Genuinely contested in the corpus.",
    ),
    Handwritten(
        "Is dense retrieval better than sparse retrieval?",
        "Contested: the corpus contains evidence both ways — dense methods "
        "outperform lexical baselines on some benchmarks while lexical methods "
        "remain competitive on others, particularly out-of-domain. A good answer "
        "presents both rather than picking a side.",
        "ambiguous",
        "The corpus genuinely disagrees with itself. Tests whether conflicting "
        "retrieved evidence is surfaced or silently resolved.",
    ),
    Handwritten(
        "How large is the model?",
        "Unanswerable as posed: no model is specified, and the corpus describes "
        "many of differing sizes. A good answer asks which model is meant.",
        "ambiguous",
        "Minimal-context question. Retrieval returns parameter counts from "
        "whichever paper ranks highest, which is arbitrary.",
    ),
    Handwritten(
        "What is the recommended chunk size for splitting documents?",
        "The corpus does not prescribe a chunk size; passage length varies by "
        "system and is not treated as a tunable recommendation. A good answer "
        "says the documents don't recommend one.",
        "ambiguous",
        "Sounds like a RAG-corpus question and is adjacent to real content on "
        "passage length — a near-miss rather than a clean absence.",
    ),
    Handwritten(
        "How many layers should the encoder have?",
        "Depends on the model: the corpus reports different depths for different "
        "architectures and does not give a general recommendation. A good answer "
        "declines to give a single number.",
        "ambiguous",
        "Layer counts appear in several papers, so retrieval succeeds while the "
        "question has no single answer.",
    ),
]


HANDWRITTEN: list[Handwritten] = UNANSWERABLE + AMBIGUOUS
