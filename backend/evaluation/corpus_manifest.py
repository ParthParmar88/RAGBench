"""The evaluation corpus: which papers, and why each one is here.

Chosen to make the retrieval experiments *falsifiable*. Two properties matter:

  1. Dense with exact identifiers ("ColBERT", "MIPS", "NDCG@10", "MS MARCO").
     Hybrid retrieval can only demonstrate a win on a corpus where exact-token
     matching beats semantic similarity. On flowing prose it would look like a
     failure no matter how well it was implemented.

  2. Mutually comparable. Several papers describe competing retrieval methods,
     so multi-document questions ("how does X differ from Y?") are natural
     rather than contrived.

All are arXiv preprints: natively PDF (the ingestion path is PDF-only), stable,
and freely downloadable. The PDFs are fetched, never committed — `manifest.json`
records SHA-256 checksums so the corpus is reproducible without redistribution.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusEntry:
    arxiv_id: str
    title: str
    filename: str
    # Normalized fragment expected on page 1. Guards against a mistyped arXiv id
    # silently fetching the wrong paper — which would corrupt every ground-truth
    # answer derived from it.
    verify: str
    why: str


CORPUS: list[CorpusEntry] = [
    CorpusEntry(
        "1706.03762", "Attention Is All You Need", "attention.pdf",
        "attentionisallyouneed",
        "Transformer architecture. Dense with exact terms (multi-head attention, "
        "positional encoding) and shares vocabulary with BERT — a good test of "
        "whether retrieval can separate two topically adjacent papers.",
    ),
    CorpusEntry(
        "1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers",
        "bert.pdf", "bertpretrainingofdeepbidirectional",
        "Bidirectional pre-training. Overlaps heavily with the Transformer and "
        "re-ranking papers, so it stresses precision rather than recall.",
    ),
    CorpusEntry(
        "2004.04906", "Dense Passage Retrieval for Open-Domain Question Answering",
        "dpr.pdf", "densepassageretrieval",
        "The dense dual-encoder baseline. Directly comparable to ColBERT, and "
        "explicitly contrasts itself with BM25 — ideal for multi-document "
        "questions and for Experiment 003's core claim.",
    ),
    CorpusEntry(
        "2004.12832", "ColBERT: Efficient and Effective Passage Search",
        "colbert.pdf", "colbert",
        "Late interaction. 'ColBERT' is a rare exact token that embeddings blur "
        "into generic BERT-similarity — the canonical case where lexical search "
        "wins and pure vector search fails.",
    ),
    CorpusEntry(
        "2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
        "rag.pdf", "retrievalaugmentedgeneration",
        "The RAG paper itself. Ties the corpus to the system under test and "
        "supplies questions about the architecture being evaluated.",
    ),
    CorpusEntry(
        "1901.04085", "Passage Re-ranking with BERT", "reranking.pdf",
        "passagereranking",
        "Cross-encoder re-ranking — the technique Experiment 004 implements. "
        "Lets the system answer questions about its own reranker.",
    ),
    CorpusEntry(
        "2104.08663", "BEIR: A Heterogeneous Benchmark for Zero-shot IR",
        "beir.pdf", "beir",
        "Retrieval evaluation methodology. Table-heavy, which also probes how "
        "badly the structure-blind chunker mangles tabular content "
        "(Experiment 002).",
    ),
    CorpusEntry(
        "2307.03172", "Lost in the Middle: How Language Models Use Long Contexts",
        "lostinthemiddle.pdf", "lostinthemiddle",
        "Position bias in long contexts. Directly relevant to why reranking "
        "order matters, not just which chunks are retrieved.",
    ),
]

PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"


def normalize(text: str) -> str:
    """Lowercase, strip everything but a-z0-9.

    PDF text extraction inserts arbitrary line breaks, ligatures and spacing
    into titles, so a substring check only survives aggressive normalization.
    """
    return "".join(c for c in text.lower() if c.isalnum())
