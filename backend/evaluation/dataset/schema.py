"""The benchmark's test-case format, and what makes a case valid.

Ground truth is anchored to **verbatim quotes**, not chunk ids.

Chunk ids are an artefact of one chunking strategy: re-ingesting with different
boundaries renumbers everything, and Experiment 002 changes chunking on purpose.
A benchmark keyed on chunk ids would break precisely when it is needed most.

A quote, by contrast, is a property of the source document. "Was a passage
containing this sentence retrieved?" is answerable under any chunking strategy,
which makes context recall comparable across every experiment.

Quotes are verified to appear in the ingested corpus (see validate_dataset.py).
A quote the LLM paraphrased rather than copied is rejected — an unverifiable
ground truth is worse than none, because it silently scores real retrievals as
failures.
"""
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
DATASET_V1 = DATASET_DIR / "v1.jsonl"

CATEGORIES = {
    "factual": "A specific fact stated in one passage. The easy baseline case.",
    "semantic": (
        "Conceptual question phrased with little vocabulary overlap with the "
        "source. Vector search should win here; BM25 alone should struggle."
    ),
    "keyword": (
        "Turns on an exact identifier or technical term. BM25 should win here; "
        "embeddings blur rare tokens into their topical neighbourhood. This is "
        "the category Experiment 003 must improve or it has failed."
    ),
    "multi_chunk": (
        "Needs two or more passages from the same paper. Tests whether the "
        "chunker severed related content — the Experiment 002 signal."
    ),
    "multi_document": (
        "Needs passages from two or more papers, usually a comparison. Tests "
        "whether top-k is spent on a single document."
    ),
    "ambiguous": (
        "Underspecified or with several defensible readings. Tests whether the "
        "system hedges appropriately instead of confidently picking one."
    ),
    "unanswerable": (
        "Plausible-sounding but not answerable from the corpus. Catches "
        "confident hallucination — a system that scores well everywhere else "
        "and fails here is not trustworthy."
    ),
}


@dataclass
class TestCase:
    # Not a pytest test class, despite the name.
    __test__ = False

    id: str
    question: str
    ground_truth: str
    category: str
    # Filenames expected to contain the answer. Empty for `unanswerable`.
    relevant_docs: list[str] = field(default_factory=list)
    # Verbatim strings from those documents. Retrieval succeeds if a returned
    # chunk contains one. Empty for `unanswerable`.
    supporting_quotes: list[str] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# PDF text uses typographic characters the model reproduces as ASCII (and
# occasionally the reverse). These differ in *rendering*, never in wording, so
# folding them cannot let a paraphrase pass — it only stops a real quote from
# being rejected over an en-dash.
_DASHES = dict.fromkeys(
    map(ord, "‐‑‒–—―−"), "-"
)
_QUOTES = {ord(c): "'" for c in "‘’‛"} | {
    ord(c): '"' for c in "“”‟"
}
# Models mark elision with a leading/trailing ellipsis. Trimming the edges is
# safe for a containment check; interior punctuation is untouched.
_EDGE_ELLIPSIS = re.compile(r"^[\s.…]+|[\s.…]+$")


def normalize(text: str) -> str:
    """Fold extraction artefacts so a genuine quote still matches its source.

    PDF extraction inserts line breaks mid-sentence, renders ligatures as single
    codepoints, and emits typographic dashes and quotes. NFKC resolves the
    ligatures and compatibility forms; the tables above handle punctuation that
    NFKC leaves alone.

    Deliberately conservative about wording: nothing here reorders, stems, or
    drops words, so a paraphrase still fails.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DASHES).translate(_QUOTES)
    text = _EDGE_ELLIPSIS.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_found_in(quote: str, texts: list[str]) -> bool:
    """True if `quote` appears in any of `texts` after normalization."""
    needle = normalize(quote)
    return any(needle in normalize(text) for text in texts)


def load_dataset(path: Path = DATASET_V1) -> list[TestCase]:
    if not path.exists():
        return []
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(TestCase(**json.loads(line)))
    return cases


def save_dataset(cases: list[TestCase], path: Path = DATASET_V1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(case.to_json() for case in cases) + "\n", encoding="utf-8"
    )
