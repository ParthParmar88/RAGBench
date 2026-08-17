"""Download the evaluation corpus and record checksums.

    python -m evaluation.fetch_corpus

Idempotent: a file already present with a matching checksum is not re-fetched.
Every download is verified against the expected title before it is accepted, so
a wrong arXiv id fails loudly instead of quietly poisoning the benchmark.

Writes `evaluation/corpus/manifest.json` — the reproducibility record. The PDFs
themselves are gitignored; anyone with the manifest can rebuild the exact corpus.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

from evaluation.corpus_manifest import CORPUS, PDF_URL, CorpusEntry, normalize

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
MANIFEST = CORPUS_DIR / "manifest.json"

# arXiv asks for unthrottled crawling to identify itself and pace requests.
HEADERS = {"User-Agent": "paraX-eval-corpus/1.0 (RAG evaluation benchmark)"}
DELAY_SECONDS = 3.0


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(entry: CorpusEntry, data: bytes) -> tuple[bool, str]:
    """Confirm the bytes are a PDF whose first page carries the expected title."""
    if not data.startswith(b"%PDF"):
        return False, "not a PDF (arXiv may have returned an error page)"

    # Imported lazily: this module is also used to plan a fetch without pypdf.
    from io import BytesIO

    from pypdf import PdfReader

    try:
        first_page = PdfReader(BytesIO(data)).pages[0].extract_text() or ""
    except Exception as exc:  # noqa: BLE001 - report, don't crash the whole run
        return False, f"unreadable PDF: {type(exc).__name__}"

    if entry.verify not in normalize(first_page):
        got = " ".join(first_page.split())[:70]
        return False, f"title mismatch — page 1 starts {got!r}"
    return True, "ok"


def fetch_one(client: httpx.Client, entry: CorpusEntry) -> dict:
    path = CORPUS_DIR / entry.filename

    if path.exists():
        data = path.read_bytes()
        ok, reason = verify(entry, data)
        if ok:
            print(f"  cached   {entry.filename:22} {len(data):>9,} bytes")
            return _record(entry, data)
        print(f"  stale    {entry.filename:22} {reason} — refetching")

    response = client.get(PDF_URL.format(arxiv_id=entry.arxiv_id))
    response.raise_for_status()
    data = response.content

    ok, reason = verify(entry, data)
    if not ok:
        raise ValueError(f"{entry.arxiv_id} ({entry.title}): {reason}")

    path.write_bytes(data)
    print(f"  fetched  {entry.filename:22} {len(data):>9,} bytes")
    return _record(entry, data)


def _record(entry: CorpusEntry, data: bytes) -> dict:
    return {
        "arxiv_id": entry.arxiv_id,
        "title": entry.title,
        "filename": entry.filename,
        "sha256": sha256(data),
        "bytes": len(data),
        "source_url": PDF_URL.format(arxiv_id=entry.arxiv_id),
        "why": entry.why,
    }


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(CORPUS)} papers into {CORPUS_DIR}\n")

    records: list[dict] = []
    failures: list[str] = []

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60.0) as client:
        for index, entry in enumerate(CORPUS):
            try:
                records.append(fetch_one(client, entry))
            except Exception as exc:  # noqa: BLE001 - collect, report at the end
                print(f"  FAILED   {entry.filename:22} {exc}")
                failures.append(f"{entry.arxiv_id} {entry.title}: {exc}")
            if index < len(CORPUS) - 1:
                time.sleep(DELAY_SECONDS)

    MANIFEST.write_text(
        json.dumps(
            {"corpus_version": "v1", "papers": records}, indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n{len(records)}/{len(CORPUS)} verified -> {MANIFEST.name}")
    if failures:
        print("\nFAILURES (corpus is incomplete — do not evaluate against it):")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()
