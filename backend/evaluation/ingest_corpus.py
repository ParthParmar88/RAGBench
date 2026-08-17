"""Ingest the fetched corpus into the live store.

    python -m evaluation.ingest_corpus
    python -m evaluation.ingest_corpus --status    # report only, ingest nothing
    python -m evaluation.ingest_corpus --reingest  # force rebuild

Deliberately routed through `services.ingestion.ingest_pdf` — the same code path
the upload endpoint uses. Evaluation must measure the real pipeline, so a
separate ingestion path for benchmarking would defeat the purpose.

Ingestion is idempotent, so re-running this is cheap and safe: unchanged papers
are skipped without re-embedding.

That idempotency is keyed on the PDF's bytes, which do not change when the
*pipeline* does. After upgrading the PDF library or changing the chunking
strategy, the stored chunks are stale even though every checksum still matches
— use `--reingest` to force a rebuild. Experiment 002 will need this.
"""
import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import func, select

from app.database import SessionLocal, init_db
from app.models import Document, KnowledgeUnit
from app.services import ingestion, vector_store

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
MANIFEST = CORPUS_DIR / "manifest.json"


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        sys.exit("No manifest.json — run `python -m evaluation.fetch_corpus` first.")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["papers"]


def report(db) -> None:
    """Show every document in the store, flagging any that aren't the corpus."""
    corpus_titles = {paper["filename"] for paper in load_manifest()}

    rows = db.execute(
        select(
            Document.title,
            Document.page_count,
            func.count(KnowledgeUnit.id),
            func.count(KnowledgeUnit.page),
        )
        .outerjoin(KnowledgeUnit, KnowledgeUnit.document_id == Document.id)
        .group_by(Document.id, Document.title, Document.page_count)
        .order_by(Document.title)
    ).all()

    print(f"\n{'document':24} {'pages':>6} {'chunks':>7} {'provenance':>11}")
    print("-" * 52)
    corpus_chunks = foreign_chunks = 0
    foreign: list[str] = []

    for title, pages, chunks, with_page in rows:
        in_corpus = title in corpus_titles
        marker = "" if in_corpus else "   <- not corpus"
        coverage = f"{with_page}/{chunks}"
        print(f"{title[:24]:24} {pages:>6} {chunks:>7} {coverage:>11}{marker}")
        if in_corpus:
            corpus_chunks += chunks
        else:
            foreign_chunks += chunks
            foreign.append(title)

    print("-" * 52)
    print(f"{'corpus':24} {'':>6} {corpus_chunks:>7}")

    if foreign:
        print(
            f"\nWARNING: {foreign_chunks} chunk(s) in the store are not part of the\n"
            f"corpus and will compete in retrieval, making results irreproducible:"
        )
        for title in foreign:
            print(f"  - {title}")
        print("Remove them before running the benchmark.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="report only")
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="re-extract and re-embed even if the PDF is unchanged",
    )
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        if not args.status:
            papers = load_manifest()
            print(f"Ingesting {len(papers)} papers\n")
            for paper in papers:
                path = CORPUS_DIR / paper["filename"]
                if not path.exists():
                    print(f"  MISSING  {paper['filename']} — re-run fetch_corpus")
                    continue
                if args.reingest:
                    # Drop first so the checksum can't short-circuit the rebuild.
                    vector_store.delete_source(db, paper["filename"])
                result = ingestion.ingest_pdf(db, paper["filename"], path.read_bytes())
                print(
                    f"  {result.status:10} {result.title:22} "
                    f"{result.page_count:>3} pages  {result.chunks_added:>4} chunks"
                )
        report(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
