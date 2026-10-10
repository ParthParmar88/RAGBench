"""Verify every test case is answerable from the ingested corpus.

    python -m evaluation.validate_dataset

Run this after any change to the dataset *or* to ingestion. A dataset that
validated under one chunking strategy still validates under another — quotes are
document properties — but a re-ingested corpus with different text extraction
can genuinely invalidate a quote, and it is better to find out here than to
silently score correct retrievals as misses.

Checks:
  - ids unique, category known, question and ground truth present
  - relevant_docs actually exist in the store
  - every supporting quote appears verbatim in its documents
  - unanswerable cases carry no docs or quotes (and every other case does)
  - category distribution is reported, so gaps are visible
"""
import argparse
import sys
from collections import Counter

from sqlalchemy import select

from app.database import SessionLocal
from app.models import Document, KnowledgeUnit
from evaluation.dataset.schema import CATEGORIES, DATASETS, load_dataset, quote_found_in


def corpus_text_by_doc(db) -> dict[str, list[str]]:
    """Every chunk's text, grouped by document title."""
    rows = db.execute(
        select(Document.title, KnowledgeUnit.content).join(
            KnowledgeUnit, KnowledgeUnit.document_id == Document.id
        )
    ).all()
    by_doc: dict[str, list[str]] = {}
    for title, content in rows:
        by_doc.setdefault(title, []).append(content)
    return by_doc


def validate(cases, by_doc) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()

    for case in cases:
        where = f"{case.id}"

        if case.id in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(case.id)

        if case.category not in CATEGORIES:
            problems.append(f"{where}: unknown category {case.category!r}")
        if not case.question.strip():
            problems.append(f"{where}: empty question")
        if not case.ground_truth.strip():
            problems.append(f"{where}: empty ground_truth")

        if case.category == "unanswerable":
            # An unanswerable case with a supporting quote is a contradiction,
            # and would make the refusal check meaningless.
            if case.relevant_docs or case.supporting_quotes:
                problems.append(f"{where}: unanswerable but cites sources")
            continue

        if case.category == "ambiguous":
            # Graded on answer behaviour (does it acknowledge the ambiguity?),
            # not on retrieval, so grounding is optional. Any quote that *is*
            # supplied still has to be real — checked below.
            pass
        elif not case.relevant_docs or not case.supporting_quotes:
            pass  # reported by the explicit checks below

        if case.category != "ambiguous":
            if not case.relevant_docs:
                problems.append(f"{where}: no relevant_docs")
            if not case.supporting_quotes:
                problems.append(f"{where}: no supporting_quotes")

        for doc in case.relevant_docs:
            if doc not in by_doc:
                problems.append(f"{where}: unknown document {doc!r}")

        texts = [t for doc in case.relevant_docs for t in by_doc.get(doc, [])]
        for quote in case.supporting_quotes:
            if not quote_found_in(quote, texts):
                excerpt = " ".join(quote.split())[:60]
                problems.append(f"{where}: quote not found in corpus — {excerpt!r}")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="v1")
    args = parser.parse_args()

    cases = load_dataset(DATASETS[args.dataset])
    if not cases:
        sys.exit("Dataset is empty — run `python -m evaluation.generate_dataset`.")

    db = SessionLocal()
    try:
        by_doc = corpus_text_by_doc(db)
    finally:
        db.close()

    problems = validate(cases, by_doc)

    print(f"{len(cases)} test cases\n")
    counts = Counter(case.category for case in cases)
    for category in CATEGORIES:
        print(f"  {category:16} {counts.get(category, 0):>3}")

    docs = Counter(doc for case in cases for doc in case.relevant_docs)
    print("\ncoverage by document:")
    for title in sorted(by_doc):
        print(f"  {title:24} {docs.get(title, 0):>3} question(s)")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("\nAll cases valid.")


if __name__ == "__main__":
    main()
