"""Run the benchmark against the current pipeline and record the result.

    python -m evaluation.run_experiment --id exp-001-baseline
    python -m evaluation.run_experiment --id smoke --limit 5 --no-ragas
    python -m evaluation.run_experiment --id exp-003-rerank --rerank

Two passes over the data:

  1. The pipeline pass runs every case through `rag.answer_question` and records
     answers, retrieved contexts, latency and token usage. Deterministic
     retrieval metrics are computed here — no judge involved.
  2. The RAGAS pass grades answer quality with an LLM judge.

Artifacts land in `results/<id>/`:

    config.json      what was run (models, settings, dataset version)
    per_query.jsonl  one row per case — required for failure analysis, since
                     averages cannot tell you *which* questions changed
    summary.json     aggregates

The judge model is pinned and recorded in config.json. An unpinned judge silently
changes the yardstick between experiments, which would make every comparison
meaningless.
"""
import argparse
import json
import platform
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.database import SessionLocal
from app.services import rag
from evaluation.dataset.schema import DATASETS, load_dataset
from evaluation.metrics import is_refusal, mean, percentile, score_retrieval

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Pinned judge. Deliberately separate from the answering model in the config so
# the two can diverge later without silently rescoring past experiments.
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_EMBEDDINGS = "text-embedding-3-small"


def run_pipeline(cases, limit=None, top_k=None) -> list[dict]:
    """Pass 1 — execute every case, record everything measurable."""
    db = SessionLocal()
    rows: list[dict] = []
    try:
        selected = cases[:limit] if limit else cases
        for index, case in enumerate(selected, start=1):
            started = time.perf_counter()
            error = None
            try:
                result = rag.answer_question(db, case.question, request_id=case.id, top_k=top_k)
            except Exception as exc:  # noqa: BLE001 - one bad case must not end the run
                error = f"{type(exc).__name__}: {exc}"
                result = None

            if result is None:
                rows.append(
                    {"id": case.id, "category": case.category,
                     "question": case.question, "error": error}
                )
                print(f"  [{index:>3}/{len(selected)}] {case.id} ERROR {error}", flush=True)
                continue

            contexts = [c.content for c in result.contexts]
            titles = [c.title for c in result.contexts]
            score = score_retrieval(case, contexts, titles)
            usage = result.usage

            rows.append({
                "id": case.id,
                "category": case.category,
                "question": case.question,
                "ground_truth": case.ground_truth,
                "answer": result.direct_answer,
                "contexts": contexts,
                "context_titles": titles,
                "context_pages": [c.page for c in result.contexts],
                "relevant_docs": case.relevant_docs,
                "supporting_quotes": case.supporting_quotes,
                "cited_sources": [s["title"] for s in result.sources],
                "retrieval": asdict(score),
                "refused": is_refusal(result.direct_answer),
                "top_score": result.retrieval.top_score,
                "top_rerank_score": result.retrieval.top_rerank_score,
                "gated": result.gated,
                "avg_score": result.retrieval.avg_score,
                "chunk_count": result.retrieval.chunk_count,
                "embedding_ms": result.timings.embedding_ms,
                "search_ms": result.timings.search_ms,
                "rerank_ms": result.timings.rerank_ms,
                "retrieval_ms": result.timings.retrieval_ms,
                "generation_ms": result.timings.generation_ms,
                "total_ms": result.timings.total_ms,
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "cost": usage.cost if usage else None,
                "wall_ms": int((time.perf_counter() - started) * 1000),
                "error": None,
            })
            flag = "HIT " if score.quote_hit else "miss"
            if case.category in ("unanswerable", "ambiguous"):
                flag = "REF " if rows[-1]["refused"] else "ans "
            print(
                f"  [{index:>3}/{len(selected)}] {case.id} {case.category:15}"
                f" {flag} {result.timings.total_ms:>5}ms",
                flush=True,
            )
    finally:
        db.close()
    return rows


def run_ragas(rows: list[dict]) -> dict:
    """Pass 2 — LLM-judged answer quality.

    Only cases with a reference answer AND retrieved context are graded.
    `unanswerable` cases are excluded: their reference is an instruction to
    refuse, which faithfulness and context-recall would score as nonsense.
    They are measured by `refusal_rate` instead.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

    gradable = [
        r for r in rows
        if not r.get("error")
        and r["category"] not in ("unanswerable", "ambiguous")
        and r.get("contexts")
    ]
    if not gradable:
        return {"error": "no gradable rows"}

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            response=r["answer"],
            reference=r["ground_truth"],
        )
        for r in gradable
    ]

    # The key is loaded from .env into settings by pydantic-settings and never
    # reaches os.environ, which is where langchain looks by default. Pass it
    # explicitly rather than mutating the environment.
    key = settings.OPENAI_API_KEY
    judge = LangchainLLMWrapper(
        ChatOpenAI(model=JUDGE_MODEL, temperature=0.0, api_key=key)
    )
    embedder = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=JUDGE_EMBEDDINGS, api_key=key)
    )

    print(f"\nRAGAS: grading {len(samples)} cases with {JUDGE_MODEL} ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge,
            embeddings=embedder,
            show_progress=True,
        )

    frame = result.to_pandas()
    per_case = {}
    scores = {}
    # How many cases each metric actually scored. A judge call that times out
    # comes back as NaN and is dropped from the mean, so a "score" can silently
    # be an average over a handful of cases. The held-out runs (exp-005) had
    # context_precision scored on 1 of 34; without this count, that mean looked
    # like any other number.
    scored = {}
    for column in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        if column not in frame.columns:
            continue
        values = [float(v) for v in frame[column].tolist() if v == v]  # drop NaN
        scores[column] = round(sum(values) / len(values), 4) if values else None
        scored[column] = len(values)
        for row, value in zip(gradable, frame[column].tolist()):
            per_case.setdefault(row["id"], {})[column] = (
                None if value != value else round(float(value), 4)
            )

    return {"graded": len(samples), "scores": scores, "scored": scored, "per_case": per_case}


def summarize(rows: list[dict], ragas_result: dict | None) -> dict:
    ok = [r for r in rows if not r.get("error")]
    errors = [r for r in rows if r.get("error")]

    # Retrieval aggregates exclude cases with nothing to retrieve.
    retrievable = [r for r in ok if r.get("supporting_quotes")]
    unanswerable = [r for r in ok if r["category"] == "unanswerable"]
    answerable = [r for r in ok if r["category"] not in ("unanswerable", "ambiguous")]

    by_category: dict[str, dict] = {}
    for row in retrievable:
        bucket = by_category.setdefault(row["category"], {"n": 0, "hits": 0, "recall": []})
        bucket["n"] += 1
        bucket["hits"] += int(row["retrieval"]["quote_hit"])
        bucket["recall"].append(row["retrieval"]["quote_recall"])
    for name, bucket in by_category.items():
        bucket["quote_hit_rate"] = round(bucket["hits"] / bucket["n"], 4)
        bucket["quote_recall"] = mean(bucket.pop("recall"))
        bucket.pop("hits")

    latency = {
        stage: {
            "mean": mean([r[stage] for r in ok]),
            "p50": percentile([r[stage] for r in ok], 50),
            "p95": percentile([r[stage] for r in ok], 95),
        }
        for stage in ("retrieval_ms", "rerank_ms", "generation_ms", "total_ms")
    }

    costs = [r["cost"] for r in ok if r["cost"] is not None]
    return {
        "cases": len(rows),
        "succeeded": len(ok),
        "errors": len(errors),
        "error_rate": round(len(errors) / len(rows), 4) if rows else 0.0,
        "retrieval": {
            "quote_hit_rate": mean([float(r["retrieval"]["quote_hit"]) for r in retrievable]),
            "quote_recall": mean([r["retrieval"]["quote_recall"] for r in retrievable]),
            "doc_precision": mean([r["retrieval"]["doc_precision"] for r in retrievable]),
            "n": len(retrievable),
            "by_category": by_category,
        },
        "behaviour": {
            # Correct refusals on questions the corpus cannot answer.
            "refusal_rate_unanswerable": mean([float(r["refused"]) for r in unanswerable]),
            # Refusals on questions it should have answered — the opposite error.
            "refusal_rate_answerable": mean([float(r["refused"]) for r in answerable]),
            "n_unanswerable": len(unanswerable),
            "n_answerable": len(answerable),
            # Answered by the relevance gate without an LLM call.
            "gated_rate": mean([float(r.get("gated", False)) for r in ok]),
        },
        "latency_ms": latency,
        "tokens": {
            "input_mean": mean([float(r["input_tokens"]) for r in ok]),
            "output_mean": mean([float(r["output_tokens"]) for r in ok]),
            "total_mean": mean([float(r["total_tokens"]) for r in ok]),
            "total_sum": sum(r["total_tokens"] for r in ok),
        },
        "cost": {
            "per_query_mean": mean(costs),
            "run_total": round(sum(costs), 6) if costs else None,
        },
        "ragas": ragas_result or {"skipped": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="experiment id, e.g. exp-001-baseline")
    parser.add_argument("--limit", type=int, help="run only the first N cases")
    parser.add_argument("--no-ragas", action="store_true", help="skip LLM-judged metrics")
    parser.add_argument("--top-k", type=int, help="override retrieval depth")
    parser.add_argument("--note", default="", help="what this run is testing")
    parser.add_argument(
        "--dataset", choices=sorted(DATASETS), default="v1",
        help="benchmark split; 'holdout' is for confirming a finished config, never for tuning",
    )
    parser.add_argument("--rerank", action="store_true", help="enable cross-encoder reranking")
    parser.add_argument("--rerank-model", help="override the cross-encoder model")
    parser.add_argument("--candidates", type=int, help="vector candidates fetched for reranking")
    parser.add_argument(
        "--min-rerank-score", type=float,
        help="relevance gate: refuse without the LLM below this cross-encoder score",
    )
    args = parser.parse_args()

    # Flags override .env for this run only. The pipeline reads settings at call
    # time, and config.json below records the values actually used.
    if args.rerank:
        settings.RERANK_ENABLED = True
    if args.rerank_model:
        settings.RERANK_MODEL = args.rerank_model
    if args.candidates:
        settings.RERANK_CANDIDATES = args.candidates
    if args.min_rerank_score is not None:
        settings.MIN_RERANK_SCORE = args.min_rerank_score
    if settings.MIN_RERANK_SCORE is not None and not settings.RERANK_ENABLED:
        sys.exit("--min-rerank-score gates on cross-encoder scores; it needs --rerank.")

    out_dir = RESULTS_DIR / args.id
    if out_dir.exists():
        sys.exit(
            f"{out_dir} already exists. Experiment results are immutable — "
            f"use a new --id (e.g. {args.id}-rerun) rather than overwriting."
        )

    cases = load_dataset(DATASETS[args.dataset])
    if not cases:
        sys.exit("Dataset is empty — run `python -m evaluation.generate_dataset`.")

    if settings.RERANK_ENABLED:
        # Model load takes seconds; outside the clock, or query 1's latency
        # would measure disk I/O rather than reranking.
        from app.services import reranker
        reranker.warm_up()

    print(f"Experiment {args.id}: {len(cases)} cases\n")
    started = time.time()
    rows = run_pipeline(cases, limit=args.limit, top_k=args.top_k)
    ragas_result = None if args.no_ragas else run_ragas(rows)
    summary = summarize(rows, ragas_result)
    elapsed = round(time.time() - started, 1)

    config = {
        "experiment_id": args.id,
        "note": args.note,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": elapsed,
        "dataset": {"path": DATASETS[args.dataset].name, "version": args.dataset, "cases": len(cases)},
        "pipeline": {
            "retrieval_mode": "vector",
            "reranking_enabled": settings.RERANK_ENABLED,
            "rerank_model": settings.RERANK_MODEL if settings.RERANK_ENABLED else None,
            "rerank_candidates": settings.RERANK_CANDIDATES if settings.RERANK_ENABLED else None,
            "min_rerank_score": settings.MIN_RERANK_SCORE,
            "langgraph_enabled": False,
            "cache_enabled": False,
            "top_k": args.top_k or settings.TOP_K,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
        },
        "models": {
            "generation": settings.OPENAI_MODEL,
            "embedding": settings.OPENAI_EMBEDDING_MODEL,
            "embedding_dim": settings.OPENAI_EMBEDDING_DIM,
            "judge": None if args.no_ragas else JUDGE_MODEL,
            "judge_embeddings": None if args.no_ragas else JUDGE_EMBEDDINGS,
        },
        "environment": {"python": platform.python_version()},
    }

    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    per_case_ragas = (ragas_result or {}).get("per_case", {})
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            row["ragas"] = per_case_ragas.get(row["id"])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n--- {args.id} ({elapsed}s) ---")
    r, b = summary["retrieval"], summary["behaviour"]
    print(f"quote hit rate   {r['quote_hit_rate']}   (n={r['n']})")
    print(f"quote recall     {r['quote_recall']}")
    print(f"doc precision    {r['doc_precision']}")
    print(f"refusal (unans)  {b['refusal_rate_unanswerable']}   (n={b['n_unanswerable']})")
    print(f"refusal (ans)    {b['refusal_rate_answerable']}   (n={b['n_answerable']})")
    for stage, values in summary["latency_ms"].items():
        print(f"{stage:16} mean {values['mean']:>7}  p50 {values['p50']:>7}  p95 {values['p95']:>7}")
    print(f"tokens/query     {summary['tokens']['total_mean']}")
    print(f"cost/query       ${summary['cost']['per_query_mean']}")
    if ragas_result and "scores" in ragas_result:
        graded = ragas_result["graded"]
        for name, value in ragas_result["scores"].items():
            count = ragas_result["scored"][name]
            # Below 90% coverage the mean is over a biased subset: do not cite it.
            flag = "" if count >= 0.9 * graded else "   <-- INCOMPLETE, judge calls failed"
            print(f"{name:16} {value}   ({count}/{graded} scored){flag}")
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
