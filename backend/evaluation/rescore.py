"""Recompute retrieval metrics for stored experiments, without re-running them.

    python -m evaluation.rescore                  # every experiment
    python -m evaluation.rescore exp-001-baseline

The deterministic metrics are pure functions of (ground truth, retrieved
contexts), and every experiment stores its retrieved contexts in
`per_query.jsonl`. So when the scorer or a ground-truth entry is corrected, the
metrics can be recomputed exactly — no API calls, no drift from re-running.

Result directories stay immutable: `summary.json` and `per_query.jsonl` are
never modified. The recomputed numbers are written next to them as
`rescored.json`, which `compare` and the chart script prefer when it exists.
RAGAS scores, latency and cost are untouched; only quote/doc retrieval metrics
are recomputed.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from evaluation.dataset.schema import DATASETS, load_dataset
from evaluation.metrics import mean, score_retrieval

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESCORED = "rescored.json"


def rescore_rows(rows: list[dict], cases: dict) -> dict:
    """Retrieval aggregates and per-case scores from stored contexts.

    Mirrors the aggregation in `run_experiment.summarize`, so a rescored number
    is directly comparable to a freshly computed one.
    """
    per_case: dict[str, dict] = {}
    changed: list[str] = []
    by_category: dict[str, dict] = {}

    for row in rows:
        case = cases.get(row["id"])
        if row.get("error") or case is None or not case.supporting_quotes:
            continue
        score = asdict(score_retrieval(case, row["contexts"], row["context_titles"]))
        per_case[row["id"]] = score
        old = row.get("retrieval") or {}
        if (old.get("quote_hit"), old.get("quote_recall")) != (score["quote_hit"], score["quote_recall"]):
            changed.append(row["id"])
        bucket = by_category.setdefault(row["category"], {"n": 0, "hits": 0, "recall": []})
        bucket["n"] += 1
        bucket["hits"] += int(score["quote_hit"])
        bucket["recall"].append(score["quote_recall"])

    for bucket in by_category.values():
        bucket["quote_hit_rate"] = round(bucket.pop("hits") / bucket["n"], 4)
        bucket["quote_recall"] = mean(bucket.pop("recall"))

    scores = list(per_case.values())
    return {
        "retrieval": {
            "quote_hit_rate": mean([float(s["quote_hit"]) for s in scores]),
            "quote_recall": mean([s["quote_recall"] for s in scores]),
            "doc_precision": mean([s["doc_precision"] for s in scores]),
            "n": len(scores),
            "by_category": by_category,
        },
        "changed_cases": changed,
        "per_case": per_case,
    }


def load_rescored(directory: Path) -> dict | None:
    """The rescored retrieval block for an experiment, if one was written."""
    path = directory / RESCORED
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", help="experiment ids (default: all)")
    args = parser.parse_args()

    directories = (
        [RESULTS_DIR / e for e in args.experiments]
        if args.experiments
        else sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())
    )
    datasets: dict[str, dict] = {}

    print(f"{'experiment':30} {'hit rate':>18} {'recall':>18}  changed")
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        version = config["dataset"]["version"]
        if version not in datasets:
            datasets[version] = {c.id: c for c in load_dataset(DATASETS[version])}
        rows = [
            json.loads(line)
            for line in (directory / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        result = rescore_rows(rows, datasets[version])
        old, new = summary["retrieval"], result["retrieval"]
        (directory / RESCORED).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"{directory.name:30} "
            f"{old['quote_hit_rate']:.4f} -> {new['quote_hit_rate']:.4f}   "
            f"{old['quote_recall']:.4f} -> {new['quote_recall']:.4f}  "
            f"{', '.join(result['changed_cases']) or '-'}"
        )


if __name__ == "__main__":
    main()
