"""Compare experiment results.

    python -m evaluation.compare exp-001-baseline exp-003-hybrid

Reports metric deltas with the direction of improvement handled correctly —
latency and cost improve by going *down*, so a naive "(new-old)/old" would
report a speedup as a regression.

Also lists cases that flipped between the two runs. This is the part that
matters for failure analysis: averages say whether something changed, flips say
*which questions* changed, and only the second tells you why.

Comparing a run against its own rerun measures the noise floor. Any difference
there is judge variance, not a real effect — so a later experiment moving a
metric by less than that is unfalsifiable, no matter how good the story sounds.
"""
import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# (path in summary.json, label, higher_is_better)
METRICS = [
    (("retrieval", "quote_hit_rate"), "quote hit rate", True),
    (("retrieval", "quote_recall"), "quote recall", True),
    (("retrieval", "doc_precision"), "doc precision", True),
    (("ragas", "scores", "faithfulness"), "faithfulness", True),
    (("ragas", "scores", "answer_relevancy"), "answer relevancy", True),
    (("ragas", "scores", "context_precision"), "context precision", True),
    (("ragas", "scores", "context_recall"), "context recall", True),
    (("behaviour", "refusal_rate_unanswerable"), "refusal (unanswerable)", True),
    (("behaviour", "refusal_rate_answerable"), "refusal (answerable)", False),
    (("latency_ms", "retrieval_ms", "mean"), "retrieval ms", False),
    (("latency_ms", "generation_ms", "mean"), "generation ms", False),
    (("latency_ms", "total_ms", "mean"), "total ms", False),
    (("latency_ms", "total_ms", "p95"), "total ms p95", False),
    (("tokens", "total_mean"), "tokens/query", False),
    (("cost", "per_query_mean"), "cost/query", False),
    (("error_rate",), "error rate", False),
]


def dig(data: dict, path: tuple):
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def load(experiment_id: str) -> tuple[dict, dict, dict]:
    directory = RESULTS_DIR / experiment_id
    if not directory.exists():
        sys.exit(f"No such experiment: {directory}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    rows = {}
    path = directory / "per_query.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return summary, config, rows


def pct_change(old: float, new: float, higher_is_better: bool) -> str:
    """Signed improvement, oriented so + always means better."""
    if old is None or new is None or old == 0:
        return "     —"
    raw = (new - old) / abs(old) * 100
    improvement = raw if higher_is_better else -raw
    return f"{improvement:+6.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()

    base_summary, base_config, base_rows = load(args.baseline)
    cand_summary, cand_config, cand_rows = load(args.candidate)

    print(f"{'':26} {args.baseline[:16]:>16} {args.candidate[:16]:>16}   change")
    print("-" * 72)
    for path, label, higher_is_better in METRICS:
        old, new = dig(base_summary, path), dig(cand_summary, path)
        if old is None and new is None:
            continue
        fmt = lambda v: "—" if v is None else (f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")
        arrow = "" if higher_is_better else "  (lower is better)"
        print(
            f"{label:26} {fmt(old):>16} {fmt(new):>16}   "
            f"{pct_change(old, new, higher_is_better)}{arrow}"
        )

    # --- per-category retrieval ---------------------------------------------
    base_cat = dig(base_summary, ("retrieval", "by_category")) or {}
    cand_cat = dig(cand_summary, ("retrieval", "by_category")) or {}
    if base_cat or cand_cat:
        print(f"\nquote hit rate by category")
        print("-" * 72)
        for name in sorted(set(base_cat) | set(cand_cat)):
            old = (base_cat.get(name) or {}).get("quote_hit_rate")
            new = (cand_cat.get(name) or {}).get("quote_hit_rate")
            n = (cand_cat.get(name) or base_cat.get(name) or {}).get("n", 0)
            fmt = lambda v: "—" if v is None else f"{v:.4f}"
            print(
                f"{name:18} n={n:<4} {fmt(old):>10} {fmt(new):>10}   "
                f"{pct_change(old, new, True)}"
            )

    # --- flips ---------------------------------------------------------------
    gained, lost = [], []
    for case_id, cand_row in cand_rows.items():
        base_row = base_rows.get(case_id)
        if not base_row or base_row.get("error") or cand_row.get("error"):
            continue
        was = (base_row.get("retrieval") or {}).get("quote_hit")
        now = (cand_row.get("retrieval") or {}).get("quote_hit")
        if was is None or now is None or was == now:
            continue
        (gained if now else lost).append((case_id, cand_row))

    print(f"\nretrieval flips: +{len(gained)} gained, -{len(lost)} lost")
    print("-" * 72)
    for label, group in (("GAINED", gained), ("LOST", lost)):
        for case_id, row in group[:8]:
            print(f"  {label:6} {case_id} [{row['category']}] {row['question'][:60]}")
    if not gained and not lost:
        print("  (none — retrieval behaved identically on every case)")


if __name__ == "__main__":
    main()
