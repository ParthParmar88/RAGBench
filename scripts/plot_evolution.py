"""Render the README's evolution chart from stored experiment results.

    python scripts/plot_evolution.py

Reads `backend/evaluation/results/<id>/summary.json` and writes a light and a
dark SVG to docs/images/. Numbers come from the result files, never typed in,
so the chart can't drift from what was measured. Standard library only.

Quality (quote hit rate) against cost (LLM tokens per query), one point per
experiment. Two paths leave the same baseline: widen retrieval, or rerank.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "backend" / "evaluation" / "results"
OUT = ROOT / "docs" / "images"

BASELINE = "exp-001-baseline"
# (experiment id, point label, label placement: dx, dy, text-anchor)
PATHS = {
    "context": {
        "name": "More context: vector top-k",
        "points": [
            ("exp-002-topk10", "Top-10", (12, 18, "start")),
            ("exp-002-topk20", "Top-20", (-12, -12, "end")),
        ],
    },
    "rerank": {
        "name": "Reranking: fetch 20, cross-encoder keeps k",
        "points": [
            ("exp-003-rerank-k5", "Rerank 20→5", (-12, 5, "end")),
            ("exp-003-rerank-k8", "Rerank 20→8", (-12, -12, "end")),
        ],
    },
}

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)",
        "rerank": "#2a78d6", "context": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)",
        "rerank": "#3987e5", "context": "#d95926",
    },
}

W, H = 880, 470
LEFT, RIGHT, TOP, BOTTOM = 78, 840, 92, 402
X_MAX = 7000
Y_MIN, Y_MAX = 80, 97.5  # headroom above 95% holds the annotation
Y_TICK_MAX = 95


def load(experiment_id: str) -> tuple[float, float]:
    """(tokens per query, quote hit rate in %) for one experiment."""
    directory = RESULTS / experiment_id
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    retrieval = summary["retrieval"]
    # `evaluation.rescore` writes corrected retrieval metrics beside the
    # original summary; prefer them so the chart reflects the current scorer.
    rescored = directory / "rescored.json"
    if rescored.exists():
        retrieval = json.loads(rescored.read_text(encoding="utf-8"))["retrieval"]
    return summary["tokens"]["total_mean"], retrieval["quote_hit_rate"] * 100


def sx(tokens: float) -> float:
    return LEFT + tokens / X_MAX * (RIGHT - LEFT)


def sy(hit: float) -> float:
    return BOTTOM - (hit - Y_MIN) / (Y_MAX - Y_MIN) * (BOTTOM - TOP)


def text(x, y, s, fill, size=13, anchor="start", weight=400):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>'
    )


def dot(x, y, fill, surface):
    # 2px surface ring keeps the dot legible where it sits on a line.
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" stroke="{surface}" stroke-width="2"/>'


def render(theme: dict) -> str:
    t = theme
    base = load(BASELINE)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="system-ui, -apple-system, \'Segoe UI\', sans-serif" role="img" '
        f'aria-label="Quote hit rate against LLM tokens per query for each experiment">',
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="{t["surface"]}" stroke="{t["border"]}"/>',
        text(LEFT - 50, 34, "Retrieval quality vs. LLM cost, per experiment", t["ink"], 16, weight=600),
    ]

    # Legend: always present for two series; identity never by color alone.
    lx = LEFT - 50
    for key in ("rerank", "context"):
        color, name = t[key], PATHS[key]["name"]
        parts.append(f'<line x1="{lx}" y1="58" x2="{lx + 18}" y2="58" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        parts.append(dot(lx + 9, 58, color, t["surface"]))
        parts.append(text(lx + 26, 62.5, name, t["ink2"]))
        lx += 26 + len(name) * 7.2 + 28

    # Gridlines and y ticks (hairline, solid, recessive).
    for hit in range(Y_MIN, Y_TICK_MAX + 1, 5):
        y = sy(hit)
        stroke = t["axis"] if hit == Y_MIN else t["grid"]
        parts.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{RIGHT}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1"/>')
        parts.append(text(LEFT - 10, y + 4.5, f"{hit}%", t["muted"], 12, "end"))
    for tokens in range(0, X_MAX + 1, 1000):
        x = sx(tokens)
        parts.append(f'<line x1="{x:.1f}" y1="{BOTTOM}" x2="{x:.1f}" y2="{BOTTOM + 5}" stroke="{t["axis"]}" stroke-width="1"/>')
        parts.append(text(x, BOTTOM + 20, f"{tokens:,}", t["muted"], 12, "middle"))
    parts.append(text((LEFT + RIGHT) / 2, BOTTOM + 44, "LLM tokens per query  (lower is cheaper)", t["ink2"], 13, "middle"))
    parts.append(
        f'<text transform="translate({LEFT - 50} {(TOP + BOTTOM) / 2}) rotate(-90)" fill="{t["ink2"]}" '
        f'font-size="13" text-anchor="middle">Right passage retrieved (quote hit rate)</text>'
    )

    # Annotation: the headline trade-off, drawn between the two finalists.
    rr8, top20 = load("exp-003-rerank-k8"), load("exp-002-topk20")
    saving = (1 - rr8[0] / top20[0]) * 100
    # Above all data, so it crosses no line.
    ay = sy(96)
    x1, x2 = sx(rr8[0]), sx(top20[0])
    parts.append(f'<line x1="{x1:.1f}" y1="{ay:.1f}" x2="{x2:.1f}" y2="{ay:.1f}" stroke="{t["muted"]}" stroke-width="1"/>')
    for x in (x1, x2):
        parts.append(f'<line x1="{x:.1f}" y1="{ay - 5:.1f}" x2="{x:.1f}" y2="{ay + 5:.1f}" stroke="{t["muted"]}" stroke-width="1"/>')
    # Leaders from the bracket ends down toward the two points it compares.
    for x, h in ((x1, rr8[1]), (x2, top20[1])):
        parts.append(f'<line x1="{x:.1f}" y1="{ay + 5:.1f}" x2="{x:.1f}" y2="{sy(h) - 24:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
    parts.append(text((x1 + x2) / 2, ay - 8, f"{saving:.0f}% fewer tokens, {top20[1] - rr8[1]:.1f} pts lower hit rate", t["ink2"], 13, "middle"))

    # Paths, then points on top so every dot's ring sits over the lines.
    points = []
    for key, path in PATHS.items():
        coords = [base] + [load(eid) for eid, _, _ in path["points"]]
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(tk):.1f},{sy(h):.1f}" for i, (tk, h) in enumerate(coords))
        parts.append(f'<path d="{d}" fill="none" stroke="{t[key]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for (eid, label, (dx, dy, anchor)), (tk, h) in zip(path["points"], coords[1:]):
            points.append((sx(tk), sy(h), t[key], f"{label} · {h:.1f}%", dx, dy, anchor))

    for x, y, color, *_ in points:
        parts.append(dot(x, y, color, t["surface"]))
    bx, by = sx(base[0]), sy(base[1])
    parts.append(dot(bx, by, t["ink"], t["surface"]))

    for x, y, _, label, dx, dy, anchor in points:
        parts.append(text(x + dx, y + dy, label, t["ink"], 13, anchor))
    parts.append(text(bx + 12, by + 20, f"Baseline, top-5 · {base[1]:.1f}%", t["ink"], 13))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        path = OUT / f"evolution-{name}.svg"
        path.write_text(render(theme), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
