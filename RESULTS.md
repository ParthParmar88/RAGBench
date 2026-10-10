# Results

Eleven experiment runs against two question sets over the same corpus (8 arXiv IR/NLP
papers, 445 chunks):

- **Tuning benchmark** (`v1`, 75 questions). Every setting was chosen on this.
- **Held-out set** (46 questions). Generated after all settings were fixed, from
  passages the tuning benchmark never used. Nothing was tuned on it.

Everything below is reproducible from
[`backend/evaluation/results/`](backend/evaluation/results/): each run stores its
config, per-question outputs and summary.

## Summary

### Tuning benchmark (61 answerable questions)

| Experiment | Chunks to LLM | Quote hit rate | Quote recall | Context precision | Context recall | Tokens / query | Latency p50 |
|---|---|---|---|---|---|---|---|
| `exp-001` baseline (vector top-5) | 5 | 0.803 | 0.713 | 0.745 | 0.839 | 1,900 | 1.7 s |
| `exp-002` vector top-10 | 10 | 0.869 | 0.779 | — | — | 3,546 | 1.8 s |
| `exp-002` vector top-20 | 20 | **0.934** | **0.844** | — | — | 6,881 | 1.9 s |
| `exp-003` rerank 20 → 5 | 5 | 0.869 | 0.779 | **0.857** | 0.853 | 1,875 | 6.0 s |
| **`exp-003` rerank 20 → 8** | 8 | 0.918 | 0.820 | **0.856** | **0.910** | 2,849 | 5.7 s |
| `exp-004` rerank 20 → 8 + relevance gate | 8 | 0.918 | 0.820 | 0.847 | 0.910 | 2,652 | 6.1 s |

### Held-out set (34 answerable questions)

| Experiment | Chunks to LLM | Quote hit rate | Quote recall | Tokens / query | Unanswerable refused | Answerable refused |
|---|---|---|---|---|---|---|
| `exp-005` baseline (vector top-5) | 5 | 0.735 | 0.603 | 1,975 | 4 / 8 | 1 / 34 |
| `exp-005` vector top-20 | 20 | **0.971** | **0.868** | 7,041 | 3 / 8 | 0 / 34 |
| **`exp-005` rerank 20 → 8** | 8 | 0.794 | 0.706 | 2,914 | 5 / 8 | 0 / 34 |
| `exp-005` rerank 20 → 8 + relevance gate | 8 | 0.794 | 0.706 | 2,648 | 4 / 8 | **3 / 34** |

- **Quote hit rate / recall.** Share of answerable questions where a passage containing a
  ground-truth quote was retrieved, and the share of all such quotes retrieved. Exact text
  matching, with no LLM judge, so the numbers come out identical on every run.
- **Context precision / recall.** RAGAS, graded by `gpt-4o-mini`. Top-k sweeps ran without
  RAGAS. On the held-out runs the judge timed out on most context-precision and a third of
  faithfulness calls, so those two are not reported there (see Corrections).
- Reranker: `cross-encoder/ms-marco-MiniLM-L12-v2`, local CPU. Generator: `gpt-4o-mini`.

## Findings

**1. The LLM judge is noisy, so I measured how noisy before trusting it.**
I ran the identical baseline twice. The deterministic retrieval metrics matched exactly.
RAGAS faithfulness moved from **0.82 to 0.92** with nothing changed, and the
unanswerable-refusal rate moved from 5/8 to 6/8. So no faithfulness claim below about
±0.10 is made anywhere in this document. Retrieval decisions were made on the
deterministic metrics, and RAGAS was used only where its run-to-run spread was small
(context precision ±0.02, context recall ±0.01).

**2. More context finds more answers, but at a steep cost. This held on both sets.**
Going from 5 to 20 chunks raised quote hit rate from 0.80 to 0.93 on the tuning benchmark
and from 0.74 to 0.97 on the held-out set. Eight questions went from miss to hit on each,
and none went the other way. It also cost **3.6× the tokens**, and correct refusals on
unanswerable questions fell on both sets (5/8 → 4/8 and 4/8 → 3/8). More loosely related
context gave the model more material to answer from when it should have refused.

**3. Reranking helps, but about half as much on unseen questions as on the tuning benchmark.**
Fetching 20 chunks by vector search and letting a cross-encoder keep 8:

| | Tuning benchmark | Held-out set |
|---|---|---|
| Quote hit rate | 0.803 → 0.918 (+11.5 pts) | 0.735 → 0.794 (+5.9 pts) |
| Quote recall | 0.713 → 0.820 | 0.603 → 0.706 |
| Context recall (RAGAS) | 0.839 → 0.910 | 0.838 → 0.895 (32 of 34 scored) |
| Questions gained / lost | 8 / 1 | 4 / 2 |
| Share of the top-20 gain recovered | 88% | 25% |
| Tokens vs top-20 | 59% fewer | 59% fewer |

The direction holds on every measure, and the token saving is identical. The size does
not hold. On the tuning benchmark reranking looked like it recovered nearly all of
top-20's recall. On the held-out set, top-20 found the right passage for 97% of questions
and reranking to 8 kept it for only 79%: the cross-encoder ranked the right passage below
eighth place for six questions where vector search had it in the top 20.

That gap is the cost of choosing the reranker model and k = 8 on the same questions they
were scored on. The honest claim is: **reranking improves retrieval by roughly 6 to 11
points of hit rate at 1.5× baseline tokens, and it does not replace retrieval depth.**
With 34 held-out questions, a 5.9-point gain is two questions net, so the held-out set
confirms the direction more firmly than the size.

**4. The cost of reranking is latency: 4 to 7 s on CPU.**
Scoring 20 pairs with a 33M-parameter cross-encoder took about 4 s per query on the
4-thread CPU during the tuning runs, and about 7 s during the held-out runs when the
machine was busier. Median latency went from 1.7 s to between 5.7 and 9.5 s. An ONNX
runtime backend barely helped (4.4 s → 4.0 s). On a GPU, or with a hosted rerank API, this
stage takes about 50 ms. This trade-off decides whether reranking belongs in a production
build.

## The relevance gate: it did not generalize

`exp-004` answers "not found" without calling the LLM when the best cross-encoder score
is below −2.5.

**On the tuning benchmark it looked promising.** It gated 5 of 75 questions and raised
correct refusals on unanswerable questions from 5/8 to 6/8. The newly caught case was one
where the model had invented an answer: asked which Python version DPR requires, it
answered "3.6 or later" in one run and "3.7 or higher" in another. Neither appears in the
paper. At the time I reported this as inconclusive, because a one-question change is the
size of the noise in finding 1 and the threshold was chosen on those same questions.

**On the held-out set it failed.** It gated four questions:

- Three were **answerable** (`h009`, `h012`, `h031`). For two of them the right passage
  had been retrieved, and the gate threw it away.
- One was unanswerable (`h042`, an out-of-domain salary question), which the LLM refuses
  on its own in every configuration.

It caught no unanswerable question the LLM was not already refusing, and it refused 9% of
answerable ones.

**Why.** The held-out unanswerable questions are mostly near-misses: "What is the context
window size of Claude 3 Opus?" against a paper that discusses Claude 1.3's context window
at length. They retrieve on-topic passages, and the cross-encoder scores those highly (up
to +7.4, far above the −2.5 threshold). A relevance score measures whether a passage is
about the question's topic. It cannot tell whether the answer is in it. The tuning
benchmark's unanswerable cases happened to include more questions whose best passage
scored low (three of eight below the threshold, against one of eight here), which is what
made a threshold look workable.

**Decision.** The gate is off in the recommended configuration. The code remains as an
option (`MIN_RERANK_SCORE`), unset by default. Unanswerable questions remain the system's
weakest behaviour: it correctly refuses about half of them.

## What the data said *not* to build

- **Hybrid (BM25 + vector) retrieval.** It was planned to fix exact-identifier queries
  (`ColBERT`, `SST-2`, `nDCG@10`). On the tuning benchmark the baseline already hit
  **13 of 14** keyword questions, leaving one question of headroom, so it was not built.
  The held-out set is less clear-cut: the baseline hit 6 of 8 keyword questions there. Both
  misses were found by vector search at depth 20, so the passages are reachable without
  lexical matching, but this decision rests on thinner evidence than it first appeared.
- **A vector-similarity relevance threshold.** Cosine scores for answerable questions
  (0.29–0.84) and unanswerable ones (0.37–0.70) overlap almost entirely, so any threshold
  would refuse real questions.
- **A bigger reranker.** Tested offline on the stored top-20 candidates, `bge-reranker-base`
  (278M params) was *worse* than MiniLM-L12 (0.869 vs 0.918 hit rate at k=8) and 3× slower.

## Corrections

Both were found while building the held-out set.

**A blank quote in the tuning benchmark flattered the baseline.** Case `q057` carried an
empty supporting quote, and the empty string is a substring of every passage, so the case
scored a retrieval hit in every experiment whatever was retrieved. The scorer now treats
an empty quote as never matching, the blank quote was removed, and every stored experiment
was rescored from its saved contexts (`python -m evaluation.rescore`). The baseline's hit
rate dropped from 0.820 to **0.803**. Every other experiment had retrieved `q057`'s real
quote, so no other number changed. An earlier version of this document reported 0.820.
Original result files are untouched; corrected metrics sit beside them in `rescored.json`.
Details are in [the evaluation README](backend/evaluation/README.md).

**RAGAS means could hide failed judge calls.** A judge call that times out returns no
score and was silently dropped from the average. On the held-out runs, context precision
was scored for 3 of 34 and 1 of 34 questions, giving "means" of 0.94 and 0.20 that
described almost nothing. The runner now records how many cases each metric scored and
flags anything under 90%. Held-out context precision and faithfulness are not reported in
this document for that reason. The tuning-benchmark runs scored 61 of 61 on every metric.

## Limitations

- **Small sets.** 61 and 34 answerable questions. One question is 1.6 points on the tuning
  benchmark and 2.9 on the held-out set. `multi_document` has 6 and 3 cases.
- **Same model generates and answers.** `gpt-4o-mini` wrote the questions in both sets and
  also answers them. Verbatim-quote grounding and hand-written adversarial cases limit
  this, but don't eliminate it.
- **Question quality is uneven.** Questions are generated and not hand-curated. A few
  held-out ones are weak: one refers to "the two passages", and three are reference-list
  lookups. They were kept, because removing questions after generation would undo the
  point of a held-out set.
- **The held-out set can only be used once this way.** Any future setting chosen by
  looking at its results would need a fresh one.
- **Chunking is unchanged from the baseline.** Fixed 1200-character windows cut sentences
  in half. Multi-chunk quote recall is 0.55 and 0.57 after reranking: most questions that
  need two passages still get only one of them.

## Reproduce

```bash
cd backend
python -m evaluation.run_experiment --id exp-001-baseline
python -m evaluation.run_experiment --id exp-002-topk20 --top-k 20 --no-ragas
python -m evaluation.run_experiment --id exp-003-rerank-k8 --rerank --top-k 8
python -m evaluation.run_experiment --id exp-004-rerank-k8-gate --rerank --top-k 8 --min-rerank-score -2.5
python -m evaluation.run_experiment --dataset holdout --id exp-005-holdout-rerank-k8 --rerank --top-k 8 --no-ragas
python -m evaluation.compare exp-001-baseline exp-003-rerank-k8
python -m evaluation.rescore        # recompute retrieval metrics from stored contexts
```

Result directories are immutable: the runner refuses to overwrite an existing id.
