# Results

Six experiments against a fixed 75-question benchmark (8 arXiv IR/NLP papers,
445 chunks). Everything below is reproducible from
[`backend/evaluation/results/`](backend/evaluation/results/): each run stores its
config, per-question outputs and summary.

## Summary

| Experiment | Chunks to LLM | Quote hit rate | Quote recall | Context precision | Context recall | Tokens / query | Latency p50 |
|---|---|---|---|---|---|---|---|
| `exp-001` baseline (vector top-5) | 5 | 0.820 | 0.721 | 0.745 | 0.839 | 1,900 | 1.7 s |
| `exp-002` vector top-10 | 10 | 0.869 | 0.779 | — | — | 3,546 | 1.8 s |
| `exp-002` vector top-20 | 20 | **0.934** | **0.844** | — | — | 6,881 | 1.9 s |
| `exp-003` rerank 20 → 5 | 5 | 0.869 | 0.779 | **0.857** | 0.853 | 1,875 | 6.0 s |
| **`exp-003` rerank 20 → 8** | 8 | 0.918 | 0.820 | **0.856** | **0.910** | 2,849 | 5.7 s |
| `exp-004` rerank 20 → 8 + relevance gate | 8 | 0.918 | 0.820 | 0.847 | 0.910 | 2,652 | 6.1 s |

- **Quote hit rate / recall.** Share of answerable questions where a passage containing the
  ground-truth quote was retrieved, and the share of all such quotes retrieved. Exact text
  matching, with no LLM judge, so the numbers come out identical on every run.
- **Context precision / recall.** RAGAS, graded by `gpt-4o-mini`. Top-k sweeps ran without
  RAGAS.
- Reranker: `cross-encoder/ms-marco-MiniLM-L12-v2`, local CPU. Generator: `gpt-4o-mini`.

## Findings

**1. The LLM judge is noisy, so I measured how noisy before trusting it.**
I ran the identical baseline twice. The deterministic retrieval metrics matched exactly
(0.8197 both times). RAGAS faithfulness moved from **0.82 to 0.92** with nothing changed,
and the unanswerable-refusal rate moved from 5/8 to 6/8. So no faithfulness claim below
about ±0.10 is made anywhere in this document. Retrieval decisions were made on the
deterministic metrics, and RAGAS was used only where its run-to-run spread was small
(context precision ±0.02, context recall ±0.01).

**2. More context finds more answers, but at a steep cost.**
Going from 5 to 20 chunks raised quote hit rate from 0.82 to 0.93. Seven questions went
from miss to hit and none went the other way. It also cost **3.6× the tokens**, p95
latency rose 52%, and correct refusals on unanswerable questions fell from 5/8 to 4/8.
More loosely related context gave the model more material to hallucinate from.

**3. Reranking recovers most of that gain at a fraction of the tokens.**
Fetching 20 chunks by vector search and letting a cross-encoder pick 8 reaches a
**0.918 hit rate** (top-20 reaches 0.934). It uses **59% fewer tokens** than top-20 and
keeps the baseline's refusal behaviour. Context precision rose from 0.745 to **0.856**
and context recall from 0.839 to **0.910**, both far outside the noise measured in
finding 1. Seven retrieval flips were gained and one was lost. At the baseline's 5-chunk
budget, reranking matches vector top-10 quality at half its tokens.

**4. The cost of that gain is latency: the reranker adds ~4 s on CPU.**
Scoring 20 pairs with a 33M-parameter cross-encoder takes ~4 s on the 4-thread CPU this
ran on, so median latency went from 1.7 s to 5.7 s. An ONNX runtime backend barely helped
(4.4 s → 4.0 s). On a GPU, or with a hosted rerank API, this stage takes about 50 ms.
This trade-off decides whether reranking belongs in a production build.

## What the data said *not* to build

- **Hybrid (BM25 + vector) retrieval.** It was planned to fix exact-identifier queries
  (`ColBERT`, `SST-2`, `nDCG@10`). But the baseline already hit **13 of 14** keyword
  questions, leaving one question of headroom. Not built.
- **A vector-similarity relevance threshold.** Cosine scores for answerable questions
  (0.29–0.84) and unanswerable ones (0.37–0.70) overlap almost entirely, so any threshold
  would refuse real questions. The gate in `exp-004` uses the cross-encoder score instead,
  which does separate them.
- **A bigger reranker.** Tested offline on the stored top-20 candidates, `bge-reranker-base`
  (278M params) was *worse* than MiniLM-L12 (0.869 vs 0.918 hit rate at k=8) and 3× slower.

## The relevance gate: an inconclusive result

`exp-004` answers "not found" without calling the LLM when the best cross-encoder
score is below −2.5. It gated 5 of 75 questions and raised correct refusals on
unanswerable questions from 5/8 to **6/8**. The newly caught case was one where the model
had invented an answer: asked which Python version DPR requires, it answered "3.6 or
later" in one run and "3.7 or higher" in another. Neither appears in the paper. It also refused one answerable question
(`q054`), whose retrieval had already failed.

A one-question change is the same size as the run-to-run noise in finding 1, and
**the threshold was chosen on this same benchmark**, with no held-out set. The gate is
therefore reported as promising, not proven. It is on in the recommended config because
it is cheap and only refused a question whose retrieval had already failed.

## Limitations

- **Small benchmark.** 61 answerable questions, so one question is 1.6 points of hit rate.
  The `multi_document` category has only 6 cases.
- **Same model generates and answers.** `gpt-4o-mini` wrote the benchmark questions and
  also answers them. Verbatim-quote grounding and hand-written adversarial cases limit
  this, but don't eliminate it.
- **Settings chosen on the benchmark.** The reranker model, k=8 and the gate threshold were
  all selected on the same benchmark they are scored on. A fresh question set would give a
  less optimistic number.

## Reproduce

```bash
cd backend
python -m evaluation.run_experiment --id exp-001-baseline
python -m evaluation.run_experiment --id exp-002-topk20 --top-k 20 --no-ragas
python -m evaluation.run_experiment --id exp-003-rerank-k8 --rerank --top-k 8
python -m evaluation.run_experiment --id exp-004-rerank-k8-gate --rerank --top-k 8 --min-rerank-score -2.5
python -m evaluation.compare exp-001-baseline exp-003-rerank-k8
```

Result directories are immutable: the runner refuses to overwrite an existing id.
