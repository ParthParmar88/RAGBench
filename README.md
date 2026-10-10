# RAGBench — A Measured RAG Pipeline

A retrieval-augmented question-answering system over PDFs. It started as a basic RAG
pipeline and was improved one version at a time, with each change measured against a
fixed benchmark before it was kept.

**FastAPI · PostgreSQL + pgvector · OpenAI · sentence-transformers · RAGAS · React**

## Evolution

Retrieval quality rose from **82% to 92%** of questions finding the right passage, while
LLM tokens per query stayed close to the baseline.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/evolution-dark.svg">
  <img alt="Quote hit rate against LLM tokens per query. Baseline top-5: 82.0% at 1,900 tokens. Adding more chunks: top-10 86.9% at 3,546, top-20 93.4% at 6,881. Reranking: 20→5 86.9% at 1,875, 20→8 91.8% at 2,849." src="docs/images/evolution-light.svg">
</picture>

There were two ways to improve on the baseline. Sending the model more chunks (orange)
works, but token cost grows almost as fast as quality. Reranking (blue) gets most of the
same gain for a fraction of the tokens.

| Version | What changed | Why | Measured result |
|---|---|---|---|
| **v1.0** Baseline | Embed → pgvector top-5 → one `gpt-4o-mini` call | Starting point | Nothing measurable yet: no tests, no benchmark |
| **v1.0.1** Safety net | Removed dead code, added a test harness | Make later changes safe | 28 tests, behaviour unchanged |
| **v1.1** Instrumentation | Per-stage latency, per-call token and cost tracking, idempotent ingestion, page provenance | You can't improve what you don't measure | 63 tests |
| **v1.2** Benchmark | 75 questions over 8 papers, ground truth tied to verbatim quotes; baseline run twice | A fixed yardstick, and a measure of the LLM judge's noise | Baseline **82.0%** hit rate. The judge varies ±0.10 on faithfulness, so decisions use exact metrics. |
| v1.2 · exp-002 | Sweep of chunks sent to the LLM: 5 → 10 → 20 | Is retrieval depth the bottleneck? | 93.4% at top-20, but **3.6× tokens**, and correct refusals fell from 5/8 to 4/8. Not shipped. |
| **v1.3** Reranking | Fetch 20, a local cross-encoder keeps the best 8 | Keep the recall gain without the token cost | **91.8%** hit rate, context recall 0.84 → **0.91**, 1.5× baseline tokens. Adds ~4 s CPU latency. |
| **v1.3** Relevance gate | Answer "not found" without calling the LLM when the best rerank score is below −2.5 | Stop confident answers to questions the documents can't answer | Unanswerable refused 5/8 → 6/8. Within noise, so reported as inconclusive. |

**Not built, on purpose:** hybrid keyword search. The baseline already answered 13 of 14
keyword questions, so there was almost nothing for it to fix.

Full analysis, including the judge-noise measurement and each experiment's caveats, is in
**[RESULTS.md](RESULTS.md)**. The chart is generated from the stored results by
[`scripts/plot_evolution.py`](scripts/plot_evolution.py).

![RAGBench answering a question with cited sources](docs/images/screenshot.png)

## Pipeline

```mermaid
flowchart LR
    Q[Question] --> E[Embed<br/>text-embedding-3-small]
    E --> S[pgvector HNSW<br/>top 20]
    S --> R[Cross-encoder rerank<br/>MiniLM-L12 → top 8]
    R --> G{Best score<br/>≥ −2.5?}
    G -- no --> N[“Not found”<br/>no LLM call]
    G -- yes --> L[gpt-4o-mini<br/>structured output]
    L --> A[Answer + citations]
```

Every query makes **one embedding call and at most one chat call**. Reranking runs
locally, and the relevance gate skips the chat call entirely when nothing retrieved is
relevant. Each stage is timed separately, and every LLM call's tokens and cost are
recorded in Postgres.

Ingestion: PDF → per-page text (`pypdf`) → 1200-character chunks with 150 overlap →
page-number resolution → batch embeddings → pgvector. Ingestion is idempotent, keyed on
the file's SHA-256.

## Evaluation

[`backend/evaluation/`](backend/evaluation/) contains the benchmark and the tools that run it:

- **Corpus.** 8 arXiv IR/NLP papers (BERT, DPR, ColBERT, RAG, BEIR, …), 445 chunks,
  pinned by SHA-256.
- **Dataset.** 75 questions in 7 categories: factual, semantic, keyword, multi-chunk,
  multi-document, ambiguous, and unanswerable. Ground truth is tied to **verbatim quotes**
  rather than chunk ids, so results stay comparable when chunking changes. 26% of
  generated candidates failed that check and were rejected.
- **Metrics.** Two kinds:
  - Deterministic retrieval metrics (quote hit rate and recall), which are exact and
    identical on every run.
  - RAGAS (faithfulness, relevancy, context precision and recall), graded by a pinned
    `gpt-4o-mini` judge.
- **Runner and comparison.** Results are written to immutable per-experiment directories,
  and `compare` reports deltas plus which questions flipped between two runs.

See [backend/evaluation/README.md](backend/evaluation/README.md) for how the dataset
was built and its known limitations.

## Run it

**Prerequisites:** Python 3.11+ (developed on 3.14), Node 18+, and PostgreSQL 14+ with the
[pgvector](https://github.com/pgvector/pgvector) extension. Create an empty database with
`CREATE DATABASE ragbench;`. Tables, the `vector` extension and indexes are created on startup.

```bash
# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp; then set OPENAI_API_KEY
uvicorn app.main:app --reload     # http://localhost:8000/docs

# Frontend (second terminal)
cd frontend
npm install
npm run dev                       # http://localhost:5173
```

```bash
# Tests (offline, no API key needed)
pip install -r requirements-dev.txt
python -m pytest

# Evaluation: fetch and ingest the corpus once, then run experiments
python -m evaluation.fetch_corpus
python -m evaluation.ingest_corpus
python -m evaluation.run_experiment --id my-run --rerank --top-k 8
python -m evaluation.compare exp-001-baseline my-run
```

## Configuration (`backend/.env`)

| Variable | Default | |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `OPENAI_MODEL` | `gpt-4o-mini` | answer generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 1536 dims |
| `TOP_K` | `5` | chunks the LLM sees (recommended: `8` with reranking) |
| `RERANK_ENABLED` | `false` | cross-encoder reranking |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L12-v2` | |
| `RERANK_CANDIDATES` | `20` | vector candidates fetched before reranking |
| `MIN_RERANK_SCORE` | unset | relevance gate (recommended: `-2.5`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `150` | characters |
| `PG_HOST`, `PG_PORT`, `PG_DATABASE`, `PG_CONNECTION_USER`, `PG_CONNECTION_PASSWORD` | `localhost`, `5432`, `ragbench`, `postgres`, `root` | |

The code defaults reproduce the original baseline exactly. `.env.example` ships the
recommended configuration (reranking on, k=8, gate at −2.5).

## Project layout

```
backend/
  app/
    api/routes/        chat, documents (upload/list/delete), usage (cost reporting)
    services/
      rag.py           query path: embed → search → rerank → gate → synthesize, timed per stage
      reranker.py      cross-encoder reranking
      synthesizer.py   grounded structured-output generation with citations
      ingestion.py     idempotent PDF ingest with page provenance
      llm.py, usage.py, pricing.py   token and cost capture on every LLM call
  evaluation/          corpus, dataset, metrics, experiment runner, results/
  tests/               112 offline tests
frontend/              React + Vite + Tailwind
scripts/               plot_evolution.py (README chart from results), snapshot.py
docs/architecture.md   design decisions and known limitations
RESULTS.md             experiment results and analysis
```
