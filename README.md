# paraX — Production-Ready RAG Chatbot (runs locally, no Docker)

A simple-but-scalable Retrieval-Augmented-Generation chatbot.

- **Backend:** FastAPI
- **Vector DB:** PostgreSQL + [pgvector](https://github.com/pgvector/pgvector)
- **LLM:** OpenAI (`gpt-4o-mini` by default) via the official `openai` SDK
- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dims)
- **Frontend:** React (Vite)

## Pipeline

### Ingestion
```
PDF upload ──▶ extract text ──▶ chunk ──▶ embed ──▶ store vectors (pgvector)
```

### Query
```
question
  │
  ├─▶ 1. Embed question
  ├─▶ 2. Vector similarity search → top-k chunks
  ├─▶ 3. Build context (question + chunks)
  └─▶ 4. Answer Synthesizer (LLM, structured output)
            → { direct_answer, sources }
```

> Exactly **one chat-LLM call per query** (the synthesizer). Retrieval uses an
> embeddings call + a pure pgvector search — no LLM.

> **Design note on steps 5 & 6.** The spec lists "generate an answer" and "pass through a
> synthesizer" as two stages. For an MVP we collapse them into a **single structured-output
> call**: the model answers *strictly from the retrieved context* and emits the structured
> response in one pass. This halves latency/cost while preserving the contract. To make it a
> true two-pass flow later, split `services/synthesizer.py` into a generator + a synthesizer —
> the interfaces are already isolated.

## Project layout

```
paraX/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + startup
│   │   ├── config.py            # settings (env-driven)
│   │   ├── database.py          # engine, session, pgvector bootstrap
│   │   ├── models.py            # SQLAlchemy ORM (Document, Chunk)
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── api/routes/
│   │   │   ├── documents.py     # upload / list / delete
│   │   │   └── chat.py          # ask a question
│   │   ├── services/
│   │   │   ├── pdf_extractor.py # PDF → page text
│   │   │   ├── chunker.py       # text → overlapping chunks
│   │   │   ├── embeddings.py    # OpenAI embeddings
│   │   │   ├── vector_store.py  # insert + similarity search
│   │   │   └── synthesizer.py   # grounded structured answer
│   │   └── core/prompts.py
│   ├── requirements.txt
│   └── .env.example
└── frontend/                    # React + Vite
```

## Prerequisites (install locally)

- **Python 3.11+**
- **Node.js 18+**
- **PostgreSQL 14+ with the `pgvector` extension**

### Installing pgvector locally

**Windows:** Install PostgreSQL via the EDB installer, then install pgvector. The simplest
route on Windows is to grab a prebuilt pgvector for your PG version, or use the
[StackBuilder]/[`pgvector` releases]. Once installed, the app runs
`CREATE EXTENSION IF NOT EXISTS vector` automatically on startup.

**macOS:** `brew install pgvector` (after `brew install postgresql@16`).

**Linux:** `sudo apt install postgresql-16-pgvector` (Debian/Ubuntu) or build from source.

### Create the database

```sql
-- psql -U postgres
CREATE DATABASE parax;
```

The `vector` extension and all tables/indexes are created automatically by the backend on
startup — you don't need to run any migrations by hand for the MVP.

## Run it (two terminals)

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
copy .env.example .env        # Windows  (macOS/Linux: cp .env.example .env)
# edit .env → set OPENAI_API_KEY and DATABASE_URL

uvicorn app.main:app --reload
```

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

- UI: http://localhost:5173 (proxies `/api` → `http://localhost:8000`)

## Environment variables (`backend/.env`)

| Variable                  | Description                                | Default                                                        |
|---------------------------|--------------------------------------------|----------------------------------------------------------------|
| `OPENAI_API_KEY`          | OpenAI API key                             | —                                                              |
| `OPENAI_MODEL`            | Chat model for answer synthesis            | `gpt-4o-mini`                                                  |
| `OPENAI_EMBEDDING_MODEL`  | Embedding model                            | `text-embedding-3-small`                                       |
| `OPENAI_EMBEDDING_DIM`    | Embedding dimension (must match the table) | `1536`                                                         |
| `PG_HOST`                 | Postgres host                              | `localhost`                                                    |
| `PG_PORT`                 | Postgres port                              | `5432`                                                         |
| `PG_DATABASE`             | Database name                              | `parax`                                                        |
| `PG_CONNECTION_USER`      | Database user                              | `postgres`                                                     |
| `PG_CONNECTION_PASSWORD`  | Database password                          | `root`                                                         |
| `PG_SCHEMA_NAME`          | Schema for tables (via `search_path`)      | `public`                                                       |
| `TOP_K`                   | Chunks retrieved per query                 | `5`                                                            |
| `CHUNK_SIZE`              | Target chunk size (characters)             | `1200`                                                         |
| `CHUNK_OVERLAP`           | Overlap between chunks (characters)        | `150`                                                          |

> Changing `OPENAI_EMBEDDING_DIM` after data has been ingested requires re-embedding everything
> (the vector column dimension is fixed at table-creation time). Drop the tables and re-upload.

## Scaling notes (MVP → production)

- **Embeddings batching** is already used on ingest. For large PDFs, move ingestion to a
  background worker (Celery/RQ/Arq) and return a job id from the upload endpoint.
- **HNSW index** on the vector column is created automatically. Add a metadata filter
  (e.g. per-tenant `document_id`) as your corpus grows.
- **Auth & multi-tenancy:** add a `user_id`/`tenant_id` column to `documents` and filter every
  query — the retrieval layer is the only place that needs to change.
- **Observability:** log the question and retrieved chunk ids per request.
