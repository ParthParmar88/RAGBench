"""FastAPI application entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, usage
from app.config import settings
from app.database import init_db
from app.services import reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Enable pgvector, create tables + index on startup.
    init_db()
    if settings.RERANK_ENABLED:
        reranker.warm_up()
    yield


app = FastAPI(title="RAGBench API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(usage.router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
