"""Chat endpoint: the RAG query (embed -> vector search -> synthesize)."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    QueryTimings,
    RetrievalInfo,
)
from app.services import rag

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def ask(payload: ChatRequest, db: Session = Depends(get_db)):
    result = rag.answer_question(db, payload.question)

    # The full retrieved context set stays out of the HTTP response on purpose:
    # it can run to tens of KB per reply and only evaluation needs it, which
    # calls `rag.answer_question` directly.
    return ChatResponse(
        direct_answer=result.direct_answer,
        sources=[Citation(**s) for s in result.sources],
        retrieval=RetrievalInfo(
            chunk_count=result.retrieval.chunk_count,
            top_score=result.retrieval.top_score,
            avg_score=result.retrieval.avg_score,
        ),
        timings=QueryTimings(
            embedding_ms=result.timings.embedding_ms,
            search_ms=result.timings.search_ms,
            retrieval_ms=result.timings.retrieval_ms,
            generation_ms=result.timings.generation_ms,
            total_ms=result.timings.total_ms,
        ),
        usage=result.usage,
    )
