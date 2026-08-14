"""Chat endpoint: the RAG query (embed -> vector search -> synthesize)."""
from uuid import uuid4

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import ChatRequest, ChatResponse, Citation
from app.services import embeddings, usage as usage_service, vector_store
from app.services.llm import UsageContext
from app.services.synthesizer import synthesize_answer

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def ask(payload: ChatRequest, db: Session = Depends(get_db)):
    # Collects token usage for the LLM call(s) in this request.
    usage = UsageContext(request_id=str(uuid4()))

    # 1. Embed the question.
    query_vector = embeddings.embed_query(payload.question)

    # 2. Vector similarity search → top-k chunks.
    chunks = vector_store.search(db, query_vector)

    # 3. Generate the grounded answer (single LLM call).
    answer, sources = synthesize_answer(
        question=payload.question,
        chunks=chunks,
        usage=usage,
    )

    # Persist token + cost usage (best-effort; never breaks the response).
    # Returns the same numbers it stored, for the client to display.
    turn_usage = usage_service.persist(db, usage)

    return ChatResponse(
        direct_answer=answer.direct_answer,
        sources=[Citation(**s) for s in sources],
        usage=turn_usage,
    )
