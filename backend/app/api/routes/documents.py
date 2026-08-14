"""Document ingestion endpoints: upload a PDF, list sources, delete a source."""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import DeleteResponse, SourceInfo, UploadResponse
from app.services import embeddings, vector_store
from app.services.chunker import chunk_text
from app.services.pdf_extractor import extract_text

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a PDF → extract → chunk → embed → store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    text = extract_text(file_bytes)
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text found (the PDF may be scanned images).",
        )

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no chunks.")

    vectors = embeddings.embed_texts(chunks)
    added = vector_store.add_knowledge_units(db, file.filename, chunks, vectors)

    return UploadResponse(title=file.filename, chunks_added=added)


@router.get("", response_model=list[SourceInfo])
def list_documents(db: Session = Depends(get_db)):
    """List ingested sources with their chunk counts."""
    return vector_store.list_sources(db)


@router.delete("/{title}", response_model=DeleteResponse)
def delete_document(title: str, db: Session = Depends(get_db)):
    """Delete every chunk belonging to a source title."""
    deleted = vector_store.delete_source(db, title)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No source named {title!r}.")

    return DeleteResponse(title=title, chunks_deleted=deleted)
