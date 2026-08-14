"""Document ingestion endpoints: upload a PDF, list sources, delete a source."""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import DeleteResponse, SourceInfo, UploadResponse
from app.services import ingestion, vector_store
from app.services.ingestion import EmptyDocumentError

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a PDF → extract → chunk → embed → store.

    Idempotent: re-uploading the same bytes under the same name is a no-op, and
    uploading changed bytes replaces the previous chunks rather than adding to
    them.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    try:
        result = ingestion.ingest_pdf(db, file.filename, file_bytes)
    except EmptyDocumentError:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found (the PDF may be scanned images).",
        )

    return UploadResponse(
        title=result.title,
        chunks_added=result.chunks_added,
        status=result.status,
        page_count=result.page_count,
    )


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
