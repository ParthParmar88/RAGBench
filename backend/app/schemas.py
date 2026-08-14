"""Pydantic request/response models for the HTTP API."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# --- Documents ---
class UploadResponse(BaseModel):
    title: str
    chunks_added: int


class SourceInfo(BaseModel):
    title: str
    chunks: int


class DeleteResponse(BaseModel):
    title: str
    chunks_deleted: int


# --- Chat ---
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Citation(BaseModel):
    number: int
    title: str
    score: float | None = None  # vector similarity of the cited chunk
    snippet: str


class TurnUsageCall(BaseModel):
    operation: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float | None


class TurnUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float | None  # null if any call lacks pricing; excludes embeddings
    currency: str
    model: str  # primary (synthesis) model for the turn
    calls: int
    request_id: str
    breakdown: list[TurnUsageCall]


class ChatResponse(BaseModel):
    direct_answer: str
    sources: list[Citation]
    usage: TurnUsage | None = None


# --- Usage / cost ---
class UsageTotals(BaseModel):
    count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float
    currency: str


class UsageBreakdown(BaseModel):
    key: str
    count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float


class UsageSummary(BaseModel):
    totals: UsageTotals
    by_model: list[UsageBreakdown]
    by_operation: list[UsageBreakdown]


class UsageRow(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    request_id: str | None
    provider: str
    model: str
    operation: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    total_cost: float | None
    currency: str
    latency_ms: int | None
