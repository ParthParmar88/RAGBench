"""Central wrapper around the OpenAI client.

Every chat call goes through `chat_parse`, which times the call, reads the
returned token usage, and appends a `UsageRecord` to a request-scoped
`UsageContext`. This is the single choke point where usage is captured — no DB
access here (the route persists the collected records).

To add another provider later, add a sibling wrapper that normalizes its usage
fields into the same `UsageRecord` shape.
"""
from dataclasses import dataclass
from time import perf_counter

from app.services.openai_client import client

PROVIDER = "openai"


@dataclass
class UsageRecord:
    operation: str
    provider: str
    model_requested: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    latency_ms: int
    raw_usage: dict | None


class UsageContext:
    """Accumulates usage records for a single API request."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.records: list[UsageRecord] = []


def chat_parse(
    *,
    model: str,
    messages: list,
    response_format,
    operation: str,
    usage: UsageContext | None = None,
):
    """Structured-output chat call with automatic usage capture."""
    started = perf_counter()
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
    )
    latency_ms = int((perf_counter() - started) * 1000)

    if usage is not None:
        _capture(usage, operation, model, completion, latency_ms)

    return completion


def _capture(
    usage: UsageContext,
    operation: str,
    model_requested: str,
    completion,
    latency_ms: int,
) -> None:
    u = getattr(completion, "usage", None)
    if u is None:
        return

    details = getattr(u, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0

    try:
        raw = u.model_dump()
    except Exception:
        raw = None

    usage.records.append(
        UsageRecord(
            operation=operation,
            provider=PROVIDER,
            model_requested=model_requested,
            model=getattr(completion, "model", model_requested) or model_requested,
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
            cached_tokens=cached,
            latency_ms=latency_ms,
            raw_usage=raw,
        )
    )
