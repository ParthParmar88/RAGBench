"""Stages 4-6: build context, generate a grounded answer, return structured output.

The spec lists answer-generation and answer-synthesis as separate stages. For the
MVP they are collapsed into one structured-output call: the model answers strictly
from the retrieved context and emits the structured response in a single pass.
"""
from pydantic import BaseModel, Field

from app.config import settings
from app.core.prompts import SYNTHESIZER_SYSTEM
from app.services import llm
from app.services.vector_store import RetrievedChunk


class SynthesizedAnswer(BaseModel):
    direct_answer: str = Field(..., description="Concise answer grounded in the context.")
    source_numbers: list[int] = Field(
        default_factory=list,
        description="The [n] passage numbers actually used to answer.",
    )


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (title: {c.title})\n{c.content}")
    return "\n\n".join(blocks)


def synthesize_answer(
    question: str,
    chunks: list[RetrievedChunk],
    usage: llm.UsageContext | None = None,
    *,
    model: str | None = None,
) -> tuple[SynthesizedAnswer, list[dict]]:
    """Return the structured answer plus a resolved list of source citations."""
    context = _build_context(chunks) if chunks else "(no relevant context found)"

    user_content = f"Question: {question}\n\nContext passages:\n{context}"

    completion = llm.chat_parse(
        model=model or settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_format=SynthesizedAnswer,
        operation="synthesize",
        usage=usage,
    )
    answer = completion.choices[0].message.parsed
    if answer is None:
        answer = SynthesizedAnswer(
            direct_answer="I couldn't produce an answer from the available context.",
            source_numbers=[],
        )

    # Resolve [n] numbers back to the underlying chunks.
    resolved: list[dict] = []
    for n in answer.source_numbers:
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            resolved.append(
                {
                    "number": n,
                    "title": c.title,
                    "score": round(c.score, 4),
                    "snippet": c.content[:280],
                }
            )
    return answer, resolved
