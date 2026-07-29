"""Stage 6: synthesize and exactly post-validate the final answer."""

from __future__ import annotations

from typing import Any

from bettercallagent.citations.output_validation import (
    OutputCitationError,
    validate_answer_citations,
)
from online.context import RunContext
from online.dependencies import OnlineDependencies
from online.prompts import answer_messages


async def run(
    context: RunContext,
    dependencies: OnlineDependencies,
) -> dict[str, Any]:
    """Generate a grounded answer and reject every unsupported citation."""
    if context.selection is None or not context.reranked:
        raise RuntimeError("Stage 5 must complete before answer generation.")
    accepted = context.selection.predicted_citations
    accepted_set = set(accepted)
    evidence = [
        candidate
        for candidate in context.reranked
        if accepted_set.intersection(candidate.citations)
    ][: dependencies.settings.top_select]
    if not evidence:
        evidence = context.reranked[: dependencies.settings.top_select]
    response = await dependencies.provider.complete(
        answer_messages(
            query=context.record.query,
            accepted_citations=accepted,
            candidates=evidence,
        ),
        model=context.model,
        purpose="answer",
        json_response=False,
        max_tokens=1_500,
        temperature=0.0,
    )
    validation = validate_answer_citations(
        response.content,
        accepted_citations=accepted,
        vocabulary=dependencies.vocabulary,
        extractor=dependencies.extractor,
    )
    if accepted and not validation.exact_mentioned:
        raise OutputCitationError("The generated answer omitted all evidence-gated citations.")
    context.answer = response.content
    context.answer_validation = validation
    context.usage_total_tokens += response.usage_total_tokens
    return {
        "kind": "final_answer_step",
        "grounded_on": list(validation.exact_mentioned),
        "document_count": len(evidence),
    }
