"""Stage 1: convert the question into a typed legal-issue analysis."""

from __future__ import annotations

from typing import Any

from bettercallagent.providers.openai_compatible import parse_json_object
from online.context import RunContext
from online.dependencies import OnlineDependencies
from online.parsing import (
    require_exact_keys,
    require_string,
    require_string_list,
)
from online.prompts import understanding_messages

_FIELDS = (
    "legal_area",
    "summary_de",
    "key_facts",
    "legal_issues",
    "meta_searchterm",
    "keywords",
)


async def run(
    context: RunContext,
    dependencies: OnlineDependencies,
) -> dict[str, Any]:
    """Analyze the legal question and enforce the configured response schema."""
    response = await dependencies.provider.complete(
        understanding_messages(context.record.query, context.history),
        model=context.model,
        purpose="understanding",
        json_response=True,
        max_tokens=1_200,
        temperature=0.0,
    )
    parsed = parse_json_object(response.content)
    require_exact_keys(parsed, _FIELDS, purpose="understanding")
    understanding: dict[str, Any] = {
        "legal_area": require_string(parsed["legal_area"], field="legal_area"),
        "summary_de": require_string(parsed["summary_de"], field="summary_de"),
        "key_facts": require_string_list(parsed["key_facts"], field="key_facts"),
        "legal_issues": require_string_list(
            parsed["legal_issues"],
            field="legal_issues",
        ),
        "meta_searchterm": require_string(
            parsed["meta_searchterm"],
            field="meta_searchterm",
        ),
        "keywords": require_string_list(parsed["keywords"], field="keywords"),
    }
    extracted_from_query = dependencies.extractor.extract(context.record.query)
    understanding["explicit_citations"] = list(extracted_from_query)
    context.understanding = understanding
    context.usage_total_tokens += response.usage_total_tokens
    return {
        "kind": "understanding",
        "route": "legal",
        "restated_question": understanding["summary_de"],
        "legal_topic": understanding["legal_area"],
        "languages_considered": ["Deutsch"],
        "key_legal_concepts": [
            *understanding["legal_issues"],
            *understanding["keywords"],
        ],
    }
