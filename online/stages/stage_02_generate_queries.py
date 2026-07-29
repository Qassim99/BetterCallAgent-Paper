"""Stage 2: construct the five deterministic retrieval views."""

from __future__ import annotations

from typing import Any

from bettercallagent.providers.openai_compatible import parse_json_object
from bettercallagent.retrieval.query_views import build_query_views
from online.context import RunContext
from online.dependencies import OnlineDependencies
from online.parsing import require_exact_keys, require_string, require_string_list
from online.prompts import query_generation_messages
from online.repository import AssetError

_FIELDS = ("meta_searchterm", "keywords")


async def run(
    context: RunContext,
    dependencies: OnlineDependencies,
) -> dict[str, Any]:
    """Generate bounded terms, then build stable framework-independent views."""
    if context.understanding is None:
        raise RuntimeError("Stage 1 must complete before query generation.")
    response = await dependencies.provider.complete(
        query_generation_messages(context.record.query, context.understanding),
        model=context.model,
        purpose="query_generation",
        json_response=True,
        max_tokens=600,
        temperature=0.0,
    )
    parsed = parse_json_object(response.content)
    require_exact_keys(parsed, _FIELDS, purpose="query generation")
    meta_searchterm = require_string(
        parsed["meta_searchterm"],
        field="meta_searchterm",
    )
    keywords = require_string_list(parsed["keywords"], field="keywords")
    views = build_query_views(
        context.record.query,
        meta_searchterm=meta_searchterm,
        keywords=keywords,
    )
    expected = dependencies.repository.retrieval_views.get(context.record.query_id)
    if expected is None:
        raise AssetError("No versioned retrieval views exist for the configured query.")
    actual_mapping = views.as_mapping()
    expected_mapping = expected.as_mapping()
    mismatched = [
        field
        for field, actual in actual_mapping.items()
        if actual.encode("utf-8") != expected_mapping[field].encode("utf-8")
    ]
    if mismatched:
        raise AssetError(
            "Generated retrieval views do not byte-match the ranking artifact "
            f"for query_id {context.record.query_id!r}; "
            f"mismatched fields={mismatched}."
        )
    context.query_views = views
    context.usage_total_tokens += response.usage_total_tokens
    search_queries = list(views.as_mapping().values())
    return {
        "kind": "query_generation",
        "search_queries": search_queries,
        "meta_searchterm_de": views.meta_searchterm,
        "keywords": keywords,
    }
