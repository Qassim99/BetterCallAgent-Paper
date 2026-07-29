"""Render compact JSON and Markdown summaries from analysis components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .error_attribution import SCHEMA_VERSION as ERROR_SCHEMA
from .gate_surrogate import SCHEMA_VERSION as SURROGATE_SCHEMA
from .perturbation_faithfulness import SCHEMA_VERSION as PERTURBATION_SCHEMA
from .ragas_style import SCHEMA_VERSION as RAGAS_STYLE_SCHEMA

REPORT_SCHEMA = "bettercallagent.interpretability.report.v1"
SCHEMA_ORDER = (
    ERROR_SCHEMA,
    RAGAS_STYLE_SCHEMA,
    SURROGATE_SCHEMA,
    PERTURBATION_SCHEMA,
)


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _metric_table(metrics: Mapping[str, Any]) -> list[str]:
    lines = ["| Metric | Value |", "|---|---:|"]
    for name, value in metrics.items():
        lines.append(f"| `{name}` | {_format_number(value)} |")
    return lines


def _render_error_attribution(component: Mapping[str, Any]) -> list[str]:
    summary = component["summary"]
    lines = [
        "## Citation error-stage attribution",
        "",
        (
            f"Queries: **{summary['query_count']}**. "
            f"Macro retrieval recall: **{summary['macro_retrieval_recall']:.6f}**. "
            f"Macro prediction F1: **{summary['macro_prediction_f1']:.6f}**."
        ),
        "",
        "| Category | Citation count |",
        "|---|---:|",
    ]
    for name, count in summary["counts"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "A retrieved-but-missed citation is intentionally attributed to the "
            "combined selector/gate stage unless a finer trace is supplied.",
        ]
    )
    return lines


def _render_ragas_style(component: Mapping[str, Any]) -> list[str]:
    return [
        "## Custom `ragas_style` citation metrics",
        "",
        ("These transparent local formulas are not results from the official RAGAS library."),
        "",
        *_metric_table(component["macro"]),
    ]


def _render_surrogate(component: Mapping[str, Any]) -> list[str]:
    fit = component["fit"]
    lines = [
        "## Logistic/additive citation-gate surrogate",
        "",
        (
            f"Rows: **{fit['row_count']}**. Fidelity accuracy: "
            f"**{fit['fidelity_accuracy']:.6f}**. Brier score: "
            f"**{fit['brier_score']:.6f}**."
        ),
        "",
        (
            "Reported terms are additive contributions to the fitted surrogate's "
            "log-odds. They describe the surrogate, not model internals or causal "
            "effects."
        ),
        "",
        "| Feature | Standardized coefficient |",
        "|---|---:|",
    ]
    for feature in fit["features"]:
        lines.append(f"| `{feature['name']}` | {feature['standardized_coefficient']:.6f} |")
    return lines


def _render_perturbation(component: Mapping[str, Any]) -> list[str]:
    return [
        "## Seeded perturbation faithfulness proxy",
        "",
        *_metric_table(component["macro"]),
        "",
        f"Limitation: {component['limitations']}",
    ]


RENDERERS = {
    ERROR_SCHEMA: _render_error_attribution,
    RAGAS_STYLE_SCHEMA: _render_ragas_style,
    SURROGATE_SCHEMA: _render_surrogate,
    PERTURBATION_SCHEMA: _render_perturbation,
}


def build_report(
    components: Sequence[Mapping[str, Any]],
    *,
    title: str = "BetterCallAgent interpretability report",
) -> tuple[dict[str, Any], str]:
    """Validate and combine known component schemas into JSON and Markdown."""

    if not title.strip():
        raise ValueError("report title must not be empty")
    if not components:
        raise ValueError("at least one report component is required")

    by_schema: dict[str, Mapping[str, Any]] = {}
    for component in components:
        schema = component.get("schema_version")
        if not isinstance(schema, str) or schema not in RENDERERS:
            raise ValueError(f"unsupported report component schema: {schema!r}")
        if schema in by_schema:
            raise ValueError(f"duplicate report component schema: {schema}")
        by_schema[schema] = component

    ordered = [by_schema[schema] for schema in SCHEMA_ORDER if schema in by_schema]
    json_report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "title": title,
        "components": ordered,
    }

    markdown_lines = [
        f"# {title}",
        "",
        (
            "This deterministic report was rendered from saved offline artifacts. "
            "It performs no retrieval, model inference, or external network calls."
        ),
        "",
    ]
    for component in ordered:
        markdown_lines.extend(RENDERERS[component["schema_version"]](component))
        markdown_lines.append("")
    return json_report, "\n".join(markdown_lines).rstrip() + "\n"
