"""Small strict-schema helpers for model-generated JSON objects."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from bettercallagent.providers.openai_compatible import ProviderError


def require_exact_keys(
    value: Mapping[str, Any],
    expected: Iterable[str],
    *,
    purpose: str,
) -> None:
    """Reject missing and unknown model-output keys."""
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise ProviderError(
            f"{purpose} response keys do not match the configured schema; "
            f"missing={sorted(expected_set - actual)}, "
            f"unknown={sorted(actual - expected_set)}."
        )


def require_string(value: object, *, field: str) -> str:
    """Return a stripped non-empty model-output string."""
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(f"{field} must be a non-empty string.")
    return value.strip()


def require_string_list(value: object, *, field: str) -> list[str]:
    """Return a unique list of non-empty model-output strings."""
    if not isinstance(value, list):
        raise ProviderError(f"{field} must be an array.")
    result = [require_string(item, field=f"{field}[]") for item in value]
    if len(result) != len(set(result)):
        raise ProviderError(f"{field} must not contain duplicates.")
    return result


def require_number(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    """Return a finite bounded number, rejecting JSON booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderError(f"{field} must be a number.")
    number = float(value)
    if not minimum <= number <= maximum:
        raise ProviderError(f"{field} must be in [{minimum}, {maximum}].")
    return number
