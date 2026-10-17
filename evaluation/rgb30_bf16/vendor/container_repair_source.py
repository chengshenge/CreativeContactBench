"""Strict JSON Schema and ranking-semantics validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


OPTION_IDS = frozenset({"A", "B", "C", "D"})
OPTION_ORDER = ("A", "B", "C", "D")
RATING_DIMENSIONS = (
    "expected_task_effectiveness",
    "embodied_feasibility",
    "functional_creativity",
)


class ResponseValidationError(ValueError):
    """A model response violates the benchmark contract."""


REPAIR_STRATEGY = "strip_markdown_json_fence_or_wrap_object"


def repair_json_container(raw: str) -> tuple[str, str | None]:
    """Apply the benchmark's opt-in, deterministic JSON-container repair."""
    candidate = raw.strip()
    if candidate.startswith("```json") and candidate.endswith("```"):
        candidate = candidate[len("```json") : -len("```")].strip()
        return candidate, "strip_markdown_json_fence"
    if not candidate.startswith("{") and candidate.startswith('"ratings"'):
        return "{" + candidate + "}", "wrap_missing_outer_object"
    return raw, None


def load_schema(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rating_totals(payload: dict[str, Any]) -> dict[str, int]:
    """Return the unweighted sum of the three required ratings per option."""
    return {
        option: sum(int(payload["ratings"][option][dimension]) for dimension in RATING_DIMENSIONS)
        for option in OPTION_ORDER
    }


def ranking_from_rating_totals(payload: dict[str, Any]) -> list[list[str]]:
    """Build canonical tie groups ordered by descending unweighted rating total."""
    totals = rating_totals(payload)
    return [
        [option for option in OPTION_ORDER if totals[option] == total]
        for total in sorted(set(totals.values()), reverse=True)
    ]


def ranking_matches_rating_totals(payload: dict[str, Any]) -> bool:
    """Accept any order inside a tie group while requiring exact score-based groups."""
    actual = [frozenset(group) for group in payload["overall_ranking"]]
    expected = [frozenset(group) for group in ranking_from_rating_totals(payload)]
    return actual == expected


def validate_response(payload: Any, schema: dict[str, Any]) -> dict[str, Any]:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ResponseValidationError(
            f"Schema validation failed at {location}: {error.message}"
        )
    flattened = [option for group in payload["overall_ranking"] for option in group]
    if len(flattened) != 4 or len(set(flattened)) != 4 or set(flattened) != OPTION_IDS:
        raise ResponseValidationError(
            "overall_ranking must contain A, B, C, and D exactly once"
        )
    if not ranking_matches_rating_totals(payload):
        totals = rating_totals(payload)
        expected = ranking_from_rating_totals(payload)
        raise ResponseValidationError(
            "overall_ranking must exactly match descending sums of the three ratings; "
            f"totals={totals}, expected={expected}"
        )
    return payload


def parse_response(raw: str, schema: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError(f"Invalid JSON: {exc}") from exc
    return validate_response(payload, schema)
