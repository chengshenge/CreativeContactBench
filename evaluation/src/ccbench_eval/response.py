"""Strict JSON Schema and ranking-semantics validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


OPTION_IDS = frozenset({"A", "B", "C", "D"})


class ResponseValidationError(ValueError):
    """A model response violates the benchmark contract."""


def load_schema(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    return payload


def parse_response(raw: str, schema: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError(f"Invalid JSON: {exc}") from exc
    return validate_response(payload, schema)
