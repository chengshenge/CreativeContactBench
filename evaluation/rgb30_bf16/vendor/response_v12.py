"""Strict JSON parsing, schema validation, and ranking semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


OPTION_IDS = frozenset({"A", "B", "C", "D"})


class ResponseValidationError(ValueError):
    """Raised when a raw evaluator response violates the response contract."""


def load_schema(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_response(payload: Any, schema: dict[str, Any]) -> dict[str, Any]:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ResponseValidationError(f"Schema validation failed at {location}: {error.message}")

    ranking = payload["overall_ranking"]
    flattened = [option_id for group in ranking for option_id in group]
    if len(flattened) != 4:
        raise ResponseValidationError(
            f"Ranking must contain exactly four option occurrences, found {len(flattened)}"
        )
    if len(flattened) != len(set(flattened)):
        raise ResponseValidationError("Each option must appear exactly once in the ranking")
    if set(flattened) != OPTION_IDS:
        missing = sorted(OPTION_IDS - set(flattened))
        unknown = sorted(set(flattened) - OPTION_IDS)
        raise ResponseValidationError(
            f"Ranking coverage mismatch; missing={missing}, unknown={unknown}"
        )
    return payload


def parse_response(raw_response: str, schema: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError(f"Invalid JSON: {exc}") from exc
    return validate_response(payload, schema)
