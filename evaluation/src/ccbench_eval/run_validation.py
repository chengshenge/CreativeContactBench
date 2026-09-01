"""Independent validation for a completed standard run directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from . import DATASET_REPO, DATASET_REVISION, EXPECTED_TASKS, PROMPT_SHA256, STANDARD_VERSION


REQUIRED = {
    "manifest.json",
    "results.jsonl",
    "summary.csv",
    "summary.md",
    "submission.json",
    "code_snapshot_sha256.json",
    "raw_responses",
}


class RunValidationError(ValueError):
    """A result directory is incomplete or not a standard submission."""


def validate_run(
    run_dir: str | Path,
    *,
    allow_partial: bool = False,
    allow_mock: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir)
    missing = sorted(name for name in REQUIRED if not (root / name).exists())
    if missing:
        raise RunValidationError(f"Missing artifacts: {missing}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    submission = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    results = [
        json.loads(line)
        for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with (root / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    errors: list[str] = []
    if manifest.get("dataset_repo_id") != DATASET_REPO:
        errors.append("dataset repo differs from the standard")
    if manifest.get("dataset_revision") != DATASET_REVISION:
        errors.append("dataset revision differs from the standard")
    if manifest.get("prompt_sha256") != PROMPT_SHA256:
        errors.append("prompt hash differs from the standard")
    if manifest.get("option_order") != "canonical" or manifest.get("repeats_per_task") != 1:
        errors.append("option order or repeat count is non-standard")
    if manifest.get("automatic_retries") != 0 or manifest.get("response_repairs") != 0:
        errors.append("retries or response repairs were enabled")
    if manifest.get("configuration", {}).get("generation", {}).get("store") is not False:
        errors.append("generation.store is not false")
    configured_run = manifest.get("configuration", {}).get("run", {})
    if any(
        configured_run.get(setting) is not False
        for setting in ("retry_api_errors", "retry_invalid_response", "repair_invalid_json")
    ):
        errors.append("configuration does not explicitly disable retry and repair")
    if manifest.get("backend") == "mock" and not allow_mock:
        errors.append("mock runs are test artifacts, not standard VLM submissions")
    if len(results) != len(summary_rows):
        errors.append("results.jsonl and summary.csv row counts differ")
    if submission.get("format") != STANDARD_VERSION:
        errors.append("submission format is not recognized")
    ids = [record.get("task_id") for record in results]
    if len(ids) != len(set(ids)):
        errors.append("duplicate task IDs found")
    if any(record.get("option_order") != "canonical" for record in results):
        errors.append("one or more records are not canonical")
    if any(record.get("dataset_revision") != DATASET_REVISION for record in results):
        errors.append("result records contain another dataset revision")
    if any(record.get("prompt_sha256") != PROMPT_SHA256 for record in results):
        errors.append("result records contain another prompt hash")
    identity = {option: option for option in "ABCD"}
    if any(record.get("displayed_to_canonical_mapping") != identity for record in results):
        errors.append("one or more records contain a non-identity option mapping")
    raw_files = [path for path in (root / "raw_responses").iterdir() if path.is_file()]
    successful = sum(record.get("raw_response") is not None for record in results)
    if len(raw_files) != successful:
        errors.append("raw response file count differs from successful calls")
    for record in results:
        relative = record.get("raw_response_path")
        if record.get("raw_response") is not None:
            raw_path = root / str(relative)
            if not raw_path.is_file() or raw_path.read_text(encoding="utf-8") != record["raw_response"]:
                errors.append(f"raw response mismatch for {record.get('task_id')}")
                break
    if not allow_partial:
        expected_ids = {f"task-{number:02d}" for number in range(1, EXPECTED_TASKS + 1)}
        if manifest.get("requested_evaluations") != EXPECTED_TASKS or set(ids) != expected_ids:
            errors.append("run does not contain exactly the 67 standard tasks")
        if len(results) != EXPECTED_TASKS:
            errors.append("result row count is not 67")
        if any(record.get("parse_status") != "valid" for record in results):
            errors.append("one or more responses are not valid JSON")
        if any(record.get("semantic_validation_status") != "valid" for record in results):
            errors.append("one or more responses failed semantic validation")
        if manifest.get("status") != "completed" or manifest.get("failures") != 0:
            errors.append("manifest does not describe a failure-free completed run")
        if manifest.get("valid_json") != EXPECTED_TASKS or manifest.get("semantically_valid") != EXPECTED_TASKS:
            errors.append("manifest validation counts are not 67/67")
        if len(submission.get("results", [])) != EXPECTED_TASKS:
            errors.append("submission.json does not contain 67 results")
        expected_requests = 0 if manifest.get("backend") == "mock" else EXPECTED_TASKS
        if manifest.get("model_requests") != expected_requests:
            errors.append(f"model request count is not {expected_requests}")
    if errors:
        raise RunValidationError("Run validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "PASS",
        "run_id": manifest.get("run_id"),
        "model": manifest.get("model_name"),
        "dataset_revision": manifest.get("dataset_revision"),
        "tasks": len(results),
        "model_requests": manifest.get("model_requests"),
        "valid_json": manifest.get("valid_json"),
        "semantically_valid": manifest.get("semantically_valid"),
        "failures": manifest.get("failures"),
        "partial": allow_partial,
        "mock": manifest.get("backend") == "mock",
    }
