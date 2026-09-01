"""Standard CSV, Markdown, submission, and code-hash artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from . import STANDARD_VERSION


OPTIONS = ("A", "B", "C", "D")
DIMENSIONS = (
    "expected_task_effectiveness",
    "embodied_feasibility",
    "functional_creativity",
)


def ranking_text(ranking: list[list[str]] | None) -> str:
    if not ranking:
        return ""
    return " > ".join(
        group[0] if len(group) == 1 else "{" + " | ".join(group) + "}"
        for group in ranking
    )


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response") or {}
    ratings = response.get("ratings") or {}
    ranking = response.get("overall_ranking")
    row: dict[str, Any] = {
        "task_id": record["task_id"],
        "parse_status": record["parse_status"],
        "semantic_validation_status": record["semantic_validation_status"],
        "overall_ranking": ranking_text(ranking),
        "top_ranked_option_or_tie": "|".join(ranking[0]) if ranking else "",
        "confidence": response.get("confidence", ""),
    }
    for option in OPTIONS:
        for dimension in DIMENSIONS:
            row[f"{option}_{dimension}"] = ratings.get(option, {}).get(dimension, "")
    for field in (
        "latency_seconds",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "provider_request_id",
    ):
        row[field] = record.get(field, "") if record.get(field) is not None else ""
    return row


def write_summary_csv(run_dir: Path, records: list[dict[str, Any]]) -> None:
    rows = [_summary_row(record) for record in records]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(
    run_dir: Path,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    valid = [record["response"] for record in records if record.get("response")]
    top_credit = Counter()
    for response in valid:
        top = response["overall_ranking"][0]
        for option in top:
            top_credit[option] += 1 / len(top)
    lines = [
        "# CreativeContactBench standard evaluation summary",
        "",
        "## Run",
        "",
        f"- Model: `{manifest['model_name']}`",
        f"- Model revision: `{manifest.get('model_revision') or 'not provided'}`",
        f"- Dataset revision: `{manifest['dataset_revision']}`",
        f"- Prompt SHA-256: `{manifest['prompt_sha256']}`",
        f"- Requested tasks: {manifest['requested_evaluations']}",
        f"- Model requests: {manifest['model_requests']}",
        f"- Valid JSON: {manifest['valid_json']}",
        f"- Semantically valid: {manifest['semantically_valid']}",
        f"- Failures: {manifest['failures']}",
        f"- Total tokens: {manifest['total_tokens']}",
        "- Option order: canonical A/B/C/D",
        "- Automatic retries: 0",
        "- Response repairs: 0",
        "",
        "## Aggregate",
        "",
        "- First-place credit: " + ", ".join(f"{option}={top_credit[option]:.1f}" for option in OPTIONS),
    ]
    confidences = [response["confidence"] for response in valid]
    if confidences:
        lines.append(f"- Mean confidence: {statistics.mean(confidences):.4f}")
    for dimension in DIMENSIONS:
        scores = [response["ratings"][option][dimension] for response in valid for option in OPTIONS]
        if scores:
            lines.append(
                f"- {dimension} mean/stddev: {statistics.mean(scores):.4f} / "
                f"{statistics.pstdev(scores):.4f}"
            )
    lines += [
        "",
        "## Per-task results",
        "",
        "| Task | Ranking | Confidence | Parse | Semantic validation |",
        "|---|---|---:|---|---|",
    ]
    for record in records:
        response = record.get("response") or {}
        lines.append(
            f"| {record['task_id']} | {ranking_text(response.get('overall_ranking')) or '—'} | "
            f"{response.get('confidence', '—')} | {record['parse_status']} | "
            f"{record['semantic_validation_status']} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_submission(
    run_dir: Path,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    payload = {
        "format": STANDARD_VERSION,
        "benchmark": "CreativeContactBench",
        "dataset": {
            "repo_id": manifest["dataset_repo_id"],
            "revision": manifest["dataset_revision"],
            "task_count": manifest["requested_evaluations"],
        },
        "evaluator": {
            "prompt_version": manifest["prompt_version"],
            "prompt_sha256": manifest["prompt_sha256"],
            "response_schema_sha256": manifest["response_schema_sha256"],
            "option_order": "canonical",
            "repeats_per_task": 1,
            "automatic_retries": 0,
            "response_repairs": 0,
        },
        "model": {
            "name": manifest["model_name"],
            "revision": manifest.get("model_revision"),
            "backend": manifest["backend"],
            "returned_model_ids": manifest.get("returned_model_ids", []),
        },
        "run": {
            "run_id": manifest["run_id"],
            "status": manifest["status"],
            "started_at": manifest["started_at"],
            "completed_at": manifest["completed_at"],
            "valid_json": manifest["valid_json"],
            "semantically_valid": manifest["semantically_valid"],
            "failures": manifest["failures"],
        },
        "results": [
            {
                "task_id": record["task_id"],
                "ratings": record["response"]["ratings"],
                "overall_ranking": record["response"]["overall_ranking"],
                "confidence": record["response"]["confidence"],
            }
            for record in records
            if record.get("response") is not None
        ],
    }
    (run_dir / "submission.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_code_snapshot(project_dir: Path, run_dir: Path) -> None:
    excluded = {
        "runs",
        "build",
        "dist",
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".secrets",
    }
    inventory: dict[str, str] = {}
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(project_dir)
        if any(part in excluded or part.endswith(".egg-info") for part in relative.parts):
            continue
        if (
            path.name.startswith(".env")
            or path.suffix in {".env", ".pyc", ".tmp"}
        ):
            continue
        inventory[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run_dir / "code_snapshot_sha256.json").write_text(
        json.dumps({"algorithm": "sha256", "files": inventory}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
