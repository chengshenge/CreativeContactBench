"""Configuration-driven canonical evaluation and artifact persistence."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .adapters import AdapterExecutionError, adapter_for
from .dataset import Task, load_dataset
from .prompt import render_prompt, sha256_file
from .reporting import (
    write_code_snapshot,
    write_submission,
    write_summary_csv,
    write_summary_markdown,
)
from .response import ResponseValidationError, load_schema, validate_response


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_no_secrets(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"api_key", "token", "password", "secret"}:
                raise ValueError(
                    f"Literal secret field {path}.{key} is forbidden; use api_key_env"
                )
            _assert_no_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_secrets(child, f"{path}[{index}]")


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    _assert_no_secrets(config)
    for section in ("dataset", "prompt", "response_schema", "run", "generation", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    run = config["run"]
    for setting in ("retry_api_errors", "retry_invalid_response", "repair_invalid_json"):
        if run.get(setting, False) is not False:
            raise ValueError(f"Standard runner requires run.{setting}: false")
    if config["generation"].get("store", False) is not False:
        raise ValueError("Standard runner requires generation.store: false")
    return config


def project_dir_for(config_path: Path) -> Path:
    for parent in config_path.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "ccbench_eval").is_dir():
            return parent
    raise ValueError(f"Could not locate evaluation project root for {config_path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _select_tasks(
    tasks: list[Task],
    configured: Any,
    override: list[str] | None,
    limit: int | None,
) -> list[Task]:
    requested = override
    if requested is None and configured != "all":
        if not isinstance(configured, list) or not all(isinstance(item, str) for item in configured):
            raise ValueError("run.task_ids must be 'all' or a list of task IDs")
        requested = configured
    if requested is not None:
        unknown = set(requested) - {task.task_id for task in tasks}
        if unknown:
            raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
        tasks = [task for task in tasks if task.task_id in set(requested)]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        tasks = tasks[:limit]
    return tasks


def run_evaluations(
    config_path: str | Path,
    *,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    output_directory: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    config_path = Path(config_path).resolve()
    project_dir = project_dir_for(config_path)
    config = load_config(config_path)
    dataset_config = config["dataset"]
    prompt_config = config["prompt"]
    run_config = config["run"]
    generation = dict(config["generation"])
    endpoint = dict(config.get("endpoint") or {})

    prompt_path = (project_dir / prompt_config["path"]).resolve()
    prompt_hash = sha256_file(prompt_path)
    if prompt_hash != prompt_config["sha256"]:
        raise ValueError(
            f"Prompt SHA-256 mismatch: configured {prompt_config['sha256']}, actual {prompt_hash}"
        )
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema_path = (project_dir / config["response_schema"]["path"]).resolve()
    schema = load_schema(schema_path)
    schema_hash = file_sha256(schema_path)

    dataset_root, all_tasks = load_dataset(
        repo_id=dataset_config["repo_id"],
        revision=dataset_config["revision"],
        expected_tasks=int(dataset_config["expected_tasks"]),
        cache_dir=dataset_config.get("cache_dir"),
    )
    selected = _select_tasks(all_tasks, run_config.get("task_ids", "all"), task_ids, limit)
    output_base = Path(output_directory or config["output"]["directory"])
    if not output_base.is_absolute():
        output_base = project_dir / output_base
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = output_base / run_id
    raw_dir = run_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=False)
    results_path = run_dir / "results.jsonl"
    started_at = utc_now()
    backend = str(run_config["backend"])
    model_name = str(run_config["model_name"])
    adapter = adapter_for(backend)
    adapter_config = {**generation, **endpoint, "model_name": model_name}
    safe_generation = {key: value for key, value in generation.items() if key != "api_key_env"}

    manifest: dict[str, Any] = {
        "format": "creativecontactbench-run-v1",
        "runner_version": __version__,
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "configuration_path": str(config_path),
        "configuration": config,
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "dataset_repo_id": dataset_config["repo_id"],
        "dataset_revision": dataset_config["revision"],
        "dataset_local_snapshot": str(dataset_root),
        "prompt_version": prompt_config["version"],
        "prompt_sha256": prompt_hash,
        "response_schema_sha256": schema_hash,
        "option_order": "canonical",
        "repeats_per_task": 1,
        "model_name": model_name,
        "model_revision": generation.get("model_revision"),
        "backend": backend,
        "generation_parameters": safe_generation,
        "requested_task_ids": [task.task_id for task in selected],
        "requested_evaluations": len(selected),
        "model_requests": 0,
        "successful_model_calls": 0,
        "valid_json": 0,
        "semantically_valid": 0,
        "failures": 0,
        "automatic_retries": 0,
        "response_repairs": 0,
        "total_tokens": 0,
        "invocation_overrides": {"task_ids": task_ids, "limit": limit},
    }
    _write_json(run_dir / "manifest.json", manifest)

    records: list[dict[str, Any]] = []
    aborted = False
    with results_path.open("w", encoding="utf-8") as results_handle:
        for task in selected:
            if aborted:
                break
            rendered = render_prompt(prompt_template, task.task_instruction, task.options)
            raw: str | None = None
            response: dict[str, Any] | None = None
            parse_status = "model_error"
            semantic_status = "not_run"
            error: str | None = None
            latency: float | None = None
            backend_metadata: dict[str, Any] = {}
            model_metadata: dict[str, Any] = {}
            request_metadata: dict[str, Any] = {}
            try:
                result = adapter.generate(rendered, task.image_path, adapter_config)
                raw = result.raw_response
                latency = result.latency_seconds
                backend_metadata = result.backend_metadata
                model_metadata = result.model_metadata
                request_metadata = result.request_metadata
                (raw_dir / f"{task.task_id}.txt").write_text(raw, encoding="utf-8")
                try:
                    payload = json.loads(raw)
                    parse_status = "valid"
                    try:
                        response = validate_response(payload, schema)
                        semantic_status = "valid"
                    except ResponseValidationError as exc:
                        semantic_status = "invalid"
                        error = str(exc)
                except json.JSONDecodeError as exc:
                    parse_status = "invalid_json"
                    error = f"Invalid JSON: {exc}"
            except AdapterExecutionError as exc:
                error = str(exc)
                backend_metadata = exc.metadata
                request_metadata = {"model_requests": exc.metadata.get("model_requests", 0)}
                if exc.metadata.get("error_category") in {"authentication", "model_access"}:
                    aborted = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            record = {
                "run_id": run_id,
                "timestamp_utc": utc_now(),
                "task_id": task.task_id,
                "dataset_repo_id": dataset_config["repo_id"],
                "dataset_revision": dataset_config["revision"],
                "image_relative_path": task.file_name,
                "image_sha256": file_sha256(task.image_path),
                "prompt_version": prompt_config["version"],
                "prompt_sha256": prompt_hash,
                "response_schema_sha256": schema_hash,
                "option_order": "canonical",
                "displayed_to_canonical_mapping": {option: option for option in "ABCD"},
                "backend": backend,
                "model_name": model_name,
                "model_revision": generation.get("model_revision"),
                "generation_parameters": safe_generation,
                "raw_response": raw,
                "raw_response_path": f"raw_responses/{task.task_id}.txt" if raw is not None else None,
                "parse_status": parse_status,
                "semantic_validation_status": semantic_status,
                "error": error,
                "response": response,
                "latency_seconds": latency,
                "backend_metadata": backend_metadata,
                "model_metadata": model_metadata,
                "request_metadata": request_metadata,
                "provider_request_id": backend_metadata.get("request_id"),
                "input_tokens": backend_metadata.get("input_tokens"),
                "cached_input_tokens": backend_metadata.get("cached_input_tokens"),
                "output_tokens": backend_metadata.get("output_tokens"),
                "reasoning_tokens": backend_metadata.get("reasoning_tokens"),
                "total_tokens": backend_metadata.get("total_tokens"),
            }
            results_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_handle.flush()
            records.append(record)

    manifest.update(
        {
            "status": "aborted" if aborted else "completed" if all(r.get("response") for r in records) else "completed_with_failures",
            "completed_at": utc_now(),
            "attempted_task_ids": [record["task_id"] for record in records],
            "model_requests": sum(int(record["request_metadata"].get("model_requests", 0)) for record in records),
            "successful_model_calls": sum(record["raw_response"] is not None for record in records),
            "valid_json": sum(record["parse_status"] == "valid" for record in records),
            "semantically_valid": sum(record["semantic_validation_status"] == "valid" for record in records),
            "failures": sum(record["semantic_validation_status"] != "valid" for record in records),
            "returned_model_ids": sorted({
                str(record["model_metadata"]["returned_model"])
                for record in records
                if record["model_metadata"].get("returned_model")
            }),
            "total_tokens": sum(int(record["total_tokens"]) for record in records if record.get("total_tokens") is not None),
        }
    )
    write_summary_csv(run_dir, records)
    write_summary_markdown(run_dir, records, manifest)
    write_submission(run_dir, records, manifest)
    write_code_snapshot(project_dir, run_dir)
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir, manifest
