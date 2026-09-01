"""Command-line interface for CreativeContactBench evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .dataset import load_dataset
from .prompt import render_prompt, sha256_file
from .run_validation import validate_run
from .runner import load_config, project_dir_for, run_evaluations


def _load_context(config_value: str):
    config_path = Path(config_value).resolve()
    project_dir = project_dir_for(config_path)
    config = load_config(config_path)
    dataset = config["dataset"]
    root, tasks = load_dataset(
        repo_id=dataset["repo_id"],
        revision=dataset["revision"],
        expected_tasks=int(dataset["expected_tasks"]),
        cache_dir=dataset.get("cache_dir"),
    )
    return config_path, project_dir, config, root, tasks


def command_validate_data(args: argparse.Namespace) -> int:
    _, _, config, root, tasks = _load_context(args.config)
    print(json.dumps({
        "status": "PASS",
        "dataset_repo_id": config["dataset"]["repo_id"],
        "dataset_revision": config["dataset"]["revision"],
        "local_snapshot": str(root),
        "record_count": len(tasks),
        "task_ids": [task.task_id for task in tasks],
    }, indent=2))
    return 0


def command_preview(args: argparse.Namespace) -> int:
    _, project_dir, config, _, tasks = _load_context(args.config)
    by_id = {task.task_id: task for task in tasks}
    if args.task_id not in by_id:
        raise ValueError(f"Unknown task ID: {args.task_id}")
    task = by_id[args.task_id]
    prompt_path = project_dir / config["prompt"]["path"]
    print(render_prompt(prompt_path.read_text(encoding="utf-8"), task.task_instruction, task.options))
    print(f"\nImage path: {task.image_path}")
    print(f"Prompt SHA-256: {sha256_file(prompt_path)}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    selected = [item.strip() for item in args.task_ids.split(",") if item.strip()] if args.task_ids else None
    run_dir, manifest = run_evaluations(
        args.config,
        task_ids=selected,
        limit=args.limit,
        output_directory=args.output_dir,
    )
    print(json.dumps({
        "run_directory": str(run_dir),
        "status": manifest["status"],
        "requested_evaluations": manifest["requested_evaluations"],
        "model_requests": manifest["model_requests"],
        "valid_json": manifest["valid_json"],
        "semantically_valid": manifest["semantically_valid"],
        "failures": manifest["failures"],
    }, indent=2))
    return 0 if manifest["failures"] == 0 else 1


def command_validate_run(args: argparse.Namespace) -> int:
    print(json.dumps(validate_run(
        args.run_dir,
        allow_partial=args.allow_partial,
        allow_mock=args.allow_mock,
    ), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_data = subparsers.add_parser("validate-data", help="download and validate the pinned dataset")
    validate_data.add_argument("--config", required=True)
    validate_data.set_defaults(func=command_validate_data)
    preview = subparsers.add_parser("preview", help="render one prompt without a model request")
    preview.add_argument("--config", required=True)
    preview.add_argument("--task-id", required=True)
    preview.set_defaults(func=command_preview)
    run = subparsers.add_parser("run", help="run canonical evaluation")
    run.add_argument("--config", required=True)
    run.add_argument("--task-ids", help="comma-separated diagnostic subset")
    run.add_argument("--limit", type=int, help="diagnostic first-N subset")
    run.add_argument("--output-dir")
    run.set_defaults(func=command_run)
    validate_result = subparsers.add_parser("validate-run", help="validate standard artifacts")
    validate_result.add_argument("--run-dir", required=True)
    validate_result.add_argument("--allow-partial", action="store_true")
    validate_result.add_argument(
        "--allow-mock",
        action="store_true",
        help="accept the deterministic mock backend for offline pipeline testing",
    )
    validate_result.set_defaults(func=command_validate_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
