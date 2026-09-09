"""Download, verify, inspect, and evaluate frozen multimodal inputs.

Outputs stay local. This entry point never uploads responses or human data.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
from pathlib import Path

from PIL import Image
from ccbench_eval.adapters import adapter_for, AdapterExecutionError
from ccbench_eval.response import parse_response, ResponseValidationError

HERE = Path(__file__).resolve().parent
CONDITIONS = ("rgb_only", "rgb_description", "rgb_tactile")
TASK_IDS = tuple(f"task-{i:02d}" for i in (1, 5, 6, 7, 8, 10, 13, 24, 25, 27, 32, 37, 38, 42, 43, 47))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Expected a relative bundle path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes bundle")
    return path


def verify(bundle, expected_sha=None):
    bundle = Path(bundle)
    if expected_sha and sha(bundle / "manifest.json") != expected_sha:
        raise ValueError("Bundle manifest differs from pinned release")
    manifest = read(bundle / "manifest.json")
    if manifest["format_version"] != "ccb-multimodal-inputs-v5":
        raise ValueError("Unsupported bundle version")
    if manifest["task_count"] != 16 or tuple(t["task_id"] for t in manifest["tasks"]) != TASK_IDS:
        raise ValueError("Expected the fixed 16-task catalog")
    if tuple(manifest["conditions"]) != CONDITIONS:
        raise ValueError("Unexpected conditions")
    if len({t["contact_group"] for t in manifest["tasks"]}) != 12:
        raise ValueError("Expected 12 contact families")
    for relative, expected in manifest["files"].items():
        path = safe_path(bundle, relative)
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError(f"Modified input: {relative}")
    region = tuple(manifest["rgb_region_xyxy"])
    if tuple(manifest["canvas_size"]) != (2048, 1024) or region != (0, 0, 1024, 1024):
        raise ValueError("Unexpected canvas or RGB crop")
    if manifest["schema_path"] not in manifest["files"]:
        raise ValueError("Schema missing from inventory")
    for task in manifest["tasks"]:
        if task["displayed_to_canonical"] != dict(zip("ABCD", "ABCD")):
            raise ValueError("This release uses canonical option order")
        if set(task["conditions"]) != set(CONDITIONS):
            raise ValueError("Missing paired condition")
        for condition in CONDITIONS:
            entry = task["conditions"][condition]
            for kind in ("prompt", "image"):
                relative = entry[kind + "_path"]
                if relative not in manifest["files"] or entry[kind + "_sha256"] != manifest["files"][relative]["sha256"]:
                    raise ValueError("Payload missing from inventory or hash mismatch")
            with Image.open(safe_path(bundle, entry["image_path"])) as im:
                im = im.convert("RGB")
                if im.size != (2048, 1024):
                    raise ValueError("Unexpected image size")
                if hashlib.sha256(im.crop(region).tobytes()).hexdigest() != task["shared_rgb_pixels_sha256"]:
                    raise ValueError("Conditions do not share the same RGB pixels")
                if condition != "rgb_tactile" and im.crop((1024, 0, 2048, 1024)).getextrema() != ((255, 255),) * 3:
                    raise ValueError("Baseline right panel must be blank")
        if task["conditions"]["rgb_only"]["image_sha256"] != task["conditions"]["rgb_description"]["image_sha256"]:
            raise ValueError("Description ablation changed the image")
    return manifest


def parse_arm(raw, schema, arm="strict"):
    if arm == "fence_only":
        match = re.fullmatch(r"```(?:json)?[ \t]*\n(.*?)\n```", raw.strip(), re.DOTALL)
        if match and "```" not in match.group(1):
            raw = match.group(1)
    elif arm != "strict":
        raise ValueError("Unknown parsing arm")
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ResponseValidationError("Duplicate JSON key")
            value[key] = item
        return value
    try:
        json.loads(raw, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ResponseValidationError("Invalid JSON") from exc
    # Strict JSON + schema + each canonical option exactly once. No score-sum ranking.
    return parse_response(raw, schema)


def jobs_for(manifest, repeats=2, seed=2026090905):
    if repeats < 1:
        raise ValueError("Repeats must be positive")
    jobs = []
    orders = list(itertools.permutations(CONDITIONS))
    for ri in range(repeats):
        tasks = sorted(manifest["tasks"], key=lambda t: t["task_id"])
        random.Random(seed + ri).shuffle(tasks)
        for ti, task in enumerate(tasks):
            for condition in orders[(ti + 3 * ri) % len(orders)]:
                jobs.append({"task_id": task["task_id"], "condition": condition, "repeat_id": f"{ri+1:02d}"})
    return jobs


def run(bundle, manifest, config, output, repeats=2, execute=False):
    output = Path(output)
    if output.resolve().is_relative_to(bundle.resolve()) or bundle.resolve().is_relative_to(output.resolve()):
        raise ValueError("Run directory and input bundle must be separate")
    # One output directory represents one immutable plan. Resuming could duplicate
    # a charged request after a crash; deliberate fresh runs use new directories.
    output.mkdir(parents=True, exist_ok=False)
    jobs = jobs_for(manifest, repeats)
    plan = {"format_version": "ccb-multimodal-run-v1", "input_manifest_sha256": sha(bundle / "manifest.json"),
            "model": config, "repeats": repeats, "jobs": jobs, "execute": execute,
            "automatic_retries": 0, "response_content_repair": False,
            "source_hashes": {"evaluate.py": sha(Path(__file__))}}
    # Record adapter/parser versions without machine-specific source paths.
    import inspect
    import ccbench_eval.adapters as adapters
    import ccbench_eval.response as response
    for module in (adapters, response):
        plan["source_hashes"][module.__name__] = sha(Path(inspect.getfile(module)))
    write(output / "plan.json", plan)
    if not execute:
        print(f"Dry run: {len(jobs)} planned requests; no model calls. Plan: {output / 'plan.json'}")
        return
    adapter = adapter_for(config["backend"])
    schema = read(bundle / manifest["schema_path"])
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    for index, job in enumerate(jobs):
        target = output / "responses" / job["task_id"] / job["condition"] / (job["repeat_id"] + ".json")
        entry = tasks[job["task_id"]]["conditions"][job["condition"]]
        record = dict(job, status="started", prompt_sha256=entry["prompt_sha256"], image_sha256=entry["image_sha256"])
        write(target, record)
        try:
            result = adapter.generate((bundle / entry["prompt_path"]).read_text(encoding="utf-8"), bundle / entry["image_path"], config)
            record.update(status="received", raw_response=result.raw_response,
                          backend_metadata=result.backend_metadata, model_metadata=result.model_metadata,
                          request_metadata=result.request_metadata, latency_seconds=result.latency_seconds)
            record["parsing"] = {}
            for arm in ("strict", "fence_only"):
                try:
                    record["parsing"][arm] = {"valid": True, "response": parse_arm(result.raw_response, schema, arm)}
                except ResponseValidationError:
                    record["parsing"][arm] = {"valid": False}
        except AdapterExecutionError as exc:
            # Do not persist exception strings, which may contain provider secrets.
            record.update(status="api_error", error_category=exc.metadata.get("error_category", "api_error"))
            write(target, record)
            raise SystemExit("Provider request failed; stopped without retry. Completed responses remain local.") from None
        write(target, record)
        print(f"Completed {index+1}/{len(jobs)}: {job['task_id']} {job['condition']} repeat {job['repeat_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "validate", "inspect", "run"))
    parser.add_argument("--release", type=Path, default=HERE / "release.json")
    parser.add_argument("--bundle", type=Path, default=HERE / "data/v5")
    parser.add_argument("--task", choices=TASK_IDS, default="task-01")
    parser.add_argument("--condition", choices=CONDITIONS, default="rgb_tactile")
    parser.add_argument("--config", type=Path, default=HERE / "configs/mock.json")
    parser.add_argument("--output", type=Path, default=HERE / "runs/dry-run")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--execute", action="store_true", help="Actually send the planned inputs to the configured provider")
    args = parser.parse_args()
    release = read(args.release)
    if args.command == "download":
        if not isinstance(release.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", release["revision"]):
            raise SystemExit("HF input upload is pending. No downloadable revision is published yet; see multimodal/README.md.")
        from huggingface_hub import snapshot_download
        import shutil
        cached = Path(snapshot_download(repo_id=release["repo_id"], repo_type="dataset", revision=release["revision"],
                                       allow_patterns=[release["path"] + "/**"])) / release["path"]
        verify(cached, release["manifest_sha256"])
        shutil.copytree(cached, args.bundle, dirs_exist_ok=False)
    manifest = verify(args.bundle, release["manifest_sha256"])
    if args.command in ("validate", "download"):
        print("Verified 16 tasks, 12 contact families, 48 input pairs, all file hashes and paired RGB pixels.")
    elif args.command == "inspect":
        task = next(t for t in manifest["tasks"] if t["task_id"] == args.task)
        entry = task["conditions"][args.condition]
        print((args.bundle / entry["prompt_path"]).read_text(encoding="utf-8"))
        print("\nIMAGE:", (args.bundle / entry["image_path"]).resolve())
    elif args.command == "run":
        config = read(args.config)
        allowed = {"backend", "model", "model_name", "api_key_env", "base_url", "reasoning_effort", "reasoning_mode",
                   "max_output_tokens", "max_tokens", "temperature", "timeout", "store", "revision"}
        if set(config) - allowed or config.get("store", False) is not False:
            raise ValueError("Unsupported config fields; credentials belong in environment variables only")
        config.setdefault("model_name", config.get("model", "mock"))
        run(args.bundle, manifest, config, args.output, args.repeats, args.execute)


if __name__ == "__main__":
    main()
