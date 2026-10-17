#!/usr/bin/env python3
"""Reproducible holistic-ranking Human--VLM agreement baseline.

This program is deliberately limited to participant holistic rankings.  It uses
the model-provided ``overall_ranking`` field, never a ranking reconstructed from
the three scalar ratings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


STRATEGIES = ("A", "B", "C", "D")
COVERAGE_FIELDS = (
    "row_type", "entity_type", "entity_id", "run_id", "model_name",
    "exact_model_version", "task_id", "selected_for_main",
    "in_human_questionnaire", "version_compatible", "record_count",
    "valid_count", "missing_count", "duplicate_count", "parse_failures",
    "semantic_failures", "repair_count", "option_order", "mapping_status",
    "dataset_revision", "prompt_version", "prompt_sha256", "image_sha256",
    "status", "issue_codes", "detail",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            rows.append({"_line_number": line_number, "_json_error": str(exc)})
        else:
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def verify_hash(root: Path, spec: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = root / spec[path_key]
    actual = sha256_file(path)
    expected = spec[hash_key]
    if actual != expected:
        raise RuntimeError(f"Frozen-input hash mismatch for {path}: {actual} != {expected}")
    return path


def canonical_dense_ranks(grouped: Any) -> dict[str, Fraction]:
    """Validate best-to-worst grouped ranking and convert to dense ranks."""
    if not isinstance(grouped, list) or not grouped:
        raise ValueError("ranking is not a non-empty list of groups")
    out: dict[str, Fraction] = {}
    for dense_rank, group in enumerate(grouped, 1):
        if not isinstance(group, list) or not group:
            raise ValueError(f"rank group {dense_rank} is not a non-empty list")
        for strategy in group:
            if strategy not in STRATEGIES:
                raise ValueError(f"unknown canonical strategy {strategy!r}")
            if strategy in out:
                raise ValueError(f"duplicate canonical strategy {strategy}")
            out[strategy] = Fraction(dense_rank)
    missing = sorted(set(STRATEGIES) - set(out))
    if missing:
        raise ValueError(f"missing canonical strategies: {','.join(missing)}")
    return out


def average_rankings(rankings: Iterable[dict[str, Fraction]]) -> dict[str, Fraction]:
    rankings = list(rankings)
    if not rankings:
        raise ValueError("cannot aggregate zero rankings")
    return {
        s: sum((r[s] for r in rankings), Fraction(0)) / len(rankings)
        for s in STRATEGIES
    }


def grouped_from_ranks(ranks: dict[str, Fraction]) -> list[list[str]]:
    groups: list[list[str]] = []
    for _, items in _group_pairs(sorted(ranks.items(), key=lambda x: (x[1], x[0]))):
        groups.append([s for s, _ in items])
    return groups


def _group_pairs(items: list[tuple[str, Fraction]]):
    if not items:
        return
    current_value = items[0][1]
    current = []
    for item in items:
        if item[1] != current_value:
            yield current_value, current
            current_value = item[1]
            current = []
        current.append(item)
    yield current_value, current


def ranking_text(ranks: dict[str, Fraction]) -> str:
    return " > ".join("=".join(group) for group in grouped_from_ranks(ranks))


def kendall_tau_b(x: dict[str, Fraction], y: dict[str, Fraction]) -> float | None:
    concordant = discordant = ties_x = ties_y = 0
    for i, left in enumerate(STRATEGIES):
        for right in STRATEGIES[i + 1:]:
            dx = (x[left] > x[right]) - (x[left] < x[right])
            dy = (y[left] > y[right]) - (y[left] < y[right])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x) *
        (concordant + discordant + ties_y)
    )
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def top_group(ranks: dict[str, Fraction]) -> set[str]:
    best = min(ranks.values())
    return {s for s, rank in ranks.items() if rank == best}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def bootstrap_model(
    task_rows: dict[str, dict[str, Any]],
    source_groups: dict[str, str],
    resamples: int,
    seed: int,
) -> dict[str, float | None]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for task_id in task_rows:
        grouped[source_groups.get(task_id, task_id)].append(task_id)
    group_ids = sorted(grouped)
    rng = random.Random(seed)
    tau_samples: list[float] = []
    top_samples: list[float] = []
    strict_samples: list[float] = []
    for _ in range(resamples):
        sampled_tasks: list[str] = []
        for group_id in (rng.choice(group_ids) for _ in group_ids):
            sampled_tasks.extend(grouped[group_id])
        taus = [task_rows[t]["tau_b"] for t in sampled_tasks if task_rows[t]["tau_b"] is not None]
        if taus:
            tau_samples.append(statistics.fmean(taus))
        eligible = [task_rows[t] for t in sampled_tasks if task_rows[t]["top1_eligible"]]
        if eligible:
            top_samples.append(statistics.fmean(float(r["top1_hit"]) for r in eligible))
            strict_samples.append(statistics.fmean(float(r["strict_top1_hit"]) for r in eligible))
    return {
        "tau_ci_low": percentile(tau_samples, 0.025),
        "tau_ci_high": percentile(tau_samples, 0.975),
        "top1_ci_low": percentile(top_samples, 0.025),
        "top1_ci_high": percentile(top_samples, 0.975),
        "strict_top1_ci_low": percentile(strict_samples, 0.025),
        "strict_top1_ci_high": percentile(strict_samples, 0.975),
    }


def bootstrap_human_aggregate(
    per_task_rows: dict[str, list[dict[str, Any]]],
    source_groups: dict[str, str],
    resamples: int,
    seed: int,
) -> dict[str, float | None]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for task_id in per_task_rows:
        grouped[source_groups.get(task_id, task_id)].append(task_id)
    group_ids = sorted(grouped)
    rng = random.Random(seed)
    tau_samples: list[float] = []
    top_samples: list[float] = []
    for _ in range(resamples):
        sampled: list[dict[str, Any]] = []
        for group_id in (rng.choice(group_ids) for _ in group_ids):
            for task_id in grouped[group_id]:
                sampled.extend(per_task_rows[task_id])
        taus = [r["tau_b"] for r in sampled if r["tau_b"] is not None]
        if taus:
            tau_samples.append(statistics.fmean(taus))
        eligible = [r for r in sampled if r["top1_eligible"]]
        if eligible:
            top_samples.append(statistics.fmean(float(r["top1_hit"]) for r in eligible))
    return {
        "tau_ci_low": percentile(tau_samples, 0.025),
        "tau_ci_high": percentile(tau_samples, 0.975),
        "top1_ci_low": percentile(top_samples, 0.025),
        "top1_ci_high": percentile(top_samples, 0.975),
    }


def as_csv(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: as_csv(row.get(k)) for k in fields})


def coverage_row(**kwargs: Any) -> dict[str, Any]:
    return {field: kwargs.get(field, "") for field in COVERAGE_FIELDS}


def exact_model_version(manifest: dict[str, Any]) -> str:
    name = manifest.get("model_name") or manifest.get("configuration", {}).get("generation", {}).get("model", "unknown")
    revision = manifest.get("model_revision")
    returned = sorted(set(manifest.get("returned_model_ids") or []))
    parts = [str(name)]
    if revision:
        parts.append(f"revision={revision}")
    if returned and returned != [name]:
        parts.append("returned=" + ",".join(returned))
    return " | ".join(parts)


def canonical_model_ranking(row: dict[str, Any]) -> tuple[dict[str, Fraction], str]:
    if row.get("canonicalized_response") is not None:
        response = row["canonicalized_response"]
        return canonical_dense_ranks(response.get("overall_ranking")), "canonicalized_response"
    mapping = row.get("displayed_to_canonical_mapping")
    identity = isinstance(mapping, dict) and all(mapping.get(s) == s for s in STRATEGIES)
    if row.get("option_order") != "canonical" or not identity:
        raise ValueError("non-identity/non-canonical record lacks canonicalized_response")
    response = row.get("response")
    if not isinstance(response, dict):
        raise ValueError("missing parsed response")
    return canonical_dense_ranks(response.get("overall_ranking")), "identity_response"


def repair_reconstructed_holistic_ranking(repair: Any) -> bool:
    """True when a repair invented/replaced holistic order from rating vectors."""
    label = str(repair or "")
    return "rebuild_ranking_from" in label or "replace_ranking_from" in label


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(c, c) for c in value)


def short_model_name(name: str) -> str:
    aliases = {
        "microsoft/Phi-4-reasoning-vision-15B": "Phi-4 Reasoning Vision 15B",
        "Qwen/Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B",
        "Qwen/Qwen3-VL-235B-A22B-Instruct": "Qwen3-VL-235B-A22B",
    }
    return aliases.get(name, name.split("/")[-1])


def draw_plot(path_png: Path, path_svg: Path, result_rows: list[dict[str, Any]], human_row: dict[str, Any]) -> None:
    ordered = sorted(result_rows, key=lambda r: r["mean_task_tau_b"], reverse=True)
    labels = [short_model_name(r["model_name"]) for r in ordered] + ["Human LOO (descriptive)"]
    rows = ordered + [human_row]
    width, margin_left, margin_right = 1500, 440, 120
    row_h, top, bottom = 49, 125, 115
    height = top + row_h * len(rows) + bottom
    x0, x1 = margin_left, width - margin_right

    def xcoord(v: float) -> float:
        return x0 + (v + 1.0) / 2.0 * (x1 - x0)

    try:
        from PIL import Image, ImageDraw, ImageFont
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        im = Image.new("RGB", (width, height), "white")
        d = ImageDraw.Draw(im)
        font = ImageFont.truetype(font_path, 23)
        small = ImageFont.truetype(font_path, 19)
        title = ImageFont.truetype(bold_path, 30)
        d.text((40, 27), "Holistic ranking agreement with Human reference", fill="#172321", font=title)
        d.text((40, 70), "Mean per-task Kendall's tau-b; 95% task-clustered bootstrap CI", fill="#596b67", font=small)
        for tick in (-1, -0.5, 0, 0.5, 1):
            x = xcoord(tick)
            d.line((x, top - 12, x, height - bottom + 12), fill="#dce4e1", width=2)
            label = f"{tick:g}"
            d.text((x - d.textlength(label, font=small) / 2, height - bottom + 25), label, fill="#52615e", font=small)
        for i, (label, row) in enumerate(zip(labels, rows)):
            y = top + i * row_h + row_h / 2
            if i == len(ordered):
                d.line((35, y - row_h / 2, width - 35, y - row_h / 2), fill="#8aa79f", width=3)
            d.text((40, y - 14), label, fill="#172321", font=font)
            low, high, mean = row["tau_ci_low"], row["tau_ci_high"], row["mean_task_tau_b"]
            color = "#17745a" if i < len(ordered) else "#a45820"
            if low is not None and high is not None:
                d.line((xcoord(low), y, xcoord(high), y), fill=color, width=5)
                d.line((xcoord(low), y - 8, xcoord(low), y + 8), fill=color, width=3)
                d.line((xcoord(high), y - 8, xcoord(high), y + 8), fill=color, width=3)
            d.ellipse((xcoord(mean) - 7, y - 7, xcoord(mean) + 7, y + 7), fill=color)
            d.text((x1 + 15, y - 12), f"{mean:.3f}", fill=color, font=small)
        d.text((40, height - 43), "Human LOO uses nine-person references and is shown separately; it is not an absolute ceiling.", fill="#596b67", font=small)
        im.save(path_png, optimize=True)
    except ImportError:
        path_png.write_text("Pillow unavailable; use the companion SVG.\n", encoding="utf-8")

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif}.label{font-size:23px;fill:#172321}.small{font-size:19px;fill:#596b67}.title{font-size:30px;font-weight:bold;fill:#172321}</style>',
        '<text x="40" y="52" class="title">Holistic ranking agreement with Human reference</text>',
        '<text x="40" y="91" class="small">Mean per-task Kendall\'s tau-b; 95% task-clustered bootstrap CI</text>',
    ]
    for tick in (-1, -0.5, 0, 0.5, 1):
        x = xcoord(tick)
        svg.append(f'<line x1="{x:.1f}" y1="{top-12}" x2="{x:.1f}" y2="{height-bottom+12}" stroke="#dce4e1" stroke-width="2"/>')
        svg.append(f'<text x="{x:.1f}" y="{height-bottom+48}" class="small" text-anchor="middle">{tick:g}</text>')
    for i, (label, row) in enumerate(zip(labels, rows)):
        y = top + i * row_h + row_h / 2
        if i == len(ordered):
            svg.append(f'<line x1="35" y1="{y-row_h/2:.1f}" x2="{width-35}" y2="{y-row_h/2:.1f}" stroke="#8aa79f" stroke-width="3"/>')
        color = "#17745a" if i < len(ordered) else "#a45820"
        svg.append(f'<text x="40" y="{y+8:.1f}" class="label">{html.escape(label)}</text>')
        low, high, mean = row["tau_ci_low"], row["tau_ci_high"], row["mean_task_tau_b"]
        if low is not None and high is not None:
            svg.append(f'<line x1="{xcoord(low):.1f}" y1="{y:.1f}" x2="{xcoord(high):.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="5"/>')
            for v in (low, high):
                svg.append(f'<line x1="{xcoord(v):.1f}" y1="{y-8:.1f}" x2="{xcoord(v):.1f}" y2="{y+8:.1f}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<circle cx="{xcoord(mean):.1f}" cy="{y:.1f}" r="7" fill="{color}"/>')
        svg.append(f'<text x="{x1+15}" y="{y+7:.1f}" font-size="19" fill="{color}">{mean:.3f}</text>')
    svg.append(f'<text x="40" y="{height-35}" class="small">Human LOO uses nine-person references and is shown separately; it is not an absolute ceiling.</text>')
    svg.append("</svg>")
    path_svg.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="human_eval_analysis/holistic_alignment_config_v1.json")
    parser.add_argument("--output", default="human_eval_analysis/results/holistic_alignment_baseline_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    config = load_json(config_path)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    human_spec = config["human"]
    task_spec = config["tasks"]
    model_spec = config["models"]
    human_path = verify_hash(root, human_spec, "export_path", "export_sha256")
    protocol_path = verify_hash(root, human_spec, "protocol_path", "protocol_sha256")
    task_path = verify_hash(root, task_spec, "human_task_manifest_path", "human_task_manifest_sha256")
    asset_path = verify_hash(root, task_spec, "human_asset_manifest_path", "human_asset_manifest_sha256")
    selection_path = verify_hash(root, model_spec, "selection_manifest_path", "selection_manifest_sha256")

    task_manifest = load_json(task_path)
    tasks_by_id = {x["task_id"]: x for x in task_manifest["tasks"]}
    human_protocol = load_json(protocol_path)
    formal_task_ids = set(human_protocol["formal_task_ids"])
    expected_presentation_positions = {
        task_id: index for index, task_id in enumerate(human_protocol["presentation_order"]["task_ids"], 1)
    }
    asset_manifest = load_json(asset_path)
    human_export = load_json(human_path)
    submissions = human_export["submissions"]
    selected_run_ids = load_json(selection_path)["source_runs"]
    exclusions = task_spec["main_set_version_exclusions"]
    coverage: list[dict[str, Any]] = []

    # Human records: retain every duplicate/invalid/missing row in the audit.
    human_rankings: dict[str, dict[str, dict[str, Fraction]]] = {}
    human_task_union: set[str] = set()
    participant_counts = Counter(str(s.get("participant_id", "")) for s in submissions)
    for submission in submissions:
        participant = submission.get("participant_id", "")
        submission_issues = []
        if not participant:
            submission_issues.append("missing_participant_id")
        if participant_counts[str(participant)] > 1:
            submission_issues.append("duplicate_participant_id")
        if submission.get("pilot") is not human_spec["pilot"]:
            submission_issues.append("pilot_flag_mismatch")
        if submission.get("dataset_revision") != human_spec["dataset_revision"]:
            submission_issues.append("dataset_revision_mismatch")
        if str(submission.get("protocol_version")) != str(human_spec["protocol_version"]):
            submission_issues.append("protocol_version_mismatch")
        submission_compatible = not submission_issues
        responses_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for response in submission.get("responses") or []:
            responses_by_task[str(response.get("task_id", ""))].append(response)
        participant_rankings: dict[str, dict[str, Fraction]] = {}
        expected = set(formal_task_ids)
        for task_id in sorted(expected | set(responses_by_task)):
            records = responses_by_task.get(task_id, [])
            valid: list[dict[str, Fraction]] = []
            errors: list[str] = []
            for response in records:
                if response.get("overall_ranking_source") != human_spec["ranking_source"]:
                    errors.append("unexpected_ranking_source")
                    continue
                if task_id in expected_presentation_positions and response.get("presentation_position") != expected_presentation_positions[task_id]:
                    errors.append("presentation_position_mismatch")
                    continue
                try:
                    valid.append(canonical_dense_ranks(response.get("overall_ranking")))
                except ValueError as exc:
                    errors.append(f"ranking_parse_failure:{exc}")
            in_questionnaire = bool(records)
            if len(records) == 1 and len(valid) == 1 and submission_compatible:
                participant_rankings[task_id] = valid[0]
                status = "valid"
            elif not records:
                status = "missing"
                errors.append("missing_response")
            elif len(records) > 1:
                status = "duplicate"
                errors.append("duplicate_response")
            else:
                status = "invalid"
            compatible = task_id not in exclusions
            coverage.append(coverage_row(
                row_type="human_task", entity_type="human", entity_id=participant,
                task_id=task_id, selected_for_main=compatible and status == "valid" and in_questionnaire,
                in_human_questionnaire=in_questionnaire, version_compatible=compatible,
                record_count=len(records), valid_count=len(valid), missing_count=int(not records),
                duplicate_count=max(0, len(records) - 1), dataset_revision=submission.get("dataset_revision", ""),
                prompt_version=f"human-protocol-{submission.get('protocol_version', '')}",
                option_order="canonical", mapping_status="identity_A_B_C_D", status=status,
                issue_codes=";".join(submission_issues + errors + (["version_incompatible"] if not compatible else [])),
                detail="; ".join(filter(None, [
                    exclusions.get(task_id, ""),
                    f"presentation_position={records[0].get('presentation_position')}" if len(records) == 1 else "",
                    f"ranking_source={records[0].get('overall_ranking_source')}" if len(records) == 1 else "",
                ])),
            ))
        human_rankings[participant] = participant_rankings
        human_task_union.update(participant_rankings)
        valid_expected = len(set(participant_rankings) & expected)
        coverage.append(coverage_row(
            row_type="human_summary", entity_type="human", entity_id=participant,
            selected_for_main=valid_expected > 0,
            in_human_questionnaire=True, version_compatible=True,
            record_count=sum(len(v) for v in responses_by_task.values()),
            valid_count=valid_expected,
            missing_count=len(expected - set(participant_rankings)),
            duplicate_count=sum(max(0, len(v) - 1) for v in responses_by_task.values()),
            dataset_revision=submission.get("dataset_revision", ""),
            prompt_version=f"human-protocol-{submission.get('protocol_version', '')}",
            option_order="canonical", mapping_status="identity_A_B_C_D",
            status="valid" if valid_expected == len(responses_by_task) == len(formal_task_ids) and submission_compatible else "has_issues",
            issue_codes=";".join(submission_issues),
            detail=f"submission_id={submission.get('id','')}; completed_at={submission.get('completed_at','')}",
        ))

    questionnaire_tasks = set.intersection(*(set(r) for r in human_rankings.values()))
    version_compatible_human_tasks = questionnaire_tasks - set(exclusions)
    for task_id, reason in exclusions.items():
        coverage.append(coverage_row(
            row_type="version_issue", entity_type="task", entity_id=task_id, task_id=task_id,
            in_human_questionnaire=task_id in questionnaire_tasks, version_compatible=False,
            status="excluded_from_main", issue_codes="version_incompatible", detail=reason,
        ))
    coverage.append(coverage_row(
        row_type="analysis_issue", entity_type="analysis", entity_id=config["analysis_id"],
        status="limitation", issue_codes="no_source_group_metadata",
        detail="No source_scene_id/source_group_id/scene_group_id exists in the task manifest; bootstrap falls back to task_id.",
    ))
    coverage.append(coverage_row(
        row_type="analysis_issue", entity_type="analysis", entity_id=config["analysis_id"],
        status="limitation", issue_codes="selected_prompt_not_current_repository_standard",
        prompt_version=model_spec["selected_run_prompt_version"], prompt_sha256=model_spec["selected_run_prompt_sha256"],
        detail=f"Selected complete runs use {model_spec['selected_run_prompt_version']}; repository current standard is {model_spec['repository_current_standard_prompt_version']}.",
    ))

    # Human reference is computed only from observed, valid holistic rankings.
    human_references: dict[str, dict[str, Fraction]] = {}
    for task_id in sorted(questionnaire_tasks):
        human_references[task_id] = average_rankings(human_rankings[p][task_id] for p in sorted(human_rankings))

    # Selected model runs and task-level aggregation.
    model_runs: list[dict[str, Any]] = []
    selected_run_root = root / model_spec["run_root"]
    for run_id in selected_run_ids:
        run_dir = selected_run_root / run_id
        manifest = load_json(run_dir / "manifest.json")
        result_rows = load_jsonl(run_dir / "results.jsonl")
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        malformed_lines = 0
        for row in result_rows:
            if row.get("_json_error"):
                malformed_lines += 1
                coverage.append(coverage_row(
                    row_type="model_record", entity_type="model", entity_id=manifest.get("model_name", ""),
                    run_id=run_id, model_name=manifest.get("model_name", ""),
                    exact_model_version=exact_model_version(manifest), status="invalid_jsonl_line",
                    parse_failures=1, issue_codes="jsonl_parse_failure",
                    detail=f"line={row.get('_line_number')}: {row.get('_json_error')}",
                ))
            else:
                by_task[str(row.get("task_id", ""))].append(row)
        aggregated: dict[str, dict[str, Fraction]] = {}
        task_audit: dict[str, dict[str, Any]] = {}
        expected_model_tasks = set(manifest.get("requested_task_ids") or manifest.get("attempted_task_ids") or tasks_by_id)
        for task_id in sorted(expected_model_tasks | set(by_task)):
            records = by_task.get(task_id, [])
            valid: list[dict[str, Fraction]] = []
            errors: list[str] = []
            mapping_modes: list[str] = []
            parse_failures = semantic_failures = repair_count = 0
            for row in records:
                if row.get("parse_status") != "valid":
                    parse_failures += 1
                    errors.append("parse_failure")
                    continue
                if row.get("semantic_validation_status") != "valid":
                    semantic_failures += 1
                    errors.append("semantic_failure")
                    continue
                if row.get("dataset_revision") != model_spec["dataset_revision"]:
                    errors.append("dataset_revision_mismatch")
                    continue
                if row.get("prompt_version") != model_spec["selected_run_prompt_version"] or row.get("prompt_sha256") != model_spec["selected_run_prompt_sha256"]:
                    errors.append("prompt_mismatch")
                    continue
                repair = row.get("response_repair") or row.get("repair") or row.get("repair_metadata")
                if repair:
                    repair_count += 1
                if repair_reconstructed_holistic_ranking(repair):
                    errors.append("holistic_ranking_reconstructed_from_dimension_ratings")
                    continue
                try:
                    ranks, mode = canonical_model_ranking(row)
                except ValueError as exc:
                    errors.append(f"ranking_parse_failure:{exc}")
                    continue
                valid.append(ranks)
                mapping_modes.append(mode)
            if valid:
                aggregated[task_id] = average_rankings(valid)
            compatible = task_id not in exclusions
            image_shas = sorted({str(r.get("image_sha256", "")) for r in records if r.get("image_sha256")})
            human_image = asset_manifest.get("image_sha256", {}).get(f"images/{task_id}.jpg", "")
            if task_id in questionnaire_tasks and image_shas and human_image and image_shas != [human_image]:
                errors.append("human_model_image_hash_mismatch")
            status = "valid" if valid else ("missing" if not records else "invalid")
            task_audit[task_id] = {
                "record_count": len(records), "valid_count": len(valid), "parse_failures": parse_failures,
                "semantic_failures": semantic_failures, "repair_count": repair_count, "errors": errors,
            }
            coverage.append(coverage_row(
                row_type="model_task", entity_type="model", entity_id=manifest.get("model_name", ""),
                run_id=run_id, model_name=manifest.get("model_name", ""), exact_model_version=exact_model_version(manifest),
                task_id=task_id, selected_for_main=compatible and task_id in version_compatible_human_tasks and bool(valid),
                in_human_questionnaire=task_id in questionnaire_tasks, version_compatible=compatible,
                record_count=len(records), valid_count=len(valid), missing_count=int(not records),
                duplicate_count=max(0, len(records) - 1), parse_failures=parse_failures,
                semantic_failures=semantic_failures, repair_count=repair_count,
                option_order=",".join(sorted({str(r.get('option_order', '')) for r in records})),
                mapping_status=",".join(sorted(set(mapping_modes))) or "unresolved",
                dataset_revision=manifest.get("dataset_revision", ""), prompt_version=manifest.get("prompt_version", ""),
                prompt_sha256=manifest.get("prompt_sha256", ""), image_sha256=",".join(image_shas),
                status=status, issue_codes=";".join(dict.fromkeys(errors + (["version_incompatible"] if not compatible else []))),
                detail=exclusions.get(task_id, ""),
            ))
        eligible_for_main = set(version_compatible_human_tasks) <= set(aggregated)
        model_runs.append({
            "run_id": run_id, "manifest": manifest, "rankings": aggregated,
            "audit": task_audit, "eligible_for_main": eligible_for_main,
        })
        for audit_row in coverage:
            if audit_row.get("row_type") == "model_task" and audit_row.get("run_id") == run_id:
                audit_row["selected_for_main"] = bool(audit_row["selected_for_main"] and eligible_for_main)
        version_unresolved = not manifest.get("model_revision") and sorted(set(manifest.get("returned_model_ids") or [])) in ([], [manifest.get("model_name")])
        coverage.append(coverage_row(
            row_type="model_summary", entity_type="model", entity_id=manifest.get("model_name", ""),
            run_id=run_id, model_name=manifest.get("model_name", ""), exact_model_version=exact_model_version(manifest),
            selected_for_main=eligible_for_main, version_compatible=True, record_count=len(result_rows), valid_count=len(aggregated),
            missing_count=len(expected_model_tasks - set(aggregated)),
            duplicate_count=sum(max(0, len(v) - 1) for v in by_task.values()),
            parse_failures=sum(x["parse_failures"] for x in task_audit.values()) + malformed_lines,
            semantic_failures=sum(x["semantic_failures"] for x in task_audit.values()),
            repair_count=sum(x["repair_count"] for x in task_audit.values()), option_order=manifest.get("option_order", ""),
            mapping_status="audited_per_record", dataset_revision=manifest.get("dataset_revision", ""),
            prompt_version=manifest.get("prompt_version", ""), prompt_sha256=manifest.get("prompt_sha256", ""),
            status=manifest.get("status", ""),
            issue_codes=";".join(filter(None, [
                "" if eligible_for_main else "missing_or_invalid_explicit_holistic_human_overlap",
                "exact_model_version_unresolved" if version_unresolved else "",
            ])),
            detail=f"requested_evaluations={manifest.get('requested_evaluations','')}; failures={manifest.get('failures','')}",
        ))

    # Compare the task text/strategy bundle actually referenced by selected run
    # manifests with the current Human task bundle.  This makes same-revision
    # hotfixes observable rather than trusting the revision string alone.
    snapshot_manifests: dict[str, tuple[Path, dict[str, dict[str, Any]]]] = {}
    for run in model_runs:
        snapshot_dir = Path(str(run["manifest"].get("dataset_local_snapshot", "")))
        metadata_path = snapshot_dir / "metadata.jsonl"
        if not metadata_path.is_file():
            coverage.append(coverage_row(
                row_type="model_snapshot_summary", entity_type="dataset_snapshot", entity_id=str(snapshot_dir),
                run_id=run["run_id"], selected_for_main=False, status="missing",
                issue_codes="snapshot_metadata_missing", detail=str(metadata_path),
            ))
            continue
        metadata_sha = sha256_file(metadata_path)
        if metadata_sha not in snapshot_manifests:
            rows = [r for r in load_jsonl(metadata_path) if not r.get("_json_error")]
            snapshot_manifests[metadata_sha] = (metadata_path, {r["task_id"]: r for r in rows})
    for metadata_sha, (metadata_path, snapshot_tasks) in snapshot_manifests.items():
        coverage.append(coverage_row(
            row_type="model_snapshot_summary", entity_type="dataset_snapshot", entity_id=metadata_sha,
            selected_for_main=True, dataset_revision=model_spec["dataset_revision"], status="audited",
            detail=str(metadata_path),
        ))
        for task_id in sorted(formal_task_ids):
            old = snapshot_tasks.get(task_id)
            current = tasks_by_id.get(task_id)
            issues = []
            if old is None:
                issues.append("model_snapshot_task_missing")
            if current is None:
                issues.append("human_task_manifest_task_missing")
            instruction_match = bool(old and current and old.get("task_instruction") == current.get("task_instruction"))
            option_match = bool(old and current and all(old.get(f"option_{s}") == current.get(f"option_{s}") for s in STRATEGIES))
            if old and current and not instruction_match:
                issues.append("human_model_instruction_mismatch")
            if old and current and not option_match:
                issues.append("human_model_strategy_text_mismatch")
            coverage.append(coverage_row(
                row_type="task_bundle_comparison", entity_type="dataset_snapshot", entity_id=metadata_sha,
                task_id=task_id, selected_for_main=task_id in version_compatible_human_tasks,
                in_human_questionnaire=True, version_compatible=task_id not in exclusions,
                record_count=int(old is not None), valid_count=int(old is not None and current is not None and instruction_match and option_match),
                missing_count=int(old is None or current is None), dataset_revision=model_spec["dataset_revision"],
                status="match" if not issues else "mismatch", issue_codes=";".join(issues),
                detail=f"instruction_match={str(instruction_match).lower()}; strategy_text_match={str(option_match).lower()}; metadata={metadata_path}",
            ))

    # Inventory every other discovered run without mixing it into analysis.
    selected_paths = {(selected_run_root / rid / "manifest.json").resolve() for rid in selected_run_ids}
    inventory_roots = [root / "CreativeContactBench/evaluation/runs", root / "creativecontactbench_eval/runs"]
    for inventory_root in inventory_roots:
        if not inventory_root.exists():
            continue
        for manifest_path in sorted(inventory_root.glob("*/manifest.json")):
            if manifest_path.resolve() in selected_paths:
                continue
            try:
                manifest = load_json(manifest_path)
            except Exception as exc:
                coverage.append(coverage_row(
                    row_type="run_inventory", entity_type="model_run", entity_id=manifest_path.parent.name,
                    run_id=manifest_path.parent.name, selected_for_main=False, status="unreadable_manifest",
                    issue_codes="manifest_parse_failure", detail=str(exc),
                ))
                continue
            same_dataset = manifest.get("dataset_revision") == model_spec["dataset_revision"]
            same_prompt = manifest.get("prompt_version") == model_spec["selected_run_prompt_version"] and manifest.get("prompt_sha256") == model_spec["selected_run_prompt_sha256"]
            reasons = ["not_in_frozen_selection_manifest"]
            if not same_dataset:
                reasons.append("dataset_revision_mismatch")
            if not same_prompt:
                reasons.append("prompt_mismatch")
            if manifest.get("status") != "completed":
                reasons.append("run_not_completed")
            coverage.append(coverage_row(
                row_type="run_inventory", entity_type="model_run", entity_id=manifest.get("model_name", manifest_path.parent.name),
                run_id=manifest.get("run_id", manifest_path.parent.name), model_name=manifest.get("model_name", ""),
                exact_model_version=exact_model_version(manifest), selected_for_main=False,
                record_count=manifest.get("requested_evaluations", ""), valid_count=manifest.get("semantically_valid", ""),
                parse_failures=(manifest.get("requested_evaluations", 0) or 0) - (manifest.get("valid_json", 0) or 0),
                semantic_failures=(manifest.get("valid_json", 0) or 0) - (manifest.get("semantically_valid", 0) or 0),
                repair_count=manifest.get("response_repairs", ""), option_order=manifest.get("option_order", ""),
                dataset_revision=manifest.get("dataset_revision", ""), prompt_version=manifest.get("prompt_version", ""),
                prompt_sha256=manifest.get("prompt_sha256", ""), status=manifest.get("status", ""),
                issue_codes=";".join(reasons), detail=str(manifest_path.relative_to(root)),
            ))

    main_model_runs = [run for run in model_runs if run["eligible_for_main"]]
    partial_model_runs = [run for run in model_runs if not run["eligible_for_main"]]
    common_tasks = set(version_compatible_human_tasks)
    for run in main_model_runs:
        common_tasks &= set(run["rankings"])
    common_tasks = set(sorted(common_tasks))
    if not common_tasks:
        raise RuntimeError("No jointly valid, version-compatible Human--model tasks")
    source_groups = {task_id: task_id for task_id in common_tasks}

    per_task_metric_rows: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    for model_index, run in enumerate(main_model_runs):
        manifest = run["manifest"]
        task_metrics: dict[str, dict[str, Any]] = {}
        for task_id in sorted(common_tasks):
            human_ref = human_references[task_id]
            model_rank = run["rankings"][task_id]
            htop, mtop = top_group(human_ref), top_group(model_rank)
            eligible = len(htop) == 1
            hit = eligible and bool(htop & mtop)
            strict_hit = hit and len(mtop) == 1
            tau = kendall_tau_b(model_rank, human_ref)
            task_metrics[task_id] = {
                "tau_b": tau, "top1_eligible": eligible, "top1_hit": hit,
                "strict_top1_hit": strict_hit,
            }
            audit = run["audit"][task_id]
            per_task_metric_rows.append({
                "entity_type": "model", "entity_id": manifest.get("model_name", ""), "run_id": run["run_id"],
                "task_id": task_id, "source_group": source_groups[task_id],
                "human_reference_ranking": ranking_text(human_ref), "entity_ranking": ranking_text(model_rank),
                "tau_b": tau, "tau_defined": tau is not None, "human_top_unique": eligible,
                "human_top": "=".join(sorted(htop)), "entity_top": "=".join(sorted(mtop)),
                "top1_hit": hit if eligible else None, "strict_top1_hit": strict_hit if eligible else None,
                "within_task_valid_records": audit["valid_count"], "repair_count": audit["repair_count"],
            })
        taus = [r["tau_b"] for r in task_metrics.values() if r["tau_b"] is not None]
        eligible = [r for r in task_metrics.values() if r["top1_eligible"]]
        ci = bootstrap_model(task_metrics, source_groups, config["uncertainty"]["resamples"], config["uncertainty"]["seed"] + model_index)
        human_overlap = set(questionnaire_tasks)
        missing_invalid = len(human_overlap - set(run["rankings"]))
        row = {
            "model_name": manifest.get("model_name", ""), "exact_model_version": exact_model_version(manifest),
            "run_id": run["run_id"], "mean_task_tau_b": statistics.fmean(taus),
            **ci, "n_common_tasks": len(common_tasks), "tau_defined_tasks": len(taus),
            "tau_undefined_tasks": len(common_tasks) - len(taus), "human_unique_top_tasks": len(eligible),
            "top1_hits": sum(r["top1_hit"] for r in eligible),
            "top1_agreement": statistics.fmean(float(r["top1_hit"]) for r in eligible) if eligible else None,
            "strict_top1_hits": sum(r["strict_top1_hit"] for r in eligible),
            "strict_top1_agreement": statistics.fmean(float(r["strict_top1_hit"]) for r in eligible) if eligible else None,
            "model_tied_top_tasks": sum(len(top_group(run["rankings"][t])) > 1 for t in common_tasks),
            "within_task_records_min": min(run["audit"][t]["valid_count"] for t in common_tasks),
            "within_task_records_max": max(run["audit"][t]["valid_count"] for t in common_tasks),
            "repairs_on_common_tasks": sum(run["audit"][t]["repair_count"] for t in common_tasks),
            "missing_or_invalid_human_tasks": missing_invalid,
            "extra_valid_nonquestionnaire_tasks": len(set(run["rankings"]) - questionnaire_tasks),
        }
        main_rows.append(row)
    main_rows.sort(key=lambda r: (-r["mean_task_tau_b"], -r["top1_agreement"], r["model_name"]))
    for rank, row in enumerate(main_rows, 1):
        row["rank_by_mean_tau_b"] = rank

    # Runs whose explicit holistic ordering is unavailable on one or more main
    # tasks are kept as non-comparable additional coverage, never forced into
    # the common-set leaderboard.
    additional_rows: list[dict[str, Any]] = []
    for model_index, run in enumerate(partial_model_runs):
        manifest = run["manifest"]
        available_tasks = sorted(version_compatible_human_tasks & set(run["rankings"]))
        task_metrics: dict[str, dict[str, Any]] = {}
        for task_id in available_tasks:
            human_ref = human_references[task_id]
            model_rank = run["rankings"][task_id]
            htop, mtop = top_group(human_ref), top_group(model_rank)
            eligible = len(htop) == 1
            metric = {
                "tau_b": kendall_tau_b(model_rank, human_ref), "top1_eligible": eligible,
                "top1_hit": eligible and bool(htop & mtop),
                "strict_top1_hit": eligible and bool(htop & mtop) and len(mtop) == 1,
            }
            task_metrics[task_id] = metric
            audit = run["audit"][task_id]
            per_task_metric_rows.append({
                "entity_type": "model_additional_coverage", "entity_id": manifest.get("model_name", ""),
                "run_id": run["run_id"], "task_id": task_id, "source_group": task_id,
                "human_reference_ranking": ranking_text(human_ref), "entity_ranking": ranking_text(model_rank),
                "tau_b": metric["tau_b"], "tau_defined": metric["tau_b"] is not None,
                "human_top_unique": eligible, "human_top": "=".join(sorted(htop)),
                "entity_top": "=".join(sorted(mtop)), "top1_hit": metric["top1_hit"] if eligible else None,
                "strict_top1_hit": metric["strict_top1_hit"] if eligible else None,
                "within_task_valid_records": audit["valid_count"], "repair_count": audit["repair_count"],
            })
        taus = [x["tau_b"] for x in task_metrics.values() if x["tau_b"] is not None]
        eligible_metrics = [x for x in task_metrics.values() if x["top1_eligible"]]
        local_groups = {task_id: task_id for task_id in available_tasks}
        ci = bootstrap_model(task_metrics, local_groups, config["uncertainty"]["resamples"], config["uncertainty"]["seed"] + 500 + model_index) if task_metrics else {
            "tau_ci_low": None, "tau_ci_high": None, "top1_ci_low": None, "top1_ci_high": None,
            "strict_top1_ci_low": None, "strict_top1_ci_high": None,
        }
        invalid_tasks = sorted(version_compatible_human_tasks - set(run["rankings"]))
        additional_rows.append({
            "model_name": manifest.get("model_name", ""), "exact_model_version": exact_model_version(manifest),
            "run_id": run["run_id"], "mean_task_tau_b": statistics.fmean(taus) if taus else None,
            **ci, "n_available_compatible_tasks": len(available_tasks), "tau_defined_tasks": len(taus),
            "tau_undefined_tasks": len(available_tasks) - len(taus),
            "human_unique_top_tasks": len(eligible_metrics),
            "top1_hits": sum(x["top1_hit"] for x in eligible_metrics),
            "top1_agreement": statistics.fmean(float(x["top1_hit"]) for x in eligible_metrics) if eligible_metrics else None,
            "strict_top1_hits": sum(x["strict_top1_hit"] for x in eligible_metrics),
            "strict_top1_agreement": statistics.fmean(float(x["strict_top1_hit"]) for x in eligible_metrics) if eligible_metrics else None,
            "missing_or_invalid_explicit_holistic_tasks": len(invalid_tasks),
            "affected_task_ids": ";".join(invalid_tasks),
            "comparison_status": "additional_coverage_only_not_main_common_set",
        })
    additional_rows.sort(key=lambda r: (-(r["mean_task_tau_b"] if r["mean_task_tau_b"] is not None else -2), r["model_name"]))

    # Leave-one-participant-out Human agreement; own observation is never in its reference.
    human_agreement_rows: list[dict[str, Any]] = []
    human_loo_per_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for participant_index, participant in enumerate(sorted(human_rankings)):
        metrics = []
        for task_id in sorted(common_tasks):
            others = [human_rankings[p][task_id] for p in sorted(human_rankings) if p != participant]
            loo_ref = average_rankings(others)
            own = human_rankings[participant][task_id]
            tau = kendall_tau_b(own, loo_ref)
            ref_top, own_top = top_group(loo_ref), top_group(own)
            eligible = len(ref_top) == 1
            metric = {
                "participant_id": participant, "task_id": task_id, "tau_b": tau,
                "top1_eligible": eligible, "top1_hit": eligible and bool(ref_top & own_top),
                "strict_top1_hit": eligible and bool(ref_top & own_top) and len(own_top) == 1,
            }
            metrics.append(metric)
            human_loo_per_task[task_id].append(metric)
            per_task_metric_rows.append({
                "entity_type": "human_loo", "entity_id": participant, "run_id": "", "task_id": task_id,
                "source_group": source_groups[task_id], "human_reference_ranking": ranking_text(loo_ref),
                "entity_ranking": ranking_text(own), "tau_b": tau, "tau_defined": tau is not None,
                "human_top_unique": eligible, "human_top": "=".join(sorted(ref_top)),
                "entity_top": "=".join(sorted(own_top)), "top1_hit": metric["top1_hit"] if eligible else None,
                "strict_top1_hit": metric["strict_top1_hit"] if eligible else None,
                "within_task_valid_records": 1, "repair_count": 0,
            })
        taus = [m["tau_b"] for m in metrics if m["tau_b"] is not None]
        eligible = [m for m in metrics if m["top1_eligible"]]
        ci = bootstrap_model({m["task_id"]: m for m in metrics}, source_groups, config["uncertainty"]["resamples"], config["uncertainty"]["seed"] + 100 + participant_index)
        human_agreement_rows.append({
            "row_type": "participant", "participant_id": participant,
            "mean_task_tau_b": statistics.fmean(taus) if taus else None,
            "tau_ci_low": ci["tau_ci_low"], "tau_ci_high": ci["tau_ci_high"],
            "n_common_tasks": len(common_tasks), "tau_defined_tasks": len(taus),
            "tau_undefined_tasks": len(common_tasks) - len(taus), "loo_reference_participants": len(human_rankings) - 1,
            "human_unique_top_tasks": len(eligible), "top1_hits": sum(m["top1_hit"] for m in eligible),
            "top1_agreement": statistics.fmean(float(m["top1_hit"]) for m in eligible) if eligible else None,
            "top1_ci_low": ci["top1_ci_low"], "top1_ci_high": ci["top1_ci_high"],
            "tied_own_top_tasks": sum(len(top_group(human_rankings[participant][t])) > 1 for t in common_tasks),
            "interpretation": "descriptive leave-one-participant-out agreement; not an absolute Human ceiling",
        })
    all_human_metrics = [r for rows in human_loo_per_task.values() for r in rows]
    all_human_taus = [r["tau_b"] for r in all_human_metrics if r["tau_b"] is not None]
    all_human_eligible = [r for r in all_human_metrics if r["top1_eligible"]]
    human_ci = bootstrap_human_aggregate(human_loo_per_task, source_groups, config["uncertainty"]["resamples"], config["uncertainty"]["seed"] + 999)
    human_aggregate = {
        "row_type": "aggregate_task_clustered", "participant_id": "ALL_LOO",
        "mean_task_tau_b": statistics.fmean(all_human_taus), **human_ci,
        "n_common_tasks": len(common_tasks), "tau_defined_tasks": len(all_human_taus),
        "tau_undefined_tasks": len(all_human_metrics) - len(all_human_taus),
        "loo_reference_participants": len(human_rankings) - 1,
        "human_unique_top_tasks": len(all_human_eligible), "top1_hits": sum(r["top1_hit"] for r in all_human_eligible),
        "top1_agreement": statistics.fmean(float(r["top1_hit"]) for r in all_human_eligible),
        "tied_own_top_tasks": sum(len(top_group(human_rankings[r["participant_id"]][r["task_id"]])) > 1 for r in all_human_metrics),
        "interpretation": "descriptive leave-one-participant-out agreement; task-clustered CI; not an absolute Human ceiling",
    }
    human_agreement_rows.append(human_aggregate)

    main_fields = [
        "rank_by_mean_tau_b", "model_name", "exact_model_version", "run_id", "mean_task_tau_b",
        "tau_ci_low", "tau_ci_high", "n_common_tasks", "tau_defined_tasks", "tau_undefined_tasks",
        "human_unique_top_tasks", "top1_hits", "top1_agreement", "top1_ci_low", "top1_ci_high",
        "strict_top1_hits", "strict_top1_agreement", "strict_top1_ci_low", "strict_top1_ci_high",
        "model_tied_top_tasks", "within_task_records_min", "within_task_records_max",
        "repairs_on_common_tasks", "missing_or_invalid_human_tasks", "extra_valid_nonquestionnaire_tasks",
    ]
    human_fields = [
        "row_type", "participant_id", "mean_task_tau_b", "tau_ci_low", "tau_ci_high", "n_common_tasks",
        "tau_defined_tasks", "tau_undefined_tasks", "loo_reference_participants", "human_unique_top_tasks",
        "top1_hits", "top1_agreement", "top1_ci_low", "top1_ci_high", "tied_own_top_tasks", "interpretation",
    ]
    per_task_fields = [
        "entity_type", "entity_id", "run_id", "task_id", "source_group", "human_reference_ranking",
        "entity_ranking", "tau_b", "tau_defined", "human_top_unique", "human_top", "entity_top",
        "top1_hit", "strict_top1_hit", "within_task_valid_records", "repair_count",
    ]
    write_csv(output / "coverage.csv", COVERAGE_FIELDS, coverage)
    write_csv(output / "main_holistic_results.csv", main_fields, main_rows)
    additional_fields = [
        "model_name", "exact_model_version", "run_id", "mean_task_tau_b", "tau_ci_low", "tau_ci_high",
        "n_available_compatible_tasks", "tau_defined_tasks", "tau_undefined_tasks", "human_unique_top_tasks",
        "top1_hits", "top1_agreement", "top1_ci_low", "top1_ci_high", "strict_top1_hits",
        "strict_top1_agreement", "strict_top1_ci_low", "strict_top1_ci_high",
        "missing_or_invalid_explicit_holistic_tasks", "affected_task_ids", "comparison_status",
    ]
    write_csv(output / "additional_holistic_results.csv", additional_fields, additional_rows)
    write_csv(output / "human_agreement.csv", human_fields, human_agreement_rows)
    write_csv(output / "per_task_holistic_metrics.csv", per_task_fields, per_task_metric_rows)

    # Publication table.
    latex = [
        r"% Generated by holistic_alignment_baseline.py; do not edit manually.",
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Holistic-ranking agreement with the ten-participant mean-rank reference on the 27-task version-compatible common set. CIs are task-clustered percentile-bootstrap intervals.}",
        r"\label{tab:holistic-human-model-agreement}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Model & $\tau_b$ $\uparrow$ & 95\% CI & Top-1 $\uparrow$ & $n$ \\", r"\midrule",
    ]
    for row in main_rows:
        latex.append(
            f"{latex_escape(short_model_name(row['model_name']))} & {row['mean_task_tau_b']:.3f} & "
            f"[{row['tau_ci_low']:.3f}, {row['tau_ci_high']:.3f}] & "
            f"{100*row['top1_agreement']:.1f}\\% & {row['n_common_tasks']} \\\\"
        )
    latex += [r"\midrule", r"\multicolumn{5}{l}{\footnotesize Human LOO is reported separately in \texttt{human\_agreement.csv}; it is not an absolute ceiling.} \\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "main_holistic_results.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")

    draw_plot(output / "holistic_agreement.png", output / "holistic_agreement.svg", main_rows, human_aggregate)

    best = main_rows[0]
    top_human_unique = best["human_unique_top_tasks"]
    all_model_tau_undefined = sum(r["tau_undefined_tasks"] for r in main_rows)
    selected_repair_total = sum(r["repairs_on_common_tasks"] for r in main_rows)
    human_task_ties = sum(len(top_group(human_references[t])) > 1 for t in common_tasks)
    summary = f"""# Holistic ranking Human--model agreement baseline

Generated from the frozen configuration [`holistic_alignment_config_v1.json`](../../holistic_alignment_config_v1.json). This round analyzes only the explicit holistic ranking; it does not replace that ordering with a sum or average of effectiveness, feasibility, and creativity ratings.

## Actual findings

- The formal Human export contains **{len(human_rankings)} participants**, each with valid explicit holistic rankings on **{len(questionnaire_tasks)} questionnaire tasks**.
- The frozen model selection contains **{len(model_runs)} completed model runs**. **{len(main_model_runs)}** have valid explicit holistic rankings on every compatible Human task and enter the common-set leaderboard; **{len(partial_model_runs)}** are reported separately because some stored rankings were reconstructed from dimension ratings rather than explicitly supplied by the model.
- The strict common, version-compatible main set contains **{len(common_tasks)} tasks**. Tasks 06, 10, and 13 are excluded for the documented image/instruction version ambiguity, but remain visible in `coverage.csv`.
- The highest mean per-task Kendall tau-b is **{best['model_name']}** at **{best['mean_task_tau_b']:.3f}** (95% task-bootstrap CI **[{best['tau_ci_low']:.3f}, {best['tau_ci_high']:.3f}]**). Its inclusive Top-1 agreement is **{100*best['top1_agreement']:.1f}%** ({best['top1_hits']}/{top_human_unique} unique-reference tasks).
- Across the common set, the ten-person reference has a tied first place on **{human_task_ties}/{len(common_tasks)} tasks**. The primary Top-1 denominator is therefore **{top_human_unique}**; tied-reference tasks, if any, are excluded rather than scored.
- The task-clustered aggregate Human leave-one-participant-out agreement is **tau-b {human_aggregate['mean_task_tau_b']:.3f}** (95% CI **[{human_aggregate['tau_ci_low']:.3f}, {human_aggregate['tau_ci_high']:.3f}]**) and Top-1 **{100*human_aggregate['top1_agreement']:.1f}%**. Each participant is compared with a reference formed from the other nine participants only.
- There are **{all_model_tau_undefined}** undefined model-task tau-b values on the main set and **{selected_repair_total}** recorded repaired responses on that set. Undefined values are reported and never replaced by zero or one.

## Coverage and quality controls

`coverage.csv` contains participant/task rows, selected model/task rows, selected-run summaries, explicit version issues, analysis limitations, and an inventory of other discovered run manifests. Missing, duplicate, parse-invalid, semantic-invalid, and repaired records have separate fields. `per_task_holistic_metrics.csv` preserves the task-level metric inputs and outputs.

All 16 selected runs use dataset revision `{model_spec['dataset_revision']}`, model prompt {model_spec['selected_run_prompt_version']} (`{model_spec['selected_run_prompt_sha256']}`), canonical A/B/C/D presentation, and one record per task unless `coverage.csv` states otherwise. Exact model names, revisions, and returned model IDs are recorded in the main table. Older pilot/incompatible runs are inventoried but not pooled.

## Limitations and concrete data to collect

1. **Unversioned hotfixes under one dataset revision.** Task 06's Human image differs from the model-run image. Tasks 10 and 13 changed instruction text after the first Human submission. Per-response viewed asset/instruction hashes were not recorded, so those tasks cannot be repaired retrospectively without assumptions. Future exports should store image SHA-256, exact instruction SHA-256, option-text SHA-256, and task-bundle version per response.
2. **Prompt generation mismatch with the repository's current standard.** The available complete selected runs use prompt {model_spec['selected_run_prompt_version']}; the repository now marks {model_spec['repository_current_standard_prompt_version']} as current. This baseline truthfully conditions on the existing v1.2 results; it does not make new API calls.
3. **No source-scene grouping metadata.** The task manifest contains none of `{', '.join(task_spec['source_group_fields_in_manifest_priority'])}`. Therefore the requested source-group bootstrap falls back to task-ID resampling. Add a stable source-scene ID before interpreting intervals as robust to correlated scene variants.
4. **Conditional Human reference.** Model intervals resample tasks while conditioning on the ten observed participants. They do not reflect participant-sampling uncertainty. The Human LOO row uses nine-person references, so it is descriptive agreement—not an absolute ceiling and not evidence by itself that a model has reached Human level.
5. **Top-1 denominator.** Top-1 excludes Human-reference ties by protocol. Consult the explicit numerator and denominator in `main_holistic_results.csv`, not only the percentage.
6. **One unresolved exact served-model version.** The `gpt-5.6-sol` run records only that alias as both requested and returned model ID and has no immutable model revision. It is retained as a run-identified result, but its exact underlying model snapshot cannot be recovered from the manifest.

## Reproduction

```bash
python human_eval_analysis/holistic_alignment_baseline.py \\
  --config human_eval_analysis/holistic_alignment_config_v1.json \\
  --output human_eval_analysis/results/holistic_alignment_baseline_v1
python -m unittest human_eval_analysis.test_holistic_alignment_baseline
```
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")

    median_tau = statistics.median(r["mean_task_tau_b"] for r in main_rows)
    paper = f"""% Generated from the frozen holistic-ranking baseline configuration and observed results.
% Intended for selective inclusion; root.tex is intentionally untouched.

\\paragraph{{Models and Evaluation Protocol.}}
We audited {len(model_runs)} VLM runs on the CreativeContactBench dataset revision \\texttt{{{model_spec['dataset_revision'][:12]}}}; {len(main_model_runs)} runs contain explicit holistic orderings on every compatible Human task and enter the main leaderboard, while {len(partial_model_runs)} runs are reported only as additional coverage. The available complete runs used the frozen evaluator prompt {model_spec['selected_run_prompt_version']} (SHA-256 \\texttt{{{model_spec['selected_run_prompt_sha256'][:12]}}}) and presented strategies in canonical A--D order. We use each model's explicit holistic ordering, rather than deriving an ordering from its three dimension ratings. For repeated or permuted observations of the same model--task pair, we first map displayed labels to canonical strategy identifiers, convert the grouped ordering to dense ranks, average ranks within task, and retain exact ties. The main comparison is the intersection of valid model results and the formal Human questionnaire after version checks ({len(common_tasks)} tasks). For each task we compute Kendall's $\\tau_b$ over the four strategies and then average across tasks. Top-1 agreement is evaluated only when the ten-participant Human mean-rank reference has a unique winner. We report percentile-bootstrap 95\\% intervals from {config['uncertainty']['resamples']:,} task-cluster resamples; because no source-scene identifier is available, the cluster is the task ID.

\\paragraph{{Overall Benchmark Performance.}}
Across the {len(main_model_runs)} common-set model runs, mean task-level $\\tau_b$ ranges from {min(r['mean_task_tau_b'] for r in main_rows):.3f} to {best['mean_task_tau_b']:.3f} (median {median_tau:.3f}). The strongest observed model by this frozen metric is {latex_escape(short_model_name(best['model_name']))}, with mean $\\tau_b={best['mean_task_tau_b']:.3f}$ (95\\% CI [{best['tau_ci_low']:.3f}, {best['tau_ci_high']:.3f}]) and inclusive Top-1 agreement of {100*best['top1_agreement']:.1f}\\% ({best['top1_hits']}/{best['human_unique_top_tasks']}). Table~\\ref{{tab:holistic-human-model-agreement}} reports common-set models, explicit denominators, tied predictions, and undefined task-level coefficients; partial-coverage runs are not ranked with them. These results apply to prompt {model_spec['selected_run_prompt_version']}; the repository's current prompt standard is {model_spec['repository_current_standard_prompt_version']}, for which no new runs were generated in this analysis.

\\paragraph{{Human Study and Ranking Agreement.}}
The formal study contains {len(human_rankings)} participants with explicit holistic rankings on {len(questionnaire_tasks)} tasks. For each strategy and task, the Human reference is its mean dense rank across participants; equal mean ranks remain tied. Tasks 06, 10, and 13 are excluded from the main agreement analysis because an image or instruction was hotfixed under the same dataset revision and the export does not record the exact per-response asset/text hashes. On the remaining {len(common_tasks)} tasks, leave-one-participant-out agreement compares each participant only with the mean-rank reference of the other nine participants. Aggregated descriptively with task-clustered uncertainty, Human LOO agreement is $\\tau_b={human_aggregate['mean_task_tau_b']:.3f}$ (95\\% CI [{human_aggregate['tau_ci_low']:.3f}, {human_aggregate['tau_ci_high']:.3f}]) and Top-1 agreement is {100*human_aggregate['top1_agreement']:.1f}\\%. This quantity is not an absolute Human ceiling, and its nine-person references differ from the ten-person reference used for the model table; we therefore do not use it alone to claim Human-level model performance. All intervals condition on the ten observed participants.
"""
    (output / "paper_snippets.tex").write_text(paper, encoding="utf-8")

    manifest_out = {
        "analysis_id": config["analysis_id"], "config_path": str(config_path.relative_to(root)),
        "config_sha256": sha256_file(config_path), "output_files": {},
        "frozen_inputs": {
            human_spec["export_path"]: human_spec["export_sha256"],
            human_spec["protocol_path"]: human_spec["protocol_sha256"],
            task_spec["human_task_manifest_path"]: task_spec["human_task_manifest_sha256"],
            task_spec["human_asset_manifest_path"]: task_spec["human_asset_manifest_sha256"],
            model_spec["selection_manifest_path"]: model_spec["selection_manifest_sha256"],
        },
        "human_participants": len(human_rankings), "human_questionnaire_tasks": len(questionnaire_tasks),
        "audited_selected_model_runs": len(model_runs), "main_leaderboard_models": len(main_model_runs),
        "additional_coverage_models": len(partial_model_runs), "main_common_tasks": sorted(common_tasks),
        "excluded_version_tasks": exclusions, "bootstrap": config["uncertainty"],
    }
    for p in sorted(output.iterdir()):
        if p.name != "analysis_manifest.json" and p.is_file():
            manifest_out["output_files"][p.name] = sha256_file(p)
    (output / "analysis_manifest.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "participants": len(human_rankings), "models": len(model_runs),
        "questionnaire_tasks": len(questionnaire_tasks), "common_tasks": len(common_tasks),
        "best_model": best["model_name"], "best_tau_b": round(best["mean_task_tau_b"], 6),
    }, indent=2))


if __name__ == "__main__":
    main()
