"""Analyze local responses against a separately supplied private human reference."""
from __future__ import annotations

import argparse
from fractions import Fraction
import itertools
import math
from pathlib import Path
import random

from ccbench_eval.response import ResponseValidationError
from evaluate import CONDITIONS, parse_arm, read, sha, verify, write


def dense(groups):
    if not isinstance(groups, list) or not groups:
        raise ValueError("Expected nonempty rank groups")
    result = {}
    for rank, group in enumerate(groups, 1):
        if not isinstance(group, list) or not group:
            raise ValueError("Expected nonempty rank group")
        for option in group:
            if not isinstance(option, str) or option not in "ABCD" or len(option) != 1 or option in result:
                raise ValueError("Invalid or duplicated canonical option")
            result[option] = Fraction(rank)
    if set(result) != set("ABCD"):
        raise ValueError("Every option must appear exactly once")
    return result


def top(ranks):
    return {k for k, v in ranks.items() if v == min(ranks.values())}


def tau_b(x, y):
    concordant = discordant = tx = ty = 0
    for a, b in itertools.combinations("ABCD", 2):
        dx = (x[a] > x[b]) - (x[a] < x[b])
        dy = (y[a] > y[b]) - (y[a] < y[b])
        if dx == dy == 0:
            continue
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx == dy:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt((concordant + discordant + tx) * (concordant + discordant + ty))
    return (concordant - discordant) / denominator if denominator else None


def references(rankings):
    if not rankings:
        raise ValueError("Cannot aggregate zero human rankings")
    ranks = [dense(r) for r in rankings]
    mean = {o: sum((r[o] for r in ranks), Fraction()) / len(ranks) for o in "ABCD"}
    votes = {o: Fraction() for o in "ABCD"}
    for r in ranks:
        best = top(r)
        for o in best:
            votes[o] -= Fraction(1, len(best))
    return {"mean_dense": mean, "plurality": votes}


def metrics(predicted, refs):
    result = {}
    for prefix, human in refs.items():
        ht, pt = top(human), top(predicted)
        result[prefix + "_strict_top1"] = int(ht == pt and len(pt) == 1) if len(ht) == 1 else None
        result[prefix + "_inclusive_top1"] = int(bool(ht & pt)) if len(ht) == 1 else None
    result["mean_dense_tau_b"] = tau_b(predicted, refs["mean_dense"])
    return result


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize(rows, resamples=5000, seed=2026090905):
    if not rows:
        return {"task_count": 0, "family_count": 0, "conditions": {}, "contrasts": {}}
    if resamples < 1:
        raise ValueError("Resamples must be positive")
    families = sorted({r["family"] for r in rows})
    by_family = {f: [r for r in rows if r["family"] == f] for f in families}
    contrasts = {"primary": ("rgb_tactile", "rgb_only"),
                 "description_ablation": ("rgb_description", "rgb_only"),
                 "tactile_increment": ("rgb_tactile", "rgb_description")}
    means = {c: sum(r[c] for r in rows) / len(rows) for c in CONDITIONS}
    draws = {name: [] for name in contrasts}
    rng = random.Random(seed)
    for _ in range(resamples):
        selected = [r for f in rng.choices(families, k=len(families)) for r in by_family[f]]
        for name, (a, b) in contrasts.items():
            draws[name].append(sum(r[a] - r[b] for r in selected) / len(selected))
    return {"task_count": len(rows), "family_count": len(families), "conditions": means,
            "contrasts": {name: {"difference": means[a] - means[b],
                                    "ci95": [percentile(draws[name], .025), percentile(draws[name], .975)]}
                          for name, (a, b) in contrasts.items()}}


def analyze(bundle, run_dir, human_file, output, arm="strict", resamples=5000):
    manifest = verify(bundle)
    plan = read(run_dir / "plan.json")
    if plan["input_manifest_sha256"] != sha(bundle / "manifest.json"):
        raise ValueError("Run and inputs differ")
    if plan["repeats"] != 2:
        raise ValueError("The v5 primary protocol requires two declared repeats")
    human = read(human_file)
    if human.get("reference_kind") != "original_scene_human_preferences":
        raise ValueError("Declare the reference scope explicitly")
    if set(human["tasks"]) != {t["task_id"] for t in manifest["tasks"]}:
        raise ValueError("Supply all 16 task references; do not silently change the evaluation subset")
    schema = read(bundle / manifest["schema_path"])
    rows, coverage = [], []
    for task in manifest["tasks"]:
        tid = task["task_id"]
        ref = references(human["tasks"][tid])
        scores = {c: {} for c in CONDITIONS}
        for condition in CONDITIONS:
            for repeat in ("01", "02"):
                path = run_dir / "responses" / tid / condition / (repeat + ".json")
                status = "not_run"
                if path.exists():
                    record = read(path)
                    if any(record.get(k) != v for k, v in (("task_id", tid), ("condition", condition), ("repeat_id", repeat))):
                        raise ValueError("Response job identity mismatch")
                    entry = task["conditions"][condition]
                    if any(record.get(k + "_sha256") != entry[k + "_sha256"] for k in ("image", "prompt")):
                        raise ValueError("Response input hash mismatch")
                    status = record["status"]
                    if status == "received":
                        try:
                            response = parse_arm(record["raw_response"], schema, arm)
                            scores[condition][repeat] = metrics(dense(response["overall_ranking"]), ref)
                            status = "valid"
                        except ResponseValidationError:
                            status = "invalid_response"
                coverage.append(dict(task_id=tid, condition=condition, repeat_id=repeat, status=status))
        metric_names = metrics(dense([["A"], ["B"], ["C"], ["D"]]), ref).keys()
        for metric in metric_names:
            matched = [r for r in ("01", "02") if all(r in scores[c] and scores[c][r][metric] is not None for c in CONDITIONS)]
            for mode, repeats in (("primary_complete_two_repeats", matched if len(matched) == 2 else []),
                                  ("matched_repeat_sensitivity", matched)):
                if repeats:
                    rows.append(dict(task_id=tid, family=task["contact_group"], metric=metric, mode=mode,
                                     **{c: sum(scores[c][r][metric] for r in repeats) / len(repeats) for c in CONDITIONS}))
    summaries = {}
    for mode in ("primary_complete_two_repeats", "matched_repeat_sensitivity"):
        summaries[mode] = {metric: summarize([r for r in rows if r["mode"] == mode and r["metric"] == metric], resamples)
                           for metric in metric_names}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "summary.json", {"reference_kind": human["reference_kind"], "human_reference_sha256": sha(human_file),
                                    "input_manifest_sha256": sha(bundle / "manifest.json"), "run_plan_sha256": sha(run_dir / "plan.json"),
                                    "analysis_source_sha256": sha(Path(__file__)), "parsing_arm": arm,
                                    "bootstrap_resamples": resamples, "bootstrap_seed": 2026090905,
                                    "bootstrap_unit": "contact_family", "summaries": summaries})
    write(output / "coverage.json", coverage)
    write(output / "task_metrics.json", rows)
    print(f"Analysis written locally: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("strict", "fence_only"), default="strict")
    args = parser.parse_args()
    analyze(args.bundle, args.run, args.human, args.output, args.arm)
