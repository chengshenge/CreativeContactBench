"""Offline checks using synthetic inputs and rankings, never private references."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev
import analyze as an
from pressure_projection import project_snapshot
from ccbench_eval.adapters import OpenAIResponsesAdapter
from ccbench_eval.response import ResponseValidationError


def payload(groups=None):
    # Deliberately equal dimension scores with a non-tied holistic ranking.
    return {"ratings": {o: {k: 3 for k in ("expected_task_effectiveness", "embodied_feasibility", "functional_creativity")} for o in "ABCD"},
            "overall_ranking": groups or [[o] for o in "ABCD"], "confidence": 3}


def inventory(bundle, manifest):
    manifest["files"] = {p.relative_to(bundle).as_posix(): {"sha256": ev.sha(p), "bytes": p.stat().st_size}
                         for p in bundle.rglob("*") if p.is_file() and p.name != "manifest.json"}
    ev.write(bundle / "manifest.json", manifest)


def fixture(bundle):
    bundle.mkdir()
    white = Image.new("RGB", (2048, 1024), "white")
    tactile = white.copy()
    tactile.paste("black", (1024, 0, 2048, 1024))
    rgb_hash = hashlib.sha256(white.crop((0, 0, 1024, 1024)).tobytes()).hexdigest()
    white.save(bundle / "baseline.png")
    tactile.save(bundle / "tactile.png")
    (bundle / "prompt.txt").write_text("SYNTHETIC offline test input")
    schema_path = "schema.json"
    ev.write(bundle / schema_path, ev.read(ev.HERE / "protocol/vlm_response_v1.0.schema.json"))
    families = {"task-10": "task-01", "task-37": "task-24", "task-27": "task-25", "task-32": "task-25"}
    tasks = []
    for tid in ev.TASK_IDS:
        entries = {}
        for c in ev.CONDITIONS:
            image = "tactile.png" if c == "rgb_tactile" else "baseline.png"
            entries[c] = {"prompt_path": "prompt.txt", "prompt_sha256": ev.sha(bundle / "prompt.txt"),
                          "image_path": image, "image_sha256": ev.sha(bundle / image)}
        tasks.append(dict(task_id=tid, contact_group=families.get(tid, tid), conditions=entries,
                          displayed_to_canonical=dict(zip("ABCD", "ABCD")), shared_rgb_pixels_sha256=rgb_hash))
    manifest = dict(format_version="ccb-multimodal-inputs-v5", task_count=16, contact_family_count=12,
                    conditions=list(ev.CONDITIONS), canvas_size=[2048, 1024], rgb_region_xyxy=[0, 0, 1024, 1024],
                    tasks=tasks, schema_path=schema_path)
    inventory(bundle, manifest)
    return manifest


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.manifest = fixture(self.bundle)
        self.schema = ev.read(self.bundle / "schema.json")

    def test_valid_bundle_and_schedule(self):
        ev.verify(self.bundle, ev.sha(self.bundle / "manifest.json"))
        jobs = ev.jobs_for(self.manifest)
        self.assertEqual(len(jobs), 96)
        self.assertEqual(len({tuple(j.values()) for j in jobs}), 96)
        self.assertEqual(jobs, ev.jobs_for(self.manifest))
        for tid in ev.TASK_IDS:
            self.assertEqual(sum(j["task_id"] == tid for j in jobs), 6)

    def test_rejects_modified_prompt(self):
        (self.bundle / "prompt.txt").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Modified input"):
            ev.verify(self.bundle)

    def test_rejects_rehashed_changed_rgb(self):
        im = Image.open(self.bundle / "tactile.png").convert("RGB")
        im.putpixel((0, 0), (0, 0, 0))
        im.save(self.bundle / "tactile.png")
        for task in self.manifest["tasks"]:
            task["conditions"]["rgb_tactile"]["image_sha256"] = ev.sha(self.bundle / "tactile.png")
        inventory(self.bundle, self.manifest)
        with self.assertRaisesRegex(ValueError, "same RGB pixels"):
            ev.verify(self.bundle)

    def test_rejects_path_escape_and_unlisted_payload(self):
        with self.assertRaises(ValueError):
            ev.safe_path(self.bundle, "../secret")
        self.manifest["tasks"][0]["conditions"]["rgb_only"]["prompt_path"] = "unlisted.txt"
        ev.write(self.bundle / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "inventory"):
            ev.verify(self.bundle)

    def test_holistic_ranking_is_not_score_sum(self):
        response = payload([["D"], ["B", "C"], ["A"]])
        self.assertEqual(ev.parse_arm(json.dumps(response), self.schema), response)

    def test_rank_uniqueness_and_exact_fence_only(self):
        bad = payload([["A"], ["A"], ["C"], ["D"]])
        with self.assertRaises(ResponseValidationError):
            ev.parse_arm(json.dumps(bad), self.schema)
        raw = json.dumps(payload())
        self.assertEqual(ev.parse_arm("```json\n" + raw + "\n```", self.schema, "fence_only"), payload())
        for arm in ("strict", "fence_only"):
            with self.assertRaises(ResponseValidationError):
                ev.parse_arm("explanation\n```json\n" + raw + "\n```", self.schema, arm)
        with self.assertRaises(ResponseValidationError):
            ev.parse_arm("```json\n" + raw + "\n```", self.schema)
        with self.assertRaises(ResponseValidationError):
            ev.parse_arm(raw[:-1] + ', "confidence": 4}', self.schema)

    def test_dry_run_does_not_create_response_files(self):
        output = self.root / "dry"
        with contextlib.redirect_stdout(io.StringIO()):
            ev.run(self.bundle, self.manifest, {"backend": "mock", "model_name": "synthetic"}, output)
        self.assertFalse((output / "responses").exists())
        self.assertEqual(len(ev.read(output / "plan.json")["jobs"]), 96)
        with self.assertRaises(FileExistsError):
            ev.run(self.bundle, self.manifest, {}, output)

    def test_official_payload_contains_only_frozen_text_and_image(self):
        from unittest.mock import patch
        factory = Mock()
        factory.return_value.responses.create.return_value = Mock(output_text=json.dumps(payload()), usage=None, model="synthetic", _request_id="synthetic")
        config = dict(model="synthetic", api_key_env="CCB_SYNTHETIC_TEST_KEY", max_output_tokens=4000,
                      reasoning_effort="low", reasoning_mode="standard", store=False)
        with patch.dict("os.environ", {"CCB_SYNTHETIC_TEST_KEY": "synthetic-placeholder"}):
            OpenAIResponsesAdapter(factory).generate("frozen prompt", self.bundle / "tactile.png", config)
        request = factory.return_value.responses.create.call_args.kwargs
        self.assertFalse(request["store"])
        self.assertNotIn("tools", request)
        self.assertEqual(len(request["input"]), 1)
        content = request["input"][0]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0], {"type": "input_text", "text": "frozen prompt"})
        self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)

    def test_local_analysis_uses_matched_denominators(self):
        run = self.root / "run"
        ev.write(run / "plan.json", dict(input_manifest_sha256=ev.sha(self.bundle / "manifest.json"), repeats=2))
        for task in self.manifest["tasks"]:
            for c in ev.CONDITIONS:
                for r in ("01", "02"):
                    entry = task["conditions"][c]
                    ev.write(run / "responses" / task["task_id"] / c / (r + ".json"),
                             dict(task_id=task["task_id"], condition=c, repeat_id=r, status="received",
                                  prompt_sha256=entry["prompt_sha256"], image_sha256=entry["image_sha256"], raw_response=json.dumps(payload())))
        (run / "responses/task-01/rgb_tactile/02.json").unlink()
        human = self.root / "synthetic-human.json"
        ev.write(human, {"reference_kind": "original_scene_human_preferences", "tasks": {tid: [payload()["overall_ranking"]] for tid in ev.TASK_IDS}})
        output = self.root / "analysis"
        with contextlib.redirect_stdout(io.StringIO()):
            an.analyze(self.bundle, run, human, output, resamples=20)
        summary = ev.read(output / "summary.json")["summaries"]
        self.assertEqual(summary["primary_complete_two_repeats"]["mean_dense_strict_top1"]["task_count"], 15)
        self.assertEqual(summary["matched_repeat_sensitivity"]["mean_dense_strict_top1"]["task_count"], 16)
        self.assertEqual(sum(r["status"] == "not_run" for r in ev.read(output / "coverage.json")), 1)


class MetricAndSensorTests(unittest.TestCase):
    def test_dense_ties_and_top1_denominator(self):
        human = an.references([[["A"], ["B", "C"], ["D"]]])
        predicted = an.dense([["A", "B"], ["C"], ["D"]])
        self.assertEqual(an.metrics(predicted, human)["mean_dense_strict_top1"], 0)
        self.assertEqual(an.metrics(predicted, human)["mean_dense_inclusive_top1"], 1)
        self.assertIsNone(an.metrics(predicted, an.references([[["A", "B", "C", "D"]]]))["mean_dense_strict_top1"])
        self.assertIsNone(an.tau_b(predicted, an.dense([["A", "B", "C", "D"]])))
        self.assertEqual(an.tau_b(predicted, predicted), 1)

    def test_family_bootstrap_retains_task_weights(self):
        rows = [dict(family="shared", rgb_only=0, rgb_description=0, rgb_tactile=0),
                dict(family="shared", rgb_only=0, rgb_description=0, rgb_tactile=0),
                dict(family="single", rgb_only=0, rgb_description=0, rgb_tactile=1)]
        summary = an.summarize(rows, resamples=100)
        self.assertEqual(summary["conditions"]["rgb_tactile"], 1 / 3)
        self.assertEqual(summary["family_count"], 2)
        self.assertEqual(summary["contrasts"]["primary"]["ci95"], [0, 1])

    def test_contact_force_is_conserved_in_pressure_bins(self):
        snapshot = {"contacts": {"position": [[0, -.00015, .04]], "normal": [[0, 1, 0]],
                                  "force_a": [[0, 4, 0]], "force_b": [[0, -4, 0]], "link_a": [1], "link_b": [99]},
                    "fingers": [{"name": "left_finger", "link_index": 1, "position_m": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0], "net_force_world_N": [0, 4, 0]},
                                {"name": "right_finger", "link_index": 2, "position_m": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0], "net_force_world_N": [0, 0, 0]}]}
        result = project_snapshot(snapshot)
        self.assertTrue(result["qa"]["passed"])
        surface = result["surfaces"]["left_finger_inner"]
        self.assertAlmostEqual(float(np.sum(surface["pressure_kPa"] * 1000 * surface["cell_area_m2"])), 4)
        snapshot["contacts"]["force_b"] = [[0, -3, 0]]
        with self.assertRaisesRegex(ValueError, "action_reaction"):
            project_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
