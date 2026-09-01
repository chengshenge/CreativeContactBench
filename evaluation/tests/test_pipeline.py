from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from ccbench_eval import PROMPT_SHA256
from ccbench_eval.adapters import OpenAICompatibleAdapter, OpenAIResponsesAdapter
from ccbench_eval.dataset import DatasetValidationError, Task, validate_local_dataset
from ccbench_eval.prompt import render_prompt, sha256_file
from ccbench_eval.response import ResponseValidationError, load_schema, validate_response
from ccbench_eval.run_validation import RunValidationError, validate_run
from ccbench_eval.runner import load_config, run_evaluations


EVALUATION_ROOT = Path(__file__).resolve().parents[1]


def valid_response() -> dict:
    return {
        "ratings": {
            option: {
                "expected_task_effectiveness": 3,
                "embodied_feasibility": 4,
                "functional_creativity": 2,
            }
            for option in "ABCD"
        },
        "overall_ranking": [["A"], ["B"], ["C"], ["D"]],
        "confidence": 4,
    }


class PromptAndResponseTests(unittest.TestCase):
    def test_prompt_hash_and_rendering(self) -> None:
        path = EVALUATION_ROOT / "prompts/vlm_evaluator_v1.2.txt"
        self.assertEqual(sha256_file(path), PROMPT_SHA256)
        rendered = render_prompt(
            path.read_text(encoding="utf-8"),
            "Move the object.",
            {option: f"Strategy {option}" for option in "ABCD"},
        )
        self.assertNotIn("{{", rendered)
        self.assertIn("Move the object.", rendered)

    def test_ranking_semantics(self) -> None:
        schema = load_schema(EVALUATION_ROOT / "schemas/vlm_response_v1.schema.json")
        self.assertEqual(validate_response(valid_response(), schema)["confidence"], 4)
        invalid = valid_response()
        invalid["overall_ranking"] = [["A"], ["A"], ["C"], ["D"]]
        with self.assertRaises(ResponseValidationError):
            validate_response(invalid, schema)


class DatasetTests(unittest.TestCase):
    def test_contiguous_dataset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            records = []
            for number in range(1, 4):
                task_id = f"task-{number:02d}"
                relative = f"images/{task_id}.png"
                Image.new("RGB", (2, 2), color=(number, number, number)).save(root / relative)
                records.append({
                    "file_name": relative,
                    "task_id": task_id,
                    "task_instruction": f"Instruction {number}",
                    "option_A": "A",
                    "option_B": "B",
                    "option_C": "C",
                    "option_D": "D",
                })
            (root / "metadata.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(validate_local_dataset(root, expected_tasks=3)), 3)

    def test_dataset_rejects_image_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            outside = root / "outside.png"
            Image.new("RGB", (2, 2)).save(outside)
            record = {
                "file_name": "../outside.png",
                "task_id": "task-01",
                "task_instruction": "Instruction",
                "option_A": "A",
                "option_B": "B",
                "option_C": "C",
                "option_D": "D",
            }
            (root / "metadata.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(DatasetValidationError):
                validate_local_dataset(root, expected_tasks=1)


class RunnerTests(unittest.TestCase):
    def test_mock_run_writes_and_validates_partial_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "task-01.png"
            Image.new("RGB", (2, 2), color=(1, 2, 3)).save(image)
            task = Task(
                file_name="images/task-01.png",
                task_id="task-01",
                task_instruction="Move the object.",
                option_A="Strategy A",
                option_B="Strategy B",
                option_C="Strategy C",
                option_D="Strategy D",
                image_path=image,
            )
            output = root / "runs"
            config = EVALUATION_ROOT / "configs/mock.yaml"
            with patch("ccbench_eval.runner.load_dataset", return_value=(root, [task])):
                run_dir, manifest = run_evaluations(config, limit=1, output_directory=output)
            self.assertEqual(manifest["semantically_valid"], 1)
            self.assertEqual(manifest["model_requests"], 0)
            self.assertEqual(
                validate_run(run_dir, allow_partial=True, allow_mock=True)["status"],
                "PASS",
            )
            with self.assertRaises(RunValidationError):
                validate_run(run_dir, allow_partial=True)
            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {
                    "manifest.json",
                    "results.jsonl",
                    "summary.csv",
                    "summary.md",
                    "submission.json",
                    "code_snapshot_sha256.json",
                    "raw_responses",
                },
            )

    def test_literal_secret_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text(
                "dataset: {}\nprompt: {}\nresponse_schema: {}\nrun: {}\n"
                "generation: {}\noutput: {}\nendpoint:\n  api_key: forbidden\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_config(path)


class AdapterTests(unittest.TestCase):
    def _image(self, root: Path) -> Path:
        path = root / "image.png"
        Image.new("RGB", (2, 2), color=(4, 5, 6)).save(path)
        return path

    def test_openai_responses_uses_one_request_and_zero_sdk_retries(self) -> None:
        calls: dict = {}

        class Usage:
            input_tokens = 10
            output_tokens = 5
            total_tokens = 15
            input_tokens_details = None
            output_tokens_details = None

        class Response:
            output_text = json.dumps(valid_response())
            usage = Usage()
            status = "completed"
            model = "test-model"
            _request_id = "req_test"

        class Responses:
            def create(self, **kwargs):
                calls["request"] = kwargs
                return Response()

        class Client:
            responses = Responses()

        def factory(**kwargs):
            calls["client"] = kwargs
            return Client()

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"CCBENCH_TEST_KEY": "test-value"}
        ):
            result = OpenAIResponsesAdapter(factory).generate(
                "prompt",
                self._image(Path(temporary)),
                {
                    "api_key_env": "CCBENCH_TEST_KEY",
                    "base_url": "https://api.openai.com/v1",
                    "model_name": "test-model",
                    "store": False,
                },
            )
        self.assertEqual(calls["client"]["max_retries"], 0)
        self.assertIs(calls["request"]["store"], False)
        self.assertEqual(result.request_metadata["model_requests"], 1)
        self.assertEqual(result.backend_metadata["request_id"], "req_test")

    def test_openai_compatible_uses_one_chat_completion(self) -> None:
        calls: dict = {}

        class Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "model": "test-open-model",
                    "choices": [{"message": {"content": json.dumps(valid_response())}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }

        class Client:
            def __init__(self, **kwargs):
                calls["client"] = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, **kwargs):
                calls["url"] = url
                calls["post"] = kwargs
                return Response()

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"CCBENCH_TEST_KEY": "test-value"}
        ), patch("ccbench_eval.adapters.httpx.Client", Client):
            result = OpenAICompatibleAdapter().generate(
                "prompt",
                self._image(Path(temporary)),
                {
                    "api_key_env": "CCBENCH_TEST_KEY",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "model_name": "test-open-model",
                    "temperature": 0,
                    "max_tokens": 100,
                    "store": False,
                },
            )
        self.assertEqual(calls["url"], "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(calls["post"]["json"]["temperature"], 0)
        self.assertEqual(result.request_metadata["model_requests"], 1)
        self.assertEqual(result.backend_metadata["total_tokens"], 15)


if __name__ == "__main__":
    unittest.main()
