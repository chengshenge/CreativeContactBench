# CreativeContactBench standard VLM evaluation

This directory is a self-contained runner for evaluating open-source and closed-source vision-language models on CreativeContactBench. It downloads a pinned Hugging Face dataset snapshot, sends exactly one image-and-text request per selected task, validates the response contract, and creates a portable standard result package.

## Standard protocol

| Field | Value |
|---|---|
| Dataset | `chengshengge/CreativeContactBench-pilot` |
| Dataset revision | `d8fc98ae30bf1233518330215a6e57f990565d94` |
| Tasks | `task-01` through `task-67` |
| Prompt | `prompts/vlm_evaluator_v1.2.txt` |
| Prompt SHA-256 | `a17d126bfd1da70d1473c997a26a53fe541bd99d5f2e561fbf49d6449b8979a1` |
| Option order | Canonical A/B/C/D |
| Evaluations | One per task; 67 total |
| Automatic retries | Disabled |
| Response repair | Disabled |

Runs using another dataset revision, prompt, option-order perturbation, repeated sampling, retries, or repaired responses are useful diagnostics but are not comparable standard submissions.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ./evaluation
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

## Validate the dataset

The command downloads only `metadata.jsonl` and task images from the pinned revision. It never writes to Hugging Face.

```bash
ccbench-eval validate-data --config evaluation/configs/mock.yaml
```

Validation requires exactly 67 unique records, continuous IDs from `task-01` to `task-67`, the expected metadata fields, and exactly one readable image per record.

## Offline smoke test

The mock backend reads the real images but makes no network model requests:

```bash
ccbench-eval run --config evaluation/configs/mock.yaml --limit 1
ccbench-eval validate-run --run-dir evaluation/runs/<RUN_ID> --allow-partial --allow-mock
```

`--limit` and task selection are intended only for smoke tests. Omit them for a standard submission.

## Closed-source model: OpenAI Responses API

Copy the template and set your model name. Never put a credential in YAML.

```bash
cp evaluation/configs/openai_responses.example.yaml evaluation/configs/my_openai_model.yaml
export OPENAI_API_KEY="..."
ccbench-eval run --config evaluation/configs/my_openai_model.yaml
```

The native adapter permits only `https://api.openai.com/v1`, uses one Responses API request per task, sets SDK retries to zero, enables no tools or Structured Outputs, and defaults to `store: false`.

## Open-source model: vLLM or another compatible server

Start a multimodal OpenAI-compatible server. For example:

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --limit-mm-per-prompt '{"image":1}'
```

Then copy the endpoint template and run the benchmark:

```bash
cp evaluation/configs/openai_compatible.example.yaml evaluation/configs/my_open_model.yaml
export CCBENCH_ENDPOINT_API_KEY="local"
ccbench-eval run --config evaluation/configs/my_open_model.yaml
```

Record an immutable model revision in `generation.model_revision` whenever the registry provides one. The runner does not start, stop, download, or modify model weights.

## Standard artifacts

Each run creates a unique directory under `evaluation/runs/`:

| Artifact | Purpose |
|---|---|
| `submission.json` | Portable public result with benchmark identity, model metadata, and canonical task results; schema: [`schemas/submission_v1.schema.json`](schemas/submission_v1.schema.json) |
| `manifest.json` | Full run configuration, counts, timestamps, hashes, generation settings, and token totals |
| `results.jsonl` | One detailed audit record per attempted task |
| `summary.csv` | Flat per-task ratings, ranking, confidence, latency, usage, and validation state |
| `summary.md` | Human-readable aggregate and per-task report |
| `raw_responses/` | Exact unmodified text returned by the model |
| `code_snapshot_sha256.json` | SHA-256 inventory of evaluator files |

`submission.json` excludes API credentials and embedded image bytes. Before sharing a full run directory, review raw responses for provider-specific sensitive content.

## Validate a standard run

```bash
ccbench-eval validate-run --run-dir evaluation/runs/<RUN_ID>
```

A standard run passes only when it contains all required artifacts, all 67 task IDs exactly once, canonical option order, the pinned dataset and prompt hashes, 67 semantically valid responses, no retries, no repairs, and no missing raw response files.

The validator rejects the deterministic mock backend by default. `--allow-mock` exists only for offline pipeline and CI verification.

## Preview a rendered prompt

```bash
ccbench-eval preview --config evaluation/configs/mock.yaml --task-id task-01
```

Preview prints the prompt and local image path but makes no model request.

## Configuration safety

- Literal config keys named `api_key`, `token`, `password`, or `secret` are rejected.
- YAML files contain only an environment-variable name, such as `OPENAI_API_KEY`.
- `.env`, `*.env`, `.secrets/`, and `evaluation/runs/` are ignored by Git.
- The runner performs no retries and no response repair; a model failure remains a failure.

## Adding another provider

Providers exposing OpenAI-compatible multimodal chat completions can use the existing endpoint adapter. For another API shape, implement the `ModelAdapter` interface in `src/ccbench_eval/adapters.py`; preserve the one-request rule and return the exact raw model text plus sanitized metadata.

## Tests

Tests are fully offline:

```bash
python -m unittest discover -s evaluation/tests -v
```

## License

The evaluation pipeline and original documentation are licensed under the
[Apache License 2.0](../LICENSE). The benchmark dataset, model outputs, and third-party
dependencies retain their own applicable licenses and terms.
