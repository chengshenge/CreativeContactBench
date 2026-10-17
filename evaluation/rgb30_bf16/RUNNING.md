# CreativeContactBench RGB30 BF16 handoff (2026-10-17)

This is the standalone **30-task paper leaderboard** pipeline, not the public generic 67-task runner. It contains the exact existing RGB30 images, task text and rendered v1.2 prompts. No model inference was run while building it. Keep this full archive within the author/collaborator team; it includes unpublished task inputs, aggregate human references and historical regression fixtures. It contains no credentials or individual human submissions. Public code alone is insufficient: obtain this author-provided archive.

## What is frozen

- The actual 30 questionnaire task IDs: 01–19, 24, 25, 27, 32, 37, 38, 42, 43, 47, 63, 64.
- Canonical A/B/C/D, one formal response per model/task; original RGB + instruction + four strategies + v1.2 rubric only.
- Preserve direct overall_ranking independently of E/F/C scores. E+F+C derived ranking is exported separately and never fills an invalid holistic answer.
- 4 configurations × 30 formal slots = 120 calls; up to 3 transient retries per model (maximum 2 attempts per task). No retries for valid/unfavorable answers, refusal, truncation, missing fields or invalid JSON. One synthetic RGB deployment check per launched server is separate from formal slots.
- Original source manifest SHA-256: `1e530ecdde8b944a6b6161890f19d98f335cd799ee4212017d674e70ae5321f2`.
- New portable manifest preserves every original task entry and byte hash. Original manifest is retained in provenance; its full private/source file list is not claimed to be distributed.
- Only RGB30 expanded is calculated. Tasks 06/10/13 retain the registered author-equivalence assumption; independently_verified_per_participant_view_hashes remains false. See protocol/version_equivalence_registration.json. No 27-task rerun, tactile, or paper edits.

## Planned configurations

| Config key | Official checkpoint | Mode | GPUs | max_tokens |
|---|---|---|---|---:|
| qwen3_8-27b-bf16 | Qwen/Qwen3.8-27B | BF16, thinking | 2 × 48GB | 16384 |
| qwen3_6-35b-a3b-bf16 | Qwen/Qwen3.6-35B-A3B | BF16, thinking | 2 × 48GB | 16384 |
| qwen3_5-9b-bf16 | Qwen/Qwen3.5-9B | BF16, thinking | 1 × 48GB | 16384 |
| qwen3-vl-8b-instruct-bf16 | Qwen/Qwen3-VL-8B-Instruct | BF16, Instruct | 1 × 48GB | 1600 |

Exact 40-character official revisions and settings are in configs/*.json. BF16 means original weights, no quantization or CPU offload. 35B-A3B must load all experts. Memory feasibility is an estimate; GPU startup has **not** been tested on the destination hardware. Context is 32768, batch concurrency 1. Thinking configurations use temperature 1, top_p .95, top_k 20, seed 42; Instruct uses greedy decoding. All other generation fields are explicit. max_tokens includes reasoning and final output. These are new configurations, not a quantization-only controlled experiment. Do not switch 8B-Instruct to a Thinking checkpoint.

## New machine: setup

Linux, Python 3.11 or 3.12, CUDA-capable NVIDIA drivers, and enough free disk for the original weights (~200GB total; recommend 300–500GB free). Two reserved 48GB GPUs for the larger models; do not stop other users' jobs. GPU model/driver compatibility with the pinned vLLM release must be checked locally.

Extract the full ZIP and `cd rgb30_bf16_handoff_20261017`. Keep its directory structure. Do not overlay it on the old experiment.

```bash
python3.11 -m venv .venv-runner
.venv-runner/bin/python -m pip install -r requirements.txt
.venv-runner/bin/python ccb_rgb30.py verify
.venv-runner/bin/python -m unittest discover -s tests -v
```

Use a separate server environment so GPU dependencies do not alter the analysis environment:

```bash
python3.11 -m venv .venv-server
.venv-server/bin/python -m pip install -r requirements-server.txt httpx jsonschema Pillow huggingface-hub
.venv-server/bin/python -m pip check
```

vLLM is pinned to 0.31.0 (published version checked during packaging). This pins a candidate deployment environment, **not a claimed validated GPU installation**. If installation or model loading fails, preserve the error and report BLOCKED; do not silently upgrade, quantize, substitute, or change the experiment configuration. Any approved backend/config change must make a new bundle/run identity before formal calls.

## Run one model

Reserve GPUs 0 and 1. In terminal A:

```bash
.venv-server/bin/python ccb_rgb30.py serve --model qwen3_8-27b-bf16 --gpus 0,1
```

This downloads the pinned public official snapshot anonymously, hashes its files, records GPU/environment/launch information, and launches a localhost-only vLLM endpoint. The first download and weight hashing can take time. It writes logs under deployment/<key>/<session>/server.log. Wait for the server to finish loading. In terminal B:

```bash
.venv-runner/bin/python ccb_rgb30.py smoke --model qwen3_8-27b-bf16
.venv-runner/bin/python ccb_rgb30.py run --model qwen3_8-27b-bf16
```

Smoke uses one synthetic plain blue image, never a benchmark question. It checks that RGB requests complete and thinking output is separated when enabled. It is a deployment/transport check, not an accuracy score. Failed smoke blocks formal inference.

When complete, Ctrl+C **your own terminal A server**, then launch the next model. Repeat smoke and run with the same key. Use `--gpus 0,1` for 35B and `--gpus 1` for 9B/8B. The four configurations run sequentially; no automatic GPU process termination is performed.

On disconnect, reconnect and rerun the same `run` command. Completed slots are never sent again. If the server restarted, use serve and smoke again first. Saved response recovery is local; missing responses consume the same fixed retry budget. Do not delete attempts to get another answer. A BLOCKED unavailable server leaves the remaining tasks not_run. A fresh independent repeat requires a new run/bundle directory.

## Analyze and return results

```bash
.venv-runner/bin/python ccb_rgb30.py analyze
.venv-runner/bin/python ccb_rgb30.py pack-results
```

Return **rgb30_bf16_results.zip** to the original machine. This includes all formal raw responses, reasoning fields when returned, parsed answers, attempt/failure records, dimensions, model versions, runtime provenance, coverage and new-result tables. It excludes weights and inputs already held by the original machine. No credentials are required for these local endpoints or for the included task inputs.

Analysis reads reference/human_consensus.json only after inference. It is an aggregate consensus grouped ordering, derived from the ten-person dense-mean reference, with no individual human records. Ordinal groups suffice for tau-b and Top-1. Existing ALL_LOO descriptive statistics are retained with provenance, not newly inferred from this aggregate. The inference request builder never reads the reference, historical fixtures, or leaderboard.

Outputs: analysis/rgb30_bf16_results.csv, .tex, per_task_metrics.csv, summary.md; runs/<key>/coverage.csv, dimension_ratings.csv, dimension_ratings_from_invalid_responses.csv, repair_failure_records.csv, model_versions.csv, raw provider responses and run_manifest.json. Missing/invalid tasks and partial coverage remain explicit. Bootstrap uses the unchanged existing implementation: 10000 task resamples, seed 20260908, conditional on the existing human reference; undefined tau is omitted. Human top ties are excluded from Top-1; model top ties fail strict but may pass inclusive.

The returned table includes only these new configurations. Merge into the complete leaderboard on the original machine after checking hashes/coverage; retain historical Q4 results as distinct configurations. Do not overwrite old answers or select rows based on whether the new score improves.

## Code and provenance

Official project: https://github.com/chengshenge/CreativeContactBench . The generic evaluation/ default is a separate 67-task workflow; do not run that default for this experiment. The public handoff code does not publish the private full archive.

vendor/response_v12.py, container_repair_source.py and holistic_alignment_baseline.py are unchanged copies of the actual previous RGB30 dependencies. Only repair_json_container is imported from container_repair_source.py; its score-derived validator is deliberately NOT used. metrics_v1.py contains unchanged extracted analysis functions. The portable request orchestration adds explicit thinking transport, raw provider preservation, pinned local deployment, and path-independent manifests. Original image bytes and prompt text are unchanged.

Original code licenses/attributions remain in LICENSE and THIRD_PARTY_NOTICES.md. Dataset content retains its own terms. See provenance/source_inventory.json, bundle_manifest.json and VALIDATION.md.
