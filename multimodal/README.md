# Multimodal contact extension — v5

This extension pairs **16 original tasks with one local contact setting each**, represented by 12 independent contact families. It adds frozen pressure/shear inputs and a reproducible three-condition evaluation workflow. It is separate from the 67-task RGB benchmark; its files and prompt must not silently replace that benchmark's pinned standard.

## Three conditions

| Condition | Original RGB | Neutral contact description and source scope | Pressure/shear maps, force annotations and legend |
|---|---|---|---|
| `rgb_only` | Yes | No | No |
| `rgb_description` | Yes | Yes | No |
| `rgb_tactile` | Yes | Yes | Yes |

All inputs use a 2048 × 1024 canvas. The left 1024 × 1024 RGB region has identical pixels across conditions. The right panel is blank white in the first two conditions, whose PNGs are byte-identical. No simulated keyframe is sent. Each request contains exactly one frozen UTF-8 prompt and one PNG. Task instructions and candidate A–D text are unchanged, in canonical order. The holistic rubric and output schema are frozen at v1.2; `overall_ranking` is used directly, not reconstructed by summing dimension ratings.

`rgb_tactile − rgb_only` measures the full additional contact-information package. `rgb_description − rgb_only` is the description ablation. `rgb_tactile − rgb_description` measures maps **plus** derived force annotations, legend and reading guidance; it does not isolate spatial map reasoning alone.

## Data and access

The input bundle is prepared for the existing **private** [Hugging Face dataset](https://huggingface.co/datasets/chengshengge/CreativeContactBench-pilot), under `multimodal/v5/`. **Upload is pending dataset write authorization.** [release.json](release.json) records the namespace and verified manifest SHA-256; its revision is null until upload succeeds. The download command refuses an unpublished revision. Once available, access requires a Hugging Face account authorized for that dataset. [catalog.json](catalog.json) lists every selected task, contact object and contact family; tasks 06, 10 and 13 are included.

The public repository contains protocol, catalog and code. The private input bundle contains:

```text
manifest.json                       # file inventory, hashes and paired payload paths
catalog.json
protocol/                           # rubric, schema, input rules and sensor specification
inputs/task-NN/{condition}.txt       # exact model-facing prompt
inputs/task-NN/{condition}.png       # exact model-facing image
samples/task-NN/task.json            # original task, A–D and neutral probe context
samples/task-NN/rgb.png
samples/task-NN/pressure_normal.png
samples/task-NN/pressure_and_shear.png
samples/task-NN/tactile_raw.npz
samples/task-NN/solver_snapshot.json
samples/task-NN/projection.json
samples/task-NN/metadata.json
```

Sample metadata and raw arrays support inspection and reproducibility; the runner never sends them to a model. No human submissions, participant identities, reference rankings, model responses, evaluation scores or result figures are distributed. Existing private dataset visibility is retained.

## Download, verify and inspect

From the repository root:

```bash
pip install -e ./evaluation
hf auth login
python multimodal/evaluate.py download
python multimodal/evaluate.py validate
python multimodal/evaluate.py inspect --task task-01 --condition rgb_tactile
```

Authenticate locally; do not put tokens in code or configuration files. `download` creates `multimodal/data/v5/` and refuses to overwrite an existing folder. `validate` checks every inventory hash, all 48 prompt/image pairs, the fixed task set and paired RGB pixels. `inspect` prints the exact prompt and image path. To inspect the full set, open the three text/PNG pairs under each task's `inputs/` directory.

## Plan and run evaluation

```bash
# No API calls: save the complete schedule and model configuration.
python multimodal/evaluate.py run --config multimodal/configs/gpt55.json --output multimodal/runs/gpt55-plan

# Offline end-to-end smoke test; these outputs are synthetic.
python multimodal/evaluate.py run --config multimodal/configs/mock.json --output multimodal/runs/mock --execute

# Actual model inference; OPENAI_API_KEY must already be set locally.
python multimodal/evaluate.py run --config multimodal/configs/gpt55.json --output multimodal/runs/gpt55 --execute
python multimodal/evaluate.py run --config multimodal/configs/gpt6.json --output multimodal/runs/gpt6 --execute
```

Each model has 16 tasks × 3 conditions × 2 repeats = **96 planned requests** (192 for two models). This is a planned workload, not a reported result. The two repeats and condition-order schedule use seed `2026090905`. No automatic retry, answer repair, tools or structured-output constraint is used. Official Responses API runs set `store=false`, low reasoning effort and a 4,000 output-token cap. Model availability depends on provider access; the explicit model IDs are recorded in the configs and are not silently substituted. Dry runs do not estimate cost. Actual runs incur the configured provider's charges and require `--execute`.

The optional `gemma.json` configuration expects a separately served OpenAI-compatible endpoint, the stated model revision, and `CCBENCH_LOCAL_API_KEY` in the environment. The runner records the revision but does not install or attest the serving process; operators must verify it. Its configuration uses temperature 0 and 1,600 output tokens.

Runs write a plan and one response record per task/condition/repeat. Raw text, returned model ID, usage and parsing status remain local in the ignored `multimodal/runs/` directory. A provider error stops the run without retry. Existing output directories are refused; interrupted requests are retained as `started` and must not be treated as successful responses. A deliberate rerun requires a fresh directory. It is not merged silently into an existing evaluation.

## Private human-reference analysis

Provide a local JSON file under `multimodal/private/` with two fields: `reference_kind` equal to `original_scene_human_preferences`, and `tasks` mapping every selected task ID to a list of participant rankings. Each participant ranking is a list of best-to-worst rank groups, containing A–D exactly once, with ties grouped together. Use canonical option IDs and exclude participant identities. No reference file or real ranking example is bundled.

```bash
python multimodal/analyze.py --bundle multimodal/data/v5 --run multimodal/runs/gpt55 --human multimodal/private/reference.json --output multimodal/runs/gpt55-analysis-strict
python multimodal/analyze.py --bundle multimodal/data/v5 --run multimodal/runs/gpt55 --human multimodal/private/reference.json --arm fence_only --output multimodal/runs/gpt55-analysis-fence
```

The main reference is the exact participant mean of dense ranks, with ties preserved using rational arithmetic. The primary metric is strict Top-1 agreement where the human reference has a unique top: a model top tie counts as failure. Secondary metrics are inclusive Top-1, Kendall tau-b and first-choice plurality (participant top ties split one vote equally). A human top tie is outside the Top-1 denominator, not silently counted as a match. Undefined tau-b remains undefined.

For each metric, primary comparisons use only tasks with all three conditions and both repeats valid and defined. Metrics are averaged within a task over repeats, then equally over tasks. A separate sensitivity analysis uses available complete three-condition repeat triplets. Coverage distinguishes not-run, interrupted, API-error and invalid-response records. Strict parsing is primary; a separately named supplement may strip only a whole-response JSON fence, applying the same rule to every model without repairing content.

Confidence intervals use 5,000 paired contact-family bootstrap resamples (seed `2026090905`), retaining all tasks and conditions within a family and equal task weighting. Intervals are conditional on the supplied human reference; secondary intervals are unadjusted. Shared families are can (01/10), books (24/37) and cup (25/27/32); the other tasks have separate families. Repeats and shared tasks are not independent new physical observations.

Existing human rankings concern the original scenes. They are not new human validation of these hypothetical probe states. Historical compatibility of task versions must be documented by the data owner; this release includes all 16 selected tasks. The released analysis implementation is portable workflow code; it does not distribute private human-reference artifacts or reproduce a paper result without those artifacts.

## Sensor construction and limits

See [sensor specification](protocol/pressure_map_specification.json). Each virtual bin is 2 × 2 mm (4 mm²). Full Genesis rigid-contact forces are projected to one declared finger face and deposited using conservative bilinear weights. Dividing the normal/tangential components by bin area yields grid-average normal pressure and signed shear in kPa. Four surfaces (left/right inner pad and fingertip end) each have normal, u-shear and v-shear channels. Scales are shared across tasks: normal 0–1500 kPa; signed shear −750–750 kPa. `F` annotations are area-integrated forces in N.

The frozen snapshots preserve contact forces and finger poses. To independently regenerate numerical maps and render a sensor image:

```bash
pip install -r multimodal/requirements-sensor.txt
python multimodal/pressure_projection.py multimodal/data/v5/samples/task-01/solver_snapshot.json multimodal/runs/reprojection/task-01
python multimodal/render_maps.py multimodal/data/v5/samples/task-01/solver_snapshot.json multimodal/runs/reprojection/task-01.png
```

Projection checks cover force conservation, action/reaction, frame conventions and contact coverage. Rasterized plot typography can vary with Matplotlib; evaluation always uses the frozen PNG bytes. This release reproduces the projection from stored solver snapshots, not the complete original scene or Genesis asset installation. Original simulation meshes are not redistributed.

Can probes use one fixed 0.35 kg proxy with an unsupported gravity hold; book probes use five 140 mm-wide rigid proxies and a force-limited approach/closure. Other targets are fixed in zero-gravity local fixtures. Dimensions, mass and friction are declared assumptions, not measurements recovered from RGB. The original RGB describes the original robot and task; the substitute Panda probe does not execute any candidate strategy.

These are simulated grid-average rigid contact signals, not calibrated deformable-skin measurements. Sidewall normal-pressure peaks are not object weight. Fixed-target contacts do not establish load support. A single frame does not measure softness, friction coefficient, temperature, fracture or task success. Zero resolved force is not proof of absent geometric contact.

## Publication boundary and provenance

Only inputs, their physical provenance and workflow code belong in this release. Never add `runs/`, `private/`, human exports, paid-call ledgers, model outputs or analysis figures to a publication commit. Code follows the repository license; existing dataset and referenced third-party asset terms are unchanged. See [third-party notices](../THIRD_PARTY_NOTICES.md).
