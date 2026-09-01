# CreativeContactBench

CreativeContactBench evaluates whether vision-language models can judge robot manipulation strategies from a scene image, a task instruction, and four candidate strategies. The public benchmark pipeline downloads a pinned dataset revision from Hugging Face, evaluates every task in canonical option order, validates the model's JSON output, and writes a reproducible standard result package.

## Evaluate a VLM

The complete evaluator is in [`evaluation/`](evaluation/README.md). It supports:

- closed-source models through the official OpenAI Responses API;
- open-source VLMs served by an OpenAI-compatible endpoint such as vLLM;
- an offline deterministic mock backend for installation and CI checks.

Quick start:

```bash
git clone https://github.com/chengshenge/CreativeContactBench.git
cd CreativeContactBench
python3 -m venv .venv
source .venv/bin/activate
pip install -e ./evaluation

ccbench-eval validate-data --config evaluation/configs/mock.yaml
ccbench-eval run --config evaluation/configs/mock.yaml --limit 1
```

For a real benchmark submission, copy one of the model configuration templates, provide credentials only through an environment variable, and run all 67 tasks without `--limit`. A completed standard run contains `submission.json`, `manifest.json`, `results.jsonl`, `summary.csv`, `summary.md`, `raw_responses/`, and `code_snapshot_sha256.json`.

See the [evaluation documentation](evaluation/README.md) for OpenAI and vLLM commands, output definitions, validation, and submission guidance.

## Dataset

- Hugging Face: [`chengshengge/CreativeContactBench-pilot`](https://huggingface.co/datasets/chengshengge/CreativeContactBench-pilot)
- Standard evaluation revision: `d8fc98ae30bf1233518330215a6e57f990565d94`
- Standard evaluator prompt: `evaluation/prompts/vlm_evaluator_v1.2.txt`
- Tasks: `task-01` through `task-67`

## Genesis Scene Kit

A reusable Genesis scene toolkit extracted from our benchmark scene-generation pipeline. It focuses on strict RayTracer rendering, reproducible camera configuration, RobotSmith single-arm assets, structured outputs, and run manifests instead of publishing every experimental revision of each benchmark task.

CreativeContactBench is an independent project. The repository-specific evaluation, scene orchestration, and rendering-helper code is maintained here; Genesis supplies the simulation and rendering runtime, while referenced robot descriptions and visual assets retain their original authorship and terms. See [Third-party notices](THIRD_PARTY_NOTICES.md) for the exact boundaries and sources.

![Ray-traced tabletop preview](docs/images/raytraced_tabletop_preview.png)

## What Is Included

- Explicit `raytracer` and `rasterizer` modes with no silent fallback for final rendering.
- Project-local Genesis, CUDA, OptiX, and LuisaRender cache directories.
- Reusable table, wall, target, transparent bowl, water surface, and visually identical block primitives.
- Optional RobotSmith xArm7 with a parallel gripper.
- RGB, GIF, and JSON manifest outputs recording the renderer, camera, runtime, and asset sources.

## Windows Quick Start

Python 3.11, an NVIDIA GPU, and a recent graphics driver are recommended. Run the following commands in PowerShell:

```powershell
git clone https://github.com/chengshenge/CreativeContactBench.git
cd CreativeContactBench

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# Select the PyTorch CUDA wheel that matches your driver. Our tested environment uses cu128.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"

# The PyPI genesis-world package does not include a ready-to-use LuisaRenderPy build.
# This script checks out our tested Genesis commit and builds the Windows/CUDA RayTracer.
.\scripts\setup_raytracer_windows.ps1

git clone https://github.com/UMass-Embodied-AGI/RobotSmith .external/RobotSmith
$env:ROBOTSMITH_ROOT = (Resolve-Path .external/RobotSmith).Path
$env:GENESIS_SOURCE_ROOT = (Resolve-Path .external/Genesis).Path
$env:LUISA_RENDER_BUILD_BIN = (Resolve-Path .external/Genesis/genesis/ext/LuisaRender/build-win-cuda/bin/Release).Path

python scripts/check_environment.py --strict-raytracer
python examples/raytraced_tabletop.py --renderer raytracer --backend gpu --spp 16
```

The example writes the following files to `outputs/raytraced_tabletop/`:

```text
scene_rgb.png
scene_preview.gif
scene_manifest.json
generated_assets/open_bowl.obj
```

The first RayTracer run compiles Genesis kernels and shaders and creates several caches. Our first minimal smoke test took approximately five minutes on Windows with an RTX 3060 Ti; later runs reused the caches. Start with a small render while validating a new machine:

```powershell
python examples/raytraced_tabletop.py --no-robot --renderer raytracer --backend gpu --width 320 --height 180 --spp 1 --settle-steps 1
```

## Test Without RobotSmith

The base scene can be rendered without downloading RobotSmith:

```powershell
python examples/raytraced_tabletop.py --no-robot --renderer raytracer --backend gpu --spp 4
```

Use rasterizer mode explicitly when you only need to inspect geometry, camera framing, or object placement:

```powershell
python examples/raytraced_tabletop.py --no-robot --renderer rasterizer --backend gpu --width 640 --height 360
```

Rasterizer output is intended for debugging and should not be used as the final photoreal benchmark RGB input.

## Linux

Use the same installation sequence, but activate the virtual environment with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the PyTorch CUDA wheel that matches your driver, then run `pip install -e ".[dev]"`. RayTracer also requires a recursive Genesis clone and a LuisaRender build following `.external/Genesis/genesis/ext/LuisaRender/BUILD.md`. After building it, set:

```bash
export GENESIS_SOURCE_ROOT="$PWD/.external/Genesis"
export LUISA_RENDER_BUILD_BIN="$GENESIS_SOURCE_ROOT/genesis/ext/LuisaRender/build/bin"
```

## Create a New Scene

1. Copy `examples/raytraced_tabletop.py`.
2. Add task-specific assets, materials, poses, and collision geometry before `scene.build()`.
3. Adjust `CameraSpec`, then validate framing with a low resolution and low SPP.
4. Increase SPP for the final render and verify `renderer_effective`, `rasterizer_fallback_used`, and the runtime information in `scene_manifest.json`.
5. Do not commit downloaded BlenderKit or Objaverse binaries. Record asset IDs, sources, and licenses in `docs/ASSETS.md`.

See [docs/PIPELINE.md](docs/PIPELINE.md) for module boundaries and [docs/ASSETS.md](docs/ASSETS.md) for external asset conventions.

## Render A Task Configuration

Photoreal scenes can be described as JSON task files instead of copy-pasted Python revisions. The repository includes
`tasks/reference_robot_paper_can_pen.json`, which records the camera crop, strict RayTracer settings, three-point
lighting, RoboWits white table, RobotSmith xArm pose, BlenderKit asset IDs, and object placement used for this image:

![Photoreal paper, can, and pen task](docs/images/reference_robot_paper_can_pen_photoreal.jpg)

After preparing the assets listed in [docs/ASSETS.md](docs/ASSETS.md), render the task with:

```powershell
python examples/render_task.py tasks/reference_robot_paper_can_pen.json
```

The default profile is `1600x1200`, `256 spp`, denoised CUDA/OptiX RayTracer. It never silently falls back to the
rasterizer. Use command-line overrides for a quick framing check or a different final resolution:

```powershell
python examples/render_task.py tasks/reference_robot_paper_can_pen.json --width 960 --height 720 --spp 32
```

The runner writes `scene_rgb.png`, `scene_manifest.json`, generated PBR maps, and the generated robot URDF under
`outputs/reference_robot_paper_can_pen/`. To create another task, copy the JSON and change assets, poses, camera, and
lighting while keeping the strict renderer contract.

## Current Limitations

- Genesis RayTracer requires a separately built LuisaRender runtime and CUDA/OptiX. A normal `pip install genesis-world` installs the Python dependencies but does not complete the RayTracer setup.
- The example water is a visible transparent surface, not a full fluid simulation.
- RobotSmith and BlenderKit assets remain subject to their upstream licenses and are not redistributed here.
- The task-configured example requires prepared third-party meshes. The JSON records exact IDs and expected paths,
  and the runner reports every missing file before Genesis starts.

## License

Code and original documentation in this repository are licensed under the
[Apache License 2.0](LICENSE). The benchmark dataset, model outputs, and third-party
software or assets retain their own applicable licenses and terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution and license scope.
