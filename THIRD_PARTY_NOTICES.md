# Third-party notices

CreativeContactBench is an independent project. Its benchmark evaluator, repository-specific orchestration code, configuration, and original documentation are maintained in this repository and licensed under the repository's [Apache License 2.0](LICENSE).

That license does not relicense external software, robot descriptions, meshes, textures, HDR environments, model outputs, or datasets. The components below retain their original authorship and applicable terms.

## Genesis World

- Project: [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world)
- Authors: Genesis Authors / Genesis AI Team
- Upstream license: [Apache License 2.0](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/LICENSE)
- Use here: runtime dependency, physics simulation API, rasterizer, and RayTracer integration.
- Repository locations: `pyproject.toml`, `genesis_scene/`, `scripts/setup_raytracer_windows.ps1`, and `examples/`.

Genesis World is installed as a dependency or cloned into the ignored `.external/Genesis` directory. Its source code is not vendored into this repository.

## LuisaCompute and LuisaRender

- Projects: [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) and [LuisaRender](https://github.com/LuisaGroup/LuisaRender)
- Authors: the LuisaGroup contributors
- Upstream licenses: [LuisaCompute Apache-2.0](https://github.com/LuisaGroup/LuisaCompute/blob/stable/LICENSE) and [LuisaRender BSD-3-Clause](https://github.com/LuisaGroup/LuisaRender/blob/next/LICENSE)
- Use here: optional compiled RayTracer backend reached through Genesis World.

These components are obtained transitively with the ignored Genesis checkout and are not vendored into this repository. The setup script only builds the upstream source tree; their respective licenses remain controlling.

## RobotSmith

- Project: [RobotSmith: Generative Robotic Tool Design for Acquisition of Complex Manipulation Skills](https://github.com/UMass-Embodied-AGI/RobotSmith)
- Authors: Chunru Lin, Haotian Yuan, Yian Wang, Xiaowen Qiu, Tsun-Hsuan Wang, Minghao Guo, Bohan Wang, Yashraj Narang, Dieter Fox, and Chuang Gan
- Use here: an externally supplied xArm URDF, HDR environment, and carpet mesh are referenced by the rendering examples.
- Repository locations: `examples/raytraced_tabletop.py`, `tasks/reference_robot_paper_can_pen.json`, `docs/ASSETS.md`, and the rendered previews listed below.

The RobotSmith checkout is expected under the ignored `.external/RobotSmith` directory and is not distributed by this repository. As of 2026-09-01, the upstream repository does not display a top-level license file. Its files are therefore not covered by CreativeContactBench's Apache-2.0 license; users must obtain and follow the applicable upstream terms before use or redistribution.

## RoboWits

- Project: [RoboWits: Unexpected Challenges for Robotic Creative Problem Solving](https://github.com/UMass-Embodied-AGI/RoboWits)
- Authors: Chunru Lin, Hongxin Zhang, Fenghao Yu, Zhehuan Chen, Thomas L. Griffiths, Yejin Choi, David Held, and Chuang Gan
- Use here: the reference scene expects a work-table mesh and related texture maps prepared outside this repository.
- Repository locations: `tasks/reference_robot_paper_can_pen.json`, `examples/render_task.py`, `docs/ASSETS.md`, and `docs/images/reference_robot_paper_can_pen_photoreal.jpg`.

The RoboWits mesh and textures are not distributed by this repository. As of 2026-09-01, the upstream repository does not display a top-level license file. These assets are not covered by CreativeContactBench's Apache-2.0 license; users must obtain and follow the applicable upstream terms before use or redistribution.

## BlenderKit assets

- Service: [BlenderKit](https://www.blenderkit.com/)
- Terms: [BlenderKit licensing FAQ](https://www.blenderkit.com/docs/licenses/licensing-faq/)
- Use here: the reference scene identifies three externally downloaded models by immutable BlenderKit asset ID.

| Scene object | BlenderKit asset ID |
|---|---|
| Transparent bowl used in internal scenes | `6bd92939-0d93-49e8-bbf6-0a8beda6de0e` |
| Steel mug used in internal scenes | `da62c1a3-ce76-4bb9-8881-6f06729a92d2` |
| A4 paper | `41aebdc7-0c53-4197-98eb-5f6d4a80116a` |
| Scanned food can | `f52953ee-cb81-434d-a6ac-0008bff15508` |
| Ballpoint pen | `2f2c9c79-09c6-44de-9b3f-ae3e2bd3230c` |

The source models are not distributed by this repository. BlenderKit assets may be offered under CC0 or Royalty-Free terms; the applicable license, creator attribution, and access tier must be checked in the asset metadata at download time. Downloaded files remain ignored by Git and must not be committed here.

## Rendered previews

The following committed images are outputs of CreativeContactBench's scene composition and rendering code, but include visual representations of external assets:

| Image | External content represented |
|---|---|
| `docs/images/raytraced_tabletop_preview.png` | RobotSmith xArm geometry and environment assets |
| `docs/images/reference_robot_paper_can_pen_photoreal.jpg` | RobotSmith xArm/environment assets, RoboWits table assets, and the BlenderKit models identified above |

The repository's Apache-2.0 license applies to the original code and documentation, not to rights in the depicted external components. Do not treat the preview images as a source for extracting or redistributing the underlying models or textures.

## Other dependencies

Python packages and optional renderer components installed from package registries or upstream repositories retain their own licenses. Their appearance in a dependency manifest does not incorporate their source code into CreativeContactBench or relicense them under Apache-2.0.

## Multimodal contact extension

The `multimodal/` pressure projection and rendering code is project-authored. It processes saved rigid-contact forces from Genesis local proxy probes and refers to the Panda hand supplied by the Genesis asset installation. No upstream Panda URDF, robot meshes or original scene asset binaries are redistributed by this code release. Existing upstream terms still apply. The separate private Hugging Face input bundle does not change dataset or source-asset licensing.
