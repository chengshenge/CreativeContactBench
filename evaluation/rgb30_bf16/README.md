# Paper RGB30 BF16 rerun pipeline

This entry point is separate from the repository's generic 67-task evaluation. It prepares four fresh official BF16 Qwen configurations on the exact 30 frozen paper inputs, using v1.2 prompts and **direct holistic rankings**. It does not alter the existing default evaluator.

The code is public. The complete author handoff contains task images/prompts, exact manifests, aggregate human consensus and historical regression fixtures; these unpublished materials are **not uploaded here**. Request `rgb30_bf16_handoff_20261017.zip` from the authors and transfer it privately. No Hugging Face dataset token or individual human submissions are needed for that archive. Do not replace its inputs with a current 67-task dataset download.

## On a new computer

```bash
git clone --branch rgb30-bf16-handoff-v1 --single-branch https://github.com/chengshenge/CreativeContactBench.git
cd CreativeContactBench
python3 evaluation/rgb30_bf16/import_author_bundle.py --zip /path/to/rgb30_bf16_handoff_20261017.zip --output /path/to/new-rgb30-work
```

The importer checks the expected archive SHA-256 before extracting, rejects unsafe paths and refuses to overwrite an existing destination. Then open `CODEX_START_HERE.md` inside the extracted directory for Codex, and follow `README.md`. The full archive is self-contained; cloning is optional when it is already available.

The four planned configs are Qwen3.8-27B BF16 thinking, Qwen3.6-35B-A3B BF16 thinking, Qwen3.5-9B BF16 thinking, and Qwen3-VL-8B-Instruct BF16. Every checkpoint is pinned to its official immutable revision. Two reserved 48GB GPUs are suggested for the larger models; actual deployment must be validated on the destination. There are 120 formal slots and at most 3 transient retries per model, plus separately recorded synthetic deployment checks.

## Validation and scope

Ten offline tests passed, including exact reproduction of the prior two Qwen configurations' 60-response historical tau/Top-1/bootstrap results, input integrity, tie rules, resume and retry caps. Historical fixtures are only in the private handoff; no old response is used for new inference. BF16 GPU deployment has **not** been run or claimed validated by packaging. See [VALIDATION.md](VALIDATION.md) and [RUNNING.md](RUNNING.md).

Task 06/10/13 comparisons retain the RGB30 author-equivalence assumption and lack independent per-participant view hashes. The new thinking and precision settings represent new configurations, not a quantization-only ablation. The private human reference is never part of a model request. Result export keeps raw responses, actual settings/versions, failures and partial coverage; return the generated `rgb30_bf16_results.zip` to the original analysis machine to integrate into the leaderboard.

No current scores, participant data, images, credentials or model weights are published by this directory. Do not use the parent evaluation README's default 67-task command for this experiment.
