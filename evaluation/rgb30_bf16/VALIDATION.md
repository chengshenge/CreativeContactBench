# Validation status

Prepared 2026-10-17. No real model calls, weight downloads, or GPU jobs were started during packaging.

## Passed locally

- 30 task IDs match the original frozen RGB30 manifest. All 90 task/image/prompt files are byte-identical; the original prompt template and response schema are unchanged.
- Ten offline tests cover input tampering, exact request image/text, separation of direct ranking from score-derived ranking, missing/duplicate fields, permitted container-only repairs, model/human top ties, undefined tau, refusal/truncation, retry eligibility, end-to-end mock execution, resume and retry cap.
- The 60 historical responses from the two previous Qwen configurations are used ONLY as offline regression fixtures. The unchanged analysis reproduces their original tau means/95% intervals, strict/inclusive hits/denominators, Top-1 intervals and tie counts to 12 decimal places. No historical result is placed under new runs/.
- End-to-end mock regression uses a temporary directory: 30 slots, 3 extra transient attempts, 26 valid mock slots, resume adds zero requests, and results export/analysis/ZIP succeeds. These mock outputs are deleted with the temporary directory and are not formal results.
- Only explicit aggregate human consensus orderings are included, not individual rankings, identities, participant submissions or credentials.

## Still requires the destination machine

- GPU model, BF16 capability, available RAM/disk, driver/CUDA and GPU interconnect checks.
- Installation/compatibility of pinned vLLM 0.31.0, four pinned official snapshots, actual peak GPU memory, and reasoning-parser behavior on this hardware.
- The synthetic image transport check, all 120 formal calls, and their resulting metrics.

No GPU deployment success or new model score is claimed by this handoff. Failures must remain BLOCKED/invalid without silent model, precision, configuration or prompt changes.
