# Analysis scripts for the post-PR checks (2026-11-19 → 2026-11-25)

Each script reads per-question predictions under `runs/delta-rule/` (local,
gitignored) and prints the tables in `NOTES.md`: EM, paired bootstrap 95%
CIs (10,000 resamples, seed 20261111, `analyze.bootstrap_diff`) and N\*.
Run from the repository root, e.g.
`.venv/bin/python experiments/issue-6-delta-rule-memory/ci/v2q_ci.py`.

| script | NOTES section |
|---|---|
| `knn_ci.py` | Upper bound for V1: uncompressed key–value memory |
| `whiten_ci.py` | Whitening V1 keys: QA test |
| `sanity3_ci.py` | Sanity check 4 (QA-pair memory) |
| `cloze_ci.py` | Sanity check 5 (cloze) |
| `pq_ci.py`, `pq_fixed_ci.py` | Sanity check 6 (+ fixed-gate row) |
| `pq_wiring.py`, `pq_what.py` | Check 6: code verification and why it fails (load Qwen) |
| `lora_ci.py` | V1-LoRA on small banks |
| `lora_beta_ci.py` | V1-LoRA + learned β (option 1) |
| `lora_tbeta_ci.py` | V1-LoRA + per-token β (option 2) |
| `lora_dtbeta_ci.py` | per-token β + bigger memory |
| `context_ci.py` | passages in the context instead of memory |
| `v2banks_ci.py` | Variant 2 on small banks (step A) |
| `v2q_ci.py` | Variant 2 step B |
