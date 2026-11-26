#!/usr/bin/env bash
# Variant 1 (frozen) refinement around the all-heads gate ~0.25 point found by
# sweep_v1_frozen.sh. Fresh calibration slice: questions 384..895 (512),
# disjoint from the probe (0..255), the first sweep (256..383) and eval.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
RUN=experiments/issue-6-delta-rule-memory/run_banks.py
COMMON=(--banks-dir runs/delta-rule/banks-v2 --split calibration --start 384 --limit 512
        --ladder 1,2,4 --output-dir runs/delta-rule/v1-refine --center-keys --rule delta)
"$PY" "$RUN" "${COMMON[@]}" --conditions closed_book,null_memory --tag base
for gate in 0.15 0.2 0.25 0.3 0.35; do
  "$PY" "$RUN" "${COMMON[@]}" --conditions memory,shuffled_memory --gate "$gate" --tag "hall-g${gate}"
done
