#!/usr/bin/env bash
# 2026-11-19 sanity checks: self-read and QA-pair memory (README amendment "Two more Variant 1 sanity checks").
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
O=runs/delta-rule/sanity3
"$PY" $E/sanity_selfread.py --configs A,B,W,WB --output $O/selfread.json
COMMON=(--banks-dir runs/delta-rule/banks-v2 --rule delta --beta 1 --alpha 1 --center-keys --sink-prefix)
declare -A CFG=([A]="" [B]="--key-expand 16 --center-queries")
for c in A B; do
  for g in 0.05 0.1 0.2 0.5 1; do
    "$PY" $E/run_banks.py "${COMMON[@]}" ${CFG[$c]} --gate $g --split calibration --limit 512 \
      --ladder 1 --conditions qa_memory,shuffled_qa_memory --output-dir $O/calib --tag $c-g$g
  done
  gate=$("$PY" $E/select_capacity_gate.py $O/calib $c 1 qa_memory --skip-collapsed)
  echo "SELECTED $c gate $gate"
  "$PY" $E/run_banks.py "${COMMON[@]}" ${CFG[$c]} --gate "$gate" --split eval --ladder 1,4,16,64 \
    --conditions closed_book,qa_memory,shuffled_qa_memory,counterfactual_qa_memory --output-dir $O/eval-$c --tag $c
  "$PY" $E/run_banks.py "${COMMON[@]}" ${CFG[$c]} --gate "$gate" --split eval --ladder 1 \
    --conditions memory --output-dir $O/eval-$c --tag $c
done
echo "SANITY3 QUEUE DONE"
