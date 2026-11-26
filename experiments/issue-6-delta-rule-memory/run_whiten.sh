#!/usr/bin/env bash
# 2026-11-19 whitened keys for Variant 1 (README amendment "Whitened keys for Variant 1").
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
COMMON=(--banks-dir runs/delta-rule/banks-v2 --rule delta --beta 1 --alpha 1 --center-keys --center-queries
        --whiten --whiten-eps 0.1 --sink-prefix)
declare -A CFG=([W]="" [WB]="--key-expand 16")
O=runs/delta-rule/whiten/1.7b
for c in W WB; do
  for g in 0.05 0.1 0.2 0.5 1; do
    "$PY" $E/run_banks.py "${COMMON[@]}" ${CFG[$c]} --gate $g --split calibration --limit 512 \
      --ladder 16 --conditions memory,shuffled_memory --output-dir $O/calib --tag $c-g$g
  done
  gate=$("$PY" $E/select_capacity_gate.py $O/calib $c)
  echo "SELECTED $c gate $gate"
  "$PY" $E/run_banks.py "${COMMON[@]}" ${CFG[$c]} --gate "$gate" --split eval --ladder 1,4,16,64 \
    --conditions closed_book,memory,shuffled_memory,counterfactual_memory --output-dir $O/eval-$c --tag $c
done
echo "WHITEN QUEUE DONE"
