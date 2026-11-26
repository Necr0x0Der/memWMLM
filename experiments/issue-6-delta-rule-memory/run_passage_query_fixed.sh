#!/usr/bin/env bash
# 2026-11-23 sanity check 6, fixed gate 0.05 for every N (README amendment "fixed-gate row").
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
O=runs/delta-rule/passage-query/1.7b
COMMON=(--banks-dir runs/delta-rule/banks-v2 --rule delta --beta 1 --alpha 1 --sink-prefix --gate 0.05)
C=passage_query_memory
declare -A CFG=([A]="--center-keys" [B]="--center-keys --center-queries --key-expand 16"
                [WB]="--center-keys --center-queries --whiten --whiten-eps 0.1 --key-expand 16")
for c in A B WB; do
  "$PY" $E/run_banks.py "${COMMON[@]}" ${CFG[$c]} --split eval --ladder 1,4,16,64 \
    --conditions $C,shuffled_$C,counterfactual_$C --output-dir $O/fixed-g0.05 --tag $c-fixed
  echo "FIXED $c DONE"
done
echo "FIXED QUEUE DONE"
