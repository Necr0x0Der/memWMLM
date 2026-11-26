#!/usr/bin/env bash
# 2026-11-18 capacity test (README amendment "Code corrections and a direct capacity test").
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
COMMON=(--banks-dir runs/delta-rule/banks-v2 --rule delta --beta 1 --alpha 1 --center-keys --sink-prefix)
declare -A CFG=([A]="" [B]="--key-expand 16 --center-queries")
for model in 1.7b 4b; do
  M=(); [[ $model == 4b ]] && M=(--model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c)
  O=runs/delta-rule/capacity/$model
  for c in A B; do
    # 1. gate on the calibration split (N = 16, first 512 questions)
    for g in 0.05 0.1 0.2 0.5 1; do
      "$PY" $E/run_banks.py "${M[@]}" "${COMMON[@]}" ${CFG[$c]} --gate $g --split calibration --limit 512 \
        --ladder 16 --conditions memory,shuffled_memory --output-dir $O/calib --tag $c-g$g
    done
    gate=$("$PY" $E/select_capacity_gate.py $O/calib $c)
    echo "SELECTED $model $c gate $gate"
    # 2. eval
    "$PY" $E/run_banks.py "${M[@]}" "${COMMON[@]}" ${CFG[$c]} --gate "$gate" --split eval --ladder 1,4,16,64 \
      --conditions closed_book,memory,shuffled_memory,counterfactual_memory --output-dir $O/eval-$c --tag $c
  done
done
echo "CAPACITY QUEUE DONE"
