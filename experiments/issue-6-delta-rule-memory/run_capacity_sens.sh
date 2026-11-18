#!/usr/bin/env bash
# 2026-11-18 1.7B capacity-test sensitivity check (README amendment of the same name).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
COMMON=(--banks-dir runs/delta-rule/banks-v2 --rule delta --beta 1 --alpha 1 --center-keys --sink-prefix)
O=runs/delta-rule/capacity/1.7b
for run in "A 0.1" "A 0.2" "B 0.2"; do
  read -r c g <<< "$run"
  X=(); [[ $c == B ]] && X=(--key-expand 16 --center-queries)
  "$PY" $E/run_banks.py "${COMMON[@]}" "${X[@]}" --gate "$g" --split eval --ladder 1,4,16,64 \
    --conditions memory,shuffled_memory,counterfactual_memory --output-dir $O/sens-$c-g$g --tag $c
done
echo "SENS QUEUE DONE"
