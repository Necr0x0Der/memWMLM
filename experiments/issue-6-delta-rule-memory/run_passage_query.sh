#!/usr/bin/env bash
# 2026-11-23 sanity check 6 (README amendment "answer the question, retrieve with the passage").
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
O=runs/delta-rule/passage-query/1.7b
COMMON=(--banks-dir runs/delta-rule/banks-v2 --rule delta --beta 1 --alpha 1 --sink-prefix)
C=passage_query_memory
declare -A CFG=([A]="--center-keys" [B]="--center-keys --center-queries --key-expand 16"
                [WB]="--center-keys --center-queries --whiten --whiten-eps 0.1 --key-expand 16")
# upper bound first: uncompressed softmax memory, no gate
"$PY" $E/run_banks.py "${COMMON[@]}" --read knn --knn-top-k 4 --split eval --ladder 1,4,16,64 \
  --conditions closed_book,$C,shuffled_$C,counterfactual_$C --output-dir $O/eval-K --tag K
for c in A B WB; do
  for n in 1 4 16 64; do
    for g in 0.05 0.1 0.2 0.5 1; do
      "$PY" $E/run_banks.py "${COMMON[@]}" ${CFG[$c]} --gate $g --split calibration --limit 512 \
        --ladder $n --conditions $C,shuffled_$C --output-dir $O/calib --tag $c-g$g
    done
    gate=$("$PY" $E/select_capacity_gate.py $O/calib $c $n $C --skip-collapsed)
    echo "SELECTED $c N=$n gate $gate"
    "$PY" $E/run_banks.py "${COMMON[@]}" ${CFG[$c]} --gate "$gate" --split eval --ladder $n \
      --conditions closed_book,$C,shuffled_$C,counterfactual_$C --output-dir $O/eval-$c --tag $c-N$n
  done
done
echo "PASSAGE QUERY QUEUE DONE"
