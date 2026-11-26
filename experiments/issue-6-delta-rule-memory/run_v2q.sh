#!/usr/bin/env bash
# 2026-11-25 Variant 2 step B (README amendment "Variant 2, step B").
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
O=runs/delta-rule/v2q
V1=runs/delta-rule/lora-banks/1.7b/lora-tbeta-none.pt
COMMON=(--banks-dir runs/delta-rule/banks-v2 --rule delta --beta 1 --alpha 1 --sink-prefix --center-keys --split eval
        --ladder 1,4,16,64 --conditions closed_book,memory,shuffled_memory,counterfactual_memory)
for c in none unrelated; do
  [[ -f $O/v2q-$c.pt ]] || "$PY" $E/train_lora_banks.py --task cloze --writer-rank 8 --learn-beta token --lr-beta 1e-4 \
    --control $c --output $O/v2q-$c.pt
  echo "TRAINED v2q $c"
done
for c in none unrelated; do
  "$PY" $E/run_banks.py "${COMMON[@]}" --lora $O/v2q-$c.pt --output-dir $O/eval-$c --tag B-$c
  echo "EVAL v2q $c DONE"
done
"$PY" $E/run_banks.py "${COMMON[@]}" --lora $O/v2q-none.pt --extra-lora $V1 --output-dir $O/eval-none-v1lora --tag B-v1lora
echo "V2Q QUEUE DONE"
