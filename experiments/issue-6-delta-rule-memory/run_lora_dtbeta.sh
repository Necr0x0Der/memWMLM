#!/usr/bin/env bash
# 2026-11-24 per-token beta + bigger memory (config B, DPFP nu 16) (README amendment "per-token β + bigger memory").
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
O=runs/delta-rule/lora-banks/1.7b
B=(--key-expand 16 --center-queries)
for c in none unrelated; do
  [[ -f $O/lora-dtbeta-$c.pt ]] || "$PY" $E/train_lora_banks.py --learn-beta token --lr-beta 1e-4 "${B[@]}" \
    --control $c --output $O/lora-dtbeta-$c.pt
  echo "TRAINED dtbeta $c"
done
for c in none unrelated; do
  "$PY" $E/run_banks.py --banks-dir runs/delta-rule/banks-v2 --rule delta --beta 1 --alpha 1 --sink-prefix \
    --center-keys "${B[@]}" --lora $O/lora-dtbeta-$c.pt --split eval --ladder 1,4,16,64 \
    --conditions closed_book,memory,shuffled_memory,counterfactual_memory --output-dir $O/eval-dtbeta-$c --tag Ld-$c
  echo "EVAL dtbeta $c DONE"
done
echo "LORA DPFP TOKEN-BETA QUEUE DONE"
