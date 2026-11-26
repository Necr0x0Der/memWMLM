#!/usr/bin/env bash
# 2026-11-24 V1-LoRA + learned beta, option 1 (README amendment "V1-LoRA + learned β, option 1").
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
O=runs/delta-rule/lora-banks/1.7b
for c in none unrelated; do
  [[ -f $O/lora-beta-$c.pt ]] || "$PY" $E/train_lora_banks.py --learn-beta head --control $c --output $O/lora-beta-$c.pt
  echo "TRAINED beta $c"
done
for c in none unrelated; do
  "$PY" $E/run_banks.py --banks-dir runs/delta-rule/banks-v2 --rule delta --beta 1 --alpha 1 --sink-prefix \
    --center-keys --lora $O/lora-beta-$c.pt --split eval --ladder 1,4,16,64 \
    --conditions closed_book,memory,shuffled_memory,counterfactual_memory --output-dir $O/eval-beta-$c --tag Lb-$c
  echo "EVAL beta $c DONE"
done
echo "LORA BETA QUEUE DONE"
