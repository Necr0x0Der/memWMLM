#!/usr/bin/env bash
# 2026-11-20 Variant 1T: evaluate trained feature maps (README amendment "Trained retrieval for the delta-rule memory").
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
FM=runs/delta-rule/featmap/1.7b/fm.pt
"$PY" $E/run_banks.py --banks-dir runs/delta-rule/banks-v2 --rule delta --beta 1 --alpha 1 --sink-prefix \
  --feature-map $FM --gate 1 --split eval --ladder 1,4,16,64 \
  --conditions closed_book,memory,shuffled_memory,counterfactual_memory \
  --output-dir runs/delta-rule/featmap/1.7b/eval-T --tag T
"$PY" $E/sanity_cloze.py --configs T --feature-map $FM --output-dir runs/delta-rule/cloze
echo "TRAINED QUEUE DONE"
