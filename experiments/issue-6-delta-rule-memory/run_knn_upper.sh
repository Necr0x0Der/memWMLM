#!/usr/bin/env bash
# 2026-11-19 upper bound for Variant 1 (README amendment "Upper bound for Variant 1").
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python experiments/issue-6-delta-rule-memory/run_banks.py --banks-dir runs/delta-rule/banks-v2 \
  --split eval --ladder 1,4,16,64,256,1024 \
  --conditions closed_book,memory,shuffled_memory,counterfactual_memory \
  --read knn --knn-top-k 4 --layers all --sink-prefix \
  --output-dir runs/delta-rule/knn-upper/1.7b --tag knn-k4
echo "KNN QUEUE DONE"
