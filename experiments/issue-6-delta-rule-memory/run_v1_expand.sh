#!/usr/bin/env bash
# Variant 1 extension: frozen Qwen heads with DPFP-expanded keys, global corpus
# (README amendment 2026-11-16). D = 256 * nu. Gate chosen afterwards on
# train_seen (analyze in select_v1_expand.py). Usage: run_v1_expand.sh "NUS"
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
for nu in ${1:-2 4 8 16}; do
  D=$((256 * nu))
  "$PY" $E/train_v1_global.py --stage frozen --key-expand "$nu" --rule delta --beta 0.5 \
    --gate 1 --gates 0.1,0.3,1,3 --output-dir runs/delta-rule/global-v1/expand/delta-D$D
  "$PY" $E/train_v1_global.py --stage frozen --key-expand "$nu" --rule hebbian --beta 1 --beta-norm tokens \
    --gate 1 --gates 2,8,32,128 --output-dir runs/delta-rule/global-v1/expand/hebb-D$D
done
