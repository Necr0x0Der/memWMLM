#!/usr/bin/env bash
# Variant 2: train on WikiText cloze, then evaluate write-once on the global
# TriviaQA corpus (README amendment 2026-11-13). Usage: run_v2.sh SEED "WIDTHS"
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
for d in $2; do
  out=runs/delta-rule/v2/d$d-s$1
  "$PY" $E/train_v2.py --output-dir "$out" --width "$d" --seed "$1"
  "$PY" $E/eval_v2.py --checkpoint "$out/interface.pt"
done
