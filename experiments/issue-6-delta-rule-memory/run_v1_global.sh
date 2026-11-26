#!/usr/bin/env bash
# Global-memory Variant 1 runs (README amendment 2026-11-13). Usage: run_v1_global.sh SEED [frozen]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
T=experiments/issue-6-delta-rule-memory/train_v1_global.py
O=runs/delta-rule/global-v1/runs
s=$1
D=(--rule delta --beta 0.5)
H=(--rule hebbian --beta 1 --beta-norm tokens)
if [[ "${2:-}" == frozen ]]; then
  "$PY" "$T" --output-dir $O/frozen-delta-g0.3 --stage frozen "${D[@]}" --gate 0.3 --references
  "$PY" "$T" --output-dir $O/frozen-hebb-g2 --stage frozen "${H[@]}" --gate 2
  "$PY" "$T" --output-dir $O/frozen-hebb-g4 --stage frozen "${H[@]}" --gate 4
fi
"$PY" "$T" --output-dir $O/gate-delta-s$s --stage gate "${D[@]}" --gate 0.3 --seed $s
"$PY" "$T" --output-dir $O/gate-hebb-s$s --stage gate "${H[@]}" --gate 2 --seed $s
"$PY" "$T" --output-dir $O/lora-delta-s$s --stage lora "${D[@]}" --gate 0.3 --seed $s
"$PY" "$T" --output-dir $O/lora-hebb-s$s --stage lora "${H[@]}" --gate 2 --seed $s
"$PY" "$T" --output-dir $O/lora-null-s$s --stage lora "${D[@]}" --gate 0.3 --seed $s --train-null
