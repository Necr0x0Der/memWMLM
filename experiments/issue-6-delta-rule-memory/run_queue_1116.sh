#!/usr/bin/env bash
# 2026-11-16 queue: pre-registered V2 at d = 4096 (chunk 512), then the V1 expanded-key sweep.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
out=runs/delta-rule/v2/d4096-s0
rm -rf "$out"
"$PY" $E/train_v2.py --output-dir "$out" --width 4096 --seed 0 --chunk-size 512
"$PY" $E/eval_v2.py --checkpoint "$out/interface.pt" --chunk-size 512
$E/run_v1_expand.sh
