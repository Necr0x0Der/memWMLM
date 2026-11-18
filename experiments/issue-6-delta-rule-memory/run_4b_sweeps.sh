#!/usr/bin/env bash
# 2026-11-17 Qwen3-4B: remaining 1.7B sweeps (README amendment "4B coverage").
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
M=(--model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c)
O=runs/delta-rule/qwen3-4b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 1. V2 pre-registered recipe at the other widths (hidden cache L18 exists)
for d in 512 2048 4096; do
  out=$O/v2/d$d-s0
  C=(); T=(); [[ $d == 4096 ]] && C=(--chunk-size 512) && T=(--micro-batch 8 --chunk-size 2048)  # 4096: OOM (README note)
  [[ -f $out/interface.pt ]] || "$PY" $E/train_v2.py "${M[@]}" --layer 18 --output-dir "$out" --width "$d" --seed 0 "${T[@]}"
  [[ -f $out/eval/meta.json ]] || "$PY" $E/eval_v2.py --checkpoint "$out/interface.pt" "${C[@]}"
  [[ -f $out/probe_v2.json ]] || "$PY" $E/probe_v2.py --checkpoint "$out/interface.pt" --sizes 1,4,64,1024 || echo "PROBE FAILED d$d"
done
# 2. V1 expanded keys, frozen (1.7B grids)
for nu in 2 4 8 16; do
  D=$((256 * nu))
  out=$O/global/expand/delta-D$D
  [[ -f $out/meta.json ]] || "$PY" $E/train_v1_global.py "${M[@]}" --stage frozen --key-expand "$nu" --rule delta --beta 0.5 \
    --gate 1 --gates 0.1,0.3,1,3 --output-dir "$out"
  out=$O/global/expand/hebb-D$D
  [[ -f $out/meta.json ]] || "$PY" $E/train_v1_global.py "${M[@]}" --stage frozen --key-expand "$nu" --rule hebbian --beta 1 --beta-norm tokens \
    --gate 1 --gates 2,8,32,128 --output-dir "$out"
done
