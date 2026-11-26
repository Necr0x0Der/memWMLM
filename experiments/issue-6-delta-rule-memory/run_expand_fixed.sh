#!/usr/bin/env bash
# 2026-11-18 global expanded keys (D 4096) with the sink and query-centring fixes (README amendment).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
F=(--key-expand 16 --sink-prefix --center-queries --stage frozen)
for model in 1.7b 4b; do
  M=(); [[ $model == 4b ]] && M=(--model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c)
  O=runs/delta-rule/expand-fixed/$model
  [[ -f $O/delta-D4096/meta.json ]] || "$PY" $E/train_v1_global.py "${M[@]}" "${F[@]}" --rule delta --beta 0.5 \
    --gate 1 --gates 0.1,0.3,1,3 --output-dir $O/delta-D4096
  [[ -f $O/hebb-D4096/meta.json ]] || "$PY" $E/train_v1_global.py "${M[@]}" "${F[@]}" --rule hebbian --beta 1 --beta-norm tokens \
    --gate 1 --gates 2,8,32,128 --output-dir $O/hebb-D4096
done
echo "EXPAND-FIXED QUEUE DONE"
