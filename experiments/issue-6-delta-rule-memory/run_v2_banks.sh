#!/usr/bin/env bash
# 2026-11-25 Variant 2 on small banks, step A (README amendment "Variant 2 on small banks").
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
O=runs/delta-rule/v2-banks
L=runs/delta-rule/lora-banks/1.7b/lora-tbeta-none.pt
declare -A CK=([d512]=runs/delta-rule/v2/d512-s0 [d1024]=runs/delta-rule/v2/d1024-s0 [d2048]=runs/delta-rule/v2/d2048-s0
               [d4096]=runs/delta-rule/v2/d4096-s0 [v2b]=runs/delta-rule/v2b-pilot/d1024-fewshot)
for c in v2b d512 d1024 d2048 d4096; do
  "$PY" $E/eval_v2_banks.py --checkpoint ${CK[$c]}/interface.pt --output-dir $O/$c
  "$PY" $E/eval_v2_banks.py --checkpoint ${CK[$c]}/interface.pt --lora $L --output-dir $O/$c-lora
  echo "V2 BANKS $c DONE"
done
echo "V2 BANKS QUEUE DONE"
