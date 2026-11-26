#!/usr/bin/env bash
# 2026-11-24 sanity check: passages in context (README amendment "passages in the context instead of memory").
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
O=runs/delta-rule/lora-banks/1.7b
COMMON=(--banks-dir runs/delta-rule/banks-v2 --split eval)
"$PY" $E/run_banks.py "${COMMON[@]}" --ladder 1 --conditions oracle_passage --output-dir $O/context-frozen-repro
echo "FROZEN REPRO DONE"
"$PY" $E/run_banks.py "${COMMON[@]}" --lora $O/lora-tbeta-none.pt --ladder 1,4,16,64 \
  --conditions closed_book,oracle_passage,counterfactual_oracle,full_context --output-dir $O/context-tbeta
echo "CONTEXT CHECK DONE"
