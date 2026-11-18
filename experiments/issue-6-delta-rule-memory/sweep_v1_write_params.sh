#!/usr/bin/env bash
# Write-parameter sweep (README amendment 2026-11-12). Calibration questions
# 384..895 only. Usage: sweep_v1_write_params.sh A|B|C [args for stage B]
#   B needs: BETA and three gates, e.g.  B 0.3 "0.2 0.3 0.5"
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
RUN=experiments/issue-6-delta-rule-memory/run_banks.py
COMMON=(--banks-dir runs/delta-rule/banks-v2 --split calibration --start 384 --limit 512
        --ladder 1,2,4,16 --output-dir runs/delta-rule/v1-write-sweep --center-keys
        --conditions memory,shuffled_memory)
case "$1" in
  A)
    for beta in 1 0.5 0.3 0.1; do
      for gate in 0.1 0.2 0.3 0.5 0.8 1.2; do
        "$PY" "$RUN" "${COMMON[@]}" --rule delta --beta "$beta" --gate "$gate" --tag "d-b${beta}-a1-g${gate}"
      done
    done ;;
  B)
    beta=$2
    for alpha in 0.999 0.99 0.95; do
      for gate in $3; do
        "$PY" "$RUN" "${COMMON[@]}" --rule delta --beta "$beta" --alpha "$alpha" --gate "$gate" \
          --tag "d-b${beta}-a${alpha}-g${gate}"
      done
    done ;;
  C)
    for gate in 0.5 1 2 4 8 16; do
      "$PY" "$RUN" "${COMMON[@]}" --rule hebbian --beta-norm tokens --gate "$gate" --tag "h-bT-a1-g${gate}"
    done ;;
esac
