#!/usr/bin/env bash
# Variant 1 (frozen) calibration sweep. Calibration (train) split only:
# questions 256..383, disjoint from the 256 used by probe_head_retrieval.py
# and from the eval (dev) set. Outputs: runs/delta-rule/v1-sweep/.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
RUN=experiments/issue-6-delta-rule-memory/run_banks.py
BANKS=runs/delta-rule/banks-v2
OUT=runs/delta-rule/v1-sweep
PROBE=runs/delta-rule/probe/head_retrieval.json
COMMON=(--banks-dir "$BANKS" --split calibration --start 256 --limit 128 --ladder 1,16
        --output-dir "$OUT" --center-keys --rule delta)

"$PY" "$RUN" "${COMMON[@]}" --conditions closed_book,null_memory --tag base
for heads in 8 32; do
  for gate in 0.5 1 2 4; do
    "$PY" "$RUN" "${COMMON[@]}" --conditions memory,shuffled_memory \
      --head-probe "$PROBE" --top-heads "$heads" --gate "$gate" --tag "h${heads}-g${gate}"
  done
done
for gate in 0.05 0.1 0.25; do
  "$PY" "$RUN" "${COMMON[@]}" --conditions memory,shuffled_memory --gate "$gate" --tag "hall-g${gate}"
done
