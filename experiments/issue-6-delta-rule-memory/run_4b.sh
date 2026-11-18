#!/usr/bin/env bash
# 2026-11-16 Qwen3-4B check (README amendment "Optional check: Qwen3-4B").
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
E=experiments/issue-6-delta-rule-memory
M=(--model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c)
O=runs/delta-rule/qwen3-4b
G=$O/global
D=(--rule delta --beta 0.5)
H=(--rule hebbian --beta 1 --beta-norm tokens)
A=(--micro-batch 8)  # 4B: whole batch of 16 exceeds 24 GB (token-weighted accumulation)
# skip runs that already finished (queue restarted after the first OOM)
V1() { local out=$1; shift; [[ -f $out/meta.json ]] || "$PY" $E/train_v1_global.py "${M[@]}" --output-dir "$out" "$@"; }
# 1-2. references + frozen V1 (gate chosen on train_seen afterwards)
V1 $G/frozen-delta --stage frozen "${D[@]}" --gate 0.3 --gates 0.1,0.3,1 --references
V1 $G/frozen-hebb --stage frozen "${H[@]}" --gate 2 --gates 1,2,4,8
# 3. adapted V1, seeds 0/1/2
for s in 0 1 2; do
  V1 $G/gate-delta-s$s --stage gate "${D[@]}" --gate 0.3 --seed $s "${A[@]}"
  V1 $G/gate-hebb-s$s --stage gate "${H[@]}" --gate 2 --seed $s "${A[@]}"
  V1 $G/lora-delta-s$s --stage lora "${D[@]}" --gate 0.3 --seed $s "${A[@]}"
  V1 $G/lora-hebb-s$s --stage lora "${H[@]}" --gate 2 --seed $s "${A[@]}"
  V1 $G/lora-null-s$s --stage lora "${D[@]}" --gate 0.3 --seed $s --train-null "${A[@]}"
done
# 4. small banks, frozen pre-registered delta
"$PY" $E/run_banks.py "${M[@]}" --banks-dir runs/delta-rule/banks-v2 --output-dir $O/banks --ladder 1,4,16 \
  --conditions closed_book,oracle_passage,counterfactual_oracle,memory,shuffled_memory,counterfactual_memory \
  --rule delta --beta 1 --alpha 1 --gate 0.2 --center-keys
# 5. V2 pre-registered recipe, d = 1024, layer 18
"$PY" $E/cache_hidden.py "${M[@]}" --layer 18
"$PY" $E/train_v2.py "${M[@]}" --layer 18 --output-dir $O/v2/d1024-s0 --width 1024 --seed 0
"$PY" $E/eval_v2.py --checkpoint $O/v2/d1024-s0/interface.pt
"$PY" $E/probe_v2.py --checkpoint $O/v2/d1024-s0/interface.pt --sizes 1,4,64,1024
# 6. V2b final configuration + decision-rule probe
"$PY" $E/train_v2.py "${M[@]}" --layer 18 --output-dir $O/v2b/d1024-fewshot --width 1024 --seed 0 \
  --steps 1000 --bank-sizes 1,4,16 --lr 1e-3 --shift-values --standardise --fewshot 3 --gap-sizes 16,64 --micro-batch 8
"$PY" $E/probe_v2.py --checkpoint $O/v2b/d1024-fewshot/interface.pt --sizes 1,4,64,1024
