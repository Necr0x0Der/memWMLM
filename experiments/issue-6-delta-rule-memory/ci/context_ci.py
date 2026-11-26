import sys
sys.path.insert(0, "experiments/issue-6-delta-rule-memory")
from pathlib import Path
from analyze import load, bootstrap_diff
R = Path("runs/delta-rule/lora-banks/1.7b"); FB = Path("runs/delta-rule/eval-baselines")
C = R / "context-tbeta"
ids = sorted(load(FB / "closed_book-N1.jsonl"))
em = lambda r, k="exact_match": 100 * sum(r[i][k] for i in ids) / len(ids)
def ci(a, b):
    lo, hi = bootstrap_diff([a[i]["exact_match"] for i in ids], [b[i]["exact_match"] for i in ids], 10000, 20261111)
    return f"{em(a)-em(b):+5.1f} [{lo:+.1f},{hi:+.1f}]"
rep = load(R / "context-frozen-repro/oracle_passage-N1.jsonl"); orig = load(FB / "oracle_passage-N1.jsonl")
print("frozen oracle repro", em(rep), "orig", em(orig), "same preds", sum(rep[i]["prediction"] == orig[i]["prediction"] for i in ids))
print("frozen closed", em(load(FB / "closed_book-N1.jsonl")), "| LoRA closed", em(load(C / "closed_book-N1.jsonl")))
fo, lo_ = load(FB / "counterfactual_oracle-N1.jsonl"), load(C / "counterfactual_oracle-N1.jsonl")
print("cf passage in context: frozen EM", em(fo), "follow", em(fo, "cf_follow_exact_match"), "| LoRA EM", em(lo_), "follow", em(lo_, "cf_follow_exact_match"))
for n in (1, 4, 16, 64):
    fz = load(FB / ("oracle_passage-N1.jsonl" if n == 1 else f"full_context-N{n}.jsonl"))
    lc = load(C / f"full_context-N{n}.jsonl")
    m2 = load(R / f"eval-tbeta-none/memory-Lt-none-N{n}.jsonl"); m4 = load(R / f"eval-dtbeta-none/memory-Ld-none-N{n}.jsonl")
    m1 = load(R / f"eval-none/memory-L-none-N{n}.jsonl")
    print(f"N={n:2} ctx frozen {em(fz):5.1f} ctx LoRA {em(lc):5.1f} | mem fixed {em(m1):5.1f} opt2 {em(m2):5.1f} big {em(m4):5.1f}"
          f" | opt2 - ctxLoRA {ci(m2, lc)} | ctxLoRA - ctxfrozen {ci(lc, fz)}")
