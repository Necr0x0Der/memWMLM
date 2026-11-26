import sys
sys.path.insert(0, "experiments/issue-6-delta-rule-memory")
from pathlib import Path
from analyze import load, bootstrap_diff
D = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/delta-rule/knn-upper/1.7b")
B = Path("runs/delta-rule/eval-baselines")
cb = load(D / "closed_book-N1.jsonl")
ids = sorted(cb)
em = lambda r: 100 * sum(r[i]["exact_match"] for i in ids) / len(ids)
def ci(a, b):
    lo, hi = bootstrap_diff([a[i]["exact_match"] for i in ids], [b[i]["exact_match"] for i in ids], 10000, 20261111)
    return f"{em(a)-em(b):+.1f} [{lo:+.1f}, {hi:+.1f}]", lo
print(f"closed-book {em(cb):.1f}; oracle passage {em(load(B/'oracle_passage-N1.jsonl')):.1f}")
nstar = 0
for n in (1, 4, 16, 64, 256, 1024):
    f = D / f"memory-knn-k4-N{n}.jsonl"
    if not f.exists(): continue
    mem, sh, cf = load(f), load(D / f"shuffled_memory-knn-k4-N{n}.jsonl"), load(D / f"counterfactual_memory-knn-k4-N{n}.jsonl")
    s_sh, lo1 = ci(mem, sh); s_cf, lo2 = ci(mem, cf); s_cb, _ = ci(mem, cb)
    valid = [i for i in ids if cf[i].get("cf_valid")]
    follow = 100 * sum(cf[i]["cf_follow_exact_match"] for i in valid) / len(valid)
    refs = []
    for name in (f"rag_bm25@1-N{n}", f"full_context-N{n}"):
        p = B / f"{name}.jsonl"
        if p.exists(): refs.append(f"{name.split('-')[0]} {em(load(p)):.1f}")
    if lo1 > 0 and lo2 > 0: nstar = n
    print(f"N={n:5d} mem {em(mem):5.1f} shuf {em(sh):5.1f} cf {em(cf):5.1f} follow {follow:4.1f} | "
          f"m-shuf {s_sh} | m-cf {s_cf} | m-closed {s_cb} | {'; '.join(refs)}")
print("N* =", nstar)
