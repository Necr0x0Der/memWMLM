import sys
sys.path.insert(0, "experiments/issue-6-delta-rule-memory")
from pathlib import Path
from analyze import load, bootstrap_diff
O = Path("runs/delta-rule/v2-banks")
def em(r, k="exact_match"): return 100 * sum(x[k] for x in r.values()) / len(r)
def ci(a, b):
    ids = sorted(a); lo, hi = bootstrap_diff([a[i]["exact_match"] for i in ids], [b[i]["exact_match"] for i in ids], 10000, 20261111)
    return em(a) - em(b), lo, hi
for run in ("v2b", "d512", "d1024", "d2048", "d4096"):
    for suf in ("", "-lora"):
        d = O / (run + suf)
        if not (d / "counterfactual_memory-N64.jsonl").exists(): print(d, "missing"); continue
        cb = load(d / "closed_book-N1.jsonl"); nstar = 0; help_ = []
        print(f"{run+suf}: closed-book {em(cb):.1f}")
        for n in (1, 4, 16, 64):
            m, s, f = (load(d / f"{c}-N{n}.jsonl") for c in ("memory", "shuffled_memory", "counterfactual_memory"))
            a, b, c = ci(m, s), ci(m, f), ci(m, cb)
            if a[1] > 0 and b[1] > 0:
                nstar = n
                if c[1] > 0: help_.append(n)
            fmt = lambda t: f"{t[0]:+5.1f} [{t[1]:+.1f},{t[2]:+.1f}]"
            print(f"  N={n:2} mem {em(m):5.1f} shuf {em(s):5.1f} cf {em(f):5.1f} fol {em(f,'cf_follow_exact_match'):3.1f} | m-s {fmt(a)} | m-cf {fmt(b)} | m-closed {fmt(c)}")
        print(f"  retrieves up to N* = {nstar}; helps at {help_}")
