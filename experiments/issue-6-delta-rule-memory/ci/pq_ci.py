import sys
sys.path.insert(0, "experiments/issue-6-delta-rule-memory")
from pathlib import Path
from analyze import load, bootstrap_diff
R = Path("runs/delta-rule/passage-query/1.7b")
C = "passage_query_memory"
cb = load(R / "eval-K/closed_book-N1.jsonl"); ids = sorted(cb)
em = lambda r: 100 * sum(r[i]["exact_match"] for i in ids) / len(ids)
def ci(a, b):
    lo, hi = bootstrap_diff([a[i]["exact_match"] for i in ids], [b[i]["exact_match"] for i in ids], 10000, 20261111)
    return f"{em(a)-em(b):+5.1f} [{lo:+.1f},{hi:+.1f}]", lo
for c in ("K", "A", "B", "WB"):
    nstar = 0
    for n in (1, 4, 16, 64):
        t = c if c == "K" else f"{c}-N{n}"
        d = R / f"eval-{c}"
        m, s, f = (load(d / f"{p}{C}-{t}-N{n}.jsonl") for p in ("", "shuffled_", "counterfactual_"))
        a, l1 = ci(m, s); b, l2 = ci(m, f); k, l3 = ci(m, cb)
        fol = 100 * sum(f[i].get("cf_follow_exact_match", 0) for i in ids) / len(ids)
        if l1 > 0 and l2 > 0: nstar = n
        print(f"{c:2} N={n:2} mem {em(m):5.1f} shuf {em(s):5.1f} cf {em(f):5.1f} follow {fol:4.1f} | m-shuf {a} | m-cf {b} | m-closed {k}")
    print(f"{c} N* = {nstar}")
