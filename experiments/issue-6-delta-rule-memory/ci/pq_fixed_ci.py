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
    return f"{em(a)-em(b):+5.1f} [{lo:+.1f},{hi:+.1f}]"
for c in ("A", "B", "WB"):
    for n in (1, 4, 16, 64):
        d = R / "fixed-g0.05"
        m, s, f = (load(d / f"{p}{C}-{c}-fixed-N{n}.jsonl") for p in ("", "shuffled_", "counterfactual_"))
        fol = 100 * sum(f[i]["cf_follow_exact_match"] for i in ids) / len(ids)
        print(f"{c:2} N={n:2} mem {em(m):5.1f} shuf {em(s):5.1f} cf {em(f):5.1f} follow {fol:3.1f} | m-shuf {ci(m,s)} | m-cf {ci(m,f)} | m-closed {ci(m,cb)}")
