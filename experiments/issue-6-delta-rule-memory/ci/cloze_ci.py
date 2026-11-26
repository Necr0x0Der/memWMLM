import json, sys
sys.path.insert(0, "experiments/issue-6-delta-rule-memory")
from pathlib import Path
from analyze import bootstrap_diff
D = Path("runs/delta-rule/cloze/eval")
L = lambda f: {r["question_id"]: r for r in map(json.loads, open(f))}
none, orc = L(D / "none.jsonl"), L(D / "oracle.jsonl"); ids = sorted(none)
rate = lambda d, k="hit": 100 * sum(d[i][k] for i in ids) / len(ids)
print(f"items {len(ids)}  none {rate(none):.1f}  oracle {rate(orc):.1f}")
def ci(a, b):
    lo, hi = bootstrap_diff([a[i]["hit"] for i in ids], [b[i]["hit"] for i in ids], 10000, 20261111)
    return f"{rate(a)-rate(b):+.1f} [{lo:+.1f}, {hi:+.1f}]", lo
for c in ("A", "B", "WB", "K"):
    nstar = 0
    for n in (1, 4, 16, 64):
        f = D / f"{c}-memory-N{n}.jsonl"
        if not f.exists(): continue
        m, s, cf = L(f), L(D / f"{c}-shuffled-N{n}.jsonl"), L(D / f"{c}-counterfactual-N{n}.jsonl")
        a, l1 = ci(m, s); b, l2 = ci(m, cf); x, _ = ci(m, none)
        if l1 > 0 and l2 > 0: nstar = n
        print(f"{c:2s} N={n:3d} mem {rate(m):5.1f} shuf {rate(s):5.1f} cf {rate(cf):5.1f} (follow {rate(cf,'follow'):5.1f}) | m-shuf {a} | m-cf {b} | m-none {x}")
    print(f"{c} N* = {nstar}")
