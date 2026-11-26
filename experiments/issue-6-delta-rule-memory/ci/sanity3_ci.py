import sys
sys.path.insert(0, "experiments/issue-6-delta-rule-memory")
from pathlib import Path
from analyze import load, bootstrap_diff
for c in ("A", "B"):
    D = Path(f"runs/delta-rule/sanity3/eval-{c}")
    if not (D / "closed_book-N1.jsonl").exists(): continue
    cb = load(D / "closed_book-N1.jsonl"); ids = sorted(cb)
    em = lambda r, k="exact_match": 100 * sum(r[i][k] for i in ids) / len(ids)
    def ci(a, b):
        lo, hi = bootstrap_diff([a[i]["exact_match"] for i in ids], [b[i]["exact_match"] for i in ids], 10000, 20261111)
        return f"{em(a)-em(b):+.1f} [{lo:+.1f}, {hi:+.1f}]", lo
    print(f"{c}: closed-book {em(cb):.1f}")
    p = D / f"memory-{c}-N1.jsonl"
    if p.exists():
        m = load(p); s, _ = ci(m, cb); print(f"  passage memory N1 {em(m):.1f} | m-closed {s}")
    nstar = 0
    for n in (1, 4, 16, 64):
        f = D / f"qa_memory-{c}-N{n}.jsonl"
        if not f.exists(): continue
        mem, sh, cf = load(f), load(D / f"shuffled_qa_memory-{c}-N{n}.jsonl"), load(D / f"counterfactual_qa_memory-{c}-N{n}.jsonl")
        a, l1 = ci(mem, sh); b, l2 = ci(mem, cf); cbs, l3 = ci(mem, cb)
        if l1 > 0 and l2 > 0: nstar = n
        print(f"  N={n:3d} qa {em(mem):5.1f} shuf {em(sh):5.1f} cf {em(cf):5.1f} (follow {em(cf, 'cf_follow_exact_match'):.1f})"
              f" | qa-shuf {a} | qa-cf {b} | qa-closed {cbs}")
    print(f"  N* = {nstar}")
