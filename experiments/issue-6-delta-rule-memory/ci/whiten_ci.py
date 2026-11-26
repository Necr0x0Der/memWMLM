import sys
sys.path.insert(0, "experiments/issue-6-delta-rule-memory")
from pathlib import Path
from analyze import load, bootstrap_diff
for c, d in (("W", "runs/delta-rule/whiten/1.7b/eval-W"), ("WB", "runs/delta-rule/whiten/1.7b/eval-WB")):
    D = Path(d)
    if not (D / "closed_book-N1.jsonl").exists(): continue
    cb = load(D / "closed_book-N1.jsonl"); ids = sorted(cb)
    em = lambda r: 100 * sum(r[i]["exact_match"] for i in ids) / len(ids)
    def ci(a, b):
        lo, hi = bootstrap_diff([a[i]["exact_match"] for i in ids], [b[i]["exact_match"] for i in ids], 10000, 20261111)
        return f"{em(a)-em(b):+.1f} [{lo:+.1f}, {hi:+.1f}]", lo
    nstar = 0
    print(f"{c}: closed-book {em(cb):.1f}")
    for n in (1, 4, 16, 64):
        f = D / f"memory-{c}-N{n}.jsonl"
        if not f.exists(): continue
        mem, sh, cf = load(f), load(D / f"shuffled_memory-{c}-N{n}.jsonl"), load(D / f"counterfactual_memory-{c}-N{n}.jsonl")
        a, l1 = ci(mem, sh); b, l2 = ci(mem, cf); cbs, _ = ci(mem, cb)
        if l1 > 0 and l2 > 0: nstar = n
        print(f"  N={n:3d} mem {em(mem):5.1f} shuf {em(sh):5.1f} cf {em(cf):5.1f} | m-shuf {a} | m-cf {b} | m-closed {cbs}")
    print(f"  N* = {nstar}")
