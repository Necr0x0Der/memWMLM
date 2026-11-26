import sys, json
sys.path.insert(0, "experiments/issue-6-delta-rule-memory")
from pathlib import Path
from analyze import load, bootstrap_diff
O = Path("runs/delta-rule/v2q")
def em(r, k="exact_match"): return 100 * sum(x[k] for x in r.values()) / len(r)
def ci(a, b):
    ids = sorted(a); lo, hi = bootstrap_diff([a[i]["exact_match"] for i in ids], [b[i]["exact_match"] for i in ids], 10000, 20261111)
    return em(a) - em(b), lo, hi
fmt = lambda t: f"{t[0]:+5.1f} [{t[1]:+.1f},{t[2]:+.1f}]"
runs = {"none": ("eval-none", "B-none"), "unrelated": ("eval-unrelated", "B-unrelated"), "v1lora": ("eval-none-v1lora", "B-v1lora")}
mem = {}
for name, (d, tag) in runs.items():
    cb = load(O / d / "closed_book-N1.jsonl"); nstar = 0
    print(f"{name}: closed-book {em(cb):.1f}")
    for n in (1, 4, 16, 64):
        m, s, f = (load(O / d / f"{c}-{tag}-N{n}.jsonl") for c in ("memory", "shuffled_memory", "counterfactual_memory"))
        mem[name, n] = m
        a, b, c = ci(m, s), ci(m, f), ci(m, cb)
        if a[1] > 0 and b[1] > 0: nstar = n
        extra = f" | m-control {fmt(ci(m, mem['unrelated', n]))}" if name == "none" and ("unrelated", n) in mem else ""
        print(f"  N={n:2} mem {em(m):5.1f} shuf {em(s):5.1f} cf {em(f):5.1f} fol {em(f,'cf_follow_exact_match'):3.1f} | m-s {fmt(a)} | m-cf {fmt(b)} | m-closed {fmt(c)}{extra}")
    print(f"  N* = {nstar}")
print("main vs control-model memory:")
for n in (1, 4, 16, 64):
    print(f"  N={n:2} {fmt(ci(mem['none', n], mem['unrelated', n]))}")
for c in ("none", "unrelated"):
    log = json.load(open(O / f"v2q-{c}.log.json"))
    print(c, "WikiText val", {k: v for k, v in log[-1].items() if k.startswith(("val_", "wrong_", "gate", "beta"))})
