import sys
sys.path.insert(0, "experiments/issue-6-delta-rule-memory")
from pathlib import Path
from analyze import load, bootstrap_diff
R = Path("runs/delta-rule/lora-banks/1.7b")
frozen = load(Path("runs/delta-rule/passage-query/1.7b/eval-K/closed_book-N1.jsonl")); ids = sorted(frozen)
em = lambda r, k="exact_match": 100 * sum(r[i][k] for i in ids) / len(ids)
def ci(a, b):
    lo, hi = bootstrap_diff([a[i]["exact_match"] for i in ids], [b[i]["exact_match"] for i in ids], 10000, 20261111)
    return f"{em(a)-em(b):+5.1f} [{lo:+.1f},{hi:+.1f}]", lo
L = {c: {n: {k: load(R / f"eval-beta-{c}/{k}-Lb-{c}-N{n}.jsonl") for k in ("memory", "shuffled_memory", "counterfactual_memory")} for n in (1, 4, 16, 64)} for c in ("none", "unrelated")}
for c in ("none", "unrelated"):
    cb = load(R / f"eval-beta-{c}/closed_book-N1.jsonl"); nstar = 0
    print(f"LoRA {c}: closed-book (LoRA on, no memory) {em(cb):.1f}, frozen {em(frozen):.1f}")
    for n in (1, 4, 16, 64):
        m, s, f = (L[c][n][k] for k in ("memory", "shuffled_memory", "counterfactual_memory"))
        a, l1 = ci(m, s); b, l2 = ci(m, f); k_, _ = ci(m, cb)
        if l1 > 0 and l2 > 0: nstar = n
        extra = f" | mem - controlLoRA mem {ci(m, L['unrelated'][n]['memory'])[0]}" if c == "none" else ""
        print(f"  N={n:2} mem {em(m):5.1f} shuf {em(s):5.1f} cf {em(f):5.1f} follow {em(f,'cf_follow_exact_match'):4.1f} | m-shuf {a} | m-cf {b} | m-closed {k_}{extra}")
    print(f"  N* = {nstar}")
print("vs fixed-beta LoRA (memory):")
F = {n: load(R / f"eval-none/memory-L-none-N{n}.jsonl") for n in (1, 4, 16, 64)}
for n in (1, 4, 16, 64):
    print(f"  N={n:2} beta {em(L['none'][n]['memory']):5.1f} fixed {em(F[n]):5.1f} diff {ci(L['none'][n]['memory'], F[n])[0]}")
