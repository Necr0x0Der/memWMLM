import json,sys,re,collections
from pathlib import Path
D=Path(sys.argv[1])
em={}
for p in D.glob('*.jsonl'):
    m=re.match(r'(memory|shuffled_memory)-(.+)-N(\d+)$',p.stem)
    if not m: continue
    rows=[json.loads(l) for l in p.read_text().split('\n') if l]
    em[(m[2],m[1],int(m[3]))]=100*sum(r['exact_match'] for r in rows)/len(rows)
tags=sorted({k[0] for k in em})
Ns=[1,2,4,16]
out=[]
for t in tags:
    try:
        mem=[em[(t,'memory',n)] for n in Ns]; sh=[em[(t,'shuffled_memory',n)] for n in Ns]
    except KeyError: continue
    score=sum(a-b for a,b in zip(mem,sh))/4
    out.append((score,sum(mem)/4,t,mem,sh))
out.sort(key=lambda x:(-x[0],-x[1]))
print(f"{'tag':22} {'score':>6} {'memEM':>6}   memory N1/2/4/16        shuffled N1/2/4/16")
for s,m,t,mem,sh in out:
    print(f"{t:22} {s:6.2f} {m:6.2f}   "+' '.join(f'{x:5.1f}' for x in mem)+'   '+' '.join(f'{x:5.1f}' for x in sh))
