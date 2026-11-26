"""What does a memory read return, for different read queries? Config A (centred keys), gate 1.
Queries: passage mean q (check 6), question's last prompt token q (normal memory),
q of the passage token right before the answer (best single 'same passage' query).
answer_rank: percentile of the answer tokens' values among the own passage's token values by
cosine to the read-out (0.5 = chance, 1 = the answer is what came back).
passage_top1 (N 4): read-out closest to the own passage's mean value (chance 0.25).
soft_answer_share: if the same query did a softmax over the stored keys, weight on answer tokens."""
import sys, types, json, torch
from torch.nn import functional as F
sys.path.insert(0, "experiments/issue-6-delta-rule-memory")
import run_banks as rb
from transformers import AutoModelForCausalLM, AutoTokenizer
from memwmlm.models.delta_qwen import DeltaMemoryConfig, DeltaMemoryQwen
from memwmlm.models.qwen import QwenChatPromptFormatter
M, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
tok = AutoTokenizer.from_pretrained(M, revision=REV, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(M, revision=REV, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
mem = DeltaMemoryQwen(model, DeltaMemoryConfig(beta=1.0, alpha=1.0, gate=1.0, center_keys=True))
fmt = QwenChatPromptFormatter(tok)
args = types.SimpleNamespace(write_batch_size=64, device="cuda", sink_prefix=True)
calib = rb.read_jsonl(rb.Path("runs/delta-rule/banks-v2/calibration.jsonl"))
k, _, m = rb.encode_passages(mem, tok, [r["passage"] for r in calib[:256]], args); mem.calibrate_key_mean(k, m); del k
rows = rb.read_jsonl(rb.Path("runs/delta-rule/banks-v2/eval.jsonl"))[:64]
sink = tok.convert_tokens_to_ids("<|endoftext|>")
G = mem.groups

def encode(bank):
    seqs, ans, pre = [], [], []
    for r in bank:
        e = tok(r["passage"], add_special_tokens=False, return_offsets_mapping=True)
        a0, a1 = r["answer_span"]
        flags = [0] + [int(s < a1 and e_ > a0) for s, e_ in e["offset_mapping"]]
        seqs.append([sink] + e["input_ids"]); ans.append(flags); pre.append(flags.index(1) - 1)
    T = max(map(len, seqs))
    ids = torch.tensor([s + [tok.pad_token_id] * (T - len(s)) for s in seqs]).cuda()
    mask = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs]).cuda()
    A = torch.tensor([a + [0] * (T - len(a)) for a in ans]).cuda().bool()
    return ids, mask, A, pre

@torch.no_grad()
def run(n):
    acc = {}
    for b0 in range(0, len(rows), n):
        bank = rows[b0:b0 + n]
        ids, mask, A, pre = encode(bank)
        mem.capture_queries = True
        with mem.capturing() as cap:
            model(input_ids=ids, attention_mask=mask)
        mem.capture_queries = False
        keys, values, wm = mem.capture_passages(ids, mask)
        state = mem.build_state(keys[:, :, wm], values[:, :, wm], wm[wm])
        B, T = ids.shape
        pid = torch.arange(B).cuda()[:, None].expand(B, T).reshape(-1)[wm]
        aflag = A.reshape(-1)[wm]
        pq = mem.passage_queries(ids, mask)
        tok.padding_side = "left"
        pr = tok([fmt(rb.render_closed(r["question"])) for r in bank], return_tensors="pt", padding=True).to("cuda")
        mem.capture_queries = True
        with mem.capturing() as qcap:
            model(**pr)
        mem.capture_queries = False
        mem.state = state
        for li, layer in enumerate(mem.layer_indices):
            qs = {"passage_mean": pq[layer],
                  "question_last": qcap[layer].queries[0][:, :, -1].float(),
                  "before_answer": torch.stack([cap[layer].queries[0][b, :, pre[b]] for b in range(B)]).float()}
            v = values[li][:, wm].float()  # [Hkv, S, D]
            c = v.mean(1, keepdim=True)
            vn = F.normalize(v - c, dim=-1)
            kraw = keys[li][:, wm].float()
            pmean = torch.stack([vn[:, pid == b].mean(1) for b in range(B)], 1)  # [Hkv, B, D]
            for name, q in qs.items():
                r = mem._read(layer, q[:, :, None, :])[:, 0]  # [B, H, D]
                rn = F.normalize(r - c.repeat_interleave(G, 0)[None, :, 0], dim=-1)  # [B, H, D]
                vh = vn.repeat_interleave(G, 0)  # [H, S, D]
                sims = torch.einsum("bhd,hsd->bhs", rn, vh)
                logits = torch.einsum("bhd,hsd->bhs", q, kraw.repeat_interleave(G, 0)) * mem.head_dim ** -0.5
                w = logits.softmax(-1)
                ranks, top1, share, share_p = [], [], [], []
                for b in range(B):
                    own = pid == b
                    s_own = sims[b][:, own]  # [H, P]
                    a = aflag[own]
                    pct = (s_own[:, a][:, :, None] > s_own[:, ~a][:, None, :]).float().mean((1, 2))
                    ranks.append(pct)
                    top1.append((torch.einsum("hd,hbd->hb", rn[b], pmean.repeat_interleave(G, 0)).argmax(-1) == b).float())
                    share.append(w[b][:, own][:, a].sum(-1)); share_p.append(w[b][:, own].sum(-1))
                d = acc.setdefault(name, {"answer_rank": [], "passage_top1": [], "soft_answer_share": [], "soft_own_passage_share": [], "answer_token_frac": []})
                d["answer_rank"].append(torch.stack(ranks).mean().item())
                d["passage_top1"].append(torch.stack(top1).mean().item())
                d["soft_answer_share"].append(torch.stack(share).mean().item())
                d["soft_own_passage_share"].append(torch.stack(share_p).mean().item())
                d["answer_token_frac"].append(aflag.float().mean().item())
        mem.state = None
    return {k: {m: round(sum(x) / len(x), 3) for m, x in d.items()} for k, d in acc.items()}

for n in (1, 4):
    print("N", n, json.dumps(run(n), indent=1))
