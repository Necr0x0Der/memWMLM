"""End-to-end wiring check of passage_query_memory in run_banks: during real generation,
every memory read (prompt pass and each decode step) must use row b's own passage mean query."""
import sys, types, torch
sys.path.insert(0, "experiments/issue-6-delta-rule-memory")
import run_banks as rb
from transformers import AutoModelForCausalLM, AutoTokenizer
from memwmlm.models.delta_qwen import DeltaMemoryConfig, DeltaMemoryQwen
from memwmlm.models.qwen import QwenChatPromptFormatter
M, REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
tok = AutoTokenizer.from_pretrained(M, revision=REV, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(M, revision=REV, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
mem = DeltaMemoryQwen(model, DeltaMemoryConfig(beta=1.0, alpha=1.0, gate=0.2, center_keys=True))
args = types.SimpleNamespace(write_batch_size=64, device="cuda", sink_prefix=True, batch_size=4, max_new_tokens=4, max_context_tokens=30000)
rows = rb.read_jsonl(rb.Path("runs/delta-rule/banks-v2/eval.jsonl"))[:4]
calib = rb.read_jsonl(rb.Path("runs/delta-rule/banks-v2/calibration.jsonl"))
k, _, m = rb.encode_passages(mem, tok, [r["passage"] for r in calib[:256]], args); mem.calibrate_key_mean(k, m); del k
expected = {i: q.clone() for i, q in rb.passage_read_queries(mem, tok, [r["passage"] for r in rows], args).items()}
calls, orig = [], mem._read
def spy(index, q):
    calls.append((index, q.shape[2], (q.float() - expected[index][:, :, None]).abs().max().item(),
                  # how different the passages' queries are from each other (row 0 vs 1)
                  (expected[index][0] - expected[index][1]).abs().max().item()))
    return orig(index, q)
mem._read = spy
class H:
    def write(self, s): print("REC", s.strip()[:160])
    def flush(self): pass
rb.run_bank("passage_query_memory", 4, 0, rows, rows, calib, model, tok, QwenChatPromptFormatter(tok), mem, args, H())
print("reads:", len(calls), "layers:", len({c[0] for c in calls}), "steps:", sorted({c[1] for c in calls}))
print("max |used query - own passage mean query|:", max(c[2] for c in calls))
print("passages differ (max |q0 - q1|, min over layers):", min(c[3] for c in calls))
print("read_queries reset afterwards:", mem.read_queries is None)
