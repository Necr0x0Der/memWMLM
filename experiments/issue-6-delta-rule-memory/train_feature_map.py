#!/usr/bin/env python3
"""Train per-head feature maps so the delta-rule read imitates the softmax memory.

Teacher = the uncompressed softmax memory (``read="knn"``, top-k 4; the
upper-bound test). For a bank of N TriviaQA *training* passages (never an
eval/calibration question of banks-v2), frozen Qwen3 runs the bank's prompts
with the teacher memory; at every layer we record the pre-RoPE queries and
what the memory adds to the attention output (joint-softmax output − plain
output). Student = delta-rule memory (beta 1) written with ``phi_k(k)`` and
read with ``phi_q(q)`` (``FeatureMaps``); loss per layer = squared error of
the read vs the teacher's contribution, divided by the contribution's mean
square (1.0 = no better than reading nothing). Qwen is frozen; no gradient
passes through it (layer-wise distillation, teacher-forced hidden states).

Prompts per bank row: the closed-book QA prompt + gold answer (no sink), and
sink + passage up to the end of the answer (cloze), so both read situations
are covered. Validation: fixed banks from held-out training passages.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_banks import read_jsonl, render_closed  # noqa: E402

from memwmlm.models.delta_qwen import DeltaMemoryConfig, DeltaMemoryQwen, FeatureMaps, prepend_sink
from memwmlm.models.qwen import QwenChatPromptFormatter


def pool_rows(banks_dir: Path, questions: Path) -> list[dict]:
    held = {r["question_id"] for f in ("eval", "calibration") for r in read_jsonl(banks_dir / f"{f}.jsonl")}
    return [r for r in read_jsonl(questions)
            if r["split"] == "train" and r.get("passage") and r["question_id"] not in held]


class Batch:
    def __init__(self, memory, tokenizer, formatter, sink, device):
        self.m, self.tok, self.fmt, self.sink, self.dev = memory, tokenizer, formatter, sink, device

    def passages(self, rows, rope):
        self.tok.padding_side = "right"
        enc = self.tok([r["passage"] for r in rows], padding=True, return_tensors="pt").to(self.dev)
        ids, mask = prepend_sink(enc["input_ids"], enc["attention_mask"], self.sink)
        k, v, m = self.m.capture_passages(ids, mask, rope=rope)
        return k[:, :, m], v[:, :, m], m[m]

    def prompts(self, rows):
        seqs = []
        for r in rows:
            seqs.append(self.tok(self.fmt(render_closed(r["question"])) + r["answer_surface"],
                                 add_special_tokens=False)["input_ids"])
            end = r["answer_span"][1]
            seqs.append([self.sink] + self.tok(r["passage"][:end], add_special_tokens=False)["input_ids"])
        width = max(map(len, seqs))
        ids = torch.tensor([[self.tok.pad_token_id] * (width - len(s)) + s for s in seqs], device=self.dev)
        mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=self.dev)
        return ids, mask

    @torch.no_grad()
    def teacher(self, rows, prompt_rows):
        """-> student keys/values [L, Hkv, T, D] and per layer (queries [H, P, D], target [H, P, D])."""

        kr, vr, mr = self.passages(rows, rope=True)
        state = self.m.build_state(kr, vr, mr)
        k, v, _ = self.passages(rows, rope=False)
        ids, mask = self.prompts(prompt_rows)
        self.m.record = {}
        with self.m.reading(state):
            self.m.qwen(input_ids=ids, attention_mask=mask)
        valid = mask.bool()
        targets = []
        for index in self.m.layer_indices:
            q, c = self.m.record[index]  # [B, H, T, D], [B, T, H, D]
            targets.append((q.permute(1, 0, 2, 3)[:, valid].float(), c.permute(2, 0, 1, 3)[:, valid]))
        self.m.record = None
        return k, v, targets


def student_loss(memory: DeltaMemoryQwen, fm: FeatureMaps, k, v, targets):
    states = memory.build_feature_state(k, v)
    losses = []
    for n, index in enumerate(memory.layer_indices):
        q, target = targets[n]
        read = torch.einsum("hpk,hkv->hpv", fm.queries(n, q), states[index].repeat_interleave(memory.groups, 0))
        losses.append(((read - target) ** 2).sum(-1).mean() / (target ** 2).sum(-1).mean().clamp_min(1e-8))
    return torch.stack(losses)  # [L]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks-dir", type=Path, default=Path("runs/delta-rule/banks-v2"))
    p.add_argument("--questions", type=Path, default=Path("runs/delta-rule/global-v1/questions.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--sizes", default="1,4,16,64")
    p.add_argument("--prompt-rows", type=int, default=8, help="bank rows whose prompts are used per step")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--val-rows", type=int, default=256)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 local_files_only=True).to(args.device).eval()
    memory = DeltaMemoryQwen(model, DeltaMemoryConfig(read="knn", knn_top_k=args.top_k, beta=1.0))
    cfg = model.config
    fm = FeatureMaps(len(memory.layer_indices), memory.num_heads, memory.num_kv_heads, memory.head_dim).to(args.device)
    memory.feature_maps = fm  # used by build_feature_state only; the knn teacher path ignores it
    batch = Batch(memory, tokenizer, QwenChatPromptFormatter(tokenizer),
                  tokenizer.convert_tokens_to_ids("<|endoftext|>"), args.device)

    rows = pool_rows(args.banks_dir, args.questions)
    random.shuffle(rows)
    val, train = rows[: args.val_rows], rows[args.val_rows:]
    sizes = [int(x) for x in args.sizes.split(",")]
    val_banks = [(n, val[i * n % len(val):][:n]) for n in sizes for i in range(4)]
    print(json.dumps({"train_rows": len(train), "val_rows": len(val), "params": sum(x.numel() for x in fm.parameters()),
                      "num_layers": cfg.num_hidden_layers}), flush=True)

    opt = torch.optim.AdamW(fm.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    log = []

    def validate(step):
        out = {}
        with torch.no_grad():
            for n in sizes:
                vals = []
                for m, bank in val_banks:
                    if m == n:
                        k, v, t = batch.teacher(bank, bank[: args.prompt_rows])
                        vals.append(student_loss(memory, fm, k, v, t))
                out[n] = torch.stack(vals).mean(0)
        row = {"step": step, **{f"val_N{n}": round(out[n].mean().item(), 4) for n in sizes},
               **{f"val_N{n}_by_layer": [round(x, 3) for x in out[n].tolist()] for n in sizes}}
        print(json.dumps({k: v for k, v in row.items() if "by_layer" not in k}), flush=True)
        return row

    log.append(validate(0))
    best = log[-1]
    started = time.time()
    running = []
    for step in range(1, args.steps + 1):
        n = random.choice(sizes)
        bank = random.sample(train, n)
        k, v, t = batch.teacher(bank, bank[: args.prompt_rows])
        loss = student_loss(memory, fm, k, v, t).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(loss.item())
        if step % 50 == 0:
            print(json.dumps({"step": step, "train_loss": round(sum(running) / len(running), 4),
                              "seconds": round(time.time() - started)}), flush=True)
            running = []
        if step % args.val_every == 0 or step == args.steps:
            log.append(validate(step))
            score = sum(log[-1][f"val_N{n}"] for n in sizes)
            if score < sum(best[f"val_N{n}"] for n in sizes) or best["step"] == 0:
                best = log[-1]
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": fm.state_dict(), "args": {k: str(v) for k, v in vars(args).items()},
                            "step": step, "shape": [len(memory.layer_indices), memory.num_heads,
                                                     memory.num_kv_heads, memory.head_dim]}, args.output)
    args.output.with_suffix(".log.json").write_text(json.dumps({"best_step": best["step"], "log": log}, indent=1))
    print(json.dumps({"best_step": best["step"], **{k: v for k, v in best.items() if "by_layer" not in k}}), flush=True)


if __name__ == "__main__":
    main()
