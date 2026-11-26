#!/usr/bin/env python3
"""V1-LoRA on small banks (README amendment 2026-11-24).

Variant 1 config A (delta rule beta 1, centred keys, D 128, sink prefix): memory is
always written by frozen Qwen (LoRA off). Trained: rank-r LoRA on q_proj and o_proj
of every layer (application side) + per-layer x per-query-head gates. Each step
draws N in --sizes, builds 8 / N banks of N random TriviaQA *training* passages
(never banks-v2 eval/calibration), and trains Qwen to answer the closed-book
prompt of the bank's questions (cross-entropy on answer surface + <|im_end|>),
each prompt reading its own bank's state.

--learn-beta: the write strength beta = sigmoid(theta) is trained per (layer, KV head)
(2026-11-24 amendment, option 1); the write is then differentiable in beta (keys and
values stay those of frozen Qwen).

--key-expand 16 --center-queries: the bigger memory of config B (DPFP, keys 4,096 wide)
with the same training (2026-11-24 amendment "per-token β + bigger memory").

--task cloze --writer-rank 8 (2026-11-25, Variant 2 step B): trained on WikiText sentence cloze only
(no TriviaQA): banks of N WikiText sentences, each prompt the V2b cloze question (3 worked examples)
about one of them, loss on the blanked word. --writer-rank adds the trainable writer (``WriterLoRA``:
Qwen's k/v_proj + low-rank update, active only for the written keys/values); the key mean is
calibrated on WikiText sentences.

--control unrelated: each prompt instead reads a bank of N *other* random
training passages (its own passage absent): the fine-tuning control.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_banks import encode_passages, read_jsonl, render_closed  # noqa: E402
from train_feature_map import pool_rows  # noqa: E402
from train_v1_global import LoRALinear, set_lora  # noqa: E402

from memwmlm.models.delta_qwen import DeltaMemoryConfig, DeltaMemoryQwen, TokenBeta, WriterLoRA
from memwmlm.models.qwen import QwenChatPromptFormatter


def install_lora(model, rank: int) -> list[LoRALinear]:
    loras = []
    for layer in model.model.layers:
        for name in ("q_proj", "o_proj"):
            wrapped = LoRALinear(getattr(layer.self_attn, name), rank, 2.0 * rank)
            setattr(layer.self_attn, name, wrapped)
            loras.append(wrapped)
    return loras


def load_lora(model, memory: DeltaMemoryQwen, path: Path) -> list[LoRALinear]:
    """Install a trained checkpoint: LoRA weights + gates. Returns the LoRA modules (enabled)."""

    ckpt = torch.load(path, map_location="cpu")
    loras = install_lora(model, ckpt["rank"])
    for i, m in enumerate(loras):
        down, up = ckpt["lora"][i]
        m.down.weight.data.copy_(down)
        m.up.weight.data.copy_(up)
    device = next(model.parameters()).device
    for index, gate in ckpt["gates"].items():
        memory.gates[int(index)] = gate.to(device)
    if ckpt.get("beta_logits") is not None:
        memory.beta_logits = ckpt["beta_logits"].to(device)
    if ckpt.get("token_beta") is not None:
        w = ckpt["token_beta"]["weight"]
        memory.beta_proj = TokenBeta(*w.shape).to(device)
        memory.beta_proj.load_state_dict(ckpt["token_beta"])
        memory.beta_proj.requires_grad_(False)
    if ckpt.get("writer") is not None:
        memory.writer = WriterLoRA(memory, ckpt["writer_rank"], 2.0 * ckpt["writer_rank"])
        memory.writer.load_state_dict(ckpt["writer"])
        memory.writer.requires_grad_(False)
    return loras


class Trainer:
    def __init__(self, model, memory, tokenizer, formatter, loras, args):
        self.model, self.m, self.tok, self.fmt, self.loras, self.args = model, memory, tokenizer, formatter, loras, args
        self.end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.prompt_of = lambda r: self.fmt(render_closed(r["question"]))  # TriviaQA closed-book prompt

    def states(self, banks: list[list[dict]]) -> dict[int, torch.Tensor]:
        """One state per bank, written by frozen Qwen -> layer -> [n_banks, Hkv, D, D]."""

        set_lora(self.loras, False)
        out = []
        for bank in banks:
            with torch.no_grad():
                k, v, mask = encode_passages(self.m, self.tok, [r["passage"] for r in bank], self.args)
            hidden = self.m.last_hidden if self.m.needs_hidden else None
            out.append(self.m.build_state(k, v, mask, hidden=hidden))  # differentiable in beta if learned
        set_lora(self.loras, True)
        return {layer: torch.stack([s[layer] for s in out]) for layer in out[0]}

    def loss(self, prompt_rows: list[dict], state: dict[int, torch.Tensor]) -> torch.Tensor:
        ids, labels = [], []
        for r in prompt_rows:
            prompt = self.tok(self.prompt_of(r))["input_ids"]
            answer = self.tok(r["answer_surface"], add_special_tokens=False)["input_ids"] + [self.end]
            ids.append(prompt + answer)
            labels.append([-100] * len(prompt) + answer)
        width = max(map(len, ids))
        pad = self.tok.pad_token_id
        dev = self.args.device
        input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in ids], device=dev)
        label_ids = torch.tensor([x + [-100] * (width - len(x)) for x in labels], device=dev)
        attention = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids], device=dev)
        with self.m.reading(state):
            hidden = self.model.model(input_ids=input_ids, attention_mask=attention, use_cache=False).last_hidden_state
        targets = label_ids[:, 1:]
        keep = targets != -100
        logits = self.model.lm_head(hidden[:, :-1][keep])
        return nn.functional.cross_entropy(logits.float(), targets[keep])

    def episode(self, rows: list[dict], n: int, rng: random.Random):
        """8 prompts; each reads its own bank (or, for the control, a bank of N other passages)."""

        n_banks = max(1, self.args.prompts // n)
        picked = rng.sample(rows, n_banks * n)
        banks = [picked[b * n:(b + 1) * n] for b in range(n_banks)]
        per_bank = self.args.prompts // n_banks
        prompt_rows, owner = [], []
        for b, bank in enumerate(banks):
            for r in bank[:per_bank]:
                prompt_rows.append(r)
                owner.append(b)
        if self.args.control == "unrelated":
            used = {r["question_id"] for r in picked}
            others = [r for r in rows if r["question_id"] not in used]
            banks = [rng.sample(others, n) for _ in range(n_banks)]
        state = self.states(banks)
        index = torch.tensor(owner, device=self.args.device)
        return prompt_rows, {layer: s[index] for layer, s in state.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks-dir", type=Path, default=Path("runs/delta-rule/banks-v2"))
    p.add_argument("--questions", type=Path, default=Path("runs/delta-rule/global-v1/questions.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--sizes", default="1,4,16")
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-gate", type=float, default=1e-2)
    p.add_argument("--gate", type=float, default=0.1, help="initial gate")
    p.add_argument("--learn-beta", choices=("off", "head", "token"), default="off",
                   help="head: one beta per layer x KV head (option 1); token: per-token beta (option 2)")
    p.add_argument("--beta-init", type=float, default=0.95)
    p.add_argument("--lr-beta", type=float, default=1e-2)
    p.add_argument("--key-expand", type=int, default=0, help="DPFP order nu (16 = config B, D 4,096); 0 = D 128")
    p.add_argument("--center-queries", action="store_true", help="centre read queries (config B)")
    p.add_argument("--task", choices=("triviaqa", "cloze"), default="triviaqa")
    p.add_argument("--cloze-data", type=Path, default=Path("runs/delta-rule/wikitext-cloze-v1"))
    p.add_argument("--fewshot", type=int, default=3, help="cloze task: worked examples in every prompt (V2b)")
    p.add_argument("--writer-rank", type=int, default=0, help="trainable writer (Variant 2 step B); 0 = off")
    p.add_argument("--control", choices=("none", "unrelated"), default="none")
    p.add_argument("--val-rows", type=int, default=256)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.write_batch_size, args.sink_prefix = 64, True
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 local_files_only=True).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    memory = DeltaMemoryQwen(model, DeltaMemoryConfig(beta=1.0, alpha=1.0, gate=args.gate, center_keys=True,
                                                      key_expand=args.key_expand, center_queries=args.center_queries))
    loras = install_lora(model, args.lora_rank)
    set_lora(loras, False)
    if args.task == "cloze":  # no TriviaQA anywhere: key mean from WikiText sentences
        calib_passages = [r["passage"] for r in read_jsonl(args.cloze_data / "train.jsonl")[:256]]
    else:
        calib_passages = [r["passage"] for r in read_jsonl(args.banks_dir / "calibration.jsonl")[:256]]
    k, _, m = encode_passages(memory, tokenizer, calib_passages, args)
    memory.calibrate_key_mean(k, m)
    del k
    gates = []
    for index in memory.layer_indices:
        gate = nn.Parameter(torch.full((memory.num_heads,), args.gate, device=args.device))
        memory.gates[index] = gate
        gates.append(gate)
    beta_params = []
    if args.learn_beta == "token":
        memory.beta_proj = TokenBeta(len(memory.layer_indices), memory.num_kv_heads, model.config.hidden_size,
                                     args.beta_init).to(args.device)
        beta_params = list(memory.beta_proj.parameters())
    elif args.learn_beta == "head":
        init = torch.logit(torch.tensor(args.beta_init)).item()
        memory.beta_logits = nn.Parameter(torch.full((len(memory.layer_indices), memory.num_kv_heads), init,
                                                     device=args.device))
        beta_params = [memory.beta_logits]
    if args.writer_rank:
        memory.writer = WriterLoRA(memory, args.writer_rank, 2.0 * args.writer_rank)
    lora_params = [p for m_ in loras for p in (m_.down.weight, m_.up.weight)]
    if memory.writer is not None:
        lora_params += list(memory.writer.parameters())
    for p_ in lora_params:
        p_.requires_grad_(True)
    trainer = Trainer(model, memory, tokenizer, QwenChatPromptFormatter(tokenizer), loras, args)

    rng = random.Random(args.seed)
    if args.task == "cloze":
        from train_v2 import ClozeFormatter, cloze_question, load_shots

        cloze_fmt = ClozeFormatter(tokenizer, load_shots(args.cloze_data, args.fewshot))
        trainer.prompt_of = lambda r: cloze_fmt(cloze_question(r["cloze"]))
        def as_rows(path, drop=0):
            rs = read_jsonl(path)
            rs = rs[: len(rs) - drop]
            return [{**r, "question_id": r["id"], "answer_surface": r["answer"]} for r in rs]
        train = as_rows(args.cloze_data / "train.jsonl", args.fewshot)  # shots never sampled
        val = as_rows(args.cloze_data / "val.jsonl")
        rng.shuffle(val)
        val = val[: args.val_rows]
    else:
        rows = pool_rows(args.banks_dir, args.questions)
        rng.shuffle(rows)
        val, train = rows[: args.val_rows], rows[args.val_rows:]
    sizes = [int(x) for x in args.sizes.split(",")]
    val_rng = random.Random(1)
    val_specs = [(n, 1000 + i) for i, n in enumerate(n for n in sizes for _ in range(4))]  # fixed val banks
    print(json.dumps({"train_rows": len(train), "val_rows": len(val), "control": args.control,
                      "trainable": sum(p_.numel() for p_ in lora_params + gates)}), flush=True)

    groups = [{"params": lora_params, "lr": args.lr_lora}, {"params": gates, "lr": args.lr_gate}]
    if beta_params:
        groups.append({"params": beta_params, "lr": args.lr_beta})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    log = []

    def validate(step):
        with torch.no_grad():
            out, wrong = {n: [] for n in sizes}, {n: [] for n in sizes}
            for n, seed in val_specs:  # same banks every time; states rebuilt with the current beta
                pr, st = trainer.episode(val, n, random.Random(seed))
                out[n].append(trainer.loss(pr, st).item())
                control, args.control = args.control, "unrelated"  # positive control: other passages
                pr, st = trainer.episode(val, n, random.Random(seed))
                args.control = control
                wrong[n].append(trainer.loss(pr, st).item())
        row = {"step": step, **{f"val_N{n}": round(sum(v) / len(v), 4) for n, v in out.items()},
               **{f"wrong_N{n}": round(sum(v) / len(v), 4) for n, v in wrong.items()},
               "gate_mean": round(torch.cat([g.detach() for g in gates]).mean().item(), 4)}
        if memory.beta_proj is not None:
            with torch.no_grad():  # beta distribution over the tokens of 4 fixed validation passages
                set_lora(loras, False)
                encode_passages(memory, tokenizer, [r["passage"] for r in val[:4]], args)
                set_lora(loras, True)
                b = memory.beta_proj(memory.last_hidden)
            row.update(beta_mean=round(b.mean().item(), 4), beta_p10=round(b.flatten().kthvalue(max(1, b.numel() // 10)).values.item(), 4),
                       beta_p90=round(b.flatten().kthvalue(max(1, 9 * b.numel() // 10)).values.item(), 4))
        if memory.beta_logits is not None:
            b = torch.sigmoid(memory.beta_logits.detach())
            row.update(beta_mean=round(b.mean().item(), 4), beta_min=round(b.min().item(), 4),
                       beta_max=round(b.max().item(), 4))
        print(json.dumps(row), flush=True)
        log.append(row)

    validate(0)
    started, running = time.time(), []
    for step in range(1, args.steps + 1):
        n = rng.choice(sizes)
        prompt_rows, state = trainer.episode(train, n, rng)
        loss = trainer.loss(prompt_rows, state)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running.append(loss.item())
        if step % 50 == 0:
            print(json.dumps({"step": step, "train_loss": round(sum(running) / len(running), 4),
                              "seconds": round(time.time() - started)}), flush=True)
            running = []
        if step % args.val_every == 0 or step == args.steps:
            validate(step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rank": args.lora_rank, "control": args.control, "step": args.steps,
                "beta_logits": memory.beta_logits.detach().cpu() if memory.beta_logits is not None else None,
                "token_beta": {k_: v_.detach().cpu() for k_, v_ in memory.beta_proj.state_dict().items()}
                if memory.beta_proj is not None else None,
                "writer": {k_: v_.detach().cpu() for k_, v_ in memory.writer.state_dict().items()}
                if memory.writer is not None else None, "writer_rank": args.writer_rank, "task": args.task,
                "lora": {i: (m_.down.weight.detach().cpu(), m_.up.weight.detach().cpu()) for i, m_ in enumerate(loras)},
                "gates": {index: g.detach().cpu() for index, g in zip(memory.layer_indices, gates)},
                "args": {k_: str(v_) for k_, v_ in vars(args).items()}}, args.output)
    args.output.with_suffix(".log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
