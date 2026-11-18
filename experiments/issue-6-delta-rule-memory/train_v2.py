#!/usr/bin/env python3
"""Train the Variant 2 delta-rule interface on WikiText cloze (no TriviaQA).

Each step is one episode: ``N`` WikiText sentences (N drawn from
``--bank-sizes``) are encoded by frozen Qwen up to layer ``--layer`` and
written, in random order, into a fresh ``d x d`` state by the trained writer;
then ``--queries`` of those sentences are asked as cropped cloze questions with
the closed-book TriviaQA prompt format (question = "Fill in the blank: ...") and
the loss is cross-entropy on the blanked span + ``<|im_end|>``. Only the
interface (W_K, W_V, w_beta, W_Q, W_O) is trained.

Validation: a fixed bank of the first ``--val-bank`` validation sentences is
written and ``--val-queries`` of them are answered greedily (exact match).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_banks import git_dirty, git_revision, normalize_answer, read_jsonl, render_closed  # noqa: E402
from run_banks import score_prediction  # noqa: E402

from memwmlm.models.delta_interface import DeltaInterface, InterfaceQwen
from memwmlm.models.qwen import QwenChatPromptFormatter

CLOZE_PREFIX = "Fill in the blank: "


def cloze_question(cloze: str) -> str:
    return render_closed(CLOZE_PREFIX + cloze)


class ClozeFormatter:
    """Chat prompt for cloze questions; ``shots`` (V2b ``--fewshot``) are prior user/assistant turns so the
    answer format is right without memory and training signal can only come from reading S."""

    def __init__(self, tokenizer, shots: list[dict]) -> None:
        self.tokenizer, self.shots = tokenizer, shots

    def __call__(self, task_text: str) -> str:
        messages = []
        for r in self.shots:
            messages += [{"role": "user", "content": cloze_question(r["cloze"])},
                         {"role": "assistant", "content": r["answer"]}]
        messages.append({"role": "user", "content": task_text})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)


def load_shots(data: Path, k: int) -> list[dict]:
    """The last ``k`` train units (never sampled into episodes when k > 0)."""

    return read_jsonl(data / "train.jsonl")[-k:] if k else []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("runs/delta-rule/wikitext-cloze-v1"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--cache", type=Path, default=None, help="default: <data>/hidden-L<layer>")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--bank-sizes", default="512,1024,2048")
    p.add_argument("--queries", type=int, default=32)
    p.add_argument("--micro-batch", type=int, default=0, help="split the queries of a step (0 = all at once)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta-bias", type=float, default=0.0)
    p.add_argument("--shift-values", action="store_true", help="Variant 2b: key token t -> value token t+1")
    p.add_argument("--gap-sizes", default="64,1024", help="val N for real-vs-wrong-S loss")
    p.add_argument("--bias-state", action="store_true", help="learned constant S0 added to every read")
    p.add_argument("--fewshot", type=int, default=0, help="worked cloze examples in every cloze prompt")
    p.add_argument("--full-cloze", action="store_true",
                   help="sanity check: query = whole stored sentence with the span blanked")
    p.add_argument("--standardise", action="store_true",
                   help="fixed per-dim standardisation of h from train-cache statistics (Variant 2b)")
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--encode-batch", type=int, default=128)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--val-bank", type=int, default=1024)
    p.add_argument("--val-queries", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 local_files_only=True).to(args.device).eval()
    return tokenizer, model


def encode(wrapped: InterfaceQwen, tokenizer, passages: list[str], batch: int, device,
           return_lengths: bool = False):
    tokenizer.padding_side = "right"
    parts, lengths = [], []
    for start in range(0, len(passages), batch):
        enc = tokenizer(passages[start : start + batch], padding=True, return_tensors="pt").to(device)
        h, n = wrapped.passage_tokens(enc["input_ids"], enc["attention_mask"], return_lengths=True)
        parts.append(h)
        lengths += n
    return (torch.cat(parts), lengths) if return_lengths else torch.cat(parts)


def cache_dir_for(data: Path, layer: int, model: str, revision: str) -> Path:
    """Hidden-state cache of ``cache_hidden.py``; the 1.7B cache keeps its original name."""

    name = f"hidden-L{layer}" if model == "Qwen/Qwen3-1.7B" else f"hidden-{model.split('/')[-1]}-L{layer}"
    path = data / name
    manifest = path / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        if (m["model"], m["revision"], m["layer"]) != (model, revision, layer):
            raise SystemExit(f"{path} was cached for {m['model']}@{m['revision']} layer {m['layer']}")
    return path


class HiddenCache:
    """Exact cached layer hidden states (``cache_hidden.py``) of one split."""

    def __init__(self, directory: Path, split: str, hidden_size: int) -> None:
        import numpy as np

        self.offsets = np.load(directory / f"{split}.offsets.npy")
        self.store = np.memmap(directory / f"{split}.bin", dtype=np.int16, mode="r",
                               shape=(int(self.offsets[-1]), hidden_size))

    def tokens(self, indices: list[int], device) -> torch.Tensor:
        import numpy as np

        parts = [self.store[self.offsets[i] : self.offsets[i + 1]] for i in indices]
        return torch.from_numpy(np.concatenate(parts)).to(device).view(torch.bfloat16)

    def lengths(self, indices: list[int]) -> list[int]:
        return [int(self.offsets[i + 1] - self.offsets[i]) for i in indices]


def write_state(wrapped, tokenizer, passages, args, grad: bool, hidden: torch.Tensor | None = None,
                lengths: list[int] | None = None) -> torch.Tensor:
    if hidden is None:
        hidden, lengths = encode(wrapped, tokenizer, passages, args.encode_batch, args.device, return_lengths=True)
    with torch.set_grad_enabled(grad):
        return wrapped.interface.write(hidden, chunk_size=args.chunk_size, checkpoint=grad, lengths=lengths)


def batch_loss(model, wrapped, tokenizer, formatter, prompts: list[str], answers: list[str], state, device,
               reduction: str = "mean"):
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids, labels = [], []
    for prompt, answer in zip(prompts, answers, strict=True):
        p = tokenizer(formatter(prompt))["input_ids"]
        a = tokenizer(answer, add_special_tokens=False)["input_ids"] + [end]
        ids.append(p + a)
        labels.append([-100] * len(p) + a)
    width = max(map(len, ids))
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in ids], device=device)
    label_ids = torch.tensor([x + [-100] * (width - len(x)) for x in labels], device=device)
    attention = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids], device=device)
    with wrapped.reading(state):
        hidden = model.model(input_ids=input_ids, attention_mask=attention, use_cache=False).last_hidden_state
    # logits only where a label follows (same loss as full logits, far less memory)
    target = label_ids[:, 1:]
    keep = target != -100
    logits = model.lm_head(hidden[:, :-1][keep])
    if reduction == "sum":  # (summed loss, target tokens) for token-weighted micro-batching
        return nn.functional.cross_entropy(logits.float(), target[keep], reduction="sum"), int(keep.sum())
    return nn.functional.cross_entropy(logits.float(), target[keep])


@torch.no_grad()
def generate(model, wrapped, tokenizer, prompts: list[str], state, device, batch: int = 32,
             max_new_tokens: int = 32) -> list[str]:
    out = []
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), batch):
        enc = tokenizer(prompts[start : start + batch], padding=True, return_tensors="pt").to(device)
        with wrapped.reading(state):
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id, top_p=None, top_k=None, temperature=None)
        out += [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen[:, enc["input_ids"].shape[1]:]]
    return out


def cloze_em(preds: list[str], answers: list[str]) -> float:
    return 100 * sum(score_prediction(p, [normalize_answer(a)])[0] for p, a in zip(preds, answers)) / len(answers)


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.data / "train.jsonl")
    val_rows = read_jsonl(args.data / "val.jsonl")
    if args.fewshot:
        train_rows = train_rows[: -args.fewshot]
    if args.full_cloze:
        for r in train_rows + val_rows:
            r["cloze"] = r["passage"].replace(r["answer"], "___", 1)
    tokenizer, model = load_model(args)
    formatter = ClozeFormatter(tokenizer, load_shots(args.data, args.fewshot))
    interface = DeltaInterface(model.config.hidden_size, args.width, beta_bias=args.beta_bias,
                               shift_values=args.shift_values, bias_state=args.bias_state).to(args.device)
    wrapped = InterfaceQwen(model, interface, args.layer)
    params = list(interface.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    sizes = [int(x) for x in args.bank_sizes.split(",")]
    rng = random.Random(args.seed)
    log = (args.output_dir / "log.jsonl").open("a")

    cache_dir = args.cache or cache_dir_for(args.data, args.layer, args.model, args.revision)
    train_cache = HiddenCache(cache_dir, "train", model.config.hidden_size)
    if args.standardise:  # statistics of the first 4,096 train sentences (fixed, not learned)
        sample = train_cache.tokens(list(range(4096)), args.device).float()
        interface.set_input_stats(sample.mean(0), sample.std(0))
        del sample
    val_cache = HiddenCache(cache_dir, "val", model.config.hidden_size)
    val_bank = val_rows[: args.val_bank]
    val_q = random.Random(1).sample(val_bank, args.val_queries)
    val_prompts = [formatter(cloze_question(r["cloze"])) for r in val_q]
    val_answers = [r["answer"] for r in val_q]

    def cached_state(cache, ids, grad):  # noqa: E306
        return write_state(wrapped, tokenizer, None, args, grad=grad, hidden=cache.tokens(ids, args.device),
                           lengths=cache.lengths(ids))

    @torch.no_grad()
    def gap(n: int) -> dict:
        """Teacher-forced loss on 32 queries with the real S vs S of n other val sentences."""
        perm = random.Random(5).sample(range(len(val_rows)), 2 * n)
        q = random.Random(6).sample(perm[:n], min(32, n))
        pr = [cloze_question(val_rows[i]["cloze"]) for i in q]
        an = [val_rows[i]["answer"] for i in q]
        real, wrong = (batch_loss(model, wrapped, tokenizer, formatter, pr, an,
                                  cached_state(val_cache, ids, False), args.device).item()
                       for ids in (perm[:n], perm[n:]))
        return {f"gap_real_N{n}": round(real, 3), f"gap_wrong_N{n}": round(wrong, 3)}

    def validate(step: int) -> dict:
        interface.eval()
        state = cached_state(val_cache, list(range(len(val_bank))), False)
        preds = generate(model, wrapped, tokenizer, val_prompts, state, args.device)
        entry = {"step": step, "val_em": round(cloze_em(preds, val_answers), 2),
                 "beta_mean_bias": float(interface.w_b.bias), "examples": list(zip(val_answers[:4], preds[:4]))}
        for n in (int(x) for x in args.gap_sizes.split(",") if x):
            entry.update(gap(n))
        if step == 0:
            null = generate(model, wrapped, tokenizer, val_prompts, None, args.device)
            entry["val_closed_book_em"] = round(cloze_em(null, val_answers), 2)
        interface.train()
        return entry

    started, losses = time.perf_counter(), []
    entry = validate(0)
    print(json.dumps(entry), flush=True)
    log.write(json.dumps(entry) + "\n")
    for step in range(1, args.steps + 1):
        n = rng.choice(sizes)
        if n >= args.queries:
            bank_ids = rng.sample(range(len(train_rows)), n)
            queries = [train_rows[i] for i in rng.sample(bank_ids, args.queries)]
            state = cached_state(train_cache, bank_ids, True)
        else:  # several independent episodes, one state per query row
            episodes = math.ceil(args.queries / n)
            pool = rng.sample(range(len(train_rows)), episodes * n)
            banks = [pool[e * n : (e + 1) * n] for e in range(episodes)]
            per_q = [(e, i) for e, b in enumerate(banks) for i in rng.sample(b, n)][: args.queries]
            states = torch.stack([cached_state(train_cache, b, True) for b in banks])
            queries = [train_rows[i] for _, i in per_q]
            state = states[torch.tensor([e for e, _ in per_q], device=args.device)]
        prompts_q = [cloze_question(r["cloze"]) for r in queries]
        answers_q = [r["answer"] for r in queries]
        optimizer.zero_grad(set_to_none=True)
        if not args.micro_batch:
            loss = batch_loss(model, wrapped, tokenizer, formatter, prompts_q, answers_q, state, args.device)
            loss.backward()
        else:  # token-weighted micro-batches; S (and its write graph) shared, kept across backward passes
            total_loss, total_tokens, size = 0.0, 0, args.micro_batch
            for m in range(0, len(queries), size):
                part_state = state if state.dim() == 2 else state[m : m + size]
                part, count = batch_loss(model, wrapped, tokenizer, formatter, prompts_q[m : m + size],
                                         answers_q[m : m + size], part_state, args.device, reduction="sum")
                part.backward(retain_graph=m + size < len(queries))
                total_loss, total_tokens = total_loss + part.item(), total_tokens + count
            for q in params:
                if q.grad is not None:
                    q.grad.div_(total_tokens)
            loss = torch.tensor(total_loss / total_tokens)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        schedule.step()
        losses.append(loss.item())
        if step % 25 == 0:
            entry = {"step": step, "loss": round(sum(losses[-25:]) / 25, 4), "N": n,
                     "lr": schedule.get_last_lr()[0], "seconds": round(time.perf_counter() - started, 1)}
            print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + "\n")
        if step % args.val_every == 0 or step == args.steps:
            entry = validate(step)
            print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            torch.save({"interface": interface.state_dict(), "args": {k: str(v) if isinstance(v, Path) else v
                                                                      for k, v in vars(args).items()},
                        "step": step}, args.output_dir / "interface.pt")
    meta = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "trainable_parameters": sum(p.numel() for p in params), "optimizer_steps": args.steps,
            "presented_queries": args.steps * args.queries, "train_seconds": round(time.perf_counter() - started, 1),
            "data_manifest": json.loads((args.data / "manifest.json").read_text())["outputs"],
            "repository_commit": git_revision(), "working_tree_dirty": git_dirty(), "torch": torch.__version__}
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
