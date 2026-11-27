#!/usr/bin/env python3
"""Train a question-independent soft prefix; evaluate source-free Qwen QA."""
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
import transformers

PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))
from extract_hidden_features import load_frozen_model, validate_dataset_manifest, git_revision
from train_linear_probes import load_feature_split, read_jsonl, sha256, encode_targets, metrics
from evaluate_full_context_qa import answer_text, parse_short_answer

MODEL = 'Qwen/Qwen3-4B'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
SCHEMA = 'source-free-soft-prefix-v1'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


class SoftPrefix(nn.Module):
    """Affine map of train-standardized features; never receives a question."""
    def __init__(self, features, hidden_size, slots, embedding_std):
        super().__init__()
        self.slots = slots
        self.hidden_size = hidden_size
        self.register_buffer('mean', features.float().mean(0))
        self.register_buffer('scale', features.float().std(0, unbiased=False).clamp_min(1e-6))
        self.projection = nn.Linear(features.shape[-1], slots * hidden_size)
        nn.init.normal_(self.projection.weight, std=embedding_std / math.sqrt(features.shape[-1]))
        nn.init.zeros_(self.projection.bias)

    def forward(self, features):
        x = (features.float() - self.mean) / self.scale
        return self.projection(x).reshape(-1, self.slots, self.hidden_size)


def prompt_ids(tokenizer, question, answers):
    # Accept only question/schema, not an episode row, source text or target.
    content = ('Answer using the preceding memory.\nQuestion: ' + question
               + '\nReturn exactly one of these lines and nothing else:\n'
               + '\n'.join('FINAL: ' + a for a in answers))
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(rendered, add_special_tokens=False)


def make_examples(rows, probes, tokenizer):
    templates = {}
    for name, spec in probes.items():
        answers = [answer_text(c) for c in spec['classes']]
        templates[name] = (prompt_ids(tokenizer, spec['question'], answers), answers)
    examples = []
    for i, row in enumerate(rows):
        for name, spec in probes.items():
            prompt, answers = templates[name]
            target = answer_text(row['targets'][name])
            answer = tokenizer.encode('FINAL: ' + target, add_special_tokens=False)
            answer.append(tokenizer.eos_token_id)
            examples.append(dict(row=i, question=name, prompt=prompt, answer=answer,
                                 allowed=answers, target=target))
    return examples


def training_batch(model, adapter, features, examples, pad_id):
    device = next(adapter.parameters()).device
    memory = adapter(features[[e['row'] for e in examples]].to(device))
    k = memory.shape[1]
    lengths = [len(e['prompt']) + len(e['answer']) for e in examples]
    ids = torch.full((len(examples), max(lengths)), pad_id, device=device, dtype=torch.long)
    mask = torch.zeros((len(examples), k + max(lengths)), device=device, dtype=torch.long)
    labels = torch.full_like(mask, -100)
    for i, e in enumerate(examples):
        ids[i, :lengths[i]] = torch.tensor(e['prompt'] + e['answer'], device=device)
        mask[i, :k + lengths[i]] = 1
        start = k + len(e['prompt'])
        labels[i, start:k + lengths[i]] = torch.tensor(e['answer'], device=device)
    text = model.get_input_embeddings()(ids)
    inputs = torch.cat((memory.to(text.dtype), text), dim=1)
    # No source input_ids or source cache; gradients must traverse frozen Qwen.
    return model(inputs_embeds=inputs, attention_mask=mask, labels=labels,
                 use_cache=False).loss, int((labels != -100).sum())


def validation_loss(model, adapter, features, examples, pad_id, batch_size):
    total, count = 0., 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            loss, n = training_batch(model, adapter, features,
                                     examples[start:start + batch_size], pad_id)
            total += loss.item() * n
            count += n
    return total / count


@torch.no_grad()
def generate_one(model, tokenizer, prompt, memory, max_new_tokens):
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    inputs = model.get_input_embeddings()(ids)
    if memory is not None:
        inputs = torch.cat((memory.to(device=device, dtype=inputs.dtype), inputs), dim=1)
    # With inputs_embeds alone, HF returns only the newly generated token IDs.
    generated = model.generate(
        inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=device),
        do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        use_cache=True)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def evaluate(model, tokenizer, adapter, features, rows, examples, probes, output, max_new_tokens):
    results = {}
    targets = encode_targets(rows, probes)
    for mode in ('memory', 'question_only'):
        predictions = {name: torch.full((len(rows),), -1, dtype=torch.long) for name in probes}
        invalid = 0
        with (output / f'test-{mode}.jsonl').open('w') as stream:
            for index, e in enumerate(examples):
                with torch.no_grad():
                    memory = (adapter(features[e['row']:e['row'] + 1].to(next(adapter.parameters()).device))
                              if mode == 'memory' else None)
                response = generate_one(model, tokenizer, e['prompt'], memory, max_new_tokens)
                parsed = parse_short_answer(response, e['allowed'])
                invalid += parsed is None
                predictions[e['question']][e['row']] = parsed if parsed is not None else -1
                row = rows[e['row']]
                stream.write(json.dumps(dict(id=row['id'], group_id=row['group_id'],
                    question=e['question'], target=e['target'], response=response,
                    prediction=None if parsed is None else e['allowed'][parsed]), ensure_ascii=False) + '\n')
                stream.flush()
                if (index + 1) % 50 == 0:
                    print(json.dumps(dict(stage='test', mode=mode, completed=index + 1)), flush=True)
        result = metrics(predictions, targets, {n: len(p['classes']) for n, p in probes.items()})
        result['joint_exact_targets_accuracy'] = result.pop('joint_exact_state_accuracy')
        result['invalid_answers'] = invalid
        result['answers'] = len(examples)
        results[mode] = result
    return results


def load_split(args, manifest, split, poolings):
    rows = read_jsonl(args.dataset_dir / f'{split}.jsonl')
    layers, tensors = load_feature_split(args.feature_dir, split, manifest['splits'][split],
                                        [r['id'] for r in rows], poolings)
    selected = [i for i, r in enumerate(rows)
                if r['instruction_id'] == args.instruction and r['fact_order'] == args.fact_order]
    if not selected:
        raise ValueError('Empty condition')
    return [rows[i] for i in selected], layers, {p: x[selected].float() for p, x in tensors.items()}


def run(args):
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no training or test evaluation was run.')
    dataset, hashes = validate_dataset_manifest(args.dataset_dir)
    fm = json.loads((args.feature_dir / 'manifest.json').read_text())
    if fm['dataset']['files'] != hashes:
        raise ValueError('Feature/dataset provenance mismatch')
    if (fm['model']['name'], fm['model']['requested_revision']) != (args.model, args.revision):
        raise ValueError('Reader and feature model/revision must match')
    if fm['tokenization']['chat_template_applied'] or fm.get('max_examples_per_split') is not None:
        raise ValueError('Require complete, raw situation-only extraction')
    poolings = args.poolings.split(',')
    train_rows, layers, train = load_split(args, fm, 'train', poolings)
    val_rows, val_layers, val = load_split(args, fm, 'validation', poolings)
    if layers != val_layers:
        raise ValueError('Layer mismatch')
    chosen_layers = [int(x) for x in args.layers.split(',')]
    if not set(chosen_layers) <= set(layers):
        raise ValueError('Requested layers absent in feature cache')
    # Metadata audit only; test features/answers are not used for selection.
    groups = {}
    for split in ('train', 'validation', 'test'):
        groups[split] = {r['group_id'] for r in read_jsonl(args.dataset_dir / f'{split}.jsonl')}
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if groups[a] & groups[b]:
            raise ValueError('Abstract-layout leakage across splits')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    provenance = dict(schema=SCHEMA, config=config, repository_commit=git_revision(),
                      script_sha256=sha256(Path(__file__)), dataset_files=hashes,
                      feature_manifest_sha256=sha256(args.feature_dir / 'manifest.json'),
                      torch_version=torch.__version__, transformers_version=transformers.__version__,
                      groups={k: len(v) for k, v in groups.items()},
                      selection='minimum validation answer-token NLL; earlier candidate/epoch breaks ties',
                      controls='question_only; oracle/null/swap deferred, no causal-use claim')
    save_json(args.output_dir / 'protocol.json', provenance)
    tokenizer, model = load_frozen_model(args.model, args.revision,
        getattr(torch, args.dtype), args.device, local_files_only=True)
    train_examples = make_examples(train_rows, dataset['probes'], tokenizer)
    val_examples = make_examples(val_rows, dataset['probes'], tokenizer)
    # A fixed sample avoids making a full FP32 copy of Qwen's large vocabulary.
    embedding_std = model.get_input_embeddings().weight[:4096].detach().float().std().item()
    best = math.inf
    started = time.monotonic()
    history = []
    for pooling in poolings:
        for layer in chosen_layers:
            torch.manual_seed(args.seed)
            rng = random.Random(args.seed)
            j = layers.index(layer)
            x, vx = train[pooling][:, j], val[pooling][:, j]
            adapter = SoftPrefix(x, model.config.hidden_size, args.slots, embedding_std).to(args.device)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.)
            for epoch in range(1, args.epochs + 1):
                order = list(range(len(train_examples)))
                rng.shuffle(order)
                total, count = 0., 0
                for offset in range(0, len(order), args.batch_size):
                    batch = [train_examples[i] for i in order[offset:offset + args.batch_size]]
                    optimizer.zero_grad(set_to_none=True)
                    loss, n = training_batch(model, adapter, x, batch, tokenizer.pad_token_id)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite training loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    total += loss.item() * n
                    count += n
                vloss = validation_loss(model, adapter, vx, val_examples, tokenizer.pad_token_id, args.batch_size)
                if not math.isfinite(vloss):
                    raise RuntimeError('Non-finite validation loss')
                item = dict(pooling=pooling, layer=layer, epoch=epoch, train_nll=total/count, validation_nll=vloss)
                history.append(item)
                print(json.dumps(item), flush=True)
                save_json(args.output_dir / 'history.json', history)
                if vloss < best:
                    best = vloss
                    torch.save(dict(state={k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                                    selected=item, config=config,
                                    trainable_parameters=sum(p.numel() for p in adapter.parameters())),
                               args.output_dir / 'selected.pt')
            del optimizer, adapter
    checkpoint = torch.load(args.output_dir / 'selected.pt', weights_only=True, map_location='cpu')
    selected = checkpoint['selected']
    # Selection is finished before test features are opened.
    test_rows, test_layers, test = load_split(args, fm, 'test', [selected['pooling']])
    if test_layers != layers:
        raise ValueError('Test layer mismatch')
    x = train[selected['pooling']][:, layers.index(selected['layer'])]
    adapter = SoftPrefix(x, model.config.hidden_size, args.slots, embedding_std).to(args.device)
    adapter.load_state_dict(checkpoint['state'])
    adapter.eval().requires_grad_(False)
    tx = test[selected['pooling']][:, layers.index(selected['layer'])]
    results = evaluate(model, tokenizer, adapter, tx, test_rows,
        make_examples(test_rows, dataset['probes'], tokenizer), dataset['probes'],
        args.output_dir, args.max_new_tokens)
    save_json(args.output_dir / 'report.json', dict(**provenance, selected=selected,
        trainable_parameters=checkpoint['trainable_parameters'], test=results,
        elapsed_seconds=time.monotonic()-started, checkpoint_sha256=sha256(args.output_dir/'selected.pt')))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--feature-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model', default=MODEL)
    p.add_argument('--revision', default=REVISION)
    p.add_argument('--instruction', choices=['none', 'imagine', 'track_mary', 'track_people'], default='none')
    p.add_argument('--fact-order', choices=['canonical', 'shuffled'], default='shuffled')
    p.add_argument('--poolings', default='mean_pool,last_token,sentence_end_mean')
    p.add_argument('--layers', default='28')
    p.add_argument('--slots', type=int, default=4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=12012)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    p.add_argument('--max-new-tokens', type=int, default=64)
    args = p.parse_args()
    if min(args.slots, args.epochs, args.batch_size, args.max_new_tokens) <= 0 or args.lr <= 0:
        p.error('Budgets and learning rate must be positive')
    if not set(args.poolings.split(',')) <= {'last_token', 'mean_pool', 'sentence_end_mean'}:
        p.error('Unknown pooling')
    return args


if __name__ == '__main__':
    run(parse_args())
