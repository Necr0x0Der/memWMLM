#!/usr/bin/env python3
"""Extract chat-matched features before generation, with/without a question."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers

from common import (MODEL, REVISION, SCHEMA, SPLITS, POOLINGS, git_revision,
                    load_frozen_model, parse_layer_spec, pool_selected_hidden_states,
                    render_prompt, save_json, sha256, subset_dataset)


def extract_context(base_model, tokenizer, rows, question, layers, output,
                    batch_size, max_length, device):
    output.mkdir(parents=True, exist_ok=False)
    shards = []
    with (output / 'prompts.jsonl').open('w') as prompt_file:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prompts = [render_prompt(tokenizer, row['text'], question) for row in batch]
            encoded = tokenizer(prompts, padding=True, truncation=False,
                                add_special_tokens=False, return_tensors='pt')
            if encoded['input_ids'].shape[1] > max_length:
                raise ValueError('Prompt exceeds --max-length; refusing truncation')
            ids = encoded['input_ids'].to(device)
            mask = encoded['attention_mask'].to(device)
            with torch.inference_mode():
                # Base transformer only: no LM head, generate, or source KV reuse.
                result = base_model(input_ids=ids, attention_mask=mask,
                                    output_hidden_states=True, use_cache=False, return_dict=True)
                last, mean = pool_selected_hidden_states(result.hidden_states, mask, layers)
            payload = dict(ids=[r['id'] for r in batch], layer_indices=torch.tensor(layers),
                           token_counts=mask.sum(1).cpu(),
                           last_token=last.cpu().to(torch.float16),
                           mean_pool=mean.cpu().to(torch.float16))
            if not all(torch.isfinite(payload[p]).all() for p in POOLINGS):
                raise ValueError('Non-finite extracted features')
            shard = output / f'shard-{len(shards):05d}.pt'
            torch.save(payload, shard)
            shards.append(dict(path=shard.name, sha256=sha256(shard), examples=len(batch)))
            for row, text in zip(batch, prompts):
                prompt_file.write(json.dumps(dict(id=row['id'], prompt=text), ensure_ascii=False) + '\n')
            print(json.dumps(dict(path=str(output), completed=start+len(batch), total=len(rows))), flush=True)
    return dict(examples=len(rows), shards=shards, prompts_sha256=sha256(output/'prompts.jsonl'))


def run(args):
    dataset, rows, audit = subset_dataset(args.dataset_dir, args.instruction, args.fact_order)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: extraction has not run')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    tokenizer, model = load_frozen_model(MODEL, REVISION, getattr(torch, args.dtype),
                                         args.device, local_files_only=True)
    layers = parse_layer_spec(args.layers, model.config.num_hidden_layers)
    contexts = {'no_question': None, **{n: p['question'] for n, p in dataset['probes'].items()}}
    args.output_dir.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, model=dict(name=MODEL, requested_revision=REVISION,
                    hidden_size=model.config.hidden_size, dtype=args.dtype),
                    dataset_manifest_sha256=sha256(args.dataset_dir/'manifest.json'),
                    dataset_files={s: sha256(args.dataset_dir/f'{s}.jsonl') for s in SPLITS},
                    condition=dict(instruction=args.instruction, fact_order=args.fact_order),
                    layers=list(layers), poolings=list(POOLINGS), audit=audit,
                    repository_commit=git_revision(), script_sha256=sha256(Path(__file__)),
                    common_sha256=sha256(Path(__file__).with_name('common.py')),
                    torch_version=torch.__version__, transformers_version=transformers.__version__,
                    tokenizer_chat_template=tokenizer.chat_template,
                    prompt=dict(chat_template=True, thinking=False, generation=False,
                                answer_options=False, last_token='last nonpadding assistant-prefix token',
                                mean_pool='all nonpadding tokens, including chat wrapper and question'),
                    settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    contexts={})
    save_json(args.output_dir/'protocol.json', manifest)
    for context, question in contexts.items():
        splits = {}
        for split in SPLITS:
            splits[split] = extract_context(model.model, tokenizer, rows[split], question, layers,
                args.output_dir/context/split, args.batch_size, args.max_length, args.device)
        manifest['contexts'][context] = dict(question=question, splits=splits)
    save_json(args.output_dir/'manifest.json', manifest)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--instruction', choices=['none', 'imagine', 'track_mary', 'track_people'], default='none')
    p.add_argument('--fact-order', choices=['canonical', 'shuffled'], default='shuffled')
    p.add_argument('--layers', default='8,16,28,36')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-length', type=int, default=512)
    args = p.parse_args()
    if min(args.batch_size, args.max_length) <= 0:
        p.error('Batch size and max length must be positive')
    return args


if __name__ == '__main__':
    run(parse_args())
