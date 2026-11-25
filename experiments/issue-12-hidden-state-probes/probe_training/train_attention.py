#!/usr/bin/env python3
"""Shared single-query attention over frozen token states, followed by linear heads."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from train import audit_dataset, paired_effects
from train_linear_probes import SPLITS, encode_targets, metrics, sha256, shuffle_targets_by_group


class AttentionProbeBank(nn.Module):
    """One query and one pooled vector for all tasks; no key/value projections."""
    def __init__(self, width, counts, mean, scale, mode='attention'):
        super().__init__()
        self.register_buffer('mean', mean.clone())
        self.register_buffer('scale', scale.clone())
        self.q = nn.Parameter(torch.zeros(width), requires_grad=mode == 'attention')
        self.heads = nn.ModuleDict({n: nn.Linear(width, c) for n, c in counts.items()})

    def pool(self, tokens, mask):
        if mask.ndim != 2 or tokens.shape[:2] != mask.shape or not mask.any(dim=1).all():
            raise ValueError('Each example must have at least one valid token')
        x = (tokens - self.mean) / self.scale
        scores = (x @ self.q) / math.sqrt(x.shape[-1])
        weights = scores.masked_fill(~mask.bool(), -torch.inf).softmax(dim=1)
        # Also mask values, so padding cannot influence the weighted sum.
        z = (weights.unsqueeze(-1) * x.masked_fill(~mask.bool().unsqueeze(-1), 0)).sum(dim=1)
        return z, weights

    def forward(self, tokens, mask):
        z, _ = self.pool(tokens, mask)
        return {n: head(z) for n, head in self.heads.items()}


def load_tokens(feature_dir, fm, split, rows, layer, verified):
    """Read only one layer into RAM; trim padding was already done by extractor."""
    expected = {r['id'] for r in rows}
    found = {}
    width = None
    records = fm['splits'][split]['shards']
    if not records:
        raise ValueError(f'No shards for {split}')
    for record in records:
        path = feature_dir / split / record['path']
        if path not in verified:
            if sha256(path) != record['sha256']:
                raise ValueError(f'Shard hash mismatch: {path}')
            verified.add(path)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if 'token_states' not in payload:
            raise ValueError('Per-token states missing. Re-extract with --save-token-states; pooled features cannot be used.')
        layers = payload['layer_indices'].tolist()
        if layers != fm['features']['layer_indices']:
            raise ValueError('Shard/manifest layers differ')
        offset = layers.index(layer)
        if len(payload['ids']) != len(payload['token_states']):
            raise ValueError('Token-state ID count mismatch')
        for i, (row_id, states) in enumerate(zip(payload['ids'], payload['token_states'])):
            if row_id in found or row_id not in expected:
                raise ValueError('Duplicate or unexpected feature ID')
            if states.ndim != 3 or states.shape[0] != len(layers) or states.shape[1] != int(payload['token_counts'][i]) or states.shape[1] == 0:
                raise ValueError('Invalid ragged token-state shape')
            if width is not None and width != states.shape[-1]:
                raise ValueError('Inconsistent hidden width')
            width = states.shape[-1]
            # Clone releases all unselected layers with the shard payload.
            found[row_id] = states[offset].clone()
        del payload
    if set(found) != expected:
        raise ValueError('Missing feature IDs')
    return [found[r['id']] for r in rows]


def batch(tokens, indices, device):
    chosen = [tokens[i] for i in indices]
    x = pad_sequence(chosen, batch_first=True).to(device=device, dtype=torch.float32)
    lengths = torch.tensor([len(t) for t in chosen], device=device)
    mask = torch.arange(x.shape[1], device=device)[None, :] < lengths[:, None]
    return x, mask


def make_bank(tokens, counts, mode, seed, device):
    # Match the train-only standardization of the mean-pool baseline.
    means = torch.stack([x.float().mean(0) for x in tokens])
    mean = means.mean(0)
    scale = means.std(0, unbiased=False).clamp_min(1e-6)
    torch.manual_seed(seed)
    return AttentionProbeBank(mean.numel(), counts, mean, scale, mode).to(device)


def predict_bank(model, tokens, batch_size, device):
    model.eval()
    results = {n: [] for n in model.heads}
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            x, mask = batch(tokens, list(range(start, min(start + batch_size, len(tokens)))), device)
            for n, logits in model(x, mask).items():
                results[n].append(logits.argmax(1).cpu())
    return {n: torch.cat(v) for n, v in results.items()}


def train_epochs(model, tokens, targets, epochs, batch_size, lr, l2, seed, device):
    """Yield each epoch; caller may score validation or retain a fixed epoch."""
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(tokens), generator=rng).tolist()
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            x, mask = batch(tokens, indices, device)
            logits = model(x, mask)
            loss = torch.stack([nn.functional.cross_entropy(logits[n], y[indices].to(device)) for n, y in targets.items()]).mean()
            penalty = torch.stack([h.weight.square().sum() for h in model.heads.values()]).mean() + model.q.square().sum()
            loss = loss + .5 * l2 * penalty
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        yield epoch


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_dir}')
    manifest, rows, audit = audit_dataset(args.dataset_dir)
    fm = json.loads((args.feature_dir / 'manifest.json').read_text())
    if fm['model']['name'] != 'Qwen/Qwen3-4B' or fm['model']['requested_revision'] != '1cfa9a7208912126459214e8b04321603b3df60c':
        raise ValueError('Expected pinned Qwen3-4B features')
    if not fm['features'].get('token_states'):
        raise ValueError('Re-extract with --save-token-states: mean/last-token features are insufficient')
    for split in SPLITS:
        if sha256(args.dataset_dir / f'{split}.jsonl') != fm['dataset']['files'][f'{split}.jsonl']:
            raise ValueError('Feature/dataset hash mismatch')
    layers = fm['features']['layer_indices'] if args.layers == 'all' else [int(x) for x in args.layers.split(',')]
    if not layers or len(set(layers)) != len(layers) or any(x not in fm['features']['layer_indices'] for x in layers):
        raise ValueError('Invalid requested layer subset')
    probes = manifest['probes']
    counts = {n: len(p['classes']) for n, p in probes.items()}
    cells, subsets, indices, ys = [], {}, {}, {}
    for instruction in sorted(manifest['instructions']):
        for order in manifest['fact_orders']:
            cell = f'{instruction}/{order}'
            cells.append(cell)
            indices[cell] = {s: [i for i, r in enumerate(rows[s]) if (r['instruction_id'], r['fact_order']) == (instruction, order)] for s in SPLITS}
            subsets[cell] = {s: [rows[s][i] for i in indices[cell][s]] for s in SPLITS}
            ys[cell] = {s: encode_targets(subsets[cell][s], probes) for s in SPLITS}
    selected, candidates, verified = {}, {c: [] for c in cells}, set()
    # All selection completes before any test hidden states are loaded/evaluated.
    for layer in layers:
        tokens = {s: load_tokens(args.feature_dir, fm, s, rows[s], layer, verified) for s in ('train', 'validation')}
        for cell in cells:
            xs = {s: [tokens[s][i] for i in indices[cell][s]] for s in tokens}
            for l2 in args.l2_values:
                model = make_bank(xs['train'], counts, args.mode, args.seed, args.device)
                for epoch in train_epochs(model, xs['train'], ys[cell]['train'], args.epochs, args.batch_size, args.lr, l2, args.seed, args.device):
                    pred = predict_bank(model, xs['validation'], args.batch_size, args.device)
                    score = metrics(pred, ys[cell]['validation'], counts)
                    config = {'layer': layer, 'l2': l2, 'epoch': epoch, 'mode': args.mode}
                    candidates[cell].append({**config, 'validation': score})
                    key = (score['mean_macro_accuracy'], score['joint_exact_state_accuracy'], l2, -epoch)
                    if cell not in selected or key > selected[cell]['key']:
                        selected[cell] = {'key': key, 'config': config, 'model': copy.deepcopy(model).cpu()}
                print(f'Selected so far: {cell}: {selected[cell]["config"]}', flush=True)
            del model, xs
        del tokens
    report = {
        'schema_version': 'shared-query-attention-probes-v1', 'model': fm['model'], 'audit': audit,
        'dataset_manifest_sha256': sha256(args.dataset_dir / 'manifest.json'),
        'feature_manifest_sha256': sha256(args.feature_dir / 'manifest.json'),
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'protocol': {'query': 'one shared q and z for every question within each instruction/order bank',
                     'inputs': 'all valid input tokens including instruction; no question; frozen Qwen',
                     'selection': 'bank validation mean macro, joint exact, stronger L2, earlier epoch, first layer',
                     'normalization': 'train situation-mean feature mean/std applied to every token',
                     'negative_control': 'train group-label shuffle at locked config; q and heads retrained',
                     'intervals': 'paired test-group bootstrap; conditional on selection; uncorrected exploratory'},
        'cells': {}, 'instruction_effects': {}, 'fact_order_effects': {}}
    args.output_dir.mkdir(parents=True)
    saved = {}
    for layer in sorted({s['config']['layer'] for s in selected.values()}):
        tokens = {s: load_tokens(args.feature_dir, fm, s, rows[s], layer, verified) for s in SPLITS}
        for cell in cells:
            selection = selected[cell]
            if selection['config']['layer'] != layer:
                continue
            config = selection['config']
            xs = {s: [tokens[s][i] for i in indices[cell][s]] for s in SPLITS}
            model = selection['model'].to(args.device)
            predictions = {s: predict_bank(model, xs[s], args.batch_size, args.device) for s in ('validation', 'test')}
            shuffled = shuffle_targets_by_group(ys[cell]['train'], subsets[cell]['train'], args.seed)
            control = make_bank(xs['train'], counts, args.mode, args.seed, args.device)
            for _ in train_epochs(control, xs['train'], shuffled, config['epoch'], args.batch_size, args.lr, config['l2'], args.seed, args.device):
                pass
            control_pred = predict_bank(control, xs['test'], args.batch_size, args.device)
            report['cells'][cell] = {'selected': config, 'candidates': candidates[cell],
                'evaluation': {s: metrics(predictions[s], ys[cell][s], counts) for s in predictions},
                'shuffled_label_control_test': metrics(control_pred, ys[cell]['test'], counts)}
            torch.save({'config': config, 'cell': cell, 'probes': probes,
                        'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()}},
                       args.output_dir / (cell.replace('/', '--') + '.pt'))
            saved[cell] = subsets[cell]['test'], predictions['test']
            model.cpu()
            del xs, control
        del tokens
    for order in manifest['fact_orders']:
        for instruction in sorted(set(manifest['instructions']) - {'none'}):
            report['instruction_effects'][f'{instruction}-minus-none/{order}'] = paired_effects(
                *saved[f'none/{order}'], *saved[f'{instruction}/{order}'], probes, args.bootstrap_samples, args.seed)
    for instruction in manifest['instructions']:
        report['fact_order_effects'][instruction] = paired_effects(*saved[f'{instruction}/canonical'],
            *saved[f'{instruction}/shuffled'], probes, args.bootstrap_samples, args.seed)
    with (args.output_dir / 'test_predictions.jsonl').open('w') as handle:
        for cell, (items, pred) in saved.items():
            for i, row in enumerate(items):
                handle.write(json.dumps({'id': row['id'], 'episode_id': row['episode_id'], 'group_id': row['group_id'],
                    'cell': cell, 'predicted_class_indices': {n: y[i].item() for n, y in pred.items()}}) + '\n')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return args.output_dir / 'report.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-dir', 'feature-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=('attention', 'mean'), default='attention')
    parser.add_argument('--layers', default='all')
    parser.add_argument('--l2-values', default='0.0001,0.001')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12012)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    args.l2_values = [float(v) for v in args.l2_values.split(',')]
    if not args.l2_values or any(not math.isfinite(v) or v < 0 for v in args.l2_values):
        parser.error('L2 values must be finite and nonnegative')
    if min(args.epochs, args.batch_size, args.threads, args.bootstrap_samples) <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('Counts and learning rate must be positive and finite')
    torch.set_num_threads(args.threads)
    print(run(args))


if __name__ == '__main__':
    main()
