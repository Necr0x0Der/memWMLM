#!/usr/bin/env python3
"""Condition-specific linear probes with independent validation selection."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_linear_probes import (  # noqa: E402
    AVAILABLE_POOLINGS, SPLITS, encode_targets, fit_probes, load_feature_split,
    metrics, predict, read_jsonl, save_probe, sha256, shuffle_targets_by_group,
)


def audit_dataset(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows = {s: read_jsonl(directory / f'{s}.jsonl') for s in SPLITS}
    cells = {(i, o) for i in manifest['instructions'] for o in manifest['fact_orders']}
    seen_groups, seen_episodes = set(), set()
    summary = {}
    for split, items in rows.items():
        if sha256(directory / f'{split}.jsonl') != manifest['files'][f'{split}.jsonl']['sha256']:
            raise ValueError(f'Dataset hash mismatch: {split}')
        groups = {r['group_id'] for r in items}
        episodes = {}
        group_targets = {}
        for r in items:
            if r['split'] != split:
                raise ValueError('Incorrect split metadata')
            episodes.setdefault(r['episode_id'], []).append(r)
            target = json.dumps(r['targets'], sort_keys=True)
            if group_targets.setdefault(r['group_id'], target) != target:
                raise ValueError('Targets vary within abstract-layout group')
        if seen_groups & groups or seen_episodes & episodes.keys():
            raise ValueError('Group/episode leakage across splits')
        seen_groups.update(groups)
        seen_episodes.update(episodes)
        for variants in episodes.values():
            if len(variants) != len(cells) or {(r['instruction_id'], r['fact_order']) for r in variants} != cells:
                raise ValueError('Missing or duplicate matched condition')
            reference = variants[0]
            for r in variants:
                if any(r[k] != reference[k] for k in ('group_id', 'targets', 'state')):
                    raise ValueError('Matched variants have different world states')
            for order in manifest['fact_orders']:
                facts = [r['facts'] for r in variants if r['fact_order'] == order]
                if any(f != facts[0] for f in facts):
                    raise ValueError('Instruction comparison confounded by different facts')
        counts = {}
        for instruction, order in sorted(cells):
            subset = [r for r in items if (r['instruction_id'], r['fact_order']) == (instruction, order)]
            targets = encode_targets(subset, manifest['probes'])
            labels = {name: dict(Counter(y.tolist())) for name, y in targets.items()}
            for name, definition in manifest['probes'].items():
                if len(labels[name]) != len(definition['classes']):
                    raise ValueError(f'Missing class: {split}/{instruction}/{order}/{name}')
            counts[f'{instruction}/{order}'] = {'rows': len(subset), 'class_counts': labels}
        summary[split] = {'rows': len(items), 'groups': len(groups), 'episodes': len(episodes), 'cells': counts}
    return manifest, rows, summary


def select_probe(train_features, validation_features, train_y, validation_y,
                 name, count, layers, l2_values, max_iter, device):
    best, records = None, []
    for pooling in train_features:
        for offset, layer in enumerate(layers):
            for l2 in l2_values:
                fitted = fit_probes(train_features[pooling][:, offset], {name: train_y},
                                    {name: count}, l2, max_iter, device)
                pred = predict(fitted, validation_features[pooling][:, offset], device)
                score = metrics(pred, {name: validation_y}, {name: count})['per_probe'][name]
                config = {'pooling': pooling, 'layer': layer, 'l2': l2}
                records.append({**config, 'validation': score})
                key = (score['macro_accuracy'], score['accuracy'], l2)
                if best is None or key > best[0]:
                    best = (key, fitted, config)
    return best[1], best[2], records


def paired_effects(base_rows, base_pred, other_rows, other_pred, probes, samples, seed):
    """Paired cluster bootstrap; differences use identical episodes and group draws."""
    lookup = {r['episode_id']: i for i, r in enumerate(other_rows)}
    if len(lookup) != len(other_rows) or set(lookup) != {r['episode_id'] for r in base_rows}:
        raise ValueError('Non-unique or unmatched episodes in comparison')
    order = torch.tensor([lookup[r['episode_id']] for r in base_rows])
    for i, r in enumerate(base_rows):
        mate = other_rows[order[i].item()]
        if r['targets'] != mate['targets'] or r['group_id'] != mate['group_id']:
            raise ValueError('Mismatched paired targets/groups')
    targets = encode_targets(base_rows, probes)
    counts = {n: len(p['classes']) for n, p in probes.items()}
    aligned = {n: y[order] for n, y in other_pred.items()}
    mary = [n for n in probes if '_left_of_' in n and 'mary' in n]
    others = [n for n in probes if '_left_of_' in n and 'mary' not in n]

    def effect(indices):
        a = metrics(base_pred, targets, counts, indices)
        b = metrics(aligned, targets, counts, indices)
        delta = {n: b['per_probe'][n]['macro_accuracy'] - a['per_probe'][n]['macro_accuracy'] for n in probes}
        result = {f'probe/{n}': d for n, d in delta.items()}
        result['mean_macro_accuracy'] = b['mean_macro_accuracy'] - a['mean_macro_accuracy']
        result['joint_exact_state_accuracy'] = b['joint_exact_state_accuracy'] - a['joint_exact_state_accuracy']
        if mary and others:
            result['mary_relations'] = sum(delta[n] for n in mary) / len(mary)
            result['bob_john_relations'] = sum(delta[n] for n in others) / len(others)
            result['selectivity_difference_in_differences'] = result['mary_relations'] - result['bob_john_relations']
        return result

    grouped = {}
    for i, r in enumerate(base_rows):
        grouped.setdefault(r['group_id'], []).append(i)
    groups, rng = sorted(grouped), random.Random(seed)
    point = effect(torch.arange(len(base_rows)))
    draws = {k: [] for k in point}
    for _ in range(samples):
        indices = torch.tensor([i for g in rng.choices(groups, k=len(groups)) for i in grouped[g]])
        for key, value in effect(indices).items():
            draws[key].append(value)
    return {k: {'delta': v, 'paired_group_bootstrap_95ci': torch.tensor(draws[k], dtype=torch.float64).quantile(torch.tensor([.025, .975], dtype=torch.float64)).tolist()} for k, v in point.items()}


def run(args):
    manifest, rows, audit = audit_dataset(args.dataset_dir)
    if args.audit_only:
        return {'audit': audit}
    if args.feature_dir is None or args.output_dir is None:
        raise ValueError('--feature-dir and --output-dir are required for training')
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_dir}')
    fm = json.loads((args.feature_dir / 'manifest.json').read_text())
    if fm['model']['name'] != 'Qwen/Qwen3-4B':
        raise ValueError('This experiment requires Qwen/Qwen3-4B features')
    if fm['model'].get('requested_revision') != '1cfa9a7208912126459214e8b04321603b3df60c':
        raise ValueError('Features must use the pinned Qwen3-4B QA revision')
    features, layers = {}, None
    for split in SPLITS:
        if fm['dataset']['files'][f'{split}.jsonl'] != sha256(args.dataset_dir / f'{split}.jsonl'):
            raise ValueError(f'Features from different dataset: {split}')
        ls, features[split] = load_feature_split(args.feature_dir, split, fm['splits'][split],
                                                 [r['id'] for r in rows[split]], args.poolings)
        if layers is not None and ls != layers:
            raise ValueError('Layer mismatch')
        layers = ls
    args.output_dir.mkdir(parents=True)
    probes = manifest['probes']
    counts = {n: len(p['classes']) for n, p in probes.items()}
    report = {'schema_version': 'instruction-specific-probes-v1', 'audit': audit,
              'model': fm['model'], 'dataset_manifest_sha256': sha256(args.dataset_dir / 'manifest.json'),
              'feature_manifest_sha256': sha256(args.feature_dir / 'manifest.json'),
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'selection_rule': 'per cell and question: validation macro accuracy, accuracy, stronger L2; first candidate on remaining ties',
              'interval_scope': 'paired test-group percentile bootstrap; conditional on fitted/selected probes; exploratory, no multiplicity correction',
              'cells': {}, 'instruction_effects': {}, 'fact_order_effects': {}}
    saved = {}
    for instruction in sorted(manifest['instructions']):
        for fact_order in manifest['fact_orders']:
            cell = f'{instruction}/{fact_order}'
            ix = {s: torch.tensor([i for i, r in enumerate(rows[s]) if (r['instruction_id'], r['fact_order']) == (instruction, fact_order)]) for s in SPLITS}
            subset = {s: [rows[s][i] for i in ix[s].tolist()] for s in SPLITS}
            ys = {s: encode_targets(subset[s], probes) for s in SPLITS}
            xs = {s: {p: x[ix[s]] for p, x in features[s].items()} for s in ('train', 'validation')}
            shuffled = shuffle_targets_by_group(ys['train'], subset['train'], args.seed)
            predictions = {s: {} for s in ('validation', 'test')}
            control = {}
            selection = {}
            for name in probes:
                fitted, config, candidates = select_probe(xs['train'], xs['validation'], ys['train'][name], ys['validation'][name],
                                                         name, counts[name], layers, args.l2_values, args.max_iter, args.device)
                p, offset = config['pooling'], layers.index(config['layer'])
                selection[name] = {'selected': config, 'candidates': candidates}
                save_probe(args.output_dir / f'{instruction}--{fact_order}--{name}.pt', fitted,
                           {**config, 'instruction': instruction, 'fact_order': fact_order, 'probe': name, 'classes': probes[name]['classes']})
                for s in predictions:
                    predictions[s][name] = predict(fitted, features[s][p][ix[s], offset], args.device)[name]
                null = fit_probes(xs['train'][p][:, offset], {name: shuffled[name]}, {name: counts[name]},
                                  config['l2'], args.max_iter, args.device)
                control[name] = predict(null, features['test'][p][ix['test'], offset], args.device)[name]
                print(f'Locked {cell}/{name}: {config}', flush=True)
            report['cells'][cell] = {'selection': selection,
                'evaluation': {s: metrics(predictions[s], ys[s], counts) for s in predictions},
                'shuffled_label_control_test': metrics(control, ys['test'], counts)}
            saved[cell] = (subset['test'], predictions['test'])
    for fact_order in manifest['fact_orders']:
        for instruction in sorted(set(manifest['instructions']) - {'none'}):
            report['instruction_effects'][f'{instruction}-minus-none/{fact_order}'] = paired_effects(
                *saved[f'none/{fact_order}'], *saved[f'{instruction}/{fact_order}'], probes, args.bootstrap_samples, args.seed)
    for instruction in manifest['instructions']:
        report['fact_order_effects'][instruction] = paired_effects(*saved[f'{instruction}/canonical'],
            *saved[f'{instruction}/shuffled'], probes, args.bootstrap_samples, args.seed)
    with (args.output_dir / 'test_predictions.jsonl').open('w') as f:
        for cell, (items, pred) in saved.items():
            for i, r in enumerate(items):
                f.write(json.dumps({'id': r['id'], 'episode_id': r['episode_id'], 'group_id': r['group_id'],
                                    'cell': cell, 'predicted_class_indices': {n: y[i].item() for n, y in pred.items()}}) + '\n')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return {'report': str(args.output_dir / 'report.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--feature-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--poolings', default='last_token,mean_pool,sentence_end_mean')
    parser.add_argument('--l2-values', default='0,1e-5,1e-4,1e-3')
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=12012)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    args.poolings = args.poolings.split(',')
    args.l2_values = [float(x) for x in args.l2_values.split(',')]
    if any(p not in AVAILABLE_POOLINGS for p in args.poolings) or not args.l2_values or any(not math.isfinite(x) or x < 0 for x in args.l2_values):
        parser.error('Invalid pooling/L2 grid')
    if min(args.max_iter, args.bootstrap_samples, args.threads) <= 0:
        parser.error('Iteration, bootstrap and thread counts must be positive')
    torch.set_num_threads(args.threads)
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
