#!/usr/bin/env python3
"""Select linear probes on validation; compare question/no-question test states."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from common import (MODEL, REVISION, SCHEMA, SPLITS, POOLINGS, encode_targets,
                    fit_probes, git_revision, load_feature_split, metrics, predict,
                    question_families, save_json, save_probe, select_probe, sha256,
                    shuffle_targets_by_group, subset_dataset, target_metrics)


def contrasts(rows, baseline, conditioned, targets, counts, asked, samples, seed):
    families = question_families(asked, list(counts))
    def delta(indices):
        before = metrics(baseline, targets, counts, indices)
        after = metrics(conditioned, targets, counts, indices)
        differences = {n: after['per_probe'][n]['macro_accuracy'] - before['per_probe'][n]['macro_accuracy']
                       for n in counts}
        result = {f'probe/{n}': d for n, d in differences.items()}
        result.update({name: sum(differences[n] for n in names)/len(names)
                       for name, names in families.items() if names})
        result['all_targets_macro'] = after['mean_macro_accuracy'] - before['mean_macro_accuracy']
        result['joint_exact_targets'] = after['joint_exact_state_accuracy'] - before['joint_exact_state_accuracy']
        return result
    grouped = {}
    for i, row in enumerate(rows):
        grouped.setdefault(row['group_id'], []).append(i)
    rng = random.Random(seed)
    point = delta(torch.arange(len(rows)))
    draws = {k: [] for k in point}
    for _ in range(samples):
        indices = torch.tensor([i for group in rng.choices(sorted(grouped), k=len(grouped)) for i in grouped[group]])
        for key, value in delta(indices).items():
            draws[key].append(value)
    return {key: dict(delta=value, paired_group_bootstrap_95ci=torch.tensor(draws[key], dtype=torch.float64)
                     .quantile(torch.tensor([.025, .975], dtype=torch.float64)).tolist())
            for key, value in point.items()}


def load_context(args, fm, rows, context, split):
    layers, tensors = load_feature_split(args.feature_dir/context, split,
        fm['contexts'][context]['splits'][split], [r['id'] for r in rows[split]], args.poolings)
    if list(layers) != fm['layers']:
        raise ValueError('Feature layer mismatch')
    if any(not torch.isfinite(x).all() for x in tensors.values()):
        raise ValueError('Non-finite features')
    return tensors


def run(args):
    dataset, rows, audit = subset_dataset(args.dataset_dir, args.instruction, args.fact_order)
    if args.audit_only:
        return dict(audit=audit, selected_rows={s: len(rs) for s, rs in rows.items()})
    if args.feature_dir is None or args.output_dir is None:
        raise ValueError('--feature-dir and --output-dir required')
    fm = json.loads((args.feature_dir/'manifest.json').read_text())
    if fm['schema'] != SCHEMA or (fm['model']['name'], fm['model']['requested_revision']) != (MODEL, REVISION):
        raise ValueError('Wrong feature protocol/model')
    if fm['condition'] != dict(instruction=args.instruction, fact_order=args.fact_order):
        raise ValueError('Condition mismatch')
    if fm['dataset_manifest_sha256'] != sha256(args.dataset_dir/'manifest.json'):
        raise ValueError('Dataset manifest mismatch')
    for split in SPLITS:
        if fm['dataset_files'][split] != sha256(args.dataset_dir/f'{split}.jsonl'):
            raise ValueError('Dataset file mismatch')
    probes = dataset['probes']
    contexts = ['no_question', *probes]
    if set(contexts) != set(fm['contexts']):
        raise ValueError('Missing or unexpected question context')
    for context in contexts:
        expected = None if context == 'no_question' else probes[context]['question']
        if fm['contexts'][context]['question'] != expected:
            raise ValueError('Question text mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts = {n: len(p['classes']) for n, p in probes.items()}
    ys = {s: encode_targets(rows[s], probes) for s in ('train', 'validation')}
    shuffled = shuffle_targets_by_group(ys['train'], rows['train'], args.seed)
    report = dict(schema=SCHEMA, settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        audit=audit, condition=fm['condition'], model=fm['model'], repository_commit=git_revision(),
        script_sha256=sha256(Path(__file__)), common_sha256=sha256(Path(__file__).with_name('common.py')),
        feature_manifest_sha256=sha256(args.feature_dir/'manifest.json'),
        dataset_manifest_sha256=sha256(args.dataset_dir/'manifest.json'),
        selection='Per context/target: validation balanced accuracy, ordinary accuracy, stronger L2, first candidate',
        interval_scope='Paired abstract-layout bootstrap; conditional on selected probes; no multiplicity correction',
        contexts={}, effects={})
    save_json(args.output_dir/'protocol.json', report)
    locked = {}
    # Lock every context and target before opening any test features.
    for context in contexts:
        x = {s: load_context(args, fm, rows, context, s) for s in ('train', 'validation')}
        selected = {}
        val_pred = {}
        locked[context] = {}
        for name, count in counts.items():
            fitted, config, candidates = select_probe(x['train'], x['validation'], ys['train'][name],
                ys['validation'][name], name, count, fm['layers'], args.l2_values, args.max_iter, args.device)
            pooling, offset = config['pooling'], fm['layers'].index(config['layer'])
            control = fit_probes(x['train'][pooling][:, offset], {name: shuffled[name]}, {name: count},
                                 config['l2'], args.max_iter, args.device)
            locked[context][name] = (fitted, control, config)
            val_pred[name] = predict(fitted, x['validation'][pooling][:, offset], args.device)[name]
            selected[name] = dict(selected=config, candidates=candidates)
            save_probe(args.output_dir/f'{context}--{name}.pt', fitted,
                       dict(**config, context=context, target=name, classes=probes[name]['classes']))
            save_probe(args.output_dir/f'{context}--{name}--shuffled-label.pt', control, config)
        report['contexts'][context] = dict(selection=selected,
            validation=target_metrics(val_pred, ys['validation'], counts))
        save_json(args.output_dir/'selection.json', report['contexts'])
        print(json.dumps(dict(locked_context=context)), flush=True)
    ys['test'] = encode_targets(rows['test'], probes)
    predictions = {}
    with (args.output_dir/'test_predictions.jsonl').open('w') as stream:
        for context in contexts:
            x = load_context(args, fm, rows, context, 'test')
            pred, null = {}, {}
            for name, (fitted, control, config) in locked[context].items():
                data = x[config['pooling']][:, fm['layers'].index(config['layer'])]
                pred[name] = predict(fitted, data, args.device)[name]
                null[name] = predict(control, data, args.device)[name]
            predictions[context] = pred
            report['contexts'][context]['test'] = target_metrics(pred, ys['test'], counts)
            report['contexts'][context]['shuffled_label_test'] = target_metrics(null, ys['test'], counts)
            for i, row in enumerate(rows['test']):
                stream.write(json.dumps(dict(id=row['id'], episode_id=row['episode_id'], group_id=row['group_id'],
                    context=context, targets={n: int(y[i]) for n, y in ys['test'].items()},
                    predictions={n: int(y[i]) for n, y in pred.items()},
                    shuffled_label_predictions={n: int(y[i]) for n, y in null.items()})) + '\n')
            stream.flush()
    for context in probes:
        report['effects'][context] = contrasts(rows['test'], predictions['no_question'], predictions[context],
            ys['test'], counts, context, args.bootstrap_samples, args.seed)
    save_json(args.output_dir/'report.json', report)
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--feature-dir', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--instruction', choices=['none', 'imagine', 'track_mary', 'track_people'], default='none')
    p.add_argument('--fact-order', choices=['canonical', 'shuffled'], default='shuffled')
    p.add_argument('--poolings', default='last_token,mean_pool')
    p.add_argument('--l2-values', default='0.0001,0.001')
    p.add_argument('--max-iter', type=int, default=100)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=12012)
    p.add_argument('--bootstrap-samples', type=int, default=2000)
    args = p.parse_args()
    args.poolings = args.poolings.split(',')
    args.l2_values = [float(x) for x in args.l2_values.split(',')]
    if not set(args.poolings) <= set(POOLINGS) or any(x < 0 for x in args.l2_values):
        p.error('Invalid pooling/L2')
    if min(args.max_iter, args.threads, args.bootstrap_samples) <= 0:
        p.error('Iteration, thread and bootstrap counts must be positive')
    torch.set_num_threads(args.threads)
    return args


if __name__ == '__main__':
    result = run(parse_args())
    print(json.dumps({'selected_rows': result.get('selected_rows'), 'complete': True}))
