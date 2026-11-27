#!/usr/bin/env python3
"""Select independent linear probes and compare continuation controls."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from continuation_common import (MODEL, REVISION, SCHEMA, SPLITS, POOLINGS, encode_targets,
                    fit_probes, git_revision, load_feature_split, metrics, predict,
                    question_families, save_json, save_probe, select_probe, sha256,
                    shuffle_targets_by_group, subset_dataset, target_metrics, contexts_for)


def contrasts(rows, predictions, targets, counts, before_contexts, after_contexts, samples, seed):
    """Average scores across fixed prompt families, never vote/ensemble predictions."""
    grouped = {}
    for i, row in enumerate(rows):
        grouped.setdefault(row['group_id'], []).append(i)
    def score(contexts, indices):
        scores = [metrics(predictions[c], targets, counts, indices) for c in contexts]
        result = {f'probe/{n}': sum(s['per_probe'][n]['macro_accuracy'] for s in scores)/len(scores)
                  for n in counts}
        result['all_targets_macro'] = sum(s['mean_macro_accuracy'] for s in scores)/len(scores)
        result['joint_exact_targets'] = sum(s['joint_exact_state_accuracy'] for s in scores)/len(scores)
        return result
    def delta(indices):
        a, b = score(before_contexts, indices), score(after_contexts, indices)
        return {k: b[k]-a[k] for k in a}
    point = delta(torch.arange(len(rows)))
    draws = {k: [] for k in point}
    rng = random.Random(seed)
    cached = {}
    for _ in range(samples):
        groups = tuple(sorted(rng.choices(sorted(grouped), k=len(grouped))))
        if groups not in cached:
            indices = torch.tensor([i for g in groups for i in grouped[g]])
            cached[groups] = delta(indices)
        for k, value in cached[groups].items():
            draws[k].append(value)
    return dict(before=before_contexts, after=after_contexts,
        metrics={k: dict(delta=v, paired_group_bootstrap_95ci=torch.tensor(draws[k], dtype=torch.float64)
                        .quantile(torch.tensor([.025, .975], dtype=torch.float64)).tolist())
                 for k, v in point.items()})


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
    definitions = contexts_for(dataset)
    contexts = list(definitions)
    if set(contexts) != set(fm['contexts']):
        raise ValueError('Missing or unexpected question context')
    for context in contexts:
        expected = definitions[context]
        if fm['contexts'][context]['continuation'] != expected:
            raise ValueError('Question text mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts = {n: len(p['classes']) for n, p in probes.items()}
    ys = {s: encode_targets(rows[s], probes) for s in ('train', 'validation')}
    shuffled = shuffle_targets_by_group(ys['train'], rows['train'], args.seed)
    report = dict(schema=SCHEMA, settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        audit=audit, condition=fm['condition'], model=fm['model'], repository_commit=git_revision(),
        script_sha256=sha256(Path(__file__)), common_sha256=sha256(Path(__file__).with_name('continuation_common.py')),
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
    questions = [c for c in contexts if '_left_of_' in c]
    neutrals = ['neutral_complete', 'neutral_sentences']
    comparisons = {f'{c}_vs_no_continuation': (['no_continuation'], [c])
                   for c in contexts if c != 'no_continuation'}
    comparisons.update({
        'questions_vs_neutrals': (neutrals, questions),
        'general_vs_neutrals': (neutrals, ['general_spatial']),
        'general_vs_questions': (questions, ['general_spatial']),
    })
    for neutral in neutrals:
        comparisons[f'questions_vs_{neutral}'] = ([neutral], questions)
        comparisons[f'general_vs_{neutral}'] = ([neutral], ['general_spatial'])
    for name, (before, after) in comparisons.items():
        report['effects'][name] = contrasts(rows['test'], predictions, ys['test'], counts,
            before, after, args.bootstrap_samples, args.seed)
    save_json(args.output_dir/'report.json', report)
    write_summary(args.output_dir/'SUMMARY.md', report)
    return report


def write_summary(path, report):
    lines = ['# Continuation controls: measured results', '',
             'Macro = mean balanced accuracy across seven targets; joint = all targets correct.',
             'These are external probe scores, not Qwen answers or proof of causal state use.', '',
             '| Continuation | Validation macro | Test macro | Joint | Shuffled-label macro |',
             '|---|---:|---:|---:|---:|']
    for name, cell in report['contexts'].items():
        values = [cell['validation']['mean_macro_accuracy'], cell['test']['mean_macro_accuracy'],
                  cell['test']['joint_exact_targets_accuracy'], cell['shuffled_label_test']['mean_macro_accuracy']]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{100*v:.2f}%' for v in values) + ' |')
    lines += ['', '## Paired contrasts', '',
              'Deltas in percentage points. Positive means the after condition is better.', '',
              '| Contrast | Macro delta | Paired 95% interval |', '|---|---:|---|']
    for name, effect in report['effects'].items():
        m = effect['metrics']['all_targets_macro']
        lo, hi = m['paired_group_bootstrap_95ci']
        lines.append(f"| {name} | {100*m['delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['', 'Only four independent test layouts; this is a previously inspected benchmark.',
              'Intervals condition on selected probes, omit training/selection uncertainty and multiplicity.',
              'Prompt-family scores are averaged; predictions are not ensembled. A small/non-significant',
              'difference does not establish equivalence. Inspect both neutral controls separately.',
              'Mean pooling includes the continuation itself. Source text is available during encoding.',
              'No automatic claim about a complete world state follows from these scores.', '']
    path.write_text('\n'.join(lines))


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
