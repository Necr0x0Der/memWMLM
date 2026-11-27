from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/issue-12-hidden-state-probes/probe_training'))
from train import audit_dataset, paired_effects, run, select_probe
from train_linear_probes import sha256


def test_questions_select_different_layers():
    torch.set_num_threads(1)
    x = torch.tensor([[[-1.], [-1.]], [[-1.], [1.]], [[1.], [-1.]], [[1.], [1.]]])
    features = {'last_token': x}
    for name, y, expected in [('a', torch.tensor([0, 0, 1, 1]), 4), ('b', torch.tensor([0, 1, 0, 1]), 8)]:
        _, config, _ = select_probe(features, features, y, y, name, 2, (4, 8), [.01], 30, 'cpu')
        assert config['layer'] == expected


def test_paired_selectivity_aligns_episodes_and_preserves_clusters():
    probes = {n: {'classes': [0, 1]} for n in ('mary_left_of_bob', 'bob_left_of_john')}
    y = torch.tensor([0, 0, 1, 1])
    rows = [{'episode_id': str(i), 'group_id': str(i // 2), 'targets': {n: y[i].item() for n in probes}} for i in range(4)]
    base = {n: 1-y for n in probes}
    other = {'mary_left_of_bob': y.flip(0), 'bob_left_of_john': (1-y).flip(0)}
    result = paired_effects(rows, base, rows[::-1], other, probes, 30, 7)
    assert result['mary_relations']['delta'] == 1
    assert result['bob_john_relations']['delta'] == 0
    assert result['selectivity_difference_in_differences']['paired_group_bootstrap_95ci'] == [1., 1.]
    with pytest.raises(ValueError, match='unmatched'):
        paired_effects(rows, base, rows[:-1], other, probes, 30, 7)


def make_fixture(tmp_path):
    dataset, features = tmp_path / 'db', tmp_path / 'features'
    dataset.mkdir()
    manifest = {'instructions': {'none': '', 'track_mary': 'Mary'}, 'fact_orders': ['canonical', 'shuffled'],
                'probes': {n: {'classes': [0, 1]} for n in ('mary_left_of_bob', 'bob_left_of_john')}, 'files': {}}
    fm = {'model': {'name': 'Qwen/Qwen3-4B', 'requested_revision': '1cfa9a7208912126459214e8b04321603b3df60c'},
          'dataset': {'files': {}}, 'splits': {}}
    for split in ('train', 'validation', 'test'):
        rows, values = [], []
        for i in range(4):
            for instruction in manifest['instructions']:
                for order in manifest['fact_orders']:
                    targets = {'mary_left_of_bob': i // 2, 'bob_left_of_john': i % 2}
                    rows.append({'id': f'{split}-{i}-{instruction}-{order}', 'split': split,
                                 'episode_id': f'{split}-{i}', 'group_id': f'{split}-{i}',
                                 'instruction_id': instruction, 'fact_order': order,
                                 'facts': [order], 'state': targets, 'targets': targets})
                    # Different instruction conditions require opposite linear weights.
                    sign = 1 if instruction == 'none' else -1
                    values.append([[(2*(i//2)-1)*sign], [(2*(i%2)-1)*sign]])
        path = dataset / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest['files'][path.name] = {'sha256': sha256(path)}
        fm['dataset']['files'][path.name] = sha256(path)
        dest = features / split
        dest.mkdir(parents=True)
        shard = dest / 'shard.pt'
        torch.save({'ids': [r['id'] for r in rows], 'layer_indices': torch.tensor([4, 8]),
                    'last_token': torch.tensor(values, dtype=torch.float32)}, shard)
        fm['splits'][split] = {'shards': [{'path': shard.name, 'sha256': sha256(shard)}]}
    (dataset / 'manifest.json').write_text(json.dumps(manifest))
    (features / 'manifest.json').write_text(json.dumps(fm))
    return dataset, features


def test_end_to_end_separate_banks_and_individual_selection(tmp_path):
    torch.set_num_threads(1)
    dataset, features = make_fixture(tmp_path)
    args = argparse.Namespace(dataset_dir=dataset, feature_dir=features, output_dir=tmp_path/'out',
        audit_only=False, poolings=['last_token'], l2_values=[.01], max_iter=30, device='cpu', seed=12, bootstrap_samples=20)
    run(args)
    report = json.loads((args.output_dir/'report.json').read_text())
    assert len(list(args.output_dir.glob('*.pt'))) == 8
    for cell in report['cells'].values():
        assert cell['selection']['mary_left_of_bob']['selected']['layer'] == 4
        assert cell['selection']['bob_left_of_john']['selected']['layer'] == 8
        assert cell['evaluation']['test']['joint_exact_state_accuracy'] == 1
    assert len(report['instruction_effects']) == 2
    assert len(report['fact_order_effects']) == 2
    with pytest.raises(FileExistsError):
        run(args)


def test_audit_rejects_missing_matched_variant(tmp_path):
    dataset, _ = make_fixture(tmp_path)
    path = dataset/'train.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[1:])+'\n')
    manifest_path = dataset/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['train.jsonl']['sha256'] = sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='matched condition'):
        audit_dataset(dataset)
