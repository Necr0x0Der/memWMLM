from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/issue-12-hidden-state-probes/probe_training'))
from train_attention import AttentionProbeBank, load_tokens, run, train_epochs
from test_instruction_specific_probes import make_fixture
from test_hidden_feature_extraction import FakeTokenizer, FakeModel
from extract_hidden_features import extract_split
from train_linear_probes import sha256


def test_zero_query_equals_masked_mean_and_padding_has_no_effect():
    model = AttentionProbeBank(2, {'a': 2, 'b': 3}, torch.zeros(2), torch.ones(2))
    x = torch.tensor([[[1., 2.], [3., 4.], [999., -999.]]])
    mask = torch.tensor([[True, True, False]])
    z, weights = model.pool(x, mask)
    assert torch.allclose(z, torch.tensor([[2., 3.]]))
    assert torch.allclose(weights, torch.tensor([[.5, .5, 0.]]))
    with torch.no_grad():
        model.q.copy_(torch.tensor([2., 1.]))
    assert torch.allclose(model.pool(x, mask)[0], model.pool(x[:, :2], mask[:, :2])[0])
    with pytest.raises(ValueError):
        model.pool(x, torch.zeros_like(mask))


def test_query_receives_gradients_from_both_heads_and_frozen_inputs_stay_fixed():
    torch.manual_seed(1)
    model = AttentionProbeBank(2, {'a': 2, 'b': 2}, torch.zeros(2), torch.ones(2))
    tokens = [torch.tensor([[1., 2.], [3., -1.]]), torch.tensor([[-2., 1.], [1., 3.]])]
    before = [t.clone() for t in tokens]
    for name in model.heads:
        model.zero_grad()
        logits = model(torch.stack(tokens), torch.ones(2, 2, dtype=torch.bool))
        nnloss = torch.nn.functional.cross_entropy(logits[name], torch.tensor([0, 1]))
        nnloss.backward()
        assert model.q.grad.norm() > 0
    for _ in train_epochs(model, tokens, {'a': torch.tensor([0, 1]), 'b': torch.tensor([1, 0])}, 3, 2, .01, 0., 12, 'cpu'):
        pass
    assert model.q.norm() > 0
    assert all(torch.equal(a, b) and b.grad is None for a, b in zip(before, tokens))
    fixed = AttentionProbeBank(2, {'a': 2}, torch.zeros(2), torch.ones(2), 'mean')
    for _ in train_epochs(fixed, tokens, {'a': torch.tensor([0, 1])}, 2, 2, .01, 0., 12, 'cpu'):
        pass
    assert fixed.q.norm() == 0 and not fixed.q.requires_grad


def test_token_extraction_keeps_unpooled_values_and_removes_padding(tmp_path):
    path = tmp_path / 'input.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in [
        {'id': 'a', 'text': 'Mary is here.'}, {'id': 'b', 'text': 'Bob.'}]))
    result = extract_split(path, tmp_path/'out', FakeTokenizer(), FakeModel(), [1, 3],
                           2, 'cpu', torch.float32, 30, save_token_states=True)
    payload = torch.load(tmp_path/'out'/result['shards'][0]['path'], weights_only=True)
    assert [tuple(t.shape) for t in payload['token_states']] == [(2, 3, 3), (2, 1, 3)]
    for i, x in enumerate(payload['token_states']):
        assert torch.allclose(x.mean(1), payload['mean_pool'][i])
        assert torch.equal(x[:, -1], payload['last_token'][i])


def token_fixture(tmp_path):
    dataset, features = make_fixture(tmp_path)
    manifest_path = features/'manifest.json'
    fm = json.loads(manifest_path.read_text())
    fm['features'] = {'layer_indices': [4, 8], 'token_states': 'ragged'}
    for split in ('train', 'validation', 'test'):
        record = fm['splits'][split]['shards'][0]
        path = features/split/record['path']
        payload = torch.load(path, weights_only=True)
        # Two layers, varying token counts, hidden width 1.
        payload['token_states'] = [x[:, None].expand(-1, 1+i%3, -1).clone() for i, x in enumerate(payload['last_token'])]
        payload['token_counts'] = torch.tensor([x.shape[1] for x in payload['token_states']])
        torch.save(payload, path)
        record['sha256'] = sha256(path)
    manifest_path.write_text(json.dumps(fm))
    return dataset, features, fm


def test_loader_alignment_and_missing_token_features(tmp_path):
    dataset, features, fm = token_fixture(tmp_path)
    rows = list(map(json.loads, (dataset/'train.jsonl').read_text().splitlines()))
    normal = load_tokens(features, fm, 'train', rows, 4, set())
    reverse = load_tokens(features, fm, 'train', rows[::-1], 4, set())
    assert all(torch.equal(a, b) for a, b in zip(normal, reverse[::-1]))
    path = features/'train/shard.pt'
    payload = torch.load(path, weights_only=True)
    del payload['token_states']
    torch.save(payload, path)
    fm['splits']['train']['shards'][0]['sha256'] = sha256(path)
    with pytest.raises(ValueError, match='Re-extract'):
        load_tokens(features, fm, 'train', rows, 4, set())


def test_end_to_end_attention_common_bank_and_locked_selection(tmp_path):
    torch.set_num_threads(1)
    dataset, features, _ = token_fixture(tmp_path)
    args = argparse.Namespace(dataset_dir=dataset, feature_dir=features, output_dir=tmp_path/'out',
        layers='all', l2_values=[.001], mode='attention', epochs=2, batch_size=4, lr=.01,
        seed=12, device='cpu', threads=1, bootstrap_samples=10)
    run(args)
    report = json.loads((args.output_dir/'report.json').read_text())
    assert len(report['cells']) == 4
    assert len(list(args.output_dir.glob('*.pt'))) == 4
    for cell in report['cells'].values():
        assert len(cell['candidates']) == 4
        best = max(cell['candidates'], key=lambda c: (c['validation']['mean_macro_accuracy'], c['validation']['joint_exact_state_accuracy'], c['l2'], -c['epoch']))
        assert cell['selected']['layer'] == best['layer']
        assert cell['selected']['epoch'] == best['epoch']
        assert cell['evaluation']['validation'] == best['validation']
    checkpoint = torch.load(next(args.output_dir.glob('*.pt')), weights_only=True)
    assert 'q' in checkpoint['state_dict']
    assert len([n for n in checkpoint['state_dict'] if n.endswith('.weight')]) == 2
    assert len(report['instruction_effects']) == 2
    with pytest.raises(FileExistsError):
        run(args)
