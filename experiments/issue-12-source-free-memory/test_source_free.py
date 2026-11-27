"""CPU checks with a tiny random Qwen, not experimental accuracy results."""
import sys
import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from run import SoftPrefix, training_batch, generate_one, make_examples
import run as experiment


def tiny_model():
    torch.manual_seed(12)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256,
        pad_token_id=0, eos_token_id=2, bos_token_id=1))
    return model.eval().requires_grad_(False)


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def encode(self, text, **kwargs):
        return [3 + ord(c) % 60 for c in text]

    def decode(self, ids, **kwargs):
        return ','.join(str(int(i)) for i in ids)


def test_gradient_through_frozen_qwen_and_masking():
    torch.set_num_threads(1)
    model = tiny_model()
    features = torch.randn(3, 16)
    adapter = SoftPrefix(features, 32, 4, .02)
    batch = [dict(row=0, prompt=[3, 4, 5], answer=[6, 2]),
             dict(row=1, prompt=[3], answer=[7, 2])]
    before = {n: p.clone() for n, p in model.named_parameters()}
    captured = []
    handle = model.register_forward_pre_hook(lambda m, a, kw: captured.append(kw), with_kwargs=True)
    loss, count = training_batch(model, adapter, features, batch, 0)
    loss.backward()
    handle.remove()
    assert torch.isfinite(loss) and count == 4
    assert adapter.projection.weight.grad.norm() > 0
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    kw = captured[0]
    assert 'input_ids' not in kw and 'past_key_values' not in kw
    assert kw['inputs_embeds'].shape == (2, 9, 32)
    assert kw['labels'][0].tolist() == [-100] * 7 + [6, 2]
    assert kw['labels'][1].tolist() == [-100] * 5 + [7, 2, -100, -100]
    assert kw['attention_mask'][1].tolist() == [1] * 7 + [0, 0]
    # Batch padding must not change answer-token loss.
    with torch.no_grad():
        a, _ = training_batch(model, adapter, features, batch[:1], 0)
        b, _ = training_batch(model, adapter, features, batch[1:], 0)
    assert torch.allclose(loss, (a+b)/2, atol=1e-6)


def test_source_and_target_cannot_leak_into_prompt():
    probes = {'p': {'question': 'Is Mary left of John?', 'classes': [0, 1]}}
    rows = [dict(text='SECRET SOURCE', targets={'p': 0}),
            dict(text='DIFFERENT SOURCE', targets={'p': 1})]
    examples = make_examples(rows, probes, Tokenizer())
    assert examples[0]['prompt'] == examples[1]['prompt']
    assert examples[0]['answer'] != examples[1]['answer']


def test_generate_matches_manual_greedy_and_no_prefix_tokens_in_output():
    torch.set_num_threads(1)
    model = tiny_model()
    model.generation_config.eos_token_id = None
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = None
    prompt = [3, 4, 5]
    memory = torch.randn(1, 4, 32) * .02
    for prefix in (None, memory):
        response = generate_one(model, tokenizer, prompt, prefix, 3)
        with torch.no_grad():
            inputs = model.get_input_embeddings()(torch.tensor([prompt]))
            if prefix is not None:
                inputs = torch.cat([prefix, inputs], 1)
            expected = []
            for _ in range(3):
                token = model(inputs_embeds=inputs, use_cache=False).logits[:, -1].argmax(-1)
                expected.append(int(token))
                inputs = torch.cat([inputs, model.get_input_embeddings()(token[:, None])], 1)
        assert response == ','.join(map(str, expected))


def test_complete_train_select_test_pipeline(tmp_path, monkeypatch):
    """Exercise real tiny-Qwen backprop/generation, verified feature joins and reports."""
    torch.set_num_threads(1)
    dataset = tmp_path / 'dataset'
    features = tmp_path / 'features'
    dataset.mkdir()
    features.mkdir()
    hashes, splits, files = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        rows = [dict(id=f'{split}-{i}', group_id=f'{split}-group-{i}',
                     text='SECRET SOURCE', instruction_id='none', fact_order='shuffled',
                     targets={'p': i % 2}) for i in range(2)]
        path = dataset / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        hashes[path.name] = experiment.sha256(path)
        files[path.name] = {'sha256': hashes[path.name]}
        (features / split).mkdir()
        shard = features / split / 'shard.pt'
        torch.save(dict(ids=[r['id'] for r in rows], layer_indices=torch.tensor([1]),
                        mean_pool=torch.randn(2, 1, 16)), shard)
        splits[split] = {'shards': [{'path': shard.name, 'sha256': experiment.sha256(shard)}]}
    (dataset / 'manifest.json').write_text(json.dumps(dict(
        files=files, probes={'p': {'classes': [0, 1], 'question': 'Left?'}})))
    (features / 'manifest.json').write_text(json.dumps(dict(
        dataset={'files': hashes}, model={'name': 'tiny', 'requested_revision': 'test'},
        tokenization={'chat_template_applied': False}, splits=splits)))
    model = tiny_model()
    before = {n: p.clone() for n, p in model.named_parameters()}
    monkeypatch.setattr(experiment, 'load_frozen_model', lambda *a, **k: (Tokenizer(), model))
    opened = []
    original_loader = experiment.load_split
    def tracked_loader(args, manifest, split, poolings):
        if split == 'test':
            assert (args.output_dir / 'selected.pt').exists()
            assert len(json.loads((args.output_dir / 'history.json').read_text())) == 2
        opened.append(split)
        return original_loader(args, manifest, split, poolings)
    monkeypatch.setattr(experiment, 'load_split', tracked_loader)
    args = argparse.Namespace(dataset_dir=dataset, feature_dir=features,
        output_dir=tmp_path/'result', device='cpu', model='tiny', revision='test',
        poolings='mean_pool', layers='1', instruction='none', fact_order='shuffled',
        dtype='float32', slots=2, seed=12, epochs=2, lr=.001, batch_size=2, max_new_tokens=2)
    experiment.run(args)
    assert opened == ['train', 'validation', 'test']
    report = json.loads((args.output_dir/'report.json').read_text())
    assert report['selected']['epoch'] in (1, 2)
    assert report['test']['memory']['answers'] == 2
    assert report['test']['question_only']['answers'] == 2
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
