"""Protocol/integration checks; tiny random Qwen does not measure real QA."""
import argparse
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
import common
import extract
import probe


class Tokenizer:
    chat_template = 'test-chat-template'
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return '<user> ' + messages[0]['content'] + ' </user> <assistant>'

    def __call__(self, texts, **kwargs):
        assert kwargs['truncation'] is False
        tokens = [[3+len(word) % 60 for word in t.split()] for t in texts]
        ids = torch.zeros(len(texts), max(map(len, tokens)), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(tokens):
            ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        return dict(input_ids=ids, attention_mask=mask)


def tiny_model():
    torch.manual_seed(12)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8)).eval().requires_grad_(False)


def test_same_chat_wrapper_and_no_answer():
    tok = Tokenizer()
    before = common.render_prompt(tok, 'FACTS')
    after = common.render_prompt(tok, 'FACTS', 'QUESTION')
    assert before == '<user> FACTS </user> <assistant>'
    assert after == '<user> FACTS\n\nQuestion: QUESTION </user> <assistant>'
    assert 'FINAL' not in after
    families = common.question_families('mary_left_of_bob', ['mary_left_of_bob', 'bob_left_of_mary', 'bob_left_of_john'])
    assert families['inverse'] == ['bob_left_of_mary']
    assert families['other_excluding_inverse'] == ['bob_left_of_john']


def test_extraction_stops_before_generation_and_padding_is_excluded(tmp_path):
    torch.set_num_threads(1)
    model = tiny_model()
    # Extraction must never call the LM wrapper, which owns the output head.
    def forbidden(*args, **kwargs):
        raise AssertionError('LM head/generation is forbidden')
    model.forward = forbidden
    model.generate = forbidden
    rows = [dict(id='a', text='Mary is here.', targets={'secret': 'SECRET'}),
            dict(id='b', text='John is in a very distant room.', targets={'secret': 'SECRET'})]
    metadata = extract.extract_context(model.model, Tokenizer(), rows, 'Where?', [1, 2],
                                       tmp_path /'features', 2, 100, 'cpu')
    saved = torch.load(tmp_path/'features/shard-00000.pt', weights_only=True)
    assert saved['last_token'].shape == (2, 2, 32)
    assert 'SECRET' not in (tmp_path/'features/prompts.jsonl').read_text()
    extract.extract_context(model.model, Tokenizer(), rows[:1], 'Where?', [1, 2],
                            tmp_path /'single', 1, 100, 'cpu')
    single = torch.load(tmp_path/'single/shard-00000.pt', weights_only=True)
    for pooling in common.POOLINGS:
        assert torch.allclose(saved[pooling][:1], single[pooling], atol=1e-3)
    assert metadata['examples'] == 2
    with pytest.raises(ValueError, match='truncation'):
        extract.extract_context(model.model, Tokenizer(), rows, 'Where?', [1], tmp_path / 'too-long', 2, 1, 'cpu')


def make_dataset(path):
    path.mkdir()
    names = ['mary_left_of_bob', 'bob_left_of_mary', 'bob_left_of_john']
    manifest = dict(instructions={'none': ''}, fact_orders=['shuffled'],
        probes={n: dict(classes=[0, 1], question=n.replace('_', ' ') + '?') for n in names}, files={})
    for split in common.SPLITS:
        rows = []
        for i in range(4):
            targets = dict(zip(names, [i//2, 1-i//2, i%2]))
            rows.append(dict(id=f'{split}-{i}', episode_id=f'{split}-{i}', group_id=f'{split}-{i}',
                split=split, instruction_id='none', fact_order='shuffled', targets=targets,
                state=targets, facts=['fact'], text='Mary Bob John ' + 'left ' * (i+1)))
        f = path/f'{split}.jsonl'
        f.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest['files'][f.name] = dict(sha256=common.sha256(f))
    common.save_json(path / 'manifest.json', manifest)


def test_full_pipeline_locks_all_probes_before_test(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    db, cache, output = tmp_path/'db', tmp_path/'cache', tmp_path/'out'
    make_dataset(db)
    model = tiny_model()
    monkeypatch.setattr(extract, 'load_frozen_model', lambda *a, **kw: (Tokenizer(), model))
    extract.run(argparse.Namespace(dataset_dir=db, output_dir=cache, instruction='none', fact_order='shuffled',
                                   device='cpu', dtype='float32', layers='1,2', batch_size=4, max_length=100))
    args = argparse.Namespace(dataset_dir=db, feature_dir=cache, output_dir=output, instruction='none',
        fact_order='shuffled', audit_only=False, poolings=['last_token', 'mean_pool'],
        l2_values=[.001], max_iter=5, device='cpu', bootstrap_samples=10, seed=12)
    original = probe.load_context
    def load(*a):
        if a[-1] == 'test':
            selection = json.loads((output/'selection.json').read_text())
            assert len(selection) == 4
            assert all(len(s['selection']) == 3 for s in selection.values())
        return original(*a)
    monkeypatch.setattr(probe, 'load_context', load)
    report = probe.run(args)
    assert len(report['contexts']) == 4 and len(report['effects']) == 3
    assert len((output/'test_predictions.jsonl').read_text().splitlines()) == 16
    assert len(list(output.glob('*.pt'))) == 24
    assert report['contexts']['no_question']['test']['examples'] == 4
    with pytest.raises(FileExistsError):
        probe.run(args)


def test_contrast_excludes_inverse():
    names = ['mary_left_of_bob', 'bob_left_of_mary', 'bob_left_of_john']
    y = torch.tensor([0, 0, 1, 1])
    rows = [dict(group_id=str(i//2)) for i in range(4)]
    targets = {n: y for n in names}
    before = {n: 1-y for n in names}
    after = {names[0]: y, names[1]: y, names[2]: 1-y}
    result = probe.contrasts(rows, before, after, targets, dict.fromkeys(names, 2), names[0], 10, 12)
    assert result['asked']['delta'] == 1
    assert result['inverse']['delta'] == 1
    assert result['other_excluding_inverse']['delta'] == 0
