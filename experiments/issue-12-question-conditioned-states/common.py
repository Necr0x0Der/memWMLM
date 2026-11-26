"""Shared, pinned protocol for question-conditioned hidden-state probes."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PARENT = Path(__file__).resolve().parents[1] / 'issue-12-hidden-state-probes'
sys.path.insert(0, str(PARENT))
from extract_hidden_features import (git_revision, load_frozen_model, parse_layer_spec,
                                    pool_selected_hidden_states, sha256)
from train_linear_probes import (encode_targets, fit_probes, load_feature_split,
                                metrics, predict, save_probe, shuffle_targets_by_group)

spec = importlib.util.spec_from_file_location('question_probe_shared_training', PARENT / 'probe_training/train.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
audit_dataset = shared.audit_dataset
select_probe = shared.select_probe
paired_effects = shared.paired_effects

MODEL = 'Qwen/Qwen3-4B'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
SCHEMA = 'question-conditioned-states-v1'
SPLITS = ('train', 'validation', 'test')
POOLINGS = ('last_token', 'mean_pool')


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def subset_dataset(directory, instruction, fact_order):
    manifest, all_rows, audit = audit_dataset(directory)
    rows = {s: [r for r in all_rows[s]
                if r['instruction_id'] == instruction and r['fact_order'] == fact_order] for s in SPLITS}
    if any(not items for items in rows.values()):
        raise ValueError('Empty instruction/order condition')
    return manifest, rows, audit


def render_prompt(tokenizer, situation, question=None):
    """Inputs are text and question only; no targets or answer options."""
    content = situation
    if question is not None:
        content += '\n\nQuestion: ' + question
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)


def question_families(asked, names):
    """Separate the asked target, its redundant inverse, and other targets."""
    inverse = None
    if '_left_of_' in asked:
        a, b = asked.split('_left_of_')
        inverse = b + '_left_of_' + a
    return {'asked': [asked],
            'inverse': [inverse] if inverse in names else [],
            'other_excluding_inverse': [n for n in names if n not in (asked, inverse)]}


def target_metrics(predictions, targets, counts):
    result = metrics(predictions, targets, counts)
    result['joint_exact_targets_accuracy'] = result.pop('joint_exact_state_accuracy')
    return result
