from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / 'experiments' / 'issue-12-hidden-state-probes'
sys.path.insert(0, str(EXPERIMENT_DIR))

from evaluate_full_context_qa import resolve_revision, validate_qa_rows
from prepare_spatial_probe_dataset import INSTRUCTIONS, generate_records, probe_definitions


def test_model_revision_is_specific_to_model():
    assert resolve_revision('Qwen/Qwen3-1.7B', None) == '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
    assert resolve_revision('Qwen/Qwen3-4B', None) == '1cfa9a7208912126459214e8b04321603b3df60c'
    assert resolve_revision('custom', 'explicit-sha') == 'explicit-sha'
    with pytest.raises(ValueError, match='--revision'):
        resolve_revision('custom', None)


def test_v2_qa_agrees_with_scoring_and_presented_fact_order():
    rows = generate_records(seed=12, episodes_per_group=1)
    manifest = {
        'schema_version': 'spatial-hidden-state-probes-v2',
        'instructions': INSTRUCTIONS,
        'probes': probe_definitions(),
    }
    validate_qa_rows(rows, manifest)
    for field, value in [('answer', 'wrong'), ('question', 'wrong'), ('allowed_answers', ['wrong'])]:
        bad = deepcopy(rows[0])
        bad['qwen_qa'][0][field] = value
        with pytest.raises(ValueError, match='QA definition/target mismatch'):
            validate_qa_rows([bad], manifest)
    bad = deepcopy(rows[0])
    bad['qwen_qa'].append(deepcopy(bad['qwen_qa'][0]))
    with pytest.raises(ValueError, match='IDs mismatch'):
        validate_qa_rows([bad], manifest)
    bad = deepcopy(rows[0])
    bad['text'] = 'different situation'
    with pytest.raises(ValueError, match='Fact order/text mismatch'):
        validate_qa_rows([bad], manifest)
    bad = deepcopy(rows[0])
    bad['fact_order'] = 'shuffled' if bad['fact_order'] == 'canonical' else 'canonical'
    with pytest.raises(ValueError, match='Fact order/text mismatch'):
        validate_qa_rows([bad], manifest)
