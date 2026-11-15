from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "issue-12-hidden-state-probes"
sys.path.insert(0, str(EXPERIMENT_DIR))

from extract_hidden_features import (  # noqa: E402
    extract_split,
    parse_layer_spec,
    pool_selected_hidden_states,
)


class FakeTokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        token_rows = [[len(token) for token in text.split()] for text in texts]
        width = max(map(len, token_rows))
        input_ids = torch.zeros((len(texts), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(token_rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeModel:
    def __call__(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        base = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 3)
        return SimpleNamespace(hidden_states=tuple(base + layer for layer in range(4)))


def test_layer_spec_uses_one_based_blocks_and_resolves_last() -> None:
    assert parse_layer_spec("4,8,24,last,28", 28) == (4, 8, 24, 28)
    try:
        parse_layer_spec("0,4", 28)
    except ValueError as error:
        assert "embedding" in str(error)
    else:
        raise AssertionError("Layer zero must not be accepted as a transformer block")


def test_last_token_and_masked_mean_pooling() -> None:
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    layer_1 = torch.tensor(
        [
            [[1.0, 10.0], [3.0, 30.0], [99.0, 99.0]],
            [[2.0, 20.0], [4.0, 40.0], [6.0, 60.0]],
        ]
    )
    last, mean = pool_selected_hidden_states(
        (torch.zeros_like(layer_1), layer_1),
        attention_mask,
        (1,),
    )
    assert last.shape == (2, 1, 2)
    assert mean.shape == (2, 1, 2)
    assert torch.equal(last[:, 0], torch.tensor([[3.0, 30.0], [6.0, 60.0]]))
    assert torch.equal(mean[:, 0], torch.tensor([[2.0, 20.0], [4.0, 40.0]]))


def test_extract_split_writes_joinable_feature_shards(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    records = [
        {"id": "example-1", "text": "aa bbbb"},
        {"id": "example-2", "text": "c ddd eeeee"},
        {"id": "example-3", "text": "ffffff gg"},
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    result = extract_split(
        dataset_path=dataset,
        output_dir=tmp_path / "features",
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        layer_indices=(1, 3),
        batch_size=2,
        device="cpu",
        output_dtype=torch.float16,
        max_length=32,
    )

    assert result["examples"] == 3
    assert len(result["shards"]) == 2
    first = torch.load(tmp_path / "features" / "shard-00000.pt", weights_only=True)
    assert first["ids"] == ["example-1", "example-2"]
    assert first["last_token"].shape == (2, 2, 3)
    assert first["mean_pool"].shape == (2, 2, 3)
    assert first["last_token"].dtype == torch.float16
    assert first["layer_indices"].tolist() == [1, 3]


def test_extract_split_can_limit_a_smoke_prefix(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        "".join(json.dumps({"id": f"example-{i}", "text": "aa bb"}) + "\n" for i in range(5)),
        encoding="utf-8",
    )
    result = extract_split(
        dataset_path=dataset,
        output_dir=tmp_path / "features",
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        layer_indices=(1,),
        batch_size=2,
        device="cpu",
        output_dtype=torch.float16,
        max_length=32,
        max_examples=3,
    )

    assert result["examples"] == 3
    ids = []
    for shard in result["shards"]:
        ids.extend(torch.load(tmp_path / "features" / shard["path"], weights_only=True)["ids"])
    assert ids == ["example-0", "example-1", "example-2"]
