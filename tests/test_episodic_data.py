"""CPU-only integration checks; requires the project's torch and transformers deps."""

from __future__ import annotations

import torch
from torch import nn

from memwmlm.models.episodic_qwen import EpisodicConfig, FrozenQwenWithEpisodicKV, KVReader


class FakeAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.o_proj = nn.Linear(width, width)

    def forward(self, hidden):
        return self.o_proj(self.q_proj(hidden))


class FakeLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = FakeAttention(width)

    def forward(self, hidden):
        return hidden + self.self_attn(hidden)


class FakeCache:
    def __init__(self, layers):
        self.layers = layers

    def __getitem__(self, index):
        return self.layers[index]


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 8, "num_key_value_heads": 2, "head_dim": 4})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer(8), FakeLayer(8)])
        self.embedding = nn.Embedding(24, 8)
        self.lm_head = nn.Linear(8, 24)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, use_cache=False, return_dict=True):
        hidden = self.embedding(input_ids)
        cache = []
        for layer in self.model.layers:
            hidden = layer(hidden)
            kv = hidden.reshape(hidden.shape[0], hidden.shape[1], 2, 4).transpose(1, 2)
            cache.append((kv, kv))
        return type("Output", (), {"logits": self.lm_head(hidden), "past_key_values": FakeCache(cache)})()


def test_reader_preserves_source_positions_and_backpropagates():
    model = FrozenQwenWithEpisodicKV(FakeQwen(), EpisodicConfig(injection_layers=2, reader_heads=2,
                                                                  gate_bias_init=0.0, lora_rank=2))
    memory = model.encode(torch.tensor([[1, 2, 3]]))
    assert len(memory) == 2 and memory[0][0].shape == (1, 2, 3, 4)
    assert all(not tensor.requires_grad for pair in memory for tensor in pair)
    logits = model(torch.tensor([[4, 5]]), memory=memory).logits
    logits.square().mean().backward()
    assert model.readers["1"].output.weight.grad is not None
    assert any(p.grad is not None for p in model.projections[0].parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.qwen.embedding.parameters())
    assert model._active is None
    model.close()


def test_reader_uses_source_values():
    reader = KVReader(hidden=8, kv_dim=8, heads=2, gate_bias=0.0)
    hidden = torch.randn(1, 2, 8)
    keys = torch.randn(1, 2, 3, 4)
    values = torch.randn(1, 2, 3, 4)
    assert not torch.allclose(reader(hidden, keys, values), reader(hidden, keys, torch.zeros_like(values)))


def test_reading_wrong_source_changes_application_logits():
    model = FrozenQwenWithEpisodicKV(FakeQwen(), EpisodicConfig(injection_layers=2,
                                                                  reader_heads=2,
                                                                  gate_bias_init=0.0))
    first = model.encode(torch.tensor([[1, 2, 3]]))
    second = model.encode(torch.tensor([[7, 8, 9]]))
    question = torch.tensor([[4, 5]])
    with torch.no_grad():
        correct = model(question, memory=first).logits
        shuffled = model(question, memory=second).logits
    assert not torch.allclose(correct, shuffled)
    model.close()
