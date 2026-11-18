import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from memwmlm.models.delta_interface import DeltaInterface, InterfaceQwen


def _tiny():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    return Qwen3ForCausalLM(config).eval()


def test_untrained_interface_and_null_state_leave_qwen_unchanged():
    qwen = _tiny()
    ids = torch.randint(0, 64, (2, 7), generator=torch.Generator().manual_seed(1))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        reference = qwen(input_ids=ids, attention_mask=mask).logits
    iface = DeltaInterface(32, 16)
    wrapped = InterfaceQwen(qwen, iface, layer=1)
    tokens = wrapped.passage_tokens(ids, mask)
    assert tokens.shape == (12, 32)
    state = iface.write(tokens)
    with torch.no_grad(), wrapped.reading(state):
        same = qwen(input_ids=ids, attention_mask=mask).logits  # W_O = 0
    torch.testing.assert_close(same, reference)
    torch.nn.init.normal_(iface.w_o.weight)
    with torch.no_grad(), wrapped.reading(torch.zeros_like(state)):
        null = qwen(input_ids=ids, attention_mask=mask).logits
    torch.testing.assert_close(null, reference)
    with torch.no_grad(), wrapped.reading(state):
        read = qwen(input_ids=ids, attention_mask=mask).logits
    assert not torch.allclose(read, reference)
    wrapped.close()


def test_gradients_reach_writer_and_reader():
    qwen = _tiny()
    ids = torch.randint(0, 64, (2, 7), generator=torch.Generator().manual_seed(2))
    mask = torch.ones_like(ids)
    iface = DeltaInterface(32, 16)
    torch.nn.init.normal_(iface.w_o.weight, std=0.1)
    wrapped = InterfaceQwen(qwen, iface, layer=1)
    state = iface.write(wrapped.passage_tokens(ids, mask), chunk_size=4, checkpoint=True)
    with wrapped.reading(state):
        qwen(input_ids=ids, attention_mask=mask).logits.sum().backward()
    for p in (iface.w_k.weight, iface.w_v.weight, iface.w_q.weight, iface.w_o.weight, iface.w_b.bias):
        assert p.grad is not None and p.grad.abs().sum() > 0
    wrapped.close()


def test_shifted_values_pair_next_token_within_passages() -> None:
    import torch
    from torch.nn import functional as F

    from memwmlm.memory import delta_rule
    from memwmlm.models.delta_interface import DeltaInterface, _rms

    torch.manual_seed(0)
    iface = DeltaInterface(16, 8, shift_values=True)
    torch.nn.init.normal_(iface.w_b.weight)
    hidden = torch.randn(7, 16)
    lengths = [3, 1, 3]  # pairs: (0,1), (1,2), (4,5), (5,6); passage of length 1 writes nothing
    got = iface.write(hidden, lengths=lengths, chunk_size=2)
    x = _rms(hidden)
    k_idx, v_idx = torch.tensor([0, 1, 4, 5]), torch.tensor([1, 2, 5, 6])
    keys = F.normalize(iface.w_k(x[k_idx]), dim=-1)
    want = delta_rule.write(iface.empty_state("cpu"), keys, iface.w_v(x[v_idx]),
                            beta=torch.sigmoid(iface.w_b(x[k_idx])).squeeze(-1))
    assert torch.allclose(got, want, atol=1e-5)


def test_standardisation_round_trips_through_state_dict() -> None:
    import torch

    from memwmlm.models.delta_interface import DeltaInterface

    a = DeltaInterface(8, 4)
    a.set_input_stats(torch.arange(8.0), torch.full((8,), 2.0))
    b = DeltaInterface(8, 4)
    b.load_state_dict(a.state_dict())
    h = torch.randn(3, 8)
    assert torch.allclose(b._norm(h), (h - torch.arange(8.0)) / 2)
