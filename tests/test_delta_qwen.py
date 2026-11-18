import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from memwmlm.models.delta_qwen import DeltaMemoryConfig, DeltaMemoryQwen, layer_subset, null_state


def _tiny_qwen() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
    )
    return Qwen3ForCausalLM(config).eval()


def _batch():
    ids = torch.randint(0, 64, (2, 7), generator=torch.Generator().manual_seed(1))
    mask = torch.ones_like(ids)
    mask[1, 5:] = 0
    return ids, mask


def test_patched_off_and_null_memory_match_unpatched_qwen():
    qwen = _tiny_qwen()
    ids, mask = _batch()
    with torch.no_grad():
        reference = qwen(input_ids=ids, attention_mask=mask).logits
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig())
    keys, values, write_mask = memory.capture_passages(ids, mask)
    state = memory.build_state(keys, values, write_mask)
    with torch.no_grad():
        off = qwen(input_ids=ids, attention_mask=mask).logits
        with memory.reading(null_state(state)):
            null = qwen(input_ids=ids, attention_mask=mask).logits
        with memory.reading(state):
            read = qwen(input_ids=ids, attention_mask=mask).logits
    torch.testing.assert_close(off, reference)
    torch.testing.assert_close(null, reference)
    assert not torch.allclose(read, reference)
    memory.close()


def test_capture_shapes_and_write_mask():
    qwen = _tiny_qwen()
    ids, mask = _batch()
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(layers=(0, 2)))
    keys, values, write_mask = memory.capture_passages(ids, mask)
    assert keys.shape == (2, 2, 14, 8) and values.shape == keys.shape
    # 7 + 5 real tokens, minus the first token of each passage.
    assert int(write_mask.sum()) == 10
    state = memory.build_state(keys, values, write_mask)
    assert set(state) == {0, 2} and state[0].shape == (2, 8, 8)


def test_generation_runs_with_memory_and_cache():
    qwen = _tiny_qwen()
    ids, mask = _batch()
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(rule="hebbian"))
    state = memory.build_state(*memory.capture_passages(ids, mask))
    with torch.no_grad(), memory.reading(state):
        out = qwen.generate(input_ids=ids[:1], max_new_tokens=3, do_sample=False, pad_token_id=0)
    assert out.shape == (1, 10)


def test_layer_subset_parsing():
    assert layer_subset(28, "all") is None
    assert layer_subset(28, "0-3") == (0, 1, 2, 3)
    assert layer_subset(28, "1,5") == (1, 5)


def test_hebbian_token_normalised_beta_is_mean_outer_product():
    qwen = _tiny_qwen()
    ids, mask = _batch()
    plain = DeltaMemoryQwen(qwen, DeltaMemoryConfig(rule="hebbian"))
    keys, values, write_mask = plain.capture_passages(ids, mask)
    total = plain.build_state(keys, values, write_mask)
    plain.close()
    normed = DeltaMemoryQwen(qwen, DeltaMemoryConfig(rule="hebbian", beta_norm="tokens"))
    mean = normed.build_state(keys, values, write_mask)
    normed.close()
    tokens = int(write_mask.sum())
    for index in total:
        torch.testing.assert_close(mean[index] * tokens, total[index])


def test_streamed_state_equals_single_write():
    qwen = _tiny_qwen()
    ids, mask = _batch()
    for rule, norm in (("delta", "none"), ("hebbian", "tokens")):
        memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(rule=rule, beta=0.5, beta_norm=norm))
        keys, values, write_mask = memory.capture_passages(ids, mask)
        keys, values = keys[:, :, write_mask], values[:, :, write_mask]
        ones = torch.ones(keys.shape[2], dtype=torch.bool)
        whole = memory.build_state(keys, values, ones)
        cut = keys.shape[2] // 2
        first = memory.build_state(keys[:, :, :cut], values[:, :, :cut], ones[:cut], total_tokens=keys.shape[2])
        both = memory.build_state(keys[:, :, cut:], values[:, :, cut:], ones[cut:], initial=first,
                                  total_tokens=keys.shape[2])
        memory.close()
        for index in whole:
            torch.testing.assert_close(both[index], whole[index])


def test_dpfp_shapes_and_expanded_delta_write_matches_sequential() -> None:
    import torch
    from torch.nn import functional as F

    from memwmlm.memory import delta_rule
    from memwmlm.models.delta_qwen import dpfp

    x = torch.randn(3, 5, 8)
    phi = dpfp(x, 2)
    assert phi.shape == (3, 5, 32) and (phi >= 0).all()
    keys = F.normalize(dpfp(torch.randn(2, 40, 8), 4), dim=-1)
    values = torch.randn(2, 40, 8)
    state0 = torch.zeros(2, 64, 8)
    chunked = delta_rule.write_chunked(state0, keys, values, torch.full((2, 40), 0.5), chunk_size=16)
    sequential = delta_rule.write(state0, keys, values, beta=0.5)
    assert torch.allclose(chunked, sequential, atol=1e-5)


def test_prepend_sink_writes_every_content_token():
    from memwmlm.models.delta_qwen import prepend_sink

    qwen = _tiny_qwen()
    ids, mask = _batch()
    sids, smask = prepend_sink(ids, mask, sink_id=63)
    assert sids.shape == (2, 8) and (sids[:, 0] == 63).all() and torch.equal(sids[:, 1:], ids)
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(layers=(0,)))
    keys, _, write_mask = memory.capture_passages(sids, smask)
    # 7 + 5 content tokens, all written; only the sink positions are skipped
    assert int(write_mask.sum()) == 12
    assert not write_mask.view(2, 8)[:, 0].any() and write_mask.view(2, 8)[0, 1:].all()
    memory.close()


def test_center_queries_uses_key_mean_and_needs_calibration():
    import pytest

    qwen = _tiny_qwen()
    ids, mask = _batch()
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(center_keys=True, center_queries=True, key_expand=2))
    keys, values, write_mask = memory.capture_passages(ids, mask)
    with pytest.raises(RuntimeError):
        memory.build_state(keys, values, write_mask)  # no key mean yet
    memory.calibrate_key_mean(keys, write_mask)
    state = memory.build_state(keys, values, write_mask)
    assert state[0].shape == (2, 2 * 8 * 2, 8)
    # query centring = subtracting the (group-repeated) key mean before DPFP
    q = torch.randn(1, 4, 3, 8)
    from torch.nn import functional as F

    from memwmlm.models.delta_qwen import dpfp
    memory.state = state
    mean = memory.key_mean[0].repeat_interleave(memory.groups, dim=0)
    expected = torch.einsum("bhtk,hkv->bhtv", F.normalize(dpfp(q - mean[None, :, None, :], 2), dim=-1),
                            state[0].repeat_interleave(memory.groups, dim=0))
    torch.testing.assert_close(memory._read(0, q), (memory.config.gate * expected).transpose(1, 2))
    memory.close()
