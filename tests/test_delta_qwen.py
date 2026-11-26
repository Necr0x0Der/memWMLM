import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from memwmlm.models.delta_qwen import (
    DeltaMemoryConfig,
    DeltaMemoryQwen,
    joint_memory_attention,
    layer_subset,
    null_state,
)


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


def _naive_joint(query, key, value, mask_add, q_pre, mem_k, mem_v, scaling, keep=None):
    """Brute force: concatenate memory tokens to the context, one softmax, per query head."""

    b, h, t, d = query.shape
    g = h // key.shape[1]
    out = torch.zeros(b, t, h, d)
    for bi in range(b):
        for hi in range(h):
            kv = hi // g
            for ti in range(t):
                lc = (key[bi, kv] @ query[bi, hi, ti]) * scaling + mask_add[bi, 0, ti]
                lm = (mem_k[kv] @ q_pre[bi, hi, ti]) * scaling
                vm = mem_v[kv]
                if keep is not None:
                    top = lm.topk(keep).indices
                    lm, vm = lm[top], vm[top]
                w = torch.softmax(torch.cat([lc, lm]), 0)
                out[bi, ti, hi] = w[: key.shape[2]] @ value[bi, kv] + w[key.shape[2]:] @ vm
    return out


def test_joint_memory_attention_matches_brute_force():
    gen = torch.Generator().manual_seed(3)
    b, h, hkv, t, s, m, d = 2, 4, 2, 3, 5, 9, 8
    query, q_pre = torch.randn(b, h, t, d, generator=gen), torch.randn(b, h, t, d, generator=gen)
    key, value = torch.randn(b, hkv, s, d, generator=gen), torch.randn(b, hkv, s, d, generator=gen)
    mem_k, mem_v = torch.randn(hkv, m, d, generator=gen), torch.randn(hkv, m, d, generator=gen)
    mask = torch.zeros(b, 1, t, s)
    mask[1, :, :, :2] = float("-inf")  # left padding of row 1
    mask = mask + torch.ones(t, s).tril(s - t).log()  # causal (0 / -inf)
    full = joint_memory_attention(query, key, value, mask, q_pre, mem_k, mem_v, 0.35, top_k=0)
    torch.testing.assert_close(full, _naive_joint(query, key, value, mask, q_pre, mem_k, mem_v, 0.35))
    top = joint_memory_attention(query, key, value, mask, q_pre, mem_k, mem_v, 0.35, top_k=3, chunk=4)
    torch.testing.assert_close(top, _naive_joint(query, key, value, mask, q_pre, mem_k, mem_v, 0.35, keep=3))
    bool_mask = joint_memory_attention(query, key, value, mask == 0, q_pre, mem_k, mem_v, 0.35, top_k=3)
    torch.testing.assert_close(bool_mask, top)


def test_knn_memory_empty_is_plain_qwen_and_generation_runs():
    qwen = _tiny_qwen()
    ids, mask = _batch()
    with torch.no_grad():
        reference = qwen(input_ids=ids, attention_mask=mask).logits
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(read="knn", knn_top_k=4))
    state = memory.build_state(*memory.capture_passages(ids, mask))
    assert state[0][0].shape == (2, 10, 8)  # [Hkv, written tokens, D]
    with torch.no_grad():
        with memory.reading(null_state(state)):
            empty = qwen(input_ids=ids, attention_mask=mask).logits
        with memory.reading(state):
            read = qwen(input_ids=ids, attention_mask=mask).logits
            out = qwen.generate(input_ids=ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
    torch.testing.assert_close(empty, reference, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(read, reference)
    assert out.shape == (2, 10)
    memory.close()


def test_knn_memory_of_one_passage_equals_passage_in_context():
    """Right-aligned stored keys + joint softmax over all of them = the passage pasted before the prompt."""

    qwen = _tiny_qwen()
    gen = torch.Generator().manual_seed(5)
    passage, prompt = torch.randint(0, 64, (1, 6), generator=gen), torch.randint(0, 64, (1, 4), generator=gen)
    with torch.no_grad():
        reference = qwen(input_ids=torch.cat([passage, prompt], 1)).logits[:, 6:]
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(read="knn", knn_top_k=0, skip_first_token=False))
    state = memory.build_state(*memory.capture_passages(passage, torch.ones_like(passage)))
    with torch.no_grad(), memory.reading(state):
        read = qwen(input_ids=prompt).logits
    torch.testing.assert_close(read, reference, rtol=1e-4, atol=1e-4)
    memory.close()


def test_whitening_spreads_keys_and_is_applied_to_both_sides():
    from memwmlm.models.delta_qwen import whitening_matrix

    gen = torch.Generator().manual_seed(7)
    x = torch.randn(4000, 6, generator=gen) * torch.tensor([10.0, 3.0, 1.0, 1.0, 0.3, 0.1])
    x = x - x.mean(0)
    w = whitening_matrix(x, eps=0.0)
    cov = (x @ w).T @ (x @ w) / (x.shape[0] - 1)
    torch.testing.assert_close(cov, torch.eye(6), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(w, w.T)

    qwen = _tiny_qwen()
    ids, mask = _batch()
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(center_keys=True, center_queries=True, whiten=True))
    keys, values, write_mask = memory.capture_passages(ids, mask)
    memory.calibrate_key_mean(keys, write_mask)
    assert memory.key_whiten[0].shape == (2, 8, 8)
    state = memory.build_state(keys, values, write_mask)
    with torch.no_grad(), memory.reading(state):
        qwen(input_ids=ids, attention_mask=mask)
    bad = DeltaMemoryQwen(_tiny_qwen(), DeltaMemoryConfig(center_keys=True, whiten=True))
    bad.calibrate_key_mean(*bad.capture_passages(ids, mask)[::2])
    state = bad.build_state(*bad.capture_passages(ids, mask))
    import pytest
    with pytest.raises(RuntimeError), torch.no_grad(), bad.reading(state):
        bad.qwen(input_ids=ids, attention_mask=mask)
    memory.close()


def test_per_example_states_match_separate_reads():
    qwen = _tiny_qwen()
    ids, mask = _batch()
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(gate=0.7))
    states = [memory.build_state(*memory.capture_passages(ids[i : i + 1], mask[i : i + 1])) for i in range(2)]
    stacked = {layer: torch.stack([s[layer] for s in states]) for layer in states[0]}
    with torch.no_grad():
        with memory.reading(stacked):
            both = qwen(input_ids=ids, attention_mask=mask).logits
        for i in range(2):
            with memory.reading(states[i]):
                one = qwen(input_ids=ids[i : i + 1], attention_mask=mask[i : i + 1]).logits
            torch.testing.assert_close(both[i : i + 1][mask[i : i + 1].bool()], one[mask[i : i + 1].bool()])
    memory.close()


def test_knn_teacher_record_is_memory_contribution():
    qwen = _tiny_qwen()
    ids, mask = _batch()
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(read="knn", knn_top_k=0))
    state = memory.build_state(*memory.capture_passages(ids, mask))
    memory.record = {}
    with torch.no_grad(), memory.reading(state):
        qwen(input_ids=ids[:1], attention_mask=mask[:1])
    q, contribution = memory.record[0]
    assert q.shape[:2] == (1, 4) and contribution.shape == (1, 7, 4, 8)
    assert contribution.abs().sum() > 0
    memory.record = {}
    from memwmlm.models.delta_qwen import null_state
    with torch.no_grad(), memory.reading(null_state(state)):
        qwen(input_ids=ids[:1], attention_mask=mask[:1])
    torch.testing.assert_close(memory.record[0][1], torch.zeros_like(contribution), atol=1e-5, rtol=0)
    memory.close()


def test_feature_map_state_and_read_match_manual_delta_rule_and_train():
    from memwmlm.memory import delta_rule
    from memwmlm.models.delta_qwen import FeatureMaps

    qwen = _tiny_qwen()
    ids, mask = _batch()
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(gate=1.0))
    fm = FeatureMaps(3, 4, 2, 8)
    with torch.no_grad():
        fm.wq.add_(0.1 * torch.randn_like(fm.wq)); fm.log_scale.add_(0.3)
    memory.feature_maps = fm
    keys, values, m = memory.capture_passages(ids, mask)
    state = memory.build_state(keys, values, m)
    k = fm.keys(1, keys[1][:, m].float())
    manual = delta_rule.write(torch.zeros(2, 16, 8), k, values[1][:, m].float(), beta=1.0, alpha=1.0, rule="delta")
    torch.testing.assert_close(state[1], manual, atol=1e-4, rtol=1e-4)
    q = torch.randn(1, 4, 3, 8)
    memory.state = state
    read = memory._read(1, q)  # [1, T, H, D]
    expect = torch.einsum("bhtk,hkv->bthv", fm.queries(1, q), manual.repeat_interleave(2, dim=0))
    torch.testing.assert_close(read, expect, atol=1e-4, rtol=1e-4)
    # gradients reach both maps through the write and the read
    st = memory.build_feature_state(keys[:, :, m], values[:, :, m])
    memory.state = st
    memory._read(1, q).pow(2).sum().backward()
    assert fm.wk.grad[1].abs().sum() > 0 and fm.wq.grad[1].abs().sum() > 0
    memory.close()


def test_passage_queries_are_masked_means_and_fix_the_memory_read():
    qwen = _tiny_qwen()
    ids, mask = _batch()
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(gate=0.7))
    state = memory.build_state(*memory.capture_passages(ids, mask))
    pq = memory.passage_queries(ids, mask)
    memory.capture_queries = True
    with memory.capturing() as captures, torch.no_grad():
        qwen(input_ids=ids, attention_mask=mask)
    memory.capture_queries = False
    layer = memory.layer_indices[0]
    q = captures[layer].queries[0].float()  # [B, H, T, D]
    m = mask.bool().clone()
    m[torch.arange(2), m.float().argmax(1)] = False
    for b in range(2):
        torch.testing.assert_close(pq[layer][b], q[b][:, m[b]].mean(1))
    # the fixed query replaces each token's own query in the memory read only
    seen = {}
    original = memory._read

    def spy(index, q_read):
        seen[index] = q_read
        return original(index, q_read)

    memory._read = spy
    memory.read_queries = pq
    with torch.no_grad(), memory.reading(state):
        qwen(input_ids=ids, attention_mask=mask)
    memory.read_queries = None
    for index in memory.layer_indices:
        torch.testing.assert_close(seen[index].float(), pq[index][:, :, None].expand_as(seen[index]).float())
    memory.close()


def test_knn_passage_query_of_own_query_matches_normal_read_for_one_token():
    qwen = _tiny_qwen()
    ids, mask = _batch()
    # memory in layer 0 only: there the token's query does not depend on earlier memory reads
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(read="knn", knn_top_k=0, layers=(0,)))
    state = memory.build_state(*memory.capture_passages(ids, mask))
    one, onem = ids[:, -1:], torch.ones_like(ids[:, -1:])
    memory.capture_queries = True
    with memory.capturing() as captures, torch.no_grad():
        qwen(input_ids=one, attention_mask=onem)
    memory.capture_queries = False
    own = {i: captures[i].queries[0][:, :, 0].float() for i in memory.layer_indices}
    with torch.no_grad(), memory.reading(state):
        ref = qwen(input_ids=one, attention_mask=onem).logits
        memory.read_queries = own
        got = qwen(input_ids=one, attention_mask=onem).logits
        memory.read_queries = None
    torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-4)
    memory.close()


def test_learned_beta_matches_fixed_beta_write_and_gets_gradients():
    qwen = _tiny_qwen()
    ids, mask = _batch()
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(beta=0.7))
    k, v, m = memory.capture_passages(ids, mask)
    fixed = memory.build_state(k, v, m)
    logit = torch.log(torch.tensor(0.7) / 0.3)
    memory.beta_logits = torch.full((len(memory.layer_indices), memory.num_kv_heads), float(logit), requires_grad=True)
    learned = memory.build_state(k, v, m)
    for index in fixed:
        torch.testing.assert_close(learned[index], fixed[index], atol=1e-4, rtol=1e-4)
    sum(s.square().sum() for s in learned.values()).backward()
    assert memory.beta_logits.grad is not None and memory.beta_logits.grad.abs().sum() > 0
    memory.close()


def test_token_beta_starts_as_constant_beta_and_learns_per_token():
    from memwmlm.models.delta_qwen import TokenBeta

    qwen = _tiny_qwen()
    ids, mask = _batch()
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(beta=0.7))
    k, v, m = memory.capture_passages(ids, mask)
    fixed = memory.build_state(k, v, m)
    memory.beta_proj = TokenBeta(len(memory.layer_indices), memory.num_kv_heads, qwen.config.hidden_size, 0.7)
    k2, v2, m2 = memory.capture_passages(ids, mask)
    assert memory.last_hidden.shape[:2] == (len(memory.layer_indices), ids.numel())
    learned = memory.build_state(k2, v2, m2, hidden=memory.last_hidden)
    for index in fixed:
        torch.testing.assert_close(learned[index], fixed[index], atol=1e-4, rtol=1e-4)
    sum(s.square().sum() for s in learned.values()).backward()
    assert memory.beta_proj.weight.grad.abs().sum() > 0  # depends on the token
    memory.close()


def test_token_beta_with_expanded_keys_matches_fixed_write_and_trains_through_the_read():
    from memwmlm.models.delta_qwen import TokenBeta

    qwen = _tiny_qwen()
    ids, mask = _batch()
    config = DeltaMemoryConfig(beta=0.7, key_expand=2, write_chunk=3, center_keys=True, center_queries=True)
    memory = DeltaMemoryQwen(qwen, config)
    k, v, m = memory.capture_passages(ids, mask)
    memory.calibrate_key_mean(k, m)
    fixed = memory.build_state(k, v, m)
    memory.beta_proj = TokenBeta(len(memory.layer_indices), memory.num_kv_heads, qwen.config.hidden_size, 0.7)
    k2, v2, m2 = memory.capture_passages(ids, mask)
    learned = memory.build_state(k2, v2, m2, hidden=memory.last_hidden)
    for index in fixed:
        assert learned[index].shape == (memory.num_kv_heads, memory.key_dim, memory.head_dim)
        torch.testing.assert_close(learned[index], fixed[index], atol=1e-4, rtol=1e-4)
    # read with grad (per-row states, checkpointed path) == read without grad; beta gets gradients
    state = {i: s[None].expand(2, *s.shape) for i, s in learned.items()}
    query_ids = ids[:2, :4]
    with torch.no_grad(), memory.reading({i: s.detach() for i, s in state.items()}):
        reference = qwen(input_ids=query_ids).logits
    with memory.reading(state):
        logits = qwen(input_ids=query_ids).logits
    torch.testing.assert_close(logits, reference, atol=1e-4, rtol=1e-4)
    logits.square().sum().backward()
    assert memory.beta_proj.weight.grad.abs().sum() > 0
    memory.close()


def test_writer_lora_starts_as_frozen_write_and_gets_gradients():
    from memwmlm.models.delta_qwen import TokenBeta, WriterLoRA

    qwen = _tiny_qwen()
    ids, mask = _batch()
    memory = DeltaMemoryQwen(qwen, DeltaMemoryConfig(beta=0.7, center_keys=True))
    k, v, m = memory.capture_passages(ids, mask)
    memory.calibrate_key_mean(k, m)
    fixed = memory.build_state(k, v, m)
    memory.writer = WriterLoRA(memory, rank=4)
    memory.beta_proj = TokenBeta(len(memory.layer_indices), memory.num_kv_heads, qwen.config.hidden_size, 0.7)
    k2, v2, m2 = memory.capture_passages(ids, mask)
    written = memory.build_state(k2, v2, m2, hidden=memory.last_hidden)
    for index in fixed:
        torch.testing.assert_close(written[index], fixed[index], atol=1e-4, rtol=1e-4)
    sum(s.square().sum() for s in written.values()).backward()
    assert memory.writer.up_k.grad.abs().sum() > 0 and memory.writer.up_v.grad.abs().sum() > 0
    memory.close()
