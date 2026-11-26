import torch

from memwmlm.memory import delta_rule


def _orthonormal_keys(n: int, d: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(0)))
    return q[:n]


def test_delta_rule_retrieves_orthonormal_keys_exactly():
    keys = _orthonormal_keys(8, 16)
    values = torch.randn(8, 4, generator=torch.Generator().manual_seed(1))
    state = delta_rule.write(delta_rule.empty_state((), 16, 4), keys, values)
    torch.testing.assert_close(delta_rule.read(state, keys), values, atol=1e-5, rtol=1e-5)


def test_delta_rule_overwrites_but_hebbian_accumulates():
    key = torch.nn.functional.normalize(torch.randn(1, 16), dim=-1)
    old, new = torch.ones(1, 4), 2 * torch.ones(1, 4)
    keys = torch.cat([key, key])
    values = torch.cat([old, new])
    empty = delta_rule.empty_state((), 16, 4)
    delta = delta_rule.read(delta_rule.write(empty, keys, values, rule="delta"), key)
    hebb = delta_rule.read(delta_rule.write(empty, keys, values, rule="hebbian"), key)
    torch.testing.assert_close(delta, new, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(hebb, old + new, atol=1e-5, rtol=1e-5)


def test_mask_skips_writes_and_decay():
    keys = _orthonormal_keys(3, 8)
    values = torch.randn(3, 2)
    mask = torch.tensor([True, False, True])
    state = delta_rule.write(delta_rule.empty_state((), 8, 2), keys, values, mask=mask, alpha=0.5)
    reference = delta_rule.write(delta_rule.empty_state((), 8, 2), keys[mask], values[mask], alpha=0.5)
    torch.testing.assert_close(state, reference)


def test_batched_heads_and_per_token_beta():
    keys = torch.nn.functional.normalize(torch.randn(2, 3, 5, 8), dim=-1)
    values = torch.randn(2, 3, 5, 4)
    beta = torch.rand(2, 3, 5)
    state = delta_rule.write(delta_rule.empty_state((2, 3), 8, 4), keys, values, beta=beta)
    single = delta_rule.write(delta_rule.empty_state((), 8, 4), keys[1, 2], values[1, 2], beta=beta[1, 2])
    torch.testing.assert_close(state[1, 2], single)


def test_write_is_differentiable():
    keys = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
    values = torch.randn(4, 3, requires_grad=True)
    state = delta_rule.write(delta_rule.empty_state((), 8, 3), keys, values)
    delta_rule.read(state, keys).sum().backward()
    assert keys.grad is not None and values.grad is not None


def test_chunked_delta_write_matches_sequential():
    write_chunked = delta_rule.write_chunked

    g = torch.Generator().manual_seed(0)
    keys = torch.nn.functional.normalize(torch.randn(2, 37, 8, generator=g, dtype=torch.float64), dim=-1)
    values = torch.randn(2, 37, 5, generator=g, dtype=torch.float64)
    beta = torch.rand(2, 37, generator=g, dtype=torch.float64)
    start = torch.randn(2, 8, 5, generator=g, dtype=torch.float64)
    expected = delta_rule.write(start, keys, values, beta=beta, rule="delta")
    for chunk in (1, 5, 16, 64):
        torch.testing.assert_close(write_chunked(start, keys, values, beta, chunk_size=chunk), expected)
    leaf = beta.clone().requires_grad_(True)
    write_chunked(start, keys, values, leaf, chunk_size=8, checkpoint=True).sum().backward()
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
