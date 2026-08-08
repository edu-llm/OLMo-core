import importlib

import pytest
import torch

fast = importlib.import_module("olmo_core.nn.mamba3.mamba3_ssd_fast")


def _ordered_prefix(rotations: torch.Tensor) -> torch.Tensor:
    carry = rotations[:, 0]
    prefixes = [carry]
    for index in range(1, rotations.shape[1]):
        carry = rotations[:, index] @ carry
        prefixes.append(carry)
    return torch.stack(prefixes, dim=1)


def _run_with_gradients(build, B, C, theta):
    b = B.clone().requires_grad_(True)
    c = C.clone().requires_grad_(True)
    t = theta.clone().requires_grad_(True)
    out_b, out_c = build(b, c, t)
    generator = torch.Generator().manual_seed(58)
    weight_b = torch.randn(out_b.shape, dtype=out_b.dtype, generator=generator)
    weight_c = torch.randn(out_c.shape, dtype=out_c.dtype, generator=generator)
    grads = torch.autograd.grad((out_b * weight_b).sum() + (out_c * weight_c).sum(), (b, c, t))
    return (out_b, out_c), grads


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        pytest.param(torch.float64, 0, 2e-10, id="float64"),
        pytest.param(torch.float32, 2e-5, 3e-5, id="float32"),
    ],
)
@pytest.mark.parametrize("seq_len", [1, 19], ids=["decode-step", "tail"])
def test_fused_quaternion_forward_and_analytic_backward_match_matrix_oracle(
    dtype, rtol, atol, seq_len
):
    torch.manual_seed(56)
    B = torch.randn(2, seq_len, 1, 2, 12, dtype=dtype)
    C = torch.randn_like(B)
    theta = torch.randn(2, seq_len, 1, 4, 3, dtype=dtype) * 0.2

    expected = _run_with_gradients(
        lambda b, c, t: fast._rotate_bc_fused(
            b, c, _ordered_prefix(fast.fast_block_rotations(t, 3))
        ),
        B,
        C,
        theta,
    )
    actual = _run_with_gradients(fast._fused_quaternion_rotate_bc, B, C, theta)

    for got, want in zip(actual[0] + actual[1], expected[0] + expected[1]):
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)


def test_fused_quaternion_preprocessing_matches_direct_bc_path():
    """Fusion changes the autograd boundary, not the direct quaternion result."""
    torch.manual_seed(57)
    B = torch.randn(2, 13, 1, 2, 12, dtype=torch.float64)
    C = torch.randn_like(B)
    theta = torch.randn(2, 13, 1, 4, 3, dtype=torch.float64) * 0.2

    expected = fast._rotate_bc_quaternion(
        B,
        C,
        fast._quaternion_prefix(fast._angles_to_quaternion(theta)),
    )
    actual = fast._fused_quaternion_rotate_bc(B, C, theta)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_fused_quaternion_saves_compact_prefix_and_never_materializes_3x3():
    torch.manual_seed(59)
    B = torch.randn(1, 23, 1, 2, 12, dtype=torch.float64, requires_grad=True)
    C = torch.randn_like(B, requires_grad=True)
    theta = (torch.randn(1, 23, 1, 4, 3, dtype=torch.float64) * 0.2).requires_grad_()
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out_b, out_c = fast._fused_quaternion_rotate_bc(B, C, theta)
        (out_b.sum() + out_c.sum()).backward()

    assert len(saved) == 4
    assert sum(tensor.numel() for tensor in saved) == (
        B.numel() + C.numel() + theta.numel() + theta.numel() // 3 * 4
    )
    assert saved[0].untyped_storage().data_ptr() == B.untyped_storage().data_ptr()
    assert saved[1].untyped_storage().data_ptr() == C.untyped_storage().data_ptr()
    assert saved[2].untyped_storage().data_ptr() == theta.untyped_storage().data_ptr()
    assert saved[3].shape[-1] == 4
    assert all(tensor.shape[-2:] != (3, 3) for tensor in saved)


@pytest.mark.parametrize("block_size,expected_calls", [(2, 0), (3, 1), (4, 0)])
def test_only_b3_quaternion_dispatch_uses_fused_preprocessing(
    monkeypatch, block_size, expected_calls
):
    calls = 0
    real_fused = fast._fused_quaternion_rotate_bc

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_fused(*args, **kwargs)

    monkeypatch.setattr(fast, "_fused_quaternion_rotate_bc", counted)
    blocks = 4
    angles = block_size * (block_size - 1) // 2
    B = torch.randn(1, 11, 1, 1, blocks * block_size, dtype=torch.float64)
    C = torch.randn_like(B)
    theta = torch.randn(1, 11, 1, blocks, angles, dtype=torch.float64) * 0.2

    fast._fast_rotate_bc_pair(B, C, theta, block_size, None, scan_impl="quaternion")
    assert calls == expected_calls


def test_fused_quaternion_preprocessing_compiles_fullgraph():
    B = torch.randn(1, 8, 1, 1, 6)
    C = torch.randn_like(B)
    theta = torch.randn(1, 8, 1, 2, 3) * 0.1

    def step(b, c, t):
        out_b, out_c = fast._fused_quaternion_rotate_bc(b, c, t)
        return out_b.square().sum() + out_c.sin().sum()

    def gradients(fn):
        inputs = tuple(tensor.clone().requires_grad_(True) for tensor in (B, C, theta))
        return torch.autograd.grad(fn(*inputs), inputs)

    torch._dynamo.reset()
    compiled = gradients(torch.compile(step, fullgraph=True, backend="eager"))
    expected = gradients(step)
    for got, want in zip(compiled, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=2e-5)
