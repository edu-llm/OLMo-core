import pytest
import torch

from olmo_core.nn.mamba3.mamba3_ssd_api import mamba3_ssd_reference
from olmo_core.nn.mamba3.mamba3_ssd_chunked import mamba3_ssd_chunked
from olmo_core.testing import requires_gpu


def _inputs(
    *,
    batch: int = 2,
    seq_len: int = 40,
    n_heads: int = 4,
    head_dim: int = 8,
    n_groups: int = 1,
    rank: int = 1,
    d_state: int = 8,
    block_size: int = 2,
    seed: int = 0,
):
    torch.manual_seed(seed)
    n_blocks = d_state // block_size
    angles = block_size * (block_size - 1) // 2
    theta_shape = (
        (batch, seq_len, n_groups, n_blocks)
        if block_size == 2
        else (batch, seq_len, n_groups, n_blocks, angles)
    )
    return dict(
        x=torch.randn(batch, seq_len, n_heads, head_dim),
        B=torch.randn(batch, seq_len, n_groups, rank, d_state),
        C=torch.randn(batch, seq_len, n_groups, rank, d_state),
        dt=torch.rand(batch, seq_len, n_heads) * 0.1 + 0.01,
        A=-torch.rand(n_heads) - 0.5,
        lam=torch.rand(batch, seq_len, n_heads),
        theta=torch.randn(*theta_shape),
        heads_per_group=n_heads // n_groups,
        block_size=block_size,
    )


@pytest.mark.parametrize("chunk_size", [4, 8, 16, 64, 256], ids=lambda q: f"q{q}")
@pytest.mark.parametrize("rank", [1, 3], ids=["siso", "mimo3"])
@pytest.mark.parametrize("n_groups", [1, 2], ids=["g1", "g2"])
def test_mamba3_ssd_chunked_matches_reference(chunk_size: int, rank: int, n_groups: int):
    """The chunked (SSD) form must reproduce the sequential reference exactly."""
    kwargs = _inputs(rank=rank, n_groups=n_groups)
    block_size = kwargs.pop("block_size")
    heads_per_group = kwargs.pop("heads_per_group")

    expected = mamba3_ssd_reference(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size
    )
    actual = mamba3_ssd_chunked(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size, chunk_size=chunk_size
    )
    # Same arithmetic, different association order; fp32 over <=40 steps.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("seq_len", [1, 7, 16, 33], ids=lambda t: f"t{t}")
def test_mamba3_ssd_chunked_handles_ragged_sequence_lengths(seq_len: int):
    """Sequence lengths that are not a multiple of the chunk size must still be exact."""
    kwargs = _inputs(seq_len=seq_len)
    block_size = kwargs.pop("block_size")
    heads_per_group = kwargs.pop("heads_per_group")

    expected = mamba3_ssd_reference(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size
    )
    actual = mamba3_ssd_chunked(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size, chunk_size=8
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_mamba3_ssd_chunked_supports_non_abelian_blocks():
    """The chunked form is rotation-agnostic: it must work for SO(b>=3) blocks too."""
    kwargs = _inputs(d_state=12, block_size=3)
    block_size = kwargs.pop("block_size")
    heads_per_group = kwargs.pop("heads_per_group")

    expected = mamba3_ssd_reference(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size
    )
    actual = mamba3_ssd_chunked(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size, chunk_size=8
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_dispatch_uses_the_chunked_path_by_default():
    """The model must actually reach the chunked kernel, not just have it sitting in the tree."""
    from olmo_core.nn.mamba3.mamba3_ssd_api import dispatch_mamba3_ssd

    kwargs = _inputs()
    block_size = kwargs.pop("block_size")
    heads_per_group = kwargs.pop("heads_per_group")

    dispatched = dispatch_mamba3_ssd(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size
    )
    chunked = mamba3_ssd_chunked(**kwargs, heads_per_group=heads_per_group, block_size=block_size)
    assert torch.equal(dispatched, chunked), "dispatch did not take the chunked path"


def test_dispatch_can_opt_out_to_the_reference_oracle():
    """The sequential reference stays reachable as the numerical oracle."""
    from olmo_core.nn.mamba3.mamba3_ssd_api import dispatch_mamba3_ssd

    kwargs = _inputs()
    block_size = kwargs.pop("block_size")
    heads_per_group = kwargs.pop("heads_per_group")

    dispatched = dispatch_mamba3_ssd(
        **kwargs,
        heads_per_group=heads_per_group,
        block_size=block_size,
        prefer_fast_kernel=False,
    )
    reference = mamba3_ssd_reference(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size
    )
    assert torch.equal(dispatched, reference), "opt-out did not take the reference path"


@requires_gpu
def test_chunked_uses_reduced_precision_under_autocast():
    """The matmuls must follow autocast, or bf16 tensor cores are left on the table."""
    kwargs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _inputs().items()}
    block_size = kwargs.pop("block_size")
    heads_per_group = kwargs.pop("heads_per_group")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = mamba3_ssd_chunked(**kwargs, heads_per_group=heads_per_group, block_size=block_size)
    assert out.dtype == torch.bfloat16, f"autocast ignored; got {out.dtype}"


@requires_gpu
def test_chunked_keeps_the_rotation_in_fp32_under_autocast():
    """
    Orthogonality drift of the b>=3 prefix product is O(T * eps): bf16 gives ~27% error at
    T=1024 and stops being a rotation at all. The rotation must stay fp32 regardless of
    autocast, so a blocked run under autocast must still track the fp32 reference.
    """
    kwargs = {
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in _inputs(seq_len=256, d_state=12, block_size=3).items()
    }
    block_size = kwargs.pop("block_size")
    heads_per_group = kwargs.pop("heads_per_group")

    expected = mamba3_ssd_reference(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = mamba3_ssd_chunked(
            **kwargs, heads_per_group=heads_per_group, block_size=block_size
        )
    # bf16 matmuls give ~1e-2 relative; a bf16 *rotation* would be off by orders more.
    torch.testing.assert_close(actual.float(), expected, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("chunk_size", [64, 256], ids=lambda q: f"q{q}")
def test_chunked_gradients_are_finite_at_realistic_decay(chunk_size: int):
    """
    Backward must not produce NaN when the decay is large enough to overflow ``exp``.

    ``A`` is initialised as ``-exp(A_log)`` with ``A_log`` up to ``log(16)``, so real models
    reach ``|A| ~ 16`` -- far larger than the ``[0.5, 1.5]`` the other tests sample. Over a
    chunk, the *non-causal* entries of ``L_t - L_s`` are large and positive, so evaluating
    ``exp`` before masking overflows to ``inf``; the forward hides it but the backward turns
    ``0 * inf`` into NaN.
    """
    kwargs = _inputs(seq_len=512, n_heads=4, d_state=16)
    kwargs.pop("block_size")
    heads_per_group = kwargs.pop("heads_per_group")
    kwargs["A"] = -torch.full((4,), 16.0)
    kwargs["dt"] = torch.full_like(kwargs["dt"], 0.1)
    # dt and A must be in the graph: the NaN is on the gradient flowing back through the
    # decay exponential, which is unreachable if only x/B/C require grad.
    for name in ("x", "B", "C", "dt", "A"):
        kwargs[name] = kwargs[name].clone().requires_grad_(True)

    out = mamba3_ssd_chunked(**kwargs, heads_per_group=heads_per_group, chunk_size=chunk_size)
    assert torch.isfinite(out).all(), "forward is not finite"
    out.sum().backward()
    for name in ("x", "B", "C", "dt", "A"):
        grad = kwargs[name].grad
        assert grad is not None and torch.isfinite(grad).all(), f"non-finite grad for {name}"


def test_mamba3_ssd_chunked_gradients_match_reference():
    """Backward must agree too, not just forward."""
    kwargs = _inputs()
    block_size = kwargs.pop("block_size")
    heads_per_group = kwargs.pop("heads_per_group")

    def grads_of(chunked: bool) -> dict:
        args = {
            k: (v.clone().requires_grad_(True) if torch.is_tensor(v) else v)
            for k, v in kwargs.items()
        }
        if chunked:
            out = mamba3_ssd_chunked(
                **args, heads_per_group=heads_per_group, block_size=block_size, chunk_size=8
            )
        else:
            out = mamba3_ssd_reference(
                **args, heads_per_group=heads_per_group, block_size=block_size
            )
        out.sum().backward()
        return {k: v.grad for k, v in args.items() if torch.is_tensor(v)}

    grads = {"ref": grads_of(chunked=False), "chunked": grads_of(chunked=True)}

    for key in grads["ref"]:
        torch.testing.assert_close(
            grads["chunked"][key],
            grads["ref"][key],
            rtol=1e-4,
            atol=1e-5,
            msg=f"grad mismatch: {key}",
        )
