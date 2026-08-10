"""
Kimi Delta Attention and its two bake-off variants.

The shipped operator's kernel (``fla.ops.kda.chunk_kda``) is Triton and runs on CUDA only, so the
CPU coverage here reaches it through the Householder layer's ``backend="torch"`` reference, which
at ``num_householder=1`` *is* the KDA recurrence. That gives an oracle comparison, a ``float64``
``gradcheck`` and an end-to-end module check without a GPU; the fused kernel itself is compared
against the same oracle in the GPU tests at the bottom.
"""

from typing import Optional, Tuple

import pytest
import torch
import torch.nn as nn

from olmo_core.nn.attention import (
    AttentionConfig,
    KimiDeltaAttention,
    KimiDeltaAttentionConfig,
    KimiDeltaHouseholder,
    KimiDeltaHouseholderConfig,
)
from olmo_core.nn.attention.base import SequenceMixerConfig
from olmo_core.nn.attention.kda_householder_torch import kda_householder_torch
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.functional import l2_normalize
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.nn.utils import no_weight_decay_param_names
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import has_fla, requires_fla
from olmo_core.utils import seed_all

# `requires_fla` carries `pytest.mark.gpu`, so it cannot be used on a test that is meant to run
# on CPU. This is the same availability check without the GPU mark.
requires_fla_cpu = pytest.mark.skipif(not has_fla, reason="Requires flash-linear-attention (fla)")

#: The study geometry: 16 layers at ``d_model=1024``, peers configured at 16 heads of 64.
STUDY_D_MODEL = 1024
STUDY_N_HEADS = 16
STUDY_HEAD_DIM = 64

#: Parameter count of each variant's bare mixer at the study geometry.
#:
#: Hand-derived from each config's own ``num_params`` algebra and asserted against a built module
#: by :func:`test_variant_parameter_counts_at_the_study_geometry`, so a number here that disagrees
#: with the module is a bug in one of the two and the test says which. The base figure is also the
#: bake-off branch's own ``_BLOCK_PARAMS["kda"]``.
STUDY_PARAMS = {
    "kda_base": 4_487_248,
    "kda_householder_r2_negeig": 6_608_976,
    "kda_gconv": 4_493_392,
}


def _study_configs() -> dict:
    """The three variants, at the frozen head geometry, with their exact constructor arguments."""
    return {
        "kda_base": KimiDeltaAttentionConfig(
            n_heads=STUDY_N_HEADS,
            head_dim=STUDY_HEAD_DIM,
        ),
        "kda_householder_r2_negeig": KimiDeltaHouseholderConfig(
            n_heads=STUDY_N_HEADS,
            head_dim=STUDY_HEAD_DIM,
            num_householder=2,
            allow_neg_eigval=True,
        ),
        "kda_gconv": KimiDeltaAttentionConfig(
            n_heads=STUDY_N_HEADS,
            head_dim=STUDY_HEAD_DIM,
            gated_conv=True,
            gate_structure="depthwise",
        ),
    }


##########################################
# The oracle                             #
##########################################


def _naive_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int,
    scale: float,
    initial_state: Optional[torch.Tensor],
    compute_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference KDA recurrence with ``R`` Householder factors per token.

    Per token: decay the state once per channel, apply ``R`` successive rank-1 delta updates
    each reading the current state, then read out. At ``R=1`` this is plain KDA.

    :param q: Queries, ``[B, T, H, K]``.
    :param k: Keys, ``[B, T * R, H, K]``, interleaved along time.
    :param v: Values, ``[B, T * R, H, V]``, interleaved along time.
    :param g: Log-space per-channel decay, ``[B, T, H, K]``, one entry per token.
    :param beta: Delta-rule step sizes, ``[B, T * R, H]``, interleaved along time.
    :param num_householder: The number of factors ``R``.
    :param scale: Query scaling factor.
    :param initial_state: Optional initial state, ``[B, H, K, V]``.
    :param compute_dtype: Dtype of all internal arithmetic.

    :returns: ``(o, S)`` with shapes ``[B, T, H, V]`` and ``[B, H, K, V]``.
    """
    R = num_householder
    B, T, H, K = q.shape
    V = v.shape[-1]

    q, k, v, g, beta = (x.to(compute_dtype) for x in (q, k, v, g, beta))
    q = q * scale

    S = torch.zeros(B, H, K, V, dtype=compute_dtype, device=q.device)
    if initial_state is not None:
        S = S + initial_state.to(compute_dtype)
    o = torch.zeros(B, T, H, V, dtype=compute_dtype, device=q.device)
    for i in range(T):
        S = S * g[:, i][..., None].exp()
        for j in range(R):
            k_ij, v_ij, b_ij = k[:, i * R + j], v[:, i * R + j], beta[:, i * R + j]
            S = S + torch.einsum(
                "b h k, b h v -> b h k v",
                b_ij[..., None] * k_ij,
                v_ij - (k_ij[..., None] * S).sum(-2),
            )
        o[:, i] = torch.einsum("b h k, b h k v -> b h v", q[:, i], S)
    return o, S


def _torch_causal_conv1d(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    backend: str = "triton",
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor]:
    """
    Depthwise causal convolution in plain torch, standing in for the Triton dispatch.

    :class:`~olmo_core.nn.convolution.CausalConv1d` has no CPU path -- it goes straight to
    ``fla``'s kernel -- so a KDA layer cannot run a forward pass on this machine without one.

    :param x: The stream, ``(batch_size, seq_len, hidden_size)``.
    :param weight: Convolution taps, ``(hidden_size, kernel_size)``.
    :param bias: Optional bias.
    :param activation: ``"silu"``, ``"swish"`` or ``None``.
    :param backend: Ignored; present to match the dispatch signature.
    :param cu_seqlens: Unsupported here.

    :returns: A one-tuple holding the output, matching what the dispatch returns.

    :raises NotImplementedError: If ``cu_seqlens`` is given.
    """
    del backend
    if cu_seqlens is not None:
        raise NotImplementedError("the CPU stand-in does not implement 'cu_seqlens'")
    hidden, kernel_size = weight.shape
    z = torch.nn.functional.conv1d(
        x.transpose(1, 2), weight.unsqueeze(1), bias, padding=kernel_size - 1, groups=hidden
    )[..., : x.shape[1]]
    z = z.transpose(1, 2)
    if activation in ("silu", "swish"):
        z = torch.nn.functional.silu(z)
    return (z,)


class _TorchRMSNormGated(nn.Module):
    """
    Sigmoid-gated RMS norm in plain torch, standing in for ``fla``'s Triton ``FusedRMSNormGated``.

    Only used to make a CPU forward pass reachable. Both the layer under test and the reference
    it is compared against call *this same object*, so the comparison isolates how the layer
    wires its pieces together and says nothing about the norm itself.
    """

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float() * torch.sigmoid(gate.float())).to(out_dtype)


@pytest.fixture
def cpu_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the two Triton-only pieces of a KDA layer so a forward pass runs on CPU."""
    monkeypatch.setattr("olmo_core.nn.convolution.dispatch_causal_conv1d", _torch_causal_conv1d)


def _make_cpu_runnable(module: nn.Module) -> None:
    """Swap the layer's fused norm for the torch stand-in, in place."""
    module.o_norm = _TorchRMSNormGated(module.o_norm.weight, module.norm_eps)  # type: ignore[assignment]


def _make_inputs(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    R: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype, device=device)

    q = randn(B, T, H, K)
    k = randn(B, T * R, H, K)
    v = randn(B, T * R, H, V)
    # A realistic decay: strictly negative, so the state contracts.
    g = -torch.nn.functional.softplus(randn(B, T, H, K))
    beta = torch.rand(B, T * R, H, generator=generator, dtype=dtype, device=device)
    return q, k, v, g, beta


def _make_module_conditioned_inputs(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    R: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    seed: int = 0,
    allow_neg_eigval: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Kernel inputs shaped the way :meth:`KimiDeltaHouseholder.forward` shapes them.

    :func:`_make_inputs` hands the recurrence raw Gaussians. That is survivable in ``float64`` at
    ``T=16``, which is all the CPU oracle tests ask of it, but it is not what the module produces
    and it does not survive a longer sequence: an unnormalized ``k`` has ``||k|| ~ sqrt(K) = 8``,
    so the delta update's ``k (k . S)`` term grows the state by ``||k||^2 ~ 64`` per step. Measured
    on this shape at ``T=64``, that reaches ``absmax = 1.3e13`` at ``R=1`` and ``1.0e38`` at
    ``R=2`` -- through the *reference* backend, on CPU, in ``float32``. Both backends then overflow
    at the same positions and a *relative* tolerance compares two piles of infinities.

    The module never feeds the kernel anything of the sort. It l2-normalizes ``q`` and ``k``, which
    is what makes the update the non-expanding reflection ``(I - beta k k^T)`` the operator is
    named for; it squashes ``beta`` through a sigmoid, doubled to ``(0, 2)`` under
    ``allow_neg_eigval``; and it builds ``g`` as ``-exp(A_log) * softplus(f_proj(x) + dt_bias)``
    with ``A_log`` initialized to ``log U(1, 16)``, a decay far stronger than
    ``-softplus(randn)``. Reproducing those four choices here keeps the same shape's output at
    ``absmax ~ 0.2``, so a parity tolerance means something and an overflow fails loudly.
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype, device=device)

    # The module l2-normalizes both, after the short convolution and before the kernel.
    q = l2_normalize(randn(B, T, H, K))
    k = l2_normalize(randn(B, T * R, H, K))
    # 'v' is the one input the module does *not* normalize: it is the value convolution's output.
    v = randn(B, T * R, H, V)

    # 'beta = sigmoid(w_b(x))', doubled into the negative-eigenvalue (reflection) regime.
    beta = randn(B, T * R, H).sigmoid()
    if allow_neg_eigval:
        beta = beta * 2.0

    # 'g = -exp(A_log) * softplus(f_proj(x) + dt_bias)', with 'A_log' at its initialized
    # distribution 'log U(1, 16)' and 'dt_bias' at its initialized zero.
    decay_rate = torch.empty(H, dtype=dtype, device=device).uniform_(1.0, 16.0, generator=generator)
    g = -decay_rate.view(1, 1, H, 1) * torch.nn.functional.softplus(randn(B, T, H, K))

    return q, k, v, g, beta


##########################################
# Parity against the oracle, on CPU      #
##########################################


@pytest.mark.parametrize("R", [1, 2, 3])
def test_kda_householder_torch_matches_the_oracle_in_float64(R: int):
    """The differentiable backend must reproduce the reference recurrence to round-off."""
    B, T, H, K, V = 2, 16, 3, 8, 8
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, R)

    out, state = kda_householder_torch(q, k, v, g, beta, num_householder=R, output_final_state=True)
    expected, expected_state = _naive_recurrence(
        q, k, v, g, beta, R, K**-0.5, None, torch.float64
    )

    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)
    assert state is not None
    torch.testing.assert_close(state, expected_state, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("R", [1, 2])
@pytest.mark.parametrize("allow_neg_eigval", [False, True])
def test_module_conditioned_inputs_keep_the_recurrence_in_range(R: int, allow_neg_eigval: bool):
    """Pin the conditioning the GPU parity test rests on, on a machine with no GPU.

    That test's tolerance is relative to the output's own scale, which only means something while
    the recurrence stays in range. This asserts the same shape and the same generator stay there
    in ``float32`` through the reference backend, so the conditioning cannot be quietly weakened
    by an edit to :func:`_make_module_conditioned_inputs` that nobody can run a GPU against.
    """
    B, T, H, K, V = 2, 64, 4, 64, 64
    q, k, v, g, beta = _make_module_conditioned_inputs(
        B, T, H, K, V, R, allow_neg_eigval=allow_neg_eigval
    )

    # The three properties the module's forward pass guarantees, checked on the inputs themselves.
    torch.testing.assert_close(k.norm(dim=-1), torch.ones_like(k[..., 0]), rtol=1e-5, atol=1e-5)
    assert 0.0 < beta.min().item()
    assert beta.max().item() < (2.0 if allow_neg_eigval else 1.0)
    assert (g < 0.0).all(), "the decay must be strictly contracting"

    out, _ = kda_householder_torch(q, k, v, g, beta, num_householder=R)

    assert torch.isfinite(out).all()
    # Raw Gaussians reach absmax 1.3e13 at R=1 and 1.0e38 at R=2 on this shape; conditioned they
    # measure 0.08 to 0.22, so a ceiling of 10 catches a regression without being brittle.
    assert out.abs().max().item() < 10.0, f"absmax {out.abs().max().item():.4g}"


@requires_fla_cpu
def test_kda_householder_torch_at_r1_matches_flas_own_kda_oracle():
    """
    At ``R=1`` the Householder recurrence *is* Kimi Delta Attention.

    This is what ties the CPU path back to the shipped operator: the comparison is against
    ``fla``'s own naive KDA kernel, not against another copy of our arithmetic. If the two
    disagree, the layer named KDA is computing something else.
    """
    from fla.ops.kda.naive import naive_recurrent_kda

    B, T, H, K, V = 2, 24, 4, 16, 16
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, 1, dtype=torch.float32)

    ours, _ = kda_householder_torch(l2_normalize(q), l2_normalize(k), v, g, beta)
    theirs, _ = naive_recurrent_kda(q=l2_normalize(q), k=l2_normalize(k), v=v, g=g, beta=beta)

    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("R", [1, 2])
def test_kda_householder_torch_gradcheck_in_float64(R: int):
    """
    Autograd derives the backward, so a ``gradcheck`` is a real check on the whole recurrence.

    ``float64`` throughout, which is what makes the perturbations meaningful.
    """
    B, T, H, K, V = 1, 5, 2, 4, 4
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, R, seed=7)
    inputs = tuple(x.clone().requires_grad_(True) for x in (q, k, v, g, beta))

    assert torch.autograd.gradcheck(
        lambda *args: kda_householder_torch(*args, num_householder=R)[0],
        inputs,
        eps=1e-6,
        atol=1e-8,
        rtol=1e-5,
    )


@requires_fla_cpu
@pytest.mark.parametrize("allow_neg_eigval", [False, True])
def test_householder_module_matches_a_hand_built_reference_on_cpu(
    allow_neg_eigval: bool, cpu_kernels: None
):
    """
    End-to-end module parity: projections, gate, the interleaved key layout, and the readout.

    Reconstructs the forward pass from the module's own submodules and the oracle recurrence, so
    a mistake in how the layer *wires* those together is caught and not just the recurrence. The
    interleave in particular -- ``[B, T, (R h d)]`` reshaped to ``[B, T * R, h, d]`` -- is the
    step that silently transposes factors against heads if it is ported wrong, and the result
    still trains.
    """
    seed_all(0)
    d_model, n_heads, head_dim, R = 64, 4, 16, 2
    B, T = 2, 12

    config = KimiDeltaHouseholderConfig(
        n_heads=n_heads,
        head_dim=head_dim,
        num_householder=R,
        allow_neg_eigval=allow_neg_eigval,
        backend="torch",
    )
    module = config.build(d_model, layer_idx=0, n_layers=1, init_device="cpu")
    module.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=1)
    _make_cpu_runnable(module)

    x = torch.randn(B, T, d_model)
    y = module(x)

    q = module.q_conv1d(x=module.w_q(x)).view(B, T, n_heads, head_dim)
    k = module.k_conv1d(x=module.w_k(x)).view(B, T, R, n_heads, head_dim)
    v = module.v_conv1d(x=module.w_v(x)).view(B, T, R, n_heads, module.head_v_dim)
    beta = module.w_b(x).sigmoid()
    if allow_neg_eigval:
        beta = beta * 2.0
    beta = beta.view(B, T, R, n_heads)

    g = -module.A_log.float().exp().unsqueeze(-1) * torch.nn.functional.softplus(
        module.f_proj(x).view(B, T, n_heads, head_dim).float()
        + module.dt_bias.float().view(n_heads, head_dim)
    )

    o_ref, _ = _naive_recurrence(
        l2_normalize(q),
        l2_normalize(k.reshape(B, T * R, n_heads, head_dim)),
        v.reshape(B, T * R, n_heads, module.head_v_dim),
        g,
        beta.reshape(B, T * R, n_heads),
        R,
        head_dim**-0.5,
        None,
        torch.float32,
    )
    gate = module.g_proj(x).view(B, T, n_heads, module.head_v_dim)
    y_ref = module.w_out(module.o_norm(o_ref.to(v.dtype), gate).view(B, T, -1))

    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)


@requires_fla_cpu
def test_householder_backward_reaches_every_parameter_on_cpu(cpu_kernels: None):
    d_model = 64
    config = KimiDeltaHouseholderConfig(n_heads=4, head_dim=16, num_householder=2, backend="torch")
    module = config.build(d_model, layer_idx=0, n_layers=1, init_device="cpu")
    module.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=1)
    _make_cpu_runnable(module)

    x = torch.randn(2, 8, d_model, requires_grad=True)
    module(x).sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"no grad for '{name}'"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for '{name}'"
        assert param.grad.abs().sum() > 0, f"dead gradient for '{name}'"


##########################################
# Parameter counts and checkpoint keys   #
##########################################


@pytest.mark.parametrize("variant", sorted(STUDY_PARAMS))
def test_variant_parameter_counts_at_the_study_geometry(variant: str):
    """
    Each variant's exact mixer parameter count at ``d_model=1024, n_heads=16, head_dim=64``.

    The width solve that matches the arms to the parameter target is done from these numbers, so
    they are asserted exactly rather than to a tolerance.
    """
    config = _study_configs()[variant]
    assert config.num_params(STUDY_D_MODEL) == STUDY_PARAMS[variant]


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(STUDY_PARAMS))
def test_timescale_parameters_are_exempt_from_weight_decay(variant: str):
    """Every arm exempts its timescale parameters, and ``A_log``/``dt_bias`` are the ones KDA owns.

    ``A_log`` sets the per-head decay rate and ``dt_bias`` the per-key-channel softplus offset, so
    decaying them moves the recurrence's horizon rather than regularizing a weight. Untagged, a
    KDA arm would shrink its own decay horizon under AdamW's 0.01 while the SSM arms it is
    compared against do not -- an optimizer difference wearing the costume of an operator one.

    The equality is exact in both directions. Nothing else here is a timescale: the gated
    convolution's ``pre_scale``/``post_scale`` are zero-initialized gains on a convolution, not
    parameters of the recurrence, so ``kda_gconv`` must tag the same two and no more.

    This pins only the mixer's half of the contract. The tag is inert until an
    :class:`~olmo_core.optim.OptimGroupOverride` names the parameter, and the pattern that does
    so lives in the wave's ledger.
    """
    module = _study_configs()[variant].build(
        STUDY_D_MODEL, layer_idx=0, n_layers=16, init_device="meta"
    )
    assert set(no_weight_decay_param_names(module)) == {"A_log", "dt_bias"}


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(STUDY_PARAMS))
def test_built_module_holds_exactly_the_declared_parameter_count(variant: str):
    """The algebra above must agree with the module it claims to describe."""
    config = _study_configs()[variant]
    module = config.build(STUDY_D_MODEL, layer_idx=0, n_layers=16, init_device="meta")
    assert sum(p.numel() for p in module.parameters()) == STUDY_PARAMS[variant]


@requires_fla_cpu
def test_gated_convolution_costs_exactly_the_gate_parameters():
    """The gated arm differs from the base arm by its gates and by nothing else."""
    configs = _study_configs()
    delta = STUDY_PARAMS["kda_gconv"] - STUDY_PARAMS["kda_base"]
    assert delta == configs["kda_gconv"].gate_params(STUDY_D_MODEL)
    # Three streams, two gates each, one scalar per channel.
    assert delta == 3 * 2 * (STUDY_N_HEADS * STUDY_HEAD_DIM)
    assert configs["kda_base"].gate_params(STUDY_D_MODEL) == 0


BASE_KEYS = {
    "w_q.weight",
    "w_k.weight",
    "w_v.weight",
    "w_b.weight",
    "f_proj.0.weight",
    "f_proj.1.weight",
    "A_log",
    "dt_bias",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "g_proj.0.weight",
    "g_proj.1.weight",
    "g_proj.1.bias",
    "o_norm.weight",
    "w_out.weight",
}

GCONV_KEYS = (BASE_KEYS - {f"{s}_conv1d.weight" for s in "qkv"}) | {
    f"{s}_conv1d.{leaf}" for s in "qkv" for leaf in ("conv.weight", "pre_scale", "post_scale")
}


@requires_fla_cpu
@pytest.mark.parametrize(
    "variant,expected",
    [
        ("kda_base", BASE_KEYS),
        ("kda_householder_r2_negeig", BASE_KEYS),
        ("kda_gconv", GCONV_KEYS),
    ],
)
def test_checkpoint_keys_are_stable(variant: str, expected: set):
    """
    A checkpoint written by one of these must be loadable by the same config later.

    Spelled out rather than snapshotted so that a rename shows up as a diff on a reviewable list.
    """
    config = _study_configs()[variant]
    module = config.build(STUDY_D_MODEL, layer_idx=0, n_layers=16, init_device="meta")
    assert set(module.state_dict()) == expected


@requires_fla_cpu
def test_householder_r2_widens_only_the_key_side():
    """
    ``R`` widens ``w_k``, ``w_v``, ``w_b`` and the k/v convolutions, and nothing else.

    The decay is applied once per token, so the query, both gates, the norm and the output
    projection are the same size as the base arm's.
    """
    base = _study_configs()["kda_base"].build(
        STUDY_D_MODEL, layer_idx=0, n_layers=16, init_device="meta"
    )
    r2 = _study_configs()["kda_householder_r2_negeig"].build(
        STUDY_D_MODEL, layer_idx=0, n_layers=16, init_device="meta"
    )

    base_state, r2_state = base.state_dict(), r2.state_dict()
    assert set(base_state) == set(r2_state)

    widened = {"w_k.weight", "w_v.weight", "w_b.weight", "k_conv1d.weight", "v_conv1d.weight"}
    for key, tensor in base_state.items():
        if key in widened:
            assert r2_state[key].shape[0] == 2 * tensor.shape[0], key
        else:
            assert r2_state[key].shape == tensor.shape, key


##########################################
# Registration and configuration         #
##########################################


@pytest.mark.parametrize("variant", sorted(STUDY_PARAMS))
def test_config_round_trips_through_the_sequence_mixer_registry(variant: str):
    """Each variant must survive being written to a config file and read back."""
    config = _study_configs()[variant]
    assert SequenceMixerConfig.from_dict(config.as_config_dict()) == config


def test_the_three_variants_are_distinct_configurations():
    """
    A variant that silently equals another would report a difference that is really noise.

    ``allow_neg_eigval`` in particular allocates nothing, so nothing about the parameter count or
    the checkpoint would give away an arm that had lost it.
    """
    configs = list(_study_configs().values())
    for i, left in enumerate(configs):
        for right in configs[i + 1 :]:
            assert left != right

    assert _study_configs()["kda_householder_r2_negeig"].allow_neg_eigval is True
    assert _study_configs()["kda_householder_r2_negeig"].num_householder == 2
    assert _study_configs()["kda_gconv"].gated_conv is True
    assert _study_configs()["kda_gconv"].gate_structure == "depthwise"
    assert _study_configs()["kda_base"].gated_conv is False
    assert _study_configs()["kda_base"].allow_neg_eigval is False


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"gate_rank": 8}, "'gate_rank' is set but 'gated_conv' is False"),
        (
            {"gated_conv_activation": "silu"},
            "'gated_conv_activation' is set but 'gated_conv' is False",
        ),
        ({"gate_structure": "lowrank"}, "'gate_structure' is 'lowrank' but 'gated_conv' is False"),
        ({"gated_conv": True, "gate_structure": "lowrank"}, "'gate_rank' is required"),
        ({"conv_activation": "gelu"}, "unsupported conv_activation"),
    ],
)
def test_incoherent_gate_options_are_refused_without_building(kwargs: dict, match: str):
    """
    A config that could never build must not be able to produce a plausible parameter count.

    ``num_params`` is what solves the FFN widths, so a number returned from an incoherent config
    would move the anchor for every arm. Checked on a bare config, so it is verifiable on CPU.
    """
    config = KimiDeltaAttentionConfig(n_heads=4, head_dim=16, **kwargs)
    with pytest.raises(ValueError, match=match):
        config.validate_gate_options()
    with pytest.raises(ValueError, match=match):
        config.gate_params(64)


@requires_fla_cpu
def test_negative_eigenvalues_double_beta_and_nothing_else(cpu_kernels: None):
    """
    ``allow_neg_eigval`` is the reflection regime, and it is a factor of two on ``beta``.

    It allocates nothing, so the two arms are parameter-identical; the check is that it changes
    the function rather than only the name.
    """
    d_model = 64
    kwargs = dict(n_heads=4, head_dim=16, num_householder=2, backend="torch")
    plain = KimiDeltaHouseholderConfig(**kwargs, allow_neg_eigval=False)
    reflect = KimiDeltaHouseholderConfig(**kwargs, allow_neg_eigval=True)

    assert plain.num_params(d_model) == reflect.num_params(d_model)

    seed_all(3)
    a = plain.build(d_model, layer_idx=0, n_layers=1, init_device="cpu")
    a.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=1)
    b = reflect.build(d_model, layer_idx=0, n_layers=1, init_device="cpu")
    b.load_state_dict(a.state_dict())
    _make_cpu_runnable(a)
    _make_cpu_runnable(b)

    x = torch.randn(1, 8, d_model)
    assert not torch.allclose(a(x), b(x))


##########################################
# The gated arm stays inside its layers  #
##########################################


def _tiny_model_config(
    mixer: SequenceMixerConfig, kda_layers: Tuple[int, ...]
) -> TransformerConfig:
    d_model = 128
    return TransformerConfig(
        d_model=d_model,
        vocab_size=256,
        n_layers=4,
        block=TransformerBlockConfig(
            sequence_mixer=AttentionConfig(n_heads=4),
            feed_forward=FeedForwardConfig(hidden_size=d_model * 2, bias=False),
            layer_norm=LayerNormConfig(name=LayerNormType.rms, bias=False),
        ),
        block_overrides={
            i: TransformerBlockConfig(
                sequence_mixer=mixer,
                feed_forward=FeedForwardConfig(hidden_size=d_model * 2, bias=False),
                layer_norm=LayerNormConfig(name=LayerNormType.rms, bias=False),
            )
            for i in kda_layers
        },
        lm_head=LMHeadConfig(bias=False),
    )


@requires_fla_cpu
def test_the_gated_convolution_arm_is_confined_to_the_layers_it_occupies():
    """
    Turning the gate on must change nothing outside the KDA slots.

    This is the condition for the arm to be droppable into an otherwise frozen model: if it
    needed a different backbone, every *other* arm in the study would have to move with it, and
    the comparison would no longer be about the gate.
    """
    kda_layers = (1, 3)
    plain = _tiny_model_config(KimiDeltaAttentionConfig(n_heads=2, head_dim=64), kda_layers).build(
        init_device="meta"
    )
    gated = _tiny_model_config(
        KimiDeltaAttentionConfig(
            n_heads=2, head_dim=64, gated_conv=True, gate_structure="depthwise"
        ),
        kda_layers,
    ).build(init_device="meta")

    plain_state, gated_state = plain.state_dict(), gated.state_dict()
    changed = set(plain_state) ^ set(gated_state)
    assert changed, "the two models are identical, so the test proves nothing"

    occupied = tuple(f"blocks.{i}.attention." for i in kda_layers)
    for key in changed:
        assert key.startswith(occupied), f"the gate reached outside its slots: '{key}'"

    for key in set(plain_state) & set(gated_state):
        assert plain_state[key].shape == gated_state[key].shape, key


#: The three variants again, narrow enough to build a whole model out of.
SMALL_VARIANTS = {
    "kda_base": KimiDeltaAttentionConfig(n_heads=2, head_dim=64),
    "kda_householder_r2_negeig": KimiDeltaHouseholderConfig(
        n_heads=2, head_dim=64, num_householder=2, allow_neg_eigval=True
    ),
    "kda_gconv": KimiDeltaAttentionConfig(
        n_heads=2, head_dim=64, gated_conv=True, gate_structure="depthwise"
    ),
}


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(SMALL_VARIANTS))
def test_transformer_init_weights_dispatches_to_the_mixer(variant: str):
    """
    The real path: ``Transformer.init_weights`` sweeps ``reset_parameters``, then dispatches.

    Calling a mixer's ``init_weights`` directly skips the sweep, so this is the only test that
    runs the two in the order training does, through the model rather than around it.
    """
    model = _tiny_model_config(SMALL_VARIANTS[variant], (1,)).build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    unwritten = [name for name, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert unwritten == [], unwritten


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(STUDY_PARAMS))
def test_each_variant_drops_into_a_block_without_touching_the_backbone(variant: str):
    """
    Every variant must be usable as a block's sequence mixer, which is how an arm builder places
    it, and must leave the attention layers around it byte-identical.
    """
    kda_layers = (1,)
    reference = _tiny_model_config(AttentionConfig(n_heads=2), ()).build(init_device="meta")
    arm = _tiny_model_config(_study_configs()[variant], kda_layers).build(init_device="meta")

    reference_state, arm_state = reference.state_dict(), arm.state_dict()
    for key, tensor in reference_state.items():
        if key.startswith("blocks.1."):
            continue
        assert key in arm_state, f"the arm dropped a backbone key: '{key}'"
        assert arm_state[key].shape == tensor.shape, key


##########################################
# The fused kernel, on a GPU             #
##########################################


@requires_fla
@requires_gpu
def test_kimi_delta_attention_matches_the_naive_oracle():
    """
    End-to-end check of the shipped operator against ``fla``'s own naive KDA kernel.

    This is the one comparison that exercises ``fla.ops.kda.chunk_kda``, the Triton kernel the
    base and gated arms actually train with.
    """
    from fla.modules.l2norm import l2norm
    from fla.ops.kda.gate import fused_kda_gate
    from fla.ops.kda.naive import naive_recurrent_kda

    seed_all(0)
    device, dtype = "cuda", torch.bfloat16
    d_model, n_heads, head_dim = 256, 4, 64
    B, T = 2, 128

    config = KimiDeltaAttentionConfig(n_heads=n_heads, head_dim=head_dim)
    module = config.build(d_model, layer_idx=0, n_layers=1, init_device=device)
    module.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=1)

    x = torch.randn(B, T, d_model, device=device, dtype=dtype)

    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)

        q = module.q_conv1d(x=module.w_q(x)).view(B, T, n_heads, head_dim)
        k = module.k_conv1d(x=module.w_k(x)).view(B, T, n_heads, head_dim)
        v = module.v_conv1d(x=module.w_v(x)).view(B, T, n_heads, module.head_v_dim)
        beta = module.w_b(x).sigmoid()
        raw = module.f_proj(x).view(B, T, n_heads, head_dim)
        g = fused_kda_gate(raw, module.A_log, module.dt_bias)
        o_ref, _ = naive_recurrent_kda(q=l2norm(q), k=l2norm(k), v=v, g=g, beta=beta)
        gate = module.g_proj(x).view(B, T, n_heads, module.head_v_dim)
        y_ref = module.w_out(module.o_norm(o_ref.to(v.dtype), gate).view(B, T, -1))

    torch.testing.assert_close(y.float(), y_ref.float(), atol=2e-2, rtol=2e-2)


@requires_fla
@requires_gpu
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(KimiDeltaAttentionConfig(n_heads=4, head_dim=64), id="base"),
        pytest.param(
            KimiDeltaAttentionConfig(
                n_heads=4, head_dim=64, gated_conv=True, gate_structure="depthwise"
            ),
            id="gconv",
        ),
    ],
)
def test_kimi_delta_attention_forward_backward_on_gpu(config: KimiDeltaAttentionConfig):
    device, dtype = "cuda", torch.bfloat16
    d_model, seq_len, batch_size = 256, 128, 2

    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)
    module.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=12)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)
        assert y.shape == x.shape
        loss = y.float().sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"no grad for '{name}'"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for '{name}'"


@requires_fla
@requires_gpu
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            KimiDeltaHouseholderConfig(n_heads=4, head_dim=64, num_householder=1), id="R1"
        ),
        pytest.param(
            KimiDeltaHouseholderConfig(
                n_heads=4, head_dim=64, num_householder=2, allow_neg_eigval=True
            ),
            id="R2-neg-eigval",
        ),
    ],
)
def test_kimi_delta_householder_forward_backward_on_gpu(config: KimiDeltaHouseholderConfig):
    """The Householder variant through the real module, on the path training takes.

    The kernel-level parity test above drives ``chunk_kda_householder`` on synthetic inputs. This
    one lets the module build its own ``q``/``k``/``v``/``g``/``beta`` from real activations and
    initialized parameters, which is the only check that would catch a recurrence that overflows
    where it matters. ``R=2`` with negative eigenvalues is the arm the study wants.
    """
    device, dtype = "cuda", torch.bfloat16
    d_model, seq_len, batch_size = 256, 128, 2

    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)
    module.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=12)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)
        assert y.shape == x.shape
        assert torch.isfinite(y.float()).all(), "the forward pass left float range"
        loss = y.float().sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"no grad for '{name}'"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for '{name}'"


@requires_fla
@requires_gpu
@pytest.mark.parametrize(
    "R, allow_neg_eigval",
    [
        pytest.param(1, False, id="R1"),
        pytest.param(2, False, id="R2"),
        pytest.param(2, True, id="R2-neg-eigval"),
    ],
)
def test_kda_householder_triton_matches_the_torch_backend(R: int, allow_neg_eigval: bool):
    """The training kernel and the reference backend must agree, on inputs the module can produce.

    Two things here are deliberate, and an earlier version of this test had neither.

    The inputs are conditioned the way the module conditions them, because raw Gaussians overflow
    the recurrence outright at this sequence length -- see
    :func:`_make_module_conditioned_inputs`. Both backends then went non-finite at the same
    positions, which is not a kernel disagreement and cannot be compared.

    And the tolerance is relative to the output's own scale rather than element-wise, with
    finiteness asserted outright. With unconditioned inputs the ``R=1`` case passed on
    ``rtol=2e-2``, which against values of ``1e14`` permits a difference of ``1e12``: a tolerance
    that large is not evidence of anything. ``R=2`` with negative eigenvalues is the arm the study
    actually wants, so it is the case that most needs to be measured rather than waved through.
    """
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    seed_all(0)
    B, T, H, K, V = 2, 64, 4, 64, 64
    q, k, v, g, beta = _make_module_conditioned_inputs(
        B, T, H, K, V, R, dtype=torch.bfloat16, device="cuda", allow_neg_eigval=allow_neg_eigval
    )

    fused, _ = chunk_kda_householder(
        q=q, k=k, v=v, g=g.float(), beta=beta, num_householder=R, backend="triton"
    )
    reference, _ = chunk_kda_householder(
        q=q, k=k, v=v, g=g.float(), beta=beta, num_householder=R, backend="torch"
    )
    fused, reference = fused.float(), reference.float()

    for name, out in (("triton", fused), ("torch", reference)):
        non_finite = int((~torch.isfinite(out)).sum())
        assert non_finite == 0, f"{name} backend: {non_finite}/{out.numel()} non-finite outputs"
    assert reference.abs().max().item() < 10.0, (
        f"the recurrence left its range (absmax {reference.abs().max().item():.4g}), so the "
        "scale-relative tolerance below would not mean anything"
    )

    peak = reference.abs().max().item()
    rms = reference.pow(2).mean().sqrt().item()
    max_diff = (fused - reference).abs().max().item()
    rms_diff = (fused - reference).pow(2).mean().sqrt().item()

    assert max_diff <= 2e-2 * peak, f"max|diff| {max_diff:.4g} exceeds 2% of peak {peak:.4g}"
    assert rms_diff <= 2e-2 * rms, f"rms|diff| {rms_diff:.4g} exceeds 2% of rms {rms:.4g}"


@requires_fla
@requires_gpu
def test_module_type_hierarchy_is_what_the_arm_builder_expects():
    """The configs must build the classes they are annotated with."""
    base = KimiDeltaAttentionConfig(n_heads=4, head_dim=64).build(
        256, layer_idx=0, n_layers=1, init_device="cuda"
    )
    householder = KimiDeltaHouseholderConfig(n_heads=4, head_dim=64).build(
        256, layer_idx=0, n_layers=1, init_device="cuda"
    )
    assert isinstance(base, KimiDeltaAttention)
    assert isinstance(householder, KimiDeltaHouseholder)
    assert isinstance(base, nn.Module) and isinstance(householder, nn.Module)


@requires_fla_cpu
def test_num_flops_per_token_is_positive_and_ordered():
    """A wider operator must not report fewer FLOPs than a narrower one."""
    configs = _study_configs()
    modules = {
        name: config.build(STUDY_D_MODEL, layer_idx=0, n_layers=16, init_device="meta")
        for name, config in configs.items()
    }
    flops = {name: module.num_flops_per_token(4096) for name, module in modules.items()}
    assert all(value > 0 for value in flops.values()), flops
    assert flops["kda_householder_r2_negeig"] > flops["kda_base"]
    assert flops["kda_gconv"] > flops["kda_base"]


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(STUDY_PARAMS))
def test_tensor_parallelism_is_refused_rather_than_silently_wrong(variant: str):
    """
    Neither layer implements TP, so it must raise rather than train something wrong.

    The mesh is never read -- ``apply_tp`` discards its arguments before raising -- so passing
    ``None`` keeps this runnable without a process group.
    """
    module = _study_configs()[variant].build(
        STUDY_D_MODEL, layer_idx=0, n_layers=16, init_device="meta"
    )
    with pytest.raises(NotImplementedError):
        module.apply_tp(None)  # type: ignore[arg-type]


def test_study_parameter_counts_are_reported_for_the_width_solve():
    """
    A single place that names every variant's cost, since the FFN respend is solved from it.

    Keeping it as an assertion rather than a print means the numbers cannot go stale silently.
    """
    assert STUDY_PARAMS == {
        "kda_base": 4_487_248,
        "kda_householder_r2_negeig": 6_608_976,
        "kda_gconv": 4_493_392,
    }
    assert STUDY_PARAMS["kda_gconv"] - STUDY_PARAMS["kda_base"] == 6_144
    assert STUDY_PARAMS["kda_householder_r2_negeig"] - STUDY_PARAMS["kda_base"] == 2_121_728
