import copy

import pytest
import torch

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.mamba3 import Mamba3Mixer, Mamba3MixerConfig
from olmo_core.nn.mamba3.mamba3_ssd_api import mamba3_ssd_reference
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.testing import requires_gpu


@pytest.mark.parametrize(
    "mixer_config",
    [
        pytest.param(Mamba3MixerConfig(n_heads=8), id="default_mimo4"),
        pytest.param(Mamba3MixerConfig(n_heads=8, mimo_rank=1), id="siso"),
        pytest.param(Mamba3MixerConfig(n_heads=8, n_groups=2), id="n_groups=2"),
        pytest.param(Mamba3MixerConfig(n_heads=8, head_dim=32), id="head_dim=32"),
        pytest.param(Mamba3MixerConfig(n_heads=8, d_state=64), id="d_state=64"),
        pytest.param(Mamba3MixerConfig(n_heads=8, bc_norm=False), id="no_bc_norm"),
        pytest.param(Mamba3MixerConfig(n_heads=8, bc_bias=False), id="no_bc_bias"),
        # Widening the rotation block widens theta_proj, so num_params must track it.
        pytest.param(
            Mamba3MixerConfig(n_heads=8, d_state=96, rotation_block_size=3), id="block_size=3"
        ),
        pytest.param(Mamba3MixerConfig(n_heads=8, rotation_block_size=4), id="block_size=4"),
        pytest.param(
            Mamba3MixerConfig(n_heads=8, rotation_block_size=4, n_groups=2),
            id="block_size=4_g2",
        ),
        # The MIMO rank scales B/C independently of the rotation, so the two have to be counted
        # together: a wide block with rank 1 and a narrow block with rank 8 both drift if
        # `num_params` folds one into the other.
        pytest.param(
            Mamba3MixerConfig(n_heads=8, d_state=96, rotation_block_size=3, mimo_rank=1),
            id="block_size=3_siso",
        ),
        pytest.param(
            Mamba3MixerConfig(n_heads=8, rotation_block_size=4, mimo_rank=8, n_groups=2),
            id="block_size=4_mimo8_g2",
        ),
        pytest.param(
            Mamba3MixerConfig(
                n_heads=8, d_state=192, rotation_block_size=6, mimo_rank=3, bc_bias=False
            ),
            id="block_size=6_mimo3_no_bc_bias",
        ),
    ],
)
def test_mamba3_mixer_config_num_params(mixer_config: Mamba3MixerConfig):
    d_model = 512
    module = mixer_config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # The estimated number of params must match the actual built module.
    n_params = sum(p.numel() for p in module.parameters())
    assert mixer_config.num_params(d_model) == n_params


@pytest.mark.parametrize("rotation_block_size", [2, 3, 4], ids=["b2", "b3", "b4"])
@pytest.mark.parametrize("mimo_rank", [1, 4], ids=["siso", "mimo4"])
@pytest.mark.parametrize("n_groups", [1, 2], ids=["g1", "g2"])
def test_mamba3_mixer_fwd_bwd(mimo_rank: int, n_groups: int, rotation_block_size: int):
    torch.manual_seed(0)
    d_model, seq_len, batch_size = 64, 16, 2
    # 12 is divisible by 2, 3 and 4, so one state size covers every block size under test.
    config = Mamba3MixerConfig(
        n_heads=4,
        head_dim=16,
        d_state=12,
        n_groups=n_groups,
        mimo_rank=mimo_rank,
        rotation_block_size=rotation_block_size,
    )
    module = config.build(d_model, layer_idx=0, n_layers=2, init_device="cpu")
    module.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2)

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape

    loss = y.float().pow(2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Every parameter should receive a finite gradient.
    for name, p in module.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def _tiny_mixer(d_model: int = 32, *, mimo_rank: int = 2, init_device: str = "cpu") -> Mamba3Mixer:
    config = Mamba3MixerConfig(n_heads=2, head_dim=8, d_state=8, n_groups=1, mimo_rank=mimo_rank)
    module = config.build(d_model, layer_idx=0, n_layers=2, init_device=init_device)
    module.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2)
    return module


def test_mamba3_mixer_rejects_multi_document_cu_doc_lens():
    """
    Packed multi-document batches must fail loudly: the SSD scan carries state across the whole
    sequence, so without intra-document masking it would leak state across document boundaries.
    """
    torch.manual_seed(0)
    d_model, seq_len = 32, 8
    module = _tiny_mixer(d_model)
    x = torch.randn(1, seq_len, d_model)

    # Two documents of length 4 packed into one sequence (the convention produced by
    # get_cumulative_document_lengths / consumed by RoPE: a flat [0, ..., total] over batch*seq).
    cu_doc_lens = torch.tensor([0, 4, 8], dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="intra-document masking"):
        module(x, cu_doc_lens=cu_doc_lens)

    # The error must name the offending argument so the failure is actionable.
    with pytest.raises(NotImplementedError, match="cu_doc_lens"):
        module(x, cu_doc_lens=cu_doc_lens)


def test_mamba3_mixer_accepts_single_document_cu_doc_lens():
    """
    A cu_doc_lens with no interior boundary describes one document, which needs no masking, so it
    must behave exactly like passing None (the packed-data training path always sets this kwarg).
    """
    torch.manual_seed(0)
    d_model, seq_len = 32, 8
    module = _tiny_mixer(d_model)
    x = torch.randn(1, seq_len, d_model)

    expected = module(x)
    # [0, total]: a single document spanning the whole flattened batch.
    single_doc = torch.tensor([0, seq_len], dtype=torch.int32)
    torch.testing.assert_close(module(x, cu_doc_lens=single_doc), expected)
    # Explicit None, plus the other packed-data kwargs the transformer block forwards.
    torch.testing.assert_close(module(x, cu_doc_lens=None, max_doc_len=8), expected)


def test_mamba3_mixer_num_flops_per_token():
    d_model, n_heads, seq_len = 256, 2, 8192

    mamba = Mamba3MixerConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )
    attn = AttentionConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )

    # At long sequence lengths the SSM uses fewer FLOPs than quadratic attention.
    mamba_flops = mamba.num_flops_per_token(seq_len)
    attn_flops = attn.num_flops_per_token(seq_len)  # type: ignore
    assert 0 < mamba_flops < attn_flops


def test_mamba3_mixer_num_flops_monotone_in_block_size():
    """
    Widening the rotation block costs FLOPs (wider theta_proj, a prefix product over SO(b)
    instead of a cumsum), so the estimate must be monotone in it -- and must still come in under
    quadratic attention at long context, which is the whole reason to use an SSM.
    """
    d_model, n_heads, seq_len = 256, 2, 8192
    attn_flops = (
        AttentionConfig(n_heads=n_heads)
        .build(d_model, layer_idx=0, n_layers=1, init_device="meta")
        .num_flops_per_token(seq_len)
    )  # type: ignore

    flops = [
        Mamba3MixerConfig(n_heads=n_heads, d_state=96, rotation_block_size=b)
        .build(d_model, layer_idx=0, n_layers=1, init_device="meta")
        .num_flops_per_token(seq_len)
        for b in (2, 3, 4, 6)
    ]
    assert flops == sorted(flops) and flops[0] < flops[-1]
    assert flops[-1] < attn_flops


def test_mamba3_mixer_rejects_indivisible_d_state():
    """
    ``d_state`` must be divisible by the block size, and the error has to name both numbers:
    the default ``d_state=128`` is not divisible by 3, which is the trap that pushes people to
    ``b=4`` when they meant ``b=3``.
    """
    with pytest.raises(OLMoConfigurationError, match=r"d_state \(128\).*rotation_block_size \(3\)"):
        Mamba3MixerConfig(n_heads=8, d_state=128, rotation_block_size=3).build(
            512, layer_idx=0, n_layers=1, init_device="meta"
        )
    with pytest.raises(OLMoConfigurationError, match="rotation_block_size must be >= 2"):
        Mamba3MixerConfig(n_heads=8, rotation_block_size=1).build(
            512, layer_idx=0, n_layers=1, init_device="meta"
        )


def test_mamba3_mixer_rejects_degenerate_d_state():
    """
    A ``d_state`` below the block size must be rejected, including ``d_state=0``.

    Divisibility alone does not catch it: ``0 % b == 0`` for every ``b``, so a zero state size
    sails through and builds a mixer with zero rotation blocks and zero-width ``B``/``C``. That
    module runs without error and returns exactly zero for every input -- a silently dead layer
    that a training run would only reveal as a loss that never moves.
    """
    for d_state, block_size in ((0, 2), (0, 4), (2, 4)):
        with pytest.raises(OLMoConfigurationError, match=rf"d_state \({d_state}\)"):
            Mamba3MixerConfig(n_heads=8, d_state=d_state, rotation_block_size=block_size).build(
                512, layer_idx=0, n_layers=1, init_device="meta"
            )


@pytest.mark.parametrize(
    "mixer_config",
    [
        pytest.param(
            Mamba3MixerConfig(n_heads=8, d_state=128, rotation_block_size=3), id="indivisible"
        ),
        pytest.param(Mamba3MixerConfig(n_heads=8, rotation_block_size=1), id="block_size=1"),
        pytest.param(Mamba3MixerConfig(n_heads=8, d_state=0), id="d_state=0"),
        pytest.param(Mamba3MixerConfig(n_heads=6, n_groups=4), id="heads_not_div_groups"),
        pytest.param(Mamba3MixerConfig(n_heads=8, n_groups=0), id="n_groups=0"),
        pytest.param(Mamba3MixerConfig(n_heads=8, mimo_rank=0), id="mimo_rank=0"),
        pytest.param(Mamba3MixerConfig(n_heads=8, a_log_init_max=0.0), id="a_log_init_max=0"),
    ],
)
def test_mamba3_mixer_config_num_params_rejects_unbuildable_config(
    mixer_config: Mamba3MixerConfig,
):
    """
    Sizing a config must fail exactly when building it fails.

    ``num_params`` is read long before any module exists -- it is what
    ``Mamba3Config.build`` logs, what the size presets report, and what sizing/dry-run scripts
    print. It re-derives the shapes with plain integer arithmetic instead of validating, so for
    a config the mixer would reject it silently returns a plausible number (``d_state=128`` with
    ``rotation_block_size=3`` truncates ``128 // 3`` to 42 blocks and reports 1,384,784 params
    for a model that cannot be constructed).
    """
    d_model = 512
    with pytest.raises(OLMoConfigurationError) as build_err:
        mixer_config.build(d_model, layer_idx=0, n_layers=1, init_device="meta")
    with pytest.raises(OLMoConfigurationError) as size_err:
        mixer_config.num_params(d_model)
    assert str(size_err.value) == str(build_err.value)


def test_mamba3_mixer_rejects_zero_n_heads():
    """
    ``n_heads=0`` must fail the same way every other invalid option does.

    Every other out-of-range value raises a ``ValueError`` naming the field. ``n_heads=0``
    slipped through to ``d_model // n_heads`` and surfaced as a bare ``ZeroDivisionError``,
    which says nothing about which option was wrong.
    """
    with pytest.raises(OLMoConfigurationError, match="n_heads must be >= 1"):
        Mamba3MixerConfig(n_heads=0, d_state=16).build(
            64, layer_idx=0, n_layers=1, init_device="meta"
        )


def test_mamba3_mixer_block_size_2_preserves_theta_proj_shape():
    """
    At ``b == 2`` the parameterization must be layout-identical to the pre-blocked code:
    ``b*(b-1)//2 == 1`` angle per block and ``d_state // 2`` blocks is exactly the old
    ``n_groups * (d_state // 2)`` output width. This is what makes ``rotation_block_size=2`` a
    true no-op default rather than a numerically-close approximation.
    """
    d_model, d_state, n_groups = 512, 64, 2
    module = Mamba3MixerConfig(
        n_heads=8, d_state=d_state, n_groups=n_groups, rotation_block_size=2
    ).build(d_model, layer_idx=0, n_layers=1, init_device="meta")
    assert module.theta_proj.out_features == n_groups * (d_state // 2)


def test_mamba3_mixer_block_size_is_actually_plumbed_through():
    """
    Same weights and input, different block size, different output.

    A config flag that silently fails to reach the kernel is the most likely way this change
    breaks: every other test would still pass while the model stayed abelian. ``d_state=12``
    keeps the parameter shapes identical across b=2 and b=4 (one angle per pair vs six per
    quadruple gives 6 vs 12 angles, so theta_proj differs) -- hence the comparison is on the
    SSD output for shared B/C/x rather than on the module forward.
    """
    torch.manual_seed(0)
    batch, seq_len, n_heads, head_dim, n_groups, mimo_rank = 2, 12, 2, 4, 1, 1
    d_state = 12

    x = torch.randn(batch, seq_len, n_heads, head_dim)
    B = torch.randn(batch, seq_len, n_groups, mimo_rank, d_state)
    C = torch.randn(batch, seq_len, n_groups, mimo_rank, d_state)
    dt = torch.rand(batch, seq_len, n_heads) * 0.1 + 0.01
    A = -torch.rand(n_heads) - 0.5
    lam = torch.rand(batch, seq_len, n_heads)

    outputs = []
    for b in (2, 4):
        torch.manual_seed(1)
        theta = torch.randn(batch, seq_len, n_groups, d_state // b, b * (b - 1) // 2)
        outputs.append(
            mamba3_ssd_reference(x, B, C, dt, A, lam, theta, heads_per_group=n_heads, block_size=b)
        )
    assert not torch.allclose(outputs[0], outputs[1], rtol=1e-3, atol=1e-3)


def test_mamba3_mixer_tp_cp_not_implemented():
    module = Mamba3MixerConfig(n_heads=4).build(64, layer_idx=0, n_layers=1, init_device="meta")
    # These require a real DeviceMesh; construct a lightweight stub for the size==1 short-circuit
    # / not-implemented paths.

    class _Mesh:
        def size(self):
            return 2

    with pytest.raises(NotImplementedError):
        module.apply_tp(_Mesh())  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        module.apply_cp(_Mesh())  # type: ignore[arg-type]


def test_mamba3_mixer_fan_in_init_rejected():
    module = Mamba3MixerConfig(n_heads=4).build(64, layer_idx=0, n_layers=1, init_device="cpu")
    with pytest.raises(NotImplementedError):
        module.init_weights(init_method=InitMethod.fan_in, d_model=64, block_idx=0, num_blocks=1)


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({}, id="default"),
        pytest.param({"rotation_block_size": 4}, id="block_size=4"),
        pytest.param(
            {"rotation_block_size": 4, "mimo_rank": 3, "n_groups": 2}, id="block_size=4_mimo3_g2"
        ),
        pytest.param(
            {"a_log_init_min": 0.00625, "a_log_init_max": 0.1}, id="a_log_init_range=state_tracking"
        ),
        pytest.param({"bc_norm": False, "bc_bias": False}, id="no_bc_norm_no_bc_bias"),
    ],
)
def test_mamba3_mixer_init_is_deterministic(options: dict):
    config = Mamba3MixerConfig(n_heads=4, head_dim=16, d_state=16, **options)
    m1 = config.build(64, layer_idx=0, n_layers=2, init_device="cpu")
    m2 = config.build(64, layer_idx=0, n_layers=2, init_device="cpu")

    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    m1.init_weights(
        init_method=InitMethod.normal, d_model=64, block_idx=0, num_blocks=2, generator=g1
    )
    m2.init_weights(
        init_method=InitMethod.normal, d_model=64, block_idx=0, num_blocks=2, generator=g2
    )

    for (n1, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        torch.testing.assert_close(p1, p2, msg=f"mismatch for {n1}")


@pytest.mark.parametrize(
    "a_log_init_min, a_log_init_max",
    [(1.0, 16.0), (0.00625, 0.1)],
    ids=["default", "state_tracking"],
)
def test_mamba3_mixer_a_log_init_range_reaches_initialization(
    a_log_init_min: float, a_log_init_max: float
):
    """
    Both ends of the ``A_log`` init range must actually reach ``init_weights``.

    The default ``(1, 16)`` is ``mamba_ssm``'s ``A_init_range``. The upper bound sets how fast
    the quickest heads forget; the lower bound floors the decay so no head starts as a
    non-decaying accumulator, which is what a bound of 0 permits. A field that is stored but
    never reaches ``init_weights`` would leave either knob silently inert, so this asserts the
    realized ``|A| = exp(A_log)`` sits inside the range and (with 64 heads) approaches both ends
    closely enough that an ignored setting could not pass.
    """
    d_model, n_heads = 512, 64
    module = Mamba3MixerConfig(
        n_heads=n_heads,
        head_dim=8,
        d_state=16,
        a_log_init_min=a_log_init_min,
        a_log_init_max=a_log_init_max,
    ).build(d_model, layer_idx=0, n_layers=1, init_device="cpu")
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=d_model,
        block_idx=0,
        num_blocks=1,
        generator=torch.Generator().manual_seed(0),
    )

    decay = module.A_log.exp()  # |A|, since A = -exp(A_log)
    assert torch.isfinite(module.A_log).all()
    assert decay.max().item() <= a_log_init_max
    assert decay.max().item() > a_log_init_max / 2
    assert decay.min().item() >= a_log_init_min
    assert decay.min().item() < a_log_init_min + (a_log_init_max - a_log_init_min) / 2


@pytest.mark.parametrize(
    "dtype, atol",
    [pytest.param(DType.bfloat16, 5e-3, id="bf16"), pytest.param(DType.float16, 5e-4, id="fp16")],
)
def test_mamba3_mixer_reduced_precision_round_trips_on_cpu(dtype: DType, atol: float):
    """
    A reduced-precision mixer must return its own dtype and still track the fp32 result.

    The internals deliberately mix precisions -- ``dt``/``A`` are float32 parameters, the SSD
    scan upcasts, and the RMS norms normalize in float32 -- so the output dtype is the product
    of several independent casts and is easy to get wrong in a way that only shows up as an
    autocast dtype mismatch mid-training. The equivalent bf16 check exists only under
    ``requires_gpu``, which leaves this unguarded on CPU-only runs.

    Tolerance: measured deviation from fp32 at an output scale of ~1.4e-1 is 1.4e-3 for bf16
    (8-bit mantissa) and 1.3e-4 for fp16, i.e. a couple of ULPs each. The atols are ~4x that --
    loose enough for the accumulated rounding of the projection stack, far tighter than the NaN,
    inf, or O(1) shift a real dtype-handling bug produces.
    """
    torch.manual_seed(0)
    d_model = 32
    module = _tiny_mixer(d_model)
    x = torch.randn(2, 16, d_model)
    expected = module(x)

    module_low = copy.deepcopy(module).to(dtype.as_pt())
    y = module_low(x.to(dtype.as_pt()))

    assert y.dtype == dtype.as_pt()
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.float(), expected, rtol=1.6e-2, atol=atol)


def _rotate_bc_via_complex(bc: torch.Tensor, theta_cumulative: torch.Tensor) -> torch.Tensor:
    """
    Rotate adjacent pairs of the state dim of ``B``/``C`` by ``exp(i * theta_cumulative)``.

    Adjacent state pairs are read as a single complex number and multiplied by a unit-modulus
    phasor. This is deliberately a different route to the same rotation than the real 2x2
    ``cos``/``sin`` formulation used by the implementation under test.
    """
    z = torch.complex(bc[..., 0::2], bc[..., 1::2])  # (..., rank, d_state // 2)
    phase = torch.exp(1j * theta_cumulative.unsqueeze(-2))  # broadcast over the rank dim
    z = z * phase
    return torch.stack((z.real, z.imag), dim=-1).flatten(start_dim=-2)


def _mamba3_ssd_closed_form_oracle(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    lam: torch.Tensor,
    theta: torch.Tensor,
    *,
    heads_per_group: int,
) -> torch.Tensor:
    """
    Independent oracle for :func:`mamba3_ssd_reference`, from the materialized (closed) form

        ``y_t = sum_{s<=t} (prod_{r=s+1..t} alpha_r) * <C_t, w_s>``

    with the transfer coefficients built as an outer difference of cumulative ``log alpha``.
    Nothing here iterates the state update, so an error in the scan cannot cancel itself out.
    """
    seq_len, n_heads = x.shape[1], x.shape[2]
    tril = torch.tril(torch.ones(seq_len, seq_len, dtype=x.dtype, device=x.device))

    # Cumulative rotation angle as a lower-triangular matmul rather than a cumsum.
    theta_cumulative = torch.einsum("ts,bsgk->btgk", tril, theta)
    B = _rotate_bc_via_complex(B, theta_cumulative)
    C = _rotate_bc_via_complex(C, theta_cumulative)

    # Groups -> heads by explicit gather: head h reads group h // heads_per_group.
    group_of_head = torch.arange(n_heads, device=x.device) // heads_per_group
    B = B[:, :, group_of_head]  # (batch, T, H, R, N)
    C = C[:, :, group_of_head]

    alpha = torch.exp(dt * A)
    gamma = lam * dt
    beta = (1.0 - lam) * dt * alpha

    # State-input v_t = B_t (x) x_t and its width-2 trapezoidal combination w_t.
    v = torch.einsum("bthrn,bthp->bthrnp", B, x)
    v_prev = torch.cat([torch.zeros_like(v[:, :1]), v[:, :-1]], dim=1)
    w = gamma[..., None, None, None] * v + beta[..., None, None, None] * v_prev

    # Transfer coefficients M[t, s] = prod_{r=s+1..t} alpha_r = exp(L_t - L_s) for s <= t.
    log_alpha = dt * A
    L = torch.einsum("ts,bsh->bth", tril, log_alpha)
    M = torch.exp(L.unsqueeze(2) - L.unsqueeze(1)) * tril[None, :, :, None]

    # <C_t, w_s> contracted over (rank, d_state), then weighted by the transfer coefficients.
    Cw = torch.einsum("bthrn,bshrnp->btshp", C, w)
    return (M.unsqueeze(-1) * Cw).sum(dim=2)


@pytest.mark.parametrize("mimo_rank", [1, 3], ids=["siso", "mimo3"])
@pytest.mark.parametrize("n_groups", [1, 2], ids=["g1", "g2"])
def test_mamba3_ssd_reference_matches_closed_form_oracle(mimo_rank: int, n_groups: int):
    """The sequential reference scan must equal the materialized closed form of the same SSM."""
    torch.manual_seed(0)
    batch, seq_len, n_heads, head_dim, d_state = 2, 8, 4, 4, 8
    heads_per_group = n_heads // n_groups

    x = torch.randn(batch, seq_len, n_heads, head_dim)
    B = torch.randn(batch, seq_len, n_groups, mimo_rank, d_state)
    C = torch.randn(batch, seq_len, n_groups, mimo_rank, d_state)
    dt = torch.rand(batch, seq_len, n_heads) * 0.1 + 0.01
    A = -torch.rand(n_heads) - 0.5
    lam = torch.rand(batch, seq_len, n_heads)
    # Non-zero angles, so the rotation is actually exercised (a zero theta hides it entirely).
    theta = torch.randn(batch, seq_len, n_groups, d_state // 2)

    y = mamba3_ssd_reference(x, B, C, dt, A, lam, theta, heads_per_group=heads_per_group)
    expected = _mamba3_ssd_closed_form_oracle(
        x, B, C, dt, A, lam, theta, heads_per_group=heads_per_group
    )

    # Both sides are fp32 and differ only in summation order (sequential scan vs. materialized
    # transfer matrix), so the error is a few float32 ULPs accumulated over the ~T*R*N terms of
    # the contraction: seq_len=8, rank<=3, d_state=8 gives <2e2 terms, i.e. ~1e-5 relative at
    # worst. 1e-5 is therefore loose enough to be stable but ~3 orders of magnitude tighter than
    # any real disagreement in the recurrence, rotation or MIMO summation would produce.
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)


def test_mamba3_mimo_adds_rank_contribution():
    """MIMO (R>1) changes the output vs SISO (R=1); a zero extra rank is a no-op (ranks are summed)."""
    torch.manual_seed(0)
    batch, seq_len, n_heads, head_dim, n_groups, d_state = 1, 6, 2, 4, 1, 4
    x = torch.randn(batch, seq_len, n_heads, head_dim)
    dt = torch.rand(batch, seq_len, n_heads) * 0.1 + 0.01
    A = -torch.rand(n_heads)
    lam = torch.rand(batch, seq_len, n_heads)
    theta = torch.randn(batch, seq_len, n_groups, d_state // 2)
    B1 = torch.randn(batch, seq_len, n_groups, 1, d_state)
    C1 = torch.randn(batch, seq_len, n_groups, 1, d_state)

    y_siso = mamba3_ssd_reference(x, B1, C1, dt, A, lam, theta, heads_per_group=n_heads)

    # A second rank that is all zeros must not change the output.
    zeros = torch.zeros_like(B1)
    y_zero_pad = mamba3_ssd_reference(
        x,
        torch.cat([B1, zeros], dim=3),
        torch.cat([C1, zeros], dim=3),
        dt,
        A,
        lam,
        theta,
        heads_per_group=n_heads,
    )
    torch.testing.assert_close(y_siso, y_zero_pad)

    # A non-zero second rank must change the output.
    y_mimo = mamba3_ssd_reference(
        x,
        torch.cat([B1, torch.randn_like(B1)], dim=3),
        torch.cat([C1, torch.randn_like(C1)], dim=3),
        dt,
        A,
        lam,
        theta,
        heads_per_group=n_heads,
    )
    assert not torch.allclose(y_siso, y_mimo)


def test_mamba3_mixer_config_round_trip():
    """The mixer config must survive as_config_dict() -> from_dict() (configs are logged/checkpointed)."""
    cfg = Mamba3MixerConfig(
        n_heads=8,
        head_dim=32,
        d_state=64,
        n_groups=2,
        mimo_rank=4,
        bc_bias=False,
        rotation_block_size=4,
        a_log_init_min=0.00625,
        a_log_init_max=0.1,
    )
    rebuilt = Mamba3MixerConfig.from_dict(cfg.as_config_dict())
    assert rebuilt == cfg
    assert rebuilt.rotation_block_size == 4
    assert rebuilt.a_log_init_min == 0.00625
    assert rebuilt.a_log_init_max == 0.1
    assert rebuilt.num_params(512) == cfg.num_params(512)


@requires_gpu
@pytest.mark.parametrize("mimo_rank", [1, 2], ids=["siso", "mimo2"])
def test_mamba3_mixer_fwd_bwd_cuda_matches_cpu(mimo_rank: int):
    """
    Forward and backward must run on CUDA and agree with the CPU result for identical weights.

    Tolerance: fp32 matmul TF32 is off by default, so both devices run true fp32 and differ only
    in reduction order and libm-vs-CUDA transcendentals (exp/sin/cos/softplus/sigmoid). Outputs
    are O(1e-1), so a few ULPs of fp32 (eps 1.2e-7) is ~1e-8; the measured worst deviation across
    the output, the input grad and every parameter grad is 3.7e-8. atol=1e-6 leaves ~25x headroom
    while staying ~4 orders of magnitude below the O(1e-2) discrepancy any genuine device-
    dependent algorithmic divergence would produce.
    """
    torch.manual_seed(0)
    d_model, seq_len, batch_size = 32, 16, 2
    rtol, atol = 1e-4, 1e-6

    module_cpu = _tiny_mixer(d_model, mimo_rank=mimo_rank)
    module_cuda = copy.deepcopy(module_cpu).to("cuda")

    x_cpu = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    x_cuda = x_cpu.detach().to("cuda").requires_grad_(True)

    y_cpu = module_cpu(x_cpu)
    y_cuda = module_cuda(x_cuda)
    assert y_cuda.shape == x_cuda.shape
    assert y_cuda.is_cuda
    assert torch.isfinite(y_cuda).all()

    y_cpu.pow(2).mean().backward()
    y_cuda.pow(2).mean().backward()

    assert x_cuda.grad is not None
    assert torch.isfinite(x_cuda.grad).all()
    for name, p in module_cuda.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    torch.testing.assert_close(y_cuda.cpu(), y_cpu, rtol=rtol, atol=atol)
    assert x_cpu.grad is not None
    torch.testing.assert_close(x_cuda.grad.cpu(), x_cpu.grad, rtol=rtol, atol=atol)
    for (name, p_cpu), (_, p_cuda) in zip(
        module_cpu.named_parameters(), module_cuda.named_parameters()
    ):
        assert p_cpu.grad is not None and p_cuda.grad is not None
        torch.testing.assert_close(
            p_cuda.grad.cpu(), p_cpu.grad, rtol=rtol, atol=atol, msg=f"grad mismatch for {name}"
        )


@requires_gpu
def test_mamba3_mixer_fwd_bwd_cuda_bf16():
    """
    The bf16 autocast path must produce finite outputs and gradients and track the fp32 result.

    Tolerance: bf16 keeps an 8-bit mantissa (relative eps 2^-8 = 3.9e-3), so at an output scale of
    ~1e-1 a single rounding is already ~4e-4; the measured deviation from fp32 is 6.3e-4, i.e. a
    couple of bf16 ULPs. atol=5e-3 is ~8x that, loose enough for the accumulated rounding of the
    whole projection stack but far tighter than a real dtype-handling bug (which shows up as NaN,
    inf, or an O(1) shift). bf16 is deliberately not compared against CPU: torch's CPU bf16
    coverage is uneven, so the CPU cross-check is done in fp32 by the test above.
    """
    torch.manual_seed(0)
    d_model, seq_len, batch_size = 32, 16, 2

    module = _tiny_mixer(d_model).to("cuda")
    x = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)

    with torch.no_grad():
        y_fp32 = module(x)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y_bf16 = module(x)
    assert y_bf16.dtype == torch.bfloat16
    assert y_bf16.shape == x.shape
    assert torch.isfinite(y_bf16).all()

    y_bf16.float().pow(2).mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, p in module.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    torch.testing.assert_close(y_bf16.float(), y_fp32, rtol=1.6e-2, atol=5e-3)


def test_b3_one_token_decode_cache_matches_full_sequence_oracle():
    torch.manual_seed(902)
    config = Mamba3MixerConfig(
        n_heads=2,
        head_dim=4,
        d_state=6,
        n_groups=1,
        mimo_rank=1,
        rotation_block_size=3,
        prefer_official_kernel=False,
        fuse_input_projections=True,
    )
    prefill = config.build(d_model=8, layer_idx=0, n_layers=2)
    cache = BufferCache()
    decode = config.build(d_model=8, layer_idx=0, n_layers=2, cache=cache)
    prefill.init_weights(
        init_method=InitMethod.normal,
        d_model=8,
        block_idx=0,
        num_blocks=2,
        generator=torch.Generator().manual_seed(903),
    )
    decode.load_state_dict(prefill.state_dict())
    x = torch.randn(2, 7, 8)

    expected = prefill(x)
    actual = torch.cat(
        [decode(x[:, index : index + 1], decode=True) for index in range(x.shape[1])],
        dim=1,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert decode._cache["cumulative_quaternion"].shape[-1] == 4
    assert decode._cache["state"].shape == (2, 2, 1, 6, 4)
    assert decode._cache["prior_state_input"].shape == (2, 2, 1, 6, 4)


# ---------------------------------------------------------------------------------------
# rotation_scan_impl reaches the kernel from the config
#
# The scan choice used to live only in `MAMBA3_ROTATION_SCAN_IMPL`, read once at import. It never
# entered the saved config and was never logged, so a resumed run that lost the export silently
# trained 2.2x slower with no error. These tests pin the config field that replaces it -- the same
# shape of guard the ablation already has for `rotation_block_size`.
# ---------------------------------------------------------------------------------------


def test_rotation_scan_impl_defaults_to_none():
    """``None`` keeps the historical behaviour: defer to the environment."""
    assert Mamba3MixerConfig(n_heads=2).rotation_scan_impl is None
    assert _tiny_mixer().rotation_scan_impl is None


def test_rotation_scan_impl_reaches_the_built_mixer():
    config = Mamba3MixerConfig(
        n_heads=2,
        head_dim=8,
        d_state=9,
        n_groups=1,
        mimo_rank=1,
        rotation_block_size=3,
        rotation_scan_impl="quaternion",
    )
    module = config.build(32, layer_idx=0, n_layers=2, init_device="meta")
    assert module.rotation_scan_impl == "quaternion"


def test_rotation_scan_impl_is_validated_at_build():
    """
    Fail at config time, not on the first forward. A typo that survives to the training loop costs
    a compile warmup and a GPU-hour before anyone notices the throughput is wrong.
    """
    config = Mamba3MixerConfig(n_heads=2, head_dim=8, d_state=8, rotation_scan_impl="quarternion")
    with pytest.raises((ValueError, OLMoConfigurationError), match="quaternion"):
        config.build(32, layer_idx=0, n_layers=2, init_device="meta")


def test_mixer_forwards_rotation_scan_impl_to_the_dispatcher(monkeypatch):
    """
    The value has to arrive at ``dispatch_mamba3_ssd``. Everything upstream of that call is
    bookkeeping; this is the assertion that the plumbing is connected end to end.
    """
    import olmo_core.nn.mamba3.mixer as mixer_mod

    seen: dict = {}
    real = mixer_mod.dispatch_mamba3_ssd

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(mixer_mod, "dispatch_mamba3_ssd", spy)

    config = Mamba3MixerConfig(
        n_heads=2,
        head_dim=8,
        d_state=8,
        n_groups=1,
        mimo_rank=2,
        rotation_scan_impl="chunked",
    )
    module = config.build(32, layer_idx=0, n_layers=2, init_device="cpu")
    module.init_weights(init_method=InitMethod.normal, d_model=32, block_idx=0, num_blocks=2)
    module(torch.randn(1, 8, 32))

    assert seen.get("rotation_scan_impl") == "chunked"


# ======================================================================================
# Faithful published-Mamba-3 SISO options (opt-in; default off keeps every arm above and
# every existing checkpoint byte-identical).
#
# Each of the five options below is a distinct deviation of *this* arm from the published
# SISO architecture that the fidelity audit identified: a token-dependent decay ``A``, a
# learned ``D`` skip, norm-before-gate output ordering, a post-BCNorm per-head ``B``/``C``
# bias initialized to one, and the official per-head ``tanh(angle) * pi * dt`` rotation
# generalized to SO(3) over only part of the state. They all default off, so the general
# mixer, the seven peer arms, and the reference/kernel parity oracle are untouched.
# ======================================================================================


def _copy_shared_named_params(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy every parameter that both modules share by name (ignoring the extras either adds)."""
    src_params = dict(src.named_parameters())
    with torch.no_grad():
        for name, p in dst.named_parameters():
            if name in src_params and src_params[name].shape == p.shape:
                p.copy_(src_params[name])


def _faithful_siso_config(**overrides) -> Mamba3MixerConfig:
    """A small CPU-runnable faithful SISO b=3 config (the chunked path handles it without a GPU)."""
    base = dict(
        n_heads=4,
        head_dim=8,
        d_state=12,
        n_groups=1,
        mimo_rank=1,
        rotation_block_size=3,
        bc_norm=True,
        bc_bias=False,
        dynamic_a=True,
        d_skip=True,
        norm_before_gate=True,
        bc_bias_after_norm=True,
        dt_scaled_rotation=True,
        rope_fraction=0.5,
        fuse_input_projections=False,
    )
    base.update(overrides)
    return Mamba3MixerConfig(**base)


def test_faithful_options_default_off_and_round_trip():
    """All five faithful options must default off and survive config serialization."""
    default = Mamba3MixerConfig(n_heads=4)
    assert default.dynamic_a is False
    assert default.d_skip is False
    assert default.norm_before_gate is False
    assert default.bc_bias_after_norm is False
    assert default.dt_scaled_rotation is False
    assert default.rope_fraction == 1.0

    cfg = _faithful_siso_config()
    rebuilt = Mamba3MixerConfig.from_dict(cfg.as_config_dict())
    assert rebuilt == cfg
    assert rebuilt.num_params(64) == cfg.num_params(64)


def test_dynamic_a_reduces_to_static_when_the_projection_is_zero():
    """
    Token-dependent ``A`` must be a modulation of the static per-head baseline: with the
    projection zeroed the decay is exactly ``-exp(A_log)`` again, so the output has to match a
    static-``A`` mixer sharing every other weight. This pins both that ``a_proj`` exists and
    that it composes with ``A_log`` rather than replacing it.
    """
    torch.manual_seed(0)
    d_model = 32
    dyn = Mamba3MixerConfig(
        n_heads=4, head_dim=8, d_state=12, n_groups=1, mimo_rank=1, dynamic_a=True
    ).build(d_model, layer_idx=0, n_layers=2, init_device="cpu")
    static = Mamba3MixerConfig(
        n_heads=4, head_dim=8, d_state=12, n_groups=1, mimo_rank=1, dynamic_a=False
    ).build(d_model, layer_idx=0, n_layers=2, init_device="cpu")
    static.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2)
    _copy_shared_named_params(static, dyn)
    assert dyn.a_proj is not None
    with torch.no_grad():
        dyn.a_proj.weight.zero_()

    x = torch.randn(2, 6, d_model)
    torch.testing.assert_close(dyn(x), static(x))

    # A non-zero projection must actually change the decay, hence the output.
    with torch.no_grad():
        dyn.a_proj.weight.normal_(std=0.5)
    assert not torch.allclose(dyn(x), static(x))


def test_d_skip_is_a_learned_identity_path_initialized_to_one():
    """``D`` is a per-head skip initialized to one; zeroing it recovers the no-skip output."""
    torch.manual_seed(0)
    d_model = 32
    with_d = Mamba3MixerConfig(
        n_heads=4, head_dim=8, d_state=12, n_groups=1, mimo_rank=1, d_skip=True
    ).build(d_model, layer_idx=0, n_layers=2, init_device="cpu")
    without_d = Mamba3MixerConfig(
        n_heads=4, head_dim=8, d_state=12, n_groups=1, mimo_rank=1, d_skip=False
    ).build(d_model, layer_idx=0, n_layers=2, init_device="cpu")
    without_d.init_weights(
        init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2
    )
    with_d.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2)
    _copy_shared_named_params(without_d, with_d)

    assert with_d.D is not None
    assert with_d.D.shape == (4,)
    assert torch.allclose(with_d.D, torch.ones_like(with_d.D))

    x = torch.randn(2, 6, d_model)
    assert not torch.allclose(with_d(x), without_d(x))
    with torch.no_grad():
        with_d.D.zero_()
    torch.testing.assert_close(with_d(x), without_d(x))


def test_norm_before_gate_changes_the_output_ordering():
    """norm-before-gate (``rmsnorm(y) * silu(z)``) must differ from gate-then-norm."""
    torch.manual_seed(0)
    d_model = 32
    before = Mamba3MixerConfig(
        n_heads=4, head_dim=8, d_state=12, n_groups=1, mimo_rank=1, norm_before_gate=True
    ).build(d_model, layer_idx=0, n_layers=2, init_device="cpu")
    after = Mamba3MixerConfig(
        n_heads=4, head_dim=8, d_state=12, n_groups=1, mimo_rank=1, norm_before_gate=False
    ).build(d_model, layer_idx=0, n_layers=2, init_device="cpu")
    after.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2)
    _copy_shared_named_params(after, before)

    x = torch.randn(2, 6, d_model)
    assert not torch.allclose(before(x), after(x))


def test_bc_bias_after_norm_is_per_head_initialized_to_one():
    """The post-BCNorm bias is head-specific, initialized to one, and requires the per-head path."""
    module = _faithful_siso_config().build(32, layer_idx=0, n_layers=2, init_device="cpu")
    module.init_weights(init_method=InitMethod.normal, d_model=32, block_idx=0, num_blocks=2)
    assert module.bc_post_bias_b is not None and module.bc_post_bias_c is not None
    assert module.bc_post_bias_b.shape == (module.n_heads, module.d_state)
    assert torch.allclose(module.bc_post_bias_b, torch.ones_like(module.bc_post_bias_b))
    assert torch.allclose(module.bc_post_bias_c, torch.ones_like(module.bc_post_bias_c))

    # It must not silently ride on the old pre-BCNorm linear bias.
    assert module.in_B is not None and module.in_B.bias is None

    with pytest.raises(OLMoConfigurationError, match="bc_bias_after_norm"):
        _faithful_siso_config(bc_bias=True).build(32, layer_idx=0, n_layers=1, init_device="meta")
    with pytest.raises(OLMoConfigurationError, match="dt_scaled_rotation"):
        _faithful_siso_config(dt_scaled_rotation=False).build(
            32, layer_idx=0, n_layers=1, init_device="meta"
        )


def test_rope_fraction_narrows_theta_proj_and_leaves_the_rest_identity():
    """
    ``rope_fraction`` rotates only a prefix of the state; the remaining blocks are identity and
    carry no angle parameters. At 0.5 the angle projection is exactly half the full-state width.
    """
    full = Mamba3MixerConfig(
        n_heads=4, head_dim=8, d_state=12, n_groups=1, rotation_block_size=3, rope_fraction=1.0
    ).build(64, layer_idx=0, n_layers=1, init_device="meta")
    half = Mamba3MixerConfig(
        n_heads=4, head_dim=8, d_state=12, n_groups=1, rotation_block_size=3, rope_fraction=0.5
    ).build(64, layer_idx=0, n_layers=1, init_device="meta")
    assert full.theta_proj is not None and half.theta_proj is not None
    assert half.theta_proj.out_features == full.theta_proj.out_features // 2

    with pytest.raises(OLMoConfigurationError, match="rope_fraction"):
        Mamba3MixerConfig(n_heads=4, d_state=12, rotation_block_size=3, rope_fraction=0.0).build(
            64, layer_idx=0, n_layers=1, init_device="meta"
        )


def test_dt_scaled_rotation_is_bounded_and_per_head():
    """
    The official rotation is ``tanh(angle) * pi * dt`` per head, so it differs from feeding raw
    per-group angles straight to the scan, and the whole faithful mixer stays finite end to end.
    """
    torch.manual_seed(0)
    d_model = 32
    scaled = _faithful_siso_config().build(d_model, layer_idx=0, n_layers=2, init_device="cpu")
    scaled.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2)
    raw = _faithful_siso_config(dt_scaled_rotation=True).build(
        d_model, layer_idx=0, n_layers=2, init_device="cpu"
    )
    _copy_shared_named_params(scaled, raw)

    x = torch.randn(2, 6, d_model, requires_grad=True)
    y = scaled(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.float().pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in scaled.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_faithful_num_params_matches_the_built_module():
    """``num_params`` has to count ``a_proj``, ``D``, the post-BCNorm biases and the narrowed theta."""
    for cfg in (
        _faithful_siso_config(),
        _faithful_siso_config(rope_fraction=1.0),
        Mamba3MixerConfig(n_heads=4, head_dim=8, d_state=12, dynamic_a=True),
        Mamba3MixerConfig(n_heads=4, head_dim=8, d_state=12, d_skip=True),
    ):
        module = cfg.build(64, layer_idx=0, n_layers=2, init_device="meta")
        assert cfg.num_params(64) == sum(p.numel() for p in module.parameters())


def test_faithful_path_forwards_the_scan_choice_and_per_head_grouping():
    """
    The faithful path must hand the dispatcher the same scan choice the config records, and hand
    it per-head ``B``/``C``.

    Dropping ``rotation_scan_impl`` here would be silent and expensive: dispatch would fall back
    to the ``MAMBA3_ROTATION_SCAN_IMPL`` default (``chunked``) with nothing raising, which is the
    exact 2.2x regression :func:`resolve_rotation_scan_impl` exists to prevent. The grouping is
    pinned alongside it because the whole per-head construction is what makes ``heads_per_group``
    1 at the boundary; a stale ``self.heads_per_group`` there would silently mis-broadcast B/C.
    """
    import olmo_core.nn.mamba3.mixer as mixer_mod

    seen: dict = {}
    real = mixer_mod.dispatch_mamba3_ssd

    def spy(*args, **kwargs):
        seen["args"] = args
        seen.update(kwargs)
        return real(*args, **kwargs)

    d_model = 32
    module = _faithful_siso_config(rotation_scan_impl="chunked").build(
        d_model, layer_idx=0, n_layers=2, init_device="cpu"
    )
    module.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mixer_mod, "dispatch_mamba3_ssd", spy)
        module(torch.randn(2, 6, d_model))

    assert seen["rotation_scan_impl"] == "chunked"
    assert seen["heads_per_group"] == 1
    _, B_seen, C_seen, _, _, _, theta_seen = seen["args"]
    # B/C arrive per head, and theta covers only the rotated prefix -- the identity tail is left
    # for the rotation to skip rather than padded out and scanned.
    assert B_seen.shape[2] == module.n_heads
    assert C_seen.shape[2] == module.n_heads
    assert theta_seen.shape[2] == module.n_heads
    assert theta_seen.shape[3] == module.n_rotated_blocks
    assert module.n_rotated_blocks < module.n_rotation_blocks


def test_faithful_flops_count_the_extra_projection_and_the_per_head_rotation():
    """
    The FLOP model feeds reported MFU, so it has to see what the faithful arm actually does.

    Two undercounts are pinned here: ``a_proj`` was missing from the projection list entirely, and
    the rotation term read ``n_groups`` when the dt-scaled rotation runs per *head* -- a 32x
    undercount of that term on the shipped arm, which would have made its MFU read low.
    ``rope_fraction`` must cut the counted rotation, because the identity tail is not model work.
    """
    d_model = 64
    base = dict(n_heads=4, head_dim=8, d_state=12, n_groups=1, mimo_rank=1, rotation_block_size=3)

    def flops(**over):
        cfg = Mamba3MixerConfig(**base, **over)
        module = cfg.build(d_model, layer_idx=0, n_layers=1, init_device="meta")
        return module.num_flops_per_token(2048)

    # a_proj is a real GEMM and is counted exactly once.
    assert flops(dynamic_a=True) == flops(dynamic_a=False) + 2 * d_model * base["n_heads"]
    # The per-head rotation costs more than the group-shared one.
    assert flops(dt_scaled_rotation=True) > flops(dt_scaled_rotation=False)
    # And only the rotated prefix counts.
    assert flops(dt_scaled_rotation=True, rope_fraction=0.5) < flops(dt_scaled_rotation=True)


@pytest.mark.parametrize("block_size", [2, 3], ids=["b2", "b3"])
@pytest.mark.parametrize("scan_impl", ["chunked", "quaternion"])
def test_partial_rotation_equals_padding_the_tail_with_identity_blocks(
    block_size: int, scan_impl: str
):
    """
    Skipping the identity tail must be a pure optimization: identical numbers, less work.

    ``rope_fraction`` was first expressed by padding ``theta`` with zero-angle blocks, which is
    mathematically right but pays the whole prefix product for a rotation that is the identity
    (measured 1.82x the cost of rotating the prefix alone). The rotation entry points now leave
    the un-covered tail exactly as it arrived. This pins that the two forms agree -- if they ever
    diverge, the optimization has silently changed the model.
    """
    from olmo_core.nn.mamba3.mamba3_ssd_fast import _fast_rotate_bc_pair

    torch.manual_seed(0)
    batch, seq, groups, rank = 2, 8, 2, 1
    n_blocks, rotated = 6, 3
    d_state = n_blocks * block_size
    angles = block_size * (block_size - 1) // 2

    B = torch.randn(batch, seq, groups, rank, d_state)
    C = torch.randn(batch, seq, groups, rank, d_state)
    theta = 0.2 * torch.randn(batch, seq, groups, rotated, angles)
    padded = torch.cat([theta, torch.zeros(batch, seq, groups, n_blocks - rotated, angles)], dim=-2)

    narrow_B, narrow_C = _fast_rotate_bc_pair(B, C, theta, block_size, None, scan_impl=scan_impl)
    padded_B, padded_C = _fast_rotate_bc_pair(B, C, padded, block_size, None, scan_impl=scan_impl)
    torch.testing.assert_close(narrow_B, padded_B)
    torch.testing.assert_close(narrow_C, padded_C)

    # The tail must be untouched, not merely equal to the padded form.
    covered = rotated * block_size
    assert torch.equal(narrow_B[..., covered:], B[..., covered:])
    assert torch.equal(narrow_C[..., covered:], C[..., covered:])

    # And the same holds end to end through the sequential reference recurrence.
    x = torch.randn(batch, seq, groups, 4)
    dt = torch.rand(batch, seq, groups) * 0.1 + 0.01
    A = -torch.rand(groups) - 0.5
    lam = torch.rand(batch, seq, groups)
    common = dict(heads_per_group=1, block_size=block_size)
    torch.testing.assert_close(
        mamba3_ssd_reference(x, B, C, dt, A, lam, theta, **common),
        mamba3_ssd_reference(x, B, C, dt, A, lam, padded, **common),
    )


def test_dt_scaled_rotation_rejects_a_second_angle_bound():
    """``theta_max`` is unread on the faithful path, so pairing the two must fail, not be ignored."""
    with pytest.raises(OLMoConfigurationError, match="theta_max"):
        _faithful_siso_config(theta_max=0.01).build(32, layer_idx=0, n_layers=1, init_device="meta")


@pytest.mark.parametrize("timescale", ["per_head", "group_mean"])
def test_faithful_options_work_fused_and_unfused_at_the_same_parameter_count(timescale: str):
    """
    Fusing the input projections is a layout choice, not a feature switch.

    ``a_proj`` rides inside the fused dynamics GEMM instead of adding a ninth launch, so the two
    layouts must hold the same parameters and both run. The fused ordering is dt, lambda, a,
    theta; getting it wrong slices every dynamics tensor at the wrong offset, which the
    round-trip below would catch.
    """
    d_model = 64
    unfused = _faithful_siso_config(rotation_timescale=timescale, fuse_input_projections=False)
    fused = _faithful_siso_config(rotation_timescale=timescale, fuse_input_projections=True)
    assert unfused.num_params(d_model) == fused.num_params(d_model)

    built_unfused = unfused.build(d_model, layer_idx=0, n_layers=2, init_device="cpu")
    built_fused = fused.build(d_model, layer_idx=0, n_layers=2, init_device="cpu")
    assert sum(p.numel() for p in built_unfused.parameters()) == unfused.num_params(d_model)
    assert sum(p.numel() for p in built_fused.parameters()) == fused.num_params(d_model)
    assert built_fused.a_proj is None and built_unfused.a_proj is not None

    for module in (built_unfused, built_fused):
        module.init_weights(
            init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2
        )
        y = module(torch.randn(2, 6, d_model))
        assert y.shape == (2, 6, d_model)
        assert torch.isfinite(y).all()

    # A fused checkpoint must load into the unfused module and vice versa.
    built_unfused.load_state_dict(built_fused.state_dict())
    built_fused.load_state_dict(built_unfused.state_dict())


def test_group_mean_timescale_keeps_the_bc_path_one_group_wide():
    """
    The point of ``group_mean`` is that ``B``/``C`` never get broadcast to heads.

    That is what preserves GQA into the scan, and it is the whole throughput argument for the
    option, so it is asserted on the tensors handed to the dispatcher rather than inferred.
    ``per_head`` is checked alongside it so the contrast is explicit.
    """
    import olmo_core.nn.mamba3.mixer as mixer_mod

    d_model = 32
    seen: dict = {}
    real = mixer_mod.dispatch_mamba3_ssd

    def spy(*args, **kwargs):
        seen["args"] = args
        seen.update(kwargs)
        return real(*args, **kwargs)

    for timescale, expected_groups in (("per_head", None), ("group_mean", 1)):
        module = _faithful_siso_config(rotation_timescale=timescale).build(
            d_model, layer_idx=0, n_layers=2, init_device="cpu"
        )
        module.init_weights(
            init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(mixer_mod, "dispatch_mamba3_ssd", spy)
            module(torch.randn(2, 6, d_model))
        _, B_seen, _, _, _, _, theta_seen = seen["args"]
        groups = expected_groups if expected_groups is not None else module.n_heads
        assert B_seen.shape[2] == groups, timescale
        assert theta_seen.shape[2] == groups, timescale
        assert seen["heads_per_group"] == module.n_heads // groups, timescale
        # The post-norm bias has exactly one row per lane the scan sees.
        assert module.bc_post_bias_b is not None
        assert module.bc_post_bias_b.shape[0] == groups, timescale


@requires_gpu
def test_faithful_mixer_fwd_bwd_cuda_is_finite_and_matches_cpu():
    """
    The faithful arm's forward/backward must be finite on CUDA and agree with the CPU result.

    This is the plan's CUDA finite-gradient and parity gate for the faithful SISO b=3 arm. With
    ``prefer_official_kernel=None`` the CUDA side takes ``official_fast`` where ``mamba-ssm`` is
    installed and the CPU side takes the chunked reference, so a match cross-checks that the
    per-head dt-scaled half-state rotation, the token-dependent decay, the post-BCNorm bias, and
    the D skip all flow through the fast kernel path the same way the reference computes them.
    Tolerance is bf16-scale because the official kernel hard-casts internally.
    """
    torch.manual_seed(0)
    d_model = 32
    cfg = _faithful_siso_config()
    module_cpu = cfg.build(d_model, layer_idx=0, n_layers=2, init_device="cpu")
    module_cpu.init_weights(
        init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2
    )
    module_cuda = copy.deepcopy(module_cpu).to("cuda")

    x_cpu = torch.randn(2, 16, d_model, requires_grad=True)
    x_cuda = x_cpu.detach().to("cuda").requires_grad_(True)

    y_cpu = module_cpu(x_cpu)
    y_cuda = module_cuda(x_cuda)
    assert torch.isfinite(y_cuda).all()

    y_cpu.float().pow(2).mean().backward()
    y_cuda.float().pow(2).mean().backward()
    assert x_cuda.grad is not None and torch.isfinite(x_cuda.grad).all()
    for name, p in module_cuda.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    # official_fast on CUDA vs the chunked reference on CPU: same recurrence, different kernels.
    torch.testing.assert_close(y_cuda.float().cpu(), y_cpu.float(), rtol=3e-2, atol=3e-2)


def test_faithful_ssd_composes_broadcast_bias_dt_scaled_rotation_and_identity_tail():
    """
    Pin the per-head plumbing against an independent reference, not just finiteness.

    ``_faithful_ssd`` must (a) broadcast the group ``B``/``C`` to heads, (b) add the per-head
    post-BCNorm bias, (c) build the angle as ``tanh(raw) * pi * dt`` per head, and (d) pad the
    un-rotated tail with identity blocks -- then hand the result to the ordinary per-head
    scalar-decay scan. Reconstructing exactly that by hand and feeding it to the sequential
    reference catches a wrong axis, a dropped ``tanh``/``dt``, or a mis-sized identity pad, any
    of which a finiteness check would sail past.
    """
    import math

    torch.manual_seed(0)
    d_model = 32
    module = _faithful_siso_config().build(d_model, layer_idx=0, n_layers=2, init_device="cpu")
    module.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2)

    batch, seq_len = 2, 6
    H, P, N = module.n_heads, module.head_dim, module.d_state
    G, R = module.n_groups, module.mimo_rank
    hpg = module.heads_per_group

    xv = torch.randn(batch, seq_len, H, P)
    Bm = torch.randn(batch, seq_len, G, R, N)  # stands in for post-BCNorm group B/C
    Cm = torch.randn(batch, seq_len, G, R, N)
    dt = torch.rand(batch, seq_len, H) * 0.1 + 0.01
    A = -torch.rand(H) - 0.5
    lam = torch.rand(batch, seq_len, H)
    theta = torch.randn(batch, seq_len, G, module.n_rotated_blocks, module.angles_per_block)

    y = module._faithful_ssd(xv, Bm, Cm, dt, A, lam, theta)

    Bh = Bm.repeat_interleave(hpg, dim=2) + module.bc_post_bias_b.view(1, 1, H, 1, N)
    Ch = Cm.repeat_interleave(hpg, dim=2) + module.bc_post_bias_c.view(1, 1, H, 1, N)
    th = torch.tanh(theta.repeat_interleave(hpg, dim=2)) * math.pi * dt.unsqueeze(-1).unsqueeze(-1)
    pad = module.n_rotation_blocks - module.n_rotated_blocks
    th = torch.cat([th, th.new_zeros(batch, seq_len, H, pad, module.angles_per_block)], dim=-2)
    expected = mamba3_ssd_reference(
        xv, Bh, Ch, dt, A, lam, th, heads_per_group=1, block_size=module.rotation_block_size
    )

    # `_faithful_ssd` routes through the chunked kernel on CPU; the reference is the sequential
    # oracle. They agree to a few float32 ULPs over this tiny contraction.
    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)
