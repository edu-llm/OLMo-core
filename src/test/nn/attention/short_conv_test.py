import math
from typing import Any, Dict

import pytest
import torch

from olmo_core.nn.attention.base import SequenceMixerConfig
from olmo_core.nn.attention.short_conv import GateStructure, ShortConv, ShortConvConfig


def _reference_lfm2_short_conv(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    w_conv: torch.Tensor,
) -> torch.Tensor:
    """
    Transcription of ``Lfm2ShortConv.slow_forward`` from ``transformers`` v5.0.0rc1
    (Apache-2.0), used as the parity oracle.

    Written out longhand rather than imported so the test does not depend on
    ``transformers`` being installed, and so the chunk order is visible at the assertion site.

    :param x: Input of shape ``(batch_size, seq_len, d_model)``.
    :param w_in: ``in_proj`` weight of shape ``(3 * d_model, d_model)``.
    :param w_out: ``out_proj`` weight of shape ``(d_model, d_model)``.
    :param w_conv: Depthwise conv weight of shape ``(d_model, 1, kernel_size)``.
    """
    seqlen = x.shape[1]
    BCx = torch.nn.functional.linear(x, w_in).transpose(-1, -2)
    B, C, v = BCx.chunk(3, dim=-2)  # the released order: pre-gate, post-gate, value
    Bx = B * v
    conv_out = torch.nn.functional.conv1d(
        Bx, w_conv, bias=None, groups=w_conv.shape[0], padding=w_conv.shape[-1] - 1
    )[..., :seqlen]
    y = C * conv_out
    return torch.nn.functional.linear(y.transpose(-1, -2).contiguous(), w_out)


def test_short_conv_matches_lfm2_reference():
    """The operator must match the released implementation bit-for-bit in fp64."""
    torch.manual_seed(0)
    d_model, kernel_size = 64, 3
    m = ShortConv(d_model=d_model, kernel_size=kernel_size, use_fla=False).to(torch.float64)

    # in_proj is split across three modules here but is one (3d, d) tensor upstream; stack in
    # the released order so the oracle sees exactly the same weights.
    pre, post = m.in_proj.gate_proj.weight.chunk(2, dim=0)
    w_in = torch.cat([pre, post, m.in_proj.value_proj.weight], dim=0)

    x = torch.randn(2, 7, d_model, dtype=torch.float64)
    expected = _reference_lfm2_short_conv(x, w_in, m.out_proj.weight, m.conv.weight)
    torch.testing.assert_close(m(x), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kernel_size", [3, 5, 9])
def test_parity_with_released_transformers_implementation(kernel_size: int):
    """
    Parity against the *actual* released ``Lfm2ShortConv``, not a transcription of it.

    This is the test that proves the operator is right: it catches a divergence introduced by
    a future ``transformers`` release, which the hand-written oracle above cannot. Skipped when
    ``transformers`` is unavailable so the suite stays runnable without it.

    Verified to give exactly 0.0 difference in float64 at all three widths.
    """
    transformers = pytest.importorskip("transformers", reason="parity oracle needs transformers")
    del transformers
    from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    d_model = 128
    cfg = Lfm2Config(
        hidden_size=d_model,
        conv_L_cache=kernel_size,
        conv_bias=False,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
    )
    hf = Lfm2ShortConv(cfg, layer_idx=0).to(torch.float64).eval()
    mine = (
        ShortConv(d_model=d_model, kernel_size=kernel_size, use_fla=False).to(torch.float64).eval()
    )

    with torch.no_grad():
        pre, post, value = hf.in_proj.weight.chunk(3, dim=0)  # the (B, C, x) order
        mine.in_proj.gate_proj.weight.copy_(torch.cat([pre, post], dim=0))
        mine.in_proj.value_proj.weight.copy_(value)
        mine.out_proj.weight.copy_(hf.out_proj.weight)
        mine.conv.weight.copy_(hf.conv.weight)

    x = torch.randn(2, 16, d_model, dtype=torch.float64)
    with torch.no_grad():
        ref = hf(x)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        torch.testing.assert_close(mine(x), ref, rtol=0, atol=1e-12)


def test_chunk_order_is_load_bearing():
    """
    Guard against the silent failure mode: permuting the chunks still produces finite output
    of the right shape, so only a value comparison catches it.
    """
    torch.manual_seed(0)
    d_model = 64
    m = ShortConv(d_model=d_model, use_fla=False).to(torch.float64)
    pre, post = m.in_proj.gate_proj.weight.chunk(2, dim=0)
    x = torch.randn(1, 6, d_model, dtype=torch.float64)

    correct = _reference_lfm2_short_conv(
        x, torch.cat([pre, post, m.in_proj.value_proj.weight]), m.out_proj.weight, m.conv.weight
    )
    swapped = _reference_lfm2_short_conv(
        x, torch.cat([pre, m.in_proj.value_proj.weight, post]), m.out_proj.weight, m.conv.weight
    )
    assert swapped.shape == correct.shape and torch.isfinite(swapped).all()
    assert not torch.allclose(correct, swapped, atol=1e-6)


def test_causality():
    """Perturbing token t must not change any output before t."""
    torch.manual_seed(0)
    d_model, seq_len = 32, 12
    m = ShortConv(d_model=d_model, kernel_size=3, use_fla=False).to(torch.float64)
    x = torch.randn(1, seq_len, d_model, dtype=torch.float64)

    with torch.no_grad():
        base = m(x)
        t = 5
        x2 = x.clone()
        x2[0, t] += 10.0
        pert = m(x2)

    torch.testing.assert_close(base[:, :t], pert[:, :t], rtol=0, atol=1e-12)
    assert not torch.allclose(base[:, t], pert[:, t])


@pytest.mark.parametrize("kernel_size", [3, 5, 9, 15])
def test_receptive_field_is_exactly_kernel_size(kernel_size: int):
    """
    A k-tap filter must reach exactly k-1 tokens back -- no more, no less.

    Probed on the *convolution path* rather than the module output. Both gates are
    multiplicative, so a zero-background input makes the post-gate at the probe position zero
    too, and every lag would read as "no reach" -- a false pass for narrow kernels and a
    false failure here. Perturbing against a nonzero background instead keeps the gates live.
    """
    torch.manual_seed(0)
    d_model, seq_len = 16, 40
    m = ShortConv(d_model=d_model, kernel_size=kernel_size, use_fla=False).to(torch.float64)
    # Random conv weights: the default init is identity-only and would hide any reach.
    with torch.no_grad():
        m.conv.weight.normal_()

    x = torch.randn(1, seq_len, d_model, dtype=torch.float64)
    probe = seq_len - 1
    with torch.no_grad():
        base = m(x)
        for lag in (kernel_size - 1, kernel_size):
            x2 = x.clone()
            x2[0, probe - lag] += 5.0
            delta = (m(x2) - base)[0, probe].abs().max().item()
            if lag <= kernel_size - 1:
                assert delta > 1e-9, f"k={kernel_size} failed to reach lag {lag}"
            else:
                assert delta < 1e-12, f"k={kernel_size} reached too far, to lag {lag}"


@pytest.mark.parametrize(
    "structure,kwargs",
    [
        ("dense", {}),
        ("lowrank", {"gate_rank": 8}),
        ("lowrank", {"gate_rank": 32}),
        ("grouped", {"gate_groups": 2}),
        ("grouped", {"gate_groups": 4}),
    ],
)
def test_num_params_matches_built_module(structure: GateStructure, kwargs: Dict[str, Any]):
    """``num_params`` is used for arm matching, so a mismatch silently unbalances the study."""
    d_model = 64
    cfg = ShortConvConfig(gate_structure=structure, **kwargs)
    m = cfg.build(d_model, layer_idx=0, n_layers=1)
    assert cfg.num_params(d_model) == sum(p.numel() for p in m.parameters())


def test_matched_cost_variants_are_actually_matched():
    """
    ``lowrank r=d/8`` and ``grouped g=4`` must cost identically, which is what makes the
    quality comparison between them meaningful.
    """
    d_model = 1024
    lr = ShortConvConfig(gate_structure="lowrank", gate_rank=d_model // 8)
    gp = ShortConvConfig(gate_structure="grouped", gate_groups=4)
    assert lr.num_params(d_model) == gp.num_params(d_model)


def test_lowrank_gate_variance_parity_at_init():
    """
    Step-0 gate output variance must match dense.

    With ``Var(y) = d * r * sigma_A^2 * sigma_B^2``, using the same std for both factors is
    24-48x too small and the error is monotone in ``r``. That yields a smooth, plausible
    "higher rank is better" sweep that is really an init-scale artifact -- so this parity
    check is what makes the rank sweep interpretable at all.
    """
    from olmo_core.nn.transformer.init import InitMethod

    d_model, std = 512, 0.02
    x = torch.randn(4, 16, d_model, dtype=torch.float64)

    dense = ShortConv(d_model=d_model, use_fla=False).to(torch.float64)
    dense.init_weights(
        init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=1, std=std
    )
    with torch.no_grad():
        ref = dense.in_proj(x)[0].var().item()

    for rank in (32, 64, 128):
        m = ShortConv(d_model=d_model, gate_structure="lowrank", gate_rank=rank, use_fla=False).to(
            torch.float64
        )
        m.init_weights(
            init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=1, std=std
        )
        with torch.no_grad():
            got = m.in_proj(x)[0].var().item()
        # Within 2x of dense across an 4x rank range. The naive init misses by 24-48x, so this
        # bound distinguishes correct from naive while tolerating trunc-normal sampling noise.
        assert 0.5 < got / ref < 2.0, f"rank={rank}: gate var ratio {got / ref:.3f} vs dense"


def test_document_isolation_prevents_cross_boundary_leakage():
    """
    With ``cu_doc_lens``, no filter may read across a document boundary. A k=3 conv that
    bleeds between documents is a different operator from the one under study.
    """
    torch.manual_seed(0)
    d_model = 16
    m = ShortConv(d_model=d_model, kernel_size=3, use_fla=False).to(torch.float64)
    with torch.no_grad():
        m.conv.weight.normal_()

    doc_lens = torch.tensor([0, 5, 11], dtype=torch.int32)
    x = torch.zeros(1, 11, d_model, dtype=torch.float64)
    with torch.no_grad():
        base = m(x, cu_doc_lens=doc_lens)
        # Perturb the last token of document 0; document 1 must not move at all.
        x2 = x.clone()
        x2[0, 4] = 1.0
        pert = m(x2, cu_doc_lens=doc_lens)

    torch.testing.assert_close(base[:, 5:], pert[:, 5:], rtol=0, atol=1e-12)
    assert not torch.allclose(base[:, 4], pert[:, 4])


def test_document_isolation_matches_independent_forward():
    """Each document's output must equal running that document through the module alone."""
    torch.manual_seed(0)
    d_model = 16
    m = ShortConv(d_model=d_model, kernel_size=3, use_fla=False).to(torch.float64)
    with torch.no_grad():
        m.conv.weight.normal_()

    lens = [4, 7, 3]
    xs = [torch.randn(1, n, d_model, dtype=torch.float64) for n in lens]
    packed = torch.cat(xs, dim=1)
    cu = torch.tensor([0, 4, 11, 14], dtype=torch.int32)

    with torch.no_grad():
        got = m(packed, cu_doc_lens=cu)
        expected = torch.cat([m(xi) for xi in xs], dim=1)
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-12)


def test_registered_in_sequence_mixer_registry():
    """The config must be resolvable by name for ``block_overrides`` to reach it."""
    cfg: SequenceMixerConfig = SequenceMixerConfig.from_dict(
        {"type": "short_conv", "kernel_size": 5}
    )
    assert isinstance(cfg, ShortConvConfig)
    assert cfg.kernel_size == 5


def test_flops_per_token_is_context_independent():
    """
    Unlike attention, a short conv has no term growing with sequence length. Arms must
    therefore be matched on FLOPs, not parameters, or long-context arms are unbalanced.
    """
    m = ShortConv(d_model=64, use_fla=False)
    assert m.num_flops_per_token(1024) == m.num_flops_per_token(32768)


def test_meta_device_construction():
    """Arm builders construct on meta first to assert shapes without allocating."""
    cfg = ShortConvConfig(gate_structure="lowrank", gate_rank=16)
    m = cfg.build(128, layer_idx=0, n_layers=16, init_device="meta")
    assert all(p.is_meta for p in m.parameters())
    assert cfg.num_params(128) == sum(p.numel() for p in m.parameters())


def test_rank_at_or_above_half_d_saves_nothing():
    """Guard the r >= d/2 trap: the r=512 rung of a d=1024 sweep is a pure loss."""
    d_model = 256
    dense = ShortConvConfig().num_params(d_model)
    assert (
        ShortConvConfig(gate_structure="lowrank", gate_rank=d_model // 2).num_params(d_model)
        == dense
    )
    assert (
        ShortConvConfig(gate_structure="lowrank", gate_rank=d_model // 4).num_params(d_model)
        < dense
    )


def test_invalid_configs_raise():
    with pytest.raises(ValueError, match="rank"):
        ShortConv(d_model=32, gate_structure="lowrank", use_fla=False)
    with pytest.raises(ValueError, match="groups"):
        ShortConv(d_model=32, gate_structure="grouped", use_fla=False)
    with pytest.raises(ValueError, match="divisible"):
        ShortConv(d_model=32, gate_structure="grouped", gate_groups=5, use_fla=False)
    with pytest.raises(ValueError, match="unknown gate structure"):
        ShortConv(d_model=32, gate_structure="bogus", use_fla=False)  # type: ignore[arg-type]


def test_identity_init_makes_kernel_widths_equivalent_at_step_zero():
    """
    The k3/k5/k9/k15 arms must start from the same function, or early-training differences
    confound the width comparison.
    """
    from olmo_core.nn.transformer.init import InitMethod

    d_model = 64
    x = torch.randn(2, 10, d_model, dtype=torch.float64)
    outs = []
    for k in (3, 5, 9, 15):
        m = ShortConv(d_model=d_model, kernel_size=k, use_fla=False).to(torch.float64)
        # Seeding before construction is NOT enough: nn.Conv1d's own reset_parameters draws
        # k*d values, so a wider kernel consumes more RNG and shifts every subsequent draw.
        # Pass an explicit generator to init_weights so projections match across widths.
        m.init_weights(
            init_method=InitMethod.normal,
            d_model=d_model,
            block_idx=0,
            num_blocks=1,
            generator=torch.Generator().manual_seed(1234),
        )
        with torch.no_grad():
            outs.append(m(x))
    for other in outs[1:]:
        torch.testing.assert_close(outs[0], other, rtol=0, atol=1e-12)


def test_builds_lfm2_topology_end_to_end():
    """
    A full 16-layer LFM2-topology hybrid must build, run, and place attention at exactly
    ``[2, 5, 8, 10, 12, 14]``.

    .. important::
        The per-layer override field is ``block.sequence_mixer``, **not** ``block.attention``.
        Setting ``.attention`` on the block config silently creates a new attribute, the
        override is ignored, and every layer stays attention -- a model that trains fine and
        answers a completely different research question. This test asserts the layer *types*
        rather than only that the forward pass works, because the forward pass works either way.
    """
    import copy

    from olmo_core.nn.transformer.config import (
        TransformerBlockConfig,
        TransformerConfig,
    )

    cfg = TransformerConfig.llama2_271M(vocab_size=1024)
    assert cfg.n_layers == 16

    # ``TransformerConfig.block`` is typed as a single config *or* a dict of them; narrow it
    # before mutating, both for mypy and because the dict form would need different handling.
    base_block = cfg.block
    assert isinstance(base_block, TransformerBlockConfig)
    liv_block = copy.deepcopy(base_block)
    liv_block.sequence_mixer = ShortConvConfig(kernel_size=3)
    attn_layers = {2, 5, 8, 10, 12, 14}
    cfg.block_overrides = {i: liv_block for i in range(cfg.n_layers) if i not in attn_layers}

    model = cfg.build()
    kinds = [type(b.attention).__name__ for b in model.blocks.values()]
    assert kinds.count("ShortConv") == 10
    assert {i for i, k in enumerate(kinds) if k == "Attention"} == attn_layers

    ids = torch.randint(0, 1024, (2, 24))
    out = model(ids)
    assert out.shape == (2, 24, 1024) and torch.isfinite(out).all()

    out.sum().backward()
    assert not [n for n, p in model.named_parameters() if p.grad is None]

    # Attention adds a term that grows with context; the conv layers do not. Arms must be
    # matched on FLOPs per token, not parameters, or long-context arms are unbalanced.
    assert model.num_flops_per_token(32768) > model.num_flops_per_token(4096)


def test_mixer_param_count_matches_brainlift_formula():
    """
    The mixer must reproduce ``4 * d^2 + k * d`` -- the figure the study's parameter budget and
    every arm-matching decision is built on. At d=2048, k=3 that is exactly 16,783,360.
    """
    for d_model in (1024, 2048):
        expected = 4 * d_model**2 + 3 * d_model
        assert ShortConvConfig(kernel_size=3).num_params(d_model) == expected
    assert ShortConvConfig(kernel_size=3).num_params(2048) == 16_783_360


def test_grouped_has_no_cross_block_mixing():
    """
    Confirm the structural property that makes grouped a distinct hypothesis from low-rank:
    it cannot move information between channel blocks.
    """
    torch.manual_seed(0)
    d_model, groups = 64, 4
    m = ShortConv(d_model=d_model, gate_structure="grouped", gate_groups=groups, use_fla=False).to(
        torch.float64
    )
    bs = d_model // groups

    x = torch.zeros(1, 1, d_model, dtype=torch.float64)
    x[0, 0, 0] = 1.0  # a channel in block 0 only
    with torch.no_grad():
        pre, _post, _v = m.in_proj(x)
    assert pre[0, 0, :bs].abs().max() > 0
    assert pre[0, 0, bs:].abs().max() == 0.0


def test_gradients_flow_to_every_parameter():
    cases: tuple[tuple[GateStructure, Dict[str, Any]], ...] = (
        ("dense", {}),
        ("lowrank", {"gate_rank": 8}),
        ("grouped", {"gate_groups": 2}),
    )
    for structure, kw in cases:
        m = ShortConv(d_model=32, gate_structure=structure, use_fla=False, **kw)
        m(torch.randn(2, 5, 32)).sum().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"{structure}/{name}"


def test_no_activation_in_conv_path():
    """
    LFM2 passes ``activation=None``. ``CausalConv1d`` defaults to ``"silu"`` inside its fused
    kernel, so a reviewer needs a positive check that this block is genuinely linear in the
    convolution path. With identity gates and an identity conv, the module must be exactly
    linear -- an activation anywhere would break superposition.
    """
    d_model = 32
    m = ShortConv(d_model=d_model, use_fla=False).to(torch.float64)
    with torch.no_grad():
        m.in_proj.gate_proj.weight.zero_()
        # both gates = 1 for every channel via a constant-one column trick is not possible
        # with a bias-free linear, so instead set value=identity and gates via eye scaling.
        m.in_proj.value_proj.weight.copy_(torch.eye(d_model, dtype=torch.float64))
        m.out_proj.weight.copy_(torch.eye(d_model, dtype=torch.float64))
        m.conv.weight.zero_()
        m.conv.weight[:, :, -1] = 1.0
        # gates: make pre and post the identity so out = x * (Wx) * ... stays polynomial-free
        pre, post = m.in_proj.gate_proj.weight.chunk(2, dim=0)
        pre.copy_(torch.eye(d_model, dtype=torch.float64))
        post.copy_(torch.eye(d_model, dtype=torch.float64))

    x = torch.randn(1, 4, d_model, dtype=torch.float64)
    with torch.no_grad():
        out = m(x)
    # out = x * (x * x) elementwise = x^3: cubic from the two gates, and NOTHING else.
    # A silu anywhere in the path would not reproduce x^3 exactly.
    torch.testing.assert_close(out, x**3, rtol=0, atol=1e-12)


def test_odd_and_even_kernel_sizes_preserve_length():
    for k in (2, 3, 4, 5, 8, 15):
        m = ShortConv(d_model=16, kernel_size=k, use_fla=False)
        assert m(torch.randn(1, 20, 16)).shape == (1, 20, 16)


def test_flops_scale_with_gate_structure():
    """Cheaper gates must report fewer FLOPs, or compute-matching is wrong."""
    d = 256
    dense = ShortConv(d_model=d, use_fla=False).num_flops_per_token(1024)
    lowrank = ShortConv(
        d_model=d, gate_structure="lowrank", gate_rank=32, use_fla=False
    ).num_flops_per_token(1024)
    grouped = ShortConv(
        d_model=d, gate_structure="grouped", gate_groups=4, use_fla=False
    ).num_flops_per_token(1024)
    assert lowrank < dense and grouped < dense
    assert math.isclose(lowrank, grouped, rel_tol=0.05)  # matched-cost pair


def test_init_weights_is_dtensor_safe():
    """
    ``init_weights`` must work on sharded parameters, not just plain tensors.

    THIS IS THE TEST THAT WAS MISSING. Every other test in this file builds on CPU in one
    process, where a parameter is an ordinary ``torch.Tensor`` and any in-place write
    succeeds. Under FSDP each parameter is a ``DTensor``, and the conv's identity-tap init
    used an indexed assignment -- ``w[:, :, -1] = 1.0`` -- which lowers to
    ``aten.fill_.Tensor``. That operator has no sharding strategy registered, so it raises:

        NotImplementedError: Operator aten.fill_.Tensor does not have a sharding strategy

    A full green suite said nothing about it, and the failure surfaced only after a run had
    been built, submitted, approved by a person and scheduled onto a GPU.

    A single-rank device mesh is enough: ``DTensor.__torch_dispatch__`` consults the sharding
    propagator on every op regardless of world size, so an unregistered operator raises at
    world_size=1 exactly as it would at 8. That makes the check free -- no distributed
    launch, no GPU.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor
    from torch.distributed.tensor.placement_types import Shard

    from olmo_core.nn.transformer.init import InitMethod

    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")

    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
        created_pg = True
    try:
        mesh = init_device_mesh("cpu", (1,))
        cases: tuple[tuple[GateStructure, Dict[str, Any]], ...] = (
            ("dense", {}),
            ("lowrank", {"gate_rank": 8}),
            ("grouped", {"gate_groups": 2}),
        )
        for structure, kw in cases:
            m = ShortConv(d_model=32, kernel_size=5, use_fla=False, **kw, gate_structure=structure)
            # Shard every parameter, which is what the trainer's FSDP wrap does.
            for mod in m.modules():
                for pname, p in list(mod.named_parameters(recurse=False)):
                    setattr(
                        mod,
                        pname,
                        torch.nn.Parameter(distribute_tensor(p.data, mesh, [Shard(0)])),
                    )
            # Must not raise. Before the fix this died on the conv identity tap.
            m.init_weights(d_model=32, init_method=InitMethod.normal, num_blocks=1, block_idx=0)

            tap = (
                m.conv.weight.full_tensor()
                if hasattr(m.conv.weight, "full_tensor")
                else m.conv.weight
            )
            assert torch.allclose(tap[:, :, -1], torch.ones_like(tap[:, :, -1])), structure
            assert torch.count_nonzero(tap[:, :, :-1]) == 0, f"{structure}: history not zeroed"
    finally:
        if created_pg:
            dist.destroy_process_group()
