import math
from dataclasses import replace
from test.nn.attention.attention_test import BF16_ATOL, BF16_RTOL
from typing import Any, Dict, cast
from unittest import mock

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.nn import functional as F

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_full_tensor, get_rank, get_world_size
from olmo_core.nn.attention import (
    AttentionConfig,
    GatedDeltaNetConfig,
    KimiDeltaAttentionConfig,
    KimiDeltaHouseholderConfig,
)
from olmo_core.nn.attention.base import SequenceMixerConfig
from olmo_core.nn.attention.recurrent import (
    GatedDeltaNet,
    KimiDeltaAttention,
    KimiDeltaHouseholder,
)
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.testing import requires_gpu, run_distributed_test
from olmo_core.testing.utils import requires_fla, requires_multi_gpu
from olmo_core.utils import get_default_device, seed_all


@requires_fla
@pytest.mark.parametrize(
    "recurrent_config",
    [
        pytest.param(GatedDeltaNetConfig(n_heads=8), id="default"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, n_v_heads=16), id="GVA"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, head_dim=32), id="head_dim=32"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, expand_v=1.0), id="expand_v=1.0"),
        pytest.param(GatedDeltaNetConfig(n_heads=8, conv_size=8, conv_bias=True), id="conv_bias"),
        pytest.param(
            GatedDeltaNetConfig(n_heads=8, allow_neg_eigval=False), id="allow_neg_eigval=False"
        ),
        pytest.param(
            GatedDeltaNetConfig(n_heads=8, gate_init="halflife"),
            id='gate_init="halflife"',
        ),
        pytest.param(
            GatedDeltaNetConfig(n_heads=8, gate_init="halflife_random"),
            id='gate_init="halflife_random"',
        ),
        pytest.param(
            GatedDeltaNetConfig(n_heads=8, gate_init="halflife_random_a"),
            id='gate_init="halflife_random_a"',
        ),
        pytest.param(
            GatedDeltaNetConfig(n_heads=8, gate_init="halflife_a"),
            id='gate_init="halflife_a"',
        ),
        pytest.param(
            GatedDeltaNetConfig(n_heads=8, gate_init="halflife_a_permuted"),
            id='gate_init="halflife_a_permuted"',
        ),
    ],
)
def test_gated_delta_net_config_num_params(recurrent_config: GatedDeltaNetConfig):
    d_model = 512
    module = recurrent_config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in module.parameters())
    assert recurrent_config.num_params(d_model) == n_params


@requires_fla
def test_gated_delta_net_halflife_gate_init():
    d_model = 512
    min_halflife = 16.0
    max_halflife = 4096.0
    config = GatedDeltaNetConfig(
        n_heads=8,
        gate_init="halflife",
        gate_min_halflife=min_halflife,
        gate_max_halflife=max_halflife,
    )
    module = config.build(d_model, layer_idx=0, n_layers=12)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=d_model,
        block_idx=0,
        num_blocks=12,
    )

    decay_rate = module.A_log.float().exp() * F.softplus(module.dt_bias.float())
    half_life = math.log(2.0) / decay_rate
    expected = torch.logspace(
        math.log10(min_halflife),
        math.log10(max_halflife),
        steps=module.n_v_heads,
        device=half_life.device,
    )

    torch.testing.assert_close(half_life, expected, rtol=1e-4, atol=1e-4)


@requires_fla
def test_gated_delta_net_halflife_random_a_gate_init():
    d_model = 512
    min_halflife = 0.5
    max_halflife = 128.0
    config = GatedDeltaNetConfig(
        n_heads=8,
        gate_init="halflife_random_a",
        gate_min_halflife=min_halflife,
        gate_max_halflife=max_halflife,
    )
    module = config.build(d_model, layer_idx=0, n_layers=12)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=d_model,
        block_idx=0,
        num_blocks=12,
        generator=torch.Generator().manual_seed(123),
    )

    A = module.A_log.float().exp()
    decay_rate = A * F.softplus(module.dt_bias.float())
    half_life = math.log(2.0) / decay_rate

    assert torch.isfinite(half_life).all()
    assert half_life.min() >= min_halflife * (1 - 1e-3)
    assert half_life.max() <= max_halflife * (1 + 1e-3)
    assert not torch.allclose(A, torch.ones_like(A))


@requires_fla
def test_gated_delta_net_halflife_a_gate_init():
    d_model = 512
    min_halflife = 0.5
    max_halflife = 128.0
    config = GatedDeltaNetConfig(
        n_heads=8,
        gate_init="halflife_a",
        gate_min_halflife=min_halflife,
        gate_max_halflife=max_halflife,
    )
    module = config.build(d_model, layer_idx=0, n_layers=12)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=d_model,
        block_idx=0,
        num_blocks=12,
        generator=torch.Generator().manual_seed(123),
    )

    A = module.A_log.float().exp()
    decay_rate = A * F.softplus(module.dt_bias.float())
    half_life = math.log(2.0) / decay_rate
    expected = torch.logspace(
        math.log10(min_halflife),
        math.log10(max_halflife),
        steps=module.n_v_heads,
        device=half_life.device,
    )

    torch.testing.assert_close(half_life, expected, rtol=1e-3, atol=1e-3)
    assert not torch.allclose(A, torch.ones_like(A))


@requires_fla
@requires_gpu
def test_gated_delta_net_fwd_bwd():
    device = "cuda"
    dtype = torch.bfloat16

    d_model, seq_len, batch_size = 256, 32, 2

    config = GatedDeltaNetConfig(n_heads=8)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)
        assert y.shape == x.shape

        loss = y.sum()
        loss.backward()
    assert x.grad is not None


@requires_fla
def test_gated_delta_net_num_flops_per_token():
    d_model, n_heads, seq_len = 256, 2, 8192

    gdn = GatedDeltaNetConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )
    attn = AttentionConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )

    # At long sequence lengths, recurrent layers use fewer FLOPs than quadratic attention.
    gdn_flops = gdn.num_flops_per_token(seq_len)
    attn_flops = attn.num_flops_per_token(seq_len)  # type: ignore
    assert 0 < gdn_flops < attn_flops


def _run_context_parallel_gdn_ulysses(
    checkpoint_dir: str,
    inputs_path: str,
    outputs_path: str,
    gdn_kwargs: Dict[str, Any],
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("cp",))

    gdn = GatedDeltaNet(init_device=device.type, **gdn_kwargs)
    gdn.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    load_model_and_optim_state(checkpoint_dir, gdn)

    # Load the input and split it across ranks on the sequence dimension.
    x = torch.load(inputs_path, map_location=device)
    rank, world_size = get_rank(), get_world_size()
    chunk_size = x.size(1) // world_size
    x_local = x[:, rank * chunk_size : (rank + 1) * chunk_size, :]

    with torch.autocast(device.type, dtype=x_local.dtype):
        local_y = gdn(x_local)
    y = DTensor.from_local(local_y, mesh, (Shard(1),))

    og_y = torch.load(outputs_path, map_location=device)
    tol_scale = 2  # requires slightly more tolerance than default
    torch.testing.assert_close(
        og_y, get_full_tensor(y), rtol=BF16_RTOL * tol_scale, atol=BF16_ATOL * tol_scale
    )


@requires_multi_gpu
@requires_fla
def test_context_parallel_gdn_ulysses(tmp_path):
    seed_all(0)
    device = get_default_device()

    # n_heads must be divisible by CP degree (world_size=2).
    gdn_kwargs: Dict[str, Any] = {"d_model": 128, "n_heads": 8}
    gdn = GatedDeltaNet(init_device=device.type, **gdn_kwargs)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, gdn_kwargs["d_model"], device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        y = gdn(x)

    outputs_path = tmp_path / "gdn_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "gdn_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, gdn)

    run_distributed_test(
        _run_context_parallel_gdn_ulysses,
        backend="nccl",
        start_method="spawn",
        func_args=(checkpoint_dir, inputs_path, outputs_path, gdn_kwargs),
    )


##################################
# Kimi Delta Attention (KDA)     #
##################################


def _mock_cp_mesh(size: int) -> DeviceMesh:
    """
    A stand-in for a :class:`DeviceMesh` of the given size, so we can exercise ``apply_cp()``
    without initializing a process group.
    """
    mesh = mock.Mock(spec=DeviceMesh)
    mesh.size.return_value = size
    return cast(DeviceMesh, mesh)


@requires_fla
@pytest.mark.parametrize(
    "recurrent_config",
    [
        pytest.param(KimiDeltaAttentionConfig(n_heads=8), id="default"),
        pytest.param(KimiDeltaAttentionConfig(n_heads=8, n_v_heads=16), id="GVA"),
        pytest.param(KimiDeltaAttentionConfig(n_heads=8, head_dim=32), id="head_dim=32"),
        pytest.param(KimiDeltaAttentionConfig(n_heads=8, expand_v=2.0), id="expand_v=2.0"),
        pytest.param(
            KimiDeltaAttentionConfig(n_heads=8, conv_size=8, conv_bias=True), id="conv_bias"
        ),
        pytest.param(
            KimiDeltaAttentionConfig(n_heads=8, allow_neg_eigval=True), id="allow_neg_eigval=True"
        ),
    ],
)
def test_kimi_delta_attention_config_num_params(recurrent_config: KimiDeltaAttentionConfig):
    d_model = 512
    module = recurrent_config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in module.parameters())
    assert recurrent_config.num_params(d_model) == n_params


def test_kimi_delta_attention_config_round_trip():
    config = KimiDeltaAttentionConfig(n_heads=8, n_v_heads=16, head_dim=32, expand_v=2.0)
    config_dict = config.as_config_dict()
    assert config_dict["type"] == "kimi_delta_attention"

    round_tripped: SequenceMixerConfig = SequenceMixerConfig.from_dict(config_dict)
    assert isinstance(round_tripped, KimiDeltaAttentionConfig)
    assert round_tripped == config


@requires_fla
def test_kimi_delta_attention_build():
    d_model, n_heads, head_dim = 256, 4, 64

    config = KimiDeltaAttentionConfig(n_heads=n_heads, head_dim=head_dim)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    assert isinstance(module, KimiDeltaAttention)
    assert module.head_k_dim == head_dim
    # 'expand_v' defaults to 1.0 for KDA, unlike GatedDeltaNet.
    assert module.head_v_dim == head_dim
    assert module.key_dim == n_heads * head_dim
    assert module.value_dim == n_heads * head_dim
    # The gate/output projections are low-rank bottlenecks through 'head_v_dim'.
    assert module.f_proj[0].out_features == module.head_v_dim  # type: ignore[index]
    assert module.f_proj[1].out_features == module.key_dim  # type: ignore[index]
    assert module.f_proj[1].bias is None  # type: ignore[index]
    assert module.g_proj[0].out_features == module.head_v_dim  # type: ignore[index]
    assert module.g_proj[1].out_features == module.value_dim  # type: ignore[index]
    assert module.g_proj[1].bias is not None  # type: ignore[index]
    assert module.o_norm.activation == "sigmoid"


@requires_fla
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("n_heads", [4, 8])
def test_kimi_delta_attention_gate_param_shapes(n_heads: int, head_dim: int):
    """
    ``A_log`` is a *per-head* scalar while ``dt_bias`` is *per-channel* and flat, which is what
    the fused KDA gate kernel expects. Guards against the ``[H, K]`` mistake.
    """
    d_model = 512

    config = KimiDeltaAttentionConfig(n_heads=n_heads, head_dim=head_dim)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    assert module.A_log.shape == (n_heads,)
    assert module.dt_bias.shape == (n_heads * head_dim,)


@requires_fla
def test_kimi_delta_attention_gate_init():
    d_model = 512

    config = KimiDeltaAttentionConfig(n_heads=8)
    module = config.build(d_model, layer_idx=0, n_layers=12)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=d_model,
        block_idx=0,
        num_blocks=12,
        generator=torch.Generator().manual_seed(123),
    )

    # Reference KDA init: 'A_log = log(U(1, 16))' and 'dt_bias = 0'.
    A = module.A_log.float().exp()
    assert torch.isfinite(A).all()
    assert A.min() >= 1.0 - 1e-4
    assert A.max() <= 16.0 + 1e-4
    torch.testing.assert_close(module.dt_bias, torch.zeros_like(module.dt_bias))


@requires_fla
def test_kimi_delta_attention_no_weight_decay_param_names():
    """
    ``A_log``/``dt_bias`` should be reachable by the ``*.A_log*`` / ``*.dt_bias*`` globs used to
    exclude them from weight decay.
    """
    from fnmatch import fnmatch

    module = KimiDeltaAttentionConfig(n_heads=8).build(
        512, layer_idx=0, n_layers=12, init_device="meta"
    )
    names = {f"blocks.0.attention.{name}" for name, _ in module.named_parameters()}

    assert {n for n in names if fnmatch(n, "*.A_log*")} == {"blocks.0.attention.A_log"}
    assert {n for n in names if fnmatch(n, "*.dt_bias*")} == {"blocks.0.attention.dt_bias"}


@requires_fla
def test_kimi_delta_attention_apply_tp_raises():
    module = KimiDeltaAttentionConfig(n_heads=8).build(
        512, layer_idx=0, n_layers=12, init_device="meta"
    )
    with pytest.raises(NotImplementedError):
        module.apply_tp(_mock_cp_mesh(2))


@requires_fla
def test_kimi_delta_attention_apply_cp():
    module = KimiDeltaAttentionConfig(n_heads=8).build(
        512, layer_idx=0, n_layers=12, init_device="meta"
    )

    # A CP world size of 1 is a no-op.
    assert module.apply_cp(_mock_cp_mesh(1), uly=UlyssesContextParallelStyle()) is None

    with pytest.raises(NotImplementedError):
        module.apply_cp(_mock_cp_mesh(2), uly=UlyssesContextParallelStyle())


@requires_fla
def test_kimi_delta_attention_num_flops_per_token():
    d_model, n_heads, seq_len = 256, 2, 8192

    kda = KimiDeltaAttentionConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )
    attn = AttentionConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )

    # At long sequence lengths, recurrent layers use fewer FLOPs than quadratic attention.
    kda_flops = kda.num_flops_per_token(seq_len)
    attn_flops = attn.num_flops_per_token(seq_len)  # type: ignore
    assert 0 < kda_flops < attn_flops


@requires_fla
@requires_gpu
def test_kimi_delta_attention_rejects_batched_cu_doc_lens():
    device = "cuda"
    d_model = 256

    module = KimiDeltaAttentionConfig(n_heads=4, head_dim=64).build(
        d_model, layer_idx=0, n_layers=1, init_device=device
    )

    x = torch.randn(2, 32, d_model, device=device, dtype=torch.bfloat16)
    cu_doc_lens = torch.tensor([0, 16, 32], dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError, match="batch size of 1"):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            module(x, cu_doc_lens=cu_doc_lens)


@requires_fla
@requires_gpu
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(KimiDeltaAttentionConfig(n_heads=4, head_dim=64), id="default"),
        pytest.param(
            KimiDeltaAttentionConfig(n_heads=4, head_dim=64, n_v_heads=8),
            id="GVA",
        ),
        pytest.param(
            KimiDeltaAttentionConfig(n_heads=4, head_dim=64, expand_v=2.0),
            id="expand_v=2.0",
        ),
    ],
)
def test_kimi_delta_attention_fwd_bwd(config: KimiDeltaAttentionConfig):
    device = "cuda"
    dtype = torch.bfloat16

    d_model, seq_len, batch_size = 256, 128, 2

    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)
    module.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=12)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    # NOTE: the config dtype defaults to float32, so autocast is required to mix fp32 params with
    # a bf16 input.
    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)
        assert y.shape == x.shape

        loss = y.float().sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Every parameter, including the gate parameters, should get finite grads.
    for name, p in module.named_parameters():
        assert p.grad is not None, f"no grad for '{name}'"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for '{name}'"
    assert module.A_log.grad is not None
    assert module.dt_bias.grad is not None


@requires_fla
@requires_gpu
def test_dispatch_chunk_kda_matches_naive():
    """
    Check ``dispatch_chunk_kda`` against the naive recurrent KDA oracle from ``fla``, for both the
    fused-gate path (raw gate input + ``A_log``/``dt_bias``) and the precomputed-gate path.
    """
    from fla.modules.l2norm import l2norm
    from fla.ops.kda.gate import fused_kda_gate
    from fla.ops.kda.naive import naive_recurrent_kda

    from olmo_core.nn.attention.flash_linear_attn_api import dispatch_chunk_kda

    seed_all(0)
    device = "cuda"
    dtype = torch.bfloat16
    B, T, H, K, V = 2, 128, 4, 64, 64

    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    raw = torch.randn(B, T, H, K, device=device, dtype=dtype)
    beta = torch.randn(B, T, H, device=device, dtype=dtype).sigmoid()
    A_log = torch.empty(H, device=device, dtype=torch.float32).uniform_(1, 16).log()
    dt_bias = torch.randn(H * K, device=device, dtype=torch.float32)

    # The naive oracle takes a *precomputed* gate and un-normalized q/k, so mirror what the
    # kernel does internally.
    g = fused_kda_gate(raw, A_log, dt_bias)
    o_naive, _ = naive_recurrent_kda(q=l2norm(q), k=l2norm(k), v=v, g=g, beta=beta)

    o_fused, final_state = dispatch_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
    )
    assert final_state is None
    assert o_fused.shape == (B, T, H, V)

    o_precomputed, _ = dispatch_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
    )

    torch.testing.assert_close(o_fused.float(), o_naive.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(o_precomputed.float(), o_naive.float(), atol=2e-2, rtol=2e-2)


@requires_fla
@requires_gpu
def test_kimi_delta_attention_matches_naive():
    """
    End-to-end check of :class:`KimiDeltaAttention` against a reconstruction of its forward pass
    built on the naive recurrent KDA oracle.
    """
    from fla.modules.l2norm import l2norm
    from fla.ops.kda.gate import fused_kda_gate
    from fla.ops.kda.naive import naive_recurrent_kda

    seed_all(0)
    device = "cuda"
    dtype = torch.bfloat16
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


##########################################
# KDA + R Householder factors            #
##########################################


@requires_fla
@pytest.mark.parametrize("num_householder", [1, 2, 4])
@pytest.mark.parametrize(
    "recurrent_config",
    [
        pytest.param(KimiDeltaHouseholderConfig(n_heads=8), id="default"),
        pytest.param(KimiDeltaHouseholderConfig(n_heads=8, n_v_heads=16), id="GVA"),
        pytest.param(KimiDeltaHouseholderConfig(n_heads=8, head_dim=32), id="head_dim=32"),
        pytest.param(KimiDeltaHouseholderConfig(n_heads=8, expand_v=2.0), id="expand_v=2.0"),
        pytest.param(
            KimiDeltaHouseholderConfig(n_heads=8, conv_size=8, conv_bias=True), id="conv_bias"
        ),
        pytest.param(
            KimiDeltaHouseholderConfig(n_heads=8, allow_neg_eigval=True), id="allow_neg_eigval=True"
        ),
    ],
)
def test_kimi_delta_householder_config_num_params(
    recurrent_config: KimiDeltaHouseholderConfig, num_householder: int
):
    d_model = 512
    config = replace(recurrent_config, num_householder=num_householder)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in module.parameters())
    assert config.num_params(d_model) == n_params


@requires_fla
def test_kimi_delta_householder_r1_matches_kda_params():
    """
    At ``num_householder=1`` nothing is widened, so the layer must have exactly the same
    parameter count as :class:`KimiDeltaAttention`.
    """
    d_model = 512
    kwargs: Dict[str, Any] = dict(n_heads=8, head_dim=64, expand_v=1.0)

    kda = KimiDeltaAttentionConfig(**kwargs)
    householder = KimiDeltaHouseholderConfig(num_householder=1, **kwargs)
    assert householder.num_params(d_model) == kda.num_params(d_model)

    # ...and the same FLOPs per token: the extra recurrent work is 'R'-proportional.
    kda_module = kda.build(d_model, layer_idx=0, n_layers=1, init_device="meta")
    hh_module = householder.build(d_model, layer_idx=0, n_layers=1, init_device="meta")
    assert hh_module.num_flops_per_token(8192) == kda_module.num_flops_per_token(8192)


def test_kimi_delta_householder_config_round_trip():
    config = KimiDeltaHouseholderConfig(
        n_heads=8, num_householder=3, n_v_heads=16, head_dim=32, expand_v=2.0
    )
    config_dict = config.as_config_dict()
    assert config_dict["type"] == "kimi_delta_householder"
    assert config_dict["num_householder"] == 3

    round_tripped: SequenceMixerConfig = SequenceMixerConfig.from_dict(config_dict)
    assert isinstance(round_tripped, KimiDeltaHouseholderConfig)
    assert round_tripped == config


@requires_fla
def test_kimi_delta_householder_build():
    d_model, n_heads, head_dim, R = 256, 4, 64, 2

    config = KimiDeltaHouseholderConfig(n_heads=n_heads, head_dim=head_dim, num_householder=R)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    assert isinstance(module, KimiDeltaHouseholder)
    assert module.num_householder == R
    assert module.head_k_dim == head_dim
    # 'expand_v' defaults to 1.0, as for KDA.
    assert module.head_v_dim == head_dim
    assert module.key_dim == n_heads * head_dim
    assert module.value_dim == n_heads * head_dim
    # The gate/output projections are low-rank bottlenecks through 'head_v_dim' and are *not*
    # widened by 'R' -- the gate is per token, not per factor.
    assert module.f_proj[0].out_features == module.head_v_dim  # type: ignore[index]
    assert module.f_proj[1].out_features == module.key_dim  # type: ignore[index]
    assert module.f_proj[1].bias is None  # type: ignore[index]
    assert module.g_proj[0].out_features == module.head_v_dim  # type: ignore[index]
    assert module.g_proj[1].out_features == module.value_dim  # type: ignore[index]
    assert module.g_proj[1].bias is not None  # type: ignore[index]
    assert module.o_norm.activation == "sigmoid"


@requires_fla
@pytest.mark.parametrize("num_householder", [1, 2, 4])
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("n_heads", [4, 8])
def test_kimi_delta_householder_gate_param_shapes(
    n_heads: int, head_dim: int, num_householder: int
):
    """
    The decay is applied *once per token*, shared by that token's ``R`` factors, so neither
    ``A_log`` (per head) nor ``dt_bias`` (per head/key-channel, flat) scales with ``R``.
    """
    d_model = 512

    config = KimiDeltaHouseholderConfig(
        n_heads=n_heads, head_dim=head_dim, num_householder=num_householder
    )
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    assert module.A_log.shape == (n_heads,)
    assert module.dt_bias.shape == (n_heads * head_dim,)
    # The forget-gate projection is per token too.
    assert module.f_proj[1].out_features == n_heads * head_dim  # type: ignore[index]


@requires_fla
@pytest.mark.parametrize("num_householder", [1, 2, 4])
def test_kimi_delta_householder_widths_scale_with_r(num_householder: int):
    """
    ``w_k`` / ``w_v`` / ``w_b`` and the ``k`` / ``v`` convolutions produce ``R`` factors per
    token; ``w_q``, ``q_conv1d`` and ``w_out`` do not.
    """
    d_model, n_heads, head_dim, R = 512, 8, 64, num_householder
    key_dim = n_heads * head_dim

    config = KimiDeltaHouseholderConfig(
        n_heads=n_heads, head_dim=head_dim, expand_v=1.0, num_householder=R
    )
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # Key side: widened by R.
    assert module.w_k.out_features == R * key_dim
    assert module.w_v.out_features == R * module.value_dim
    assert module.w_b.out_features == R * n_heads
    assert module.k_conv1d.hidden_size == R * key_dim
    assert module.v_conv1d.hidden_size == R * module.value_dim

    # Query side and output: unchanged by R.
    assert module.w_q.out_features == key_dim
    assert module.q_conv1d.hidden_size == key_dim
    assert module.w_out.in_features == module.value_dim
    assert module.w_out.out_features == d_model
    assert module.o_norm.weight.shape == (module.head_v_dim,)


@requires_fla
def test_kimi_delta_householder_rejects_bad_num_householder():
    with pytest.raises(AssertionError):
        KimiDeltaHouseholderConfig(n_heads=8, num_householder=0).build(
            512, layer_idx=0, n_layers=12, init_device="meta"
        )


@requires_fla
def test_kimi_delta_householder_gate_init():
    d_model = 512

    config = KimiDeltaHouseholderConfig(n_heads=8, num_householder=2)
    module = config.build(d_model, layer_idx=0, n_layers=12)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=d_model,
        block_idx=0,
        num_blocks=12,
        generator=torch.Generator().manual_seed(123),
    )

    # Reference KDA init: 'A_log = log(U(1, 16))' and 'dt_bias = 0'.
    A = module.A_log.float().exp()
    assert torch.isfinite(A).all()
    assert A.min() >= 1.0 - 1e-4
    assert A.max() <= 16.0 + 1e-4
    torch.testing.assert_close(module.dt_bias, torch.zeros_like(module.dt_bias))

    # Every other parameter should be finite too.
    for name, p in module.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite init for '{name}'"


@requires_fla
def test_kimi_delta_householder_no_weight_decay_param_names():
    """
    ``A_log``/``dt_bias`` should be reachable by the ``*.A_log*`` / ``*.dt_bias*`` globs used to
    exclude them from weight decay.
    """
    from fnmatch import fnmatch

    module = KimiDeltaHouseholderConfig(n_heads=8, num_householder=2).build(
        512, layer_idx=0, n_layers=12, init_device="meta"
    )
    names = {f"blocks.0.attention.{name}" for name, _ in module.named_parameters()}

    assert {n for n in names if fnmatch(n, "*.A_log*")} == {"blocks.0.attention.A_log"}
    assert {n for n in names if fnmatch(n, "*.dt_bias*")} == {"blocks.0.attention.dt_bias"}


@requires_fla
def test_kimi_delta_householder_apply_tp_raises():
    module = KimiDeltaHouseholderConfig(n_heads=8, num_householder=2).build(
        512, layer_idx=0, n_layers=12, init_device="meta"
    )
    with pytest.raises(NotImplementedError):
        module.apply_tp(_mock_cp_mesh(2))


@requires_fla
def test_kimi_delta_householder_apply_cp():
    module = KimiDeltaHouseholderConfig(n_heads=8, num_householder=2).build(
        512, layer_idx=0, n_layers=12, init_device="meta"
    )

    # A CP world size of 1 is a no-op.
    assert module.apply_cp(_mock_cp_mesh(1), uly=UlyssesContextParallelStyle()) is None

    # Anything larger is rejected: the interleaved 'T * R' layout doesn't line up with the
    # Ulysses all-to-all.
    with pytest.raises(NotImplementedError, match="T \\* R"):
        module.apply_cp(_mock_cp_mesh(2), uly=UlyssesContextParallelStyle())


@requires_fla
@pytest.mark.parametrize("num_householder", [1, 2, 4])
def test_kimi_delta_householder_num_flops_per_token(num_householder: int):
    d_model, n_heads, seq_len = 256, 2, 8192

    householder = KimiDeltaHouseholderConfig(
        n_heads=n_heads, num_householder=num_householder
    ).build(d_model, layer_idx=0, n_layers=1, init_device="meta")
    attn = AttentionConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )

    # At long sequence lengths, recurrent layers use fewer FLOPs than quadratic attention.
    householder_flops = householder.num_flops_per_token(seq_len)
    attn_flops = attn.num_flops_per_token(seq_len)  # type: ignore
    assert 0 < householder_flops < attn_flops

    # FLOPs must be strictly increasing in R.
    if num_householder > 1:
        smaller = KimiDeltaHouseholderConfig(n_heads=n_heads, num_householder=1).build(
            d_model, layer_idx=0, n_layers=1, init_device="meta"
        )
        assert householder_flops > smaller.num_flops_per_token(seq_len)


@requires_fla
def test_kimi_delta_householder_interleave_ordering():
    """
    The ``[B, T, R * H * D] -> [B, T * R, H, D]`` reshape used in the forward pass must place the
    ``R`` factors of token ``t`` at interleaved positions ``t * R + r``, which is what the kernel
    indexes. This is the einops ``'b t (n h d) -> b (t n) h d'`` layout.
    """
    B, T, R, H, D = 2, 3, 3, 2, 4
    x = torch.arange(B * T * R * H * D).reshape(B, T, R * H * D)

    actual = x.view(B, T, R, H, D).reshape(B, T * R, H, D)

    expected = torch.empty(B, T * R, H, D, dtype=x.dtype)
    for b in range(B):
        for t in range(T):
            for r in range(R):
                for h in range(H):
                    for d in range(D):
                        expected[b, t * R + r, h, d] = x[b, t, r * (H * D) + h * D + d]

    torch.testing.assert_close(actual, expected)

    # And the same for the 2-D 'beta' path.
    beta = torch.arange(B * T * R * H).reshape(B, T, R * H)
    actual_beta = beta.view(B, T, R, H).reshape(B, T * R, H)
    expected_beta = torch.empty(B, T * R, H, dtype=beta.dtype)
    for b in range(B):
        for t in range(T):
            for r in range(R):
                for h in range(H):
                    expected_beta[b, t * R + r, h] = beta[b, t, r * H + h]
    torch.testing.assert_close(actual_beta, expected_beta)


@requires_fla
@requires_gpu
def test_kimi_delta_householder_rejects_batched_cu_doc_lens():
    device = "cuda"
    d_model = 256

    module = KimiDeltaHouseholderConfig(n_heads=4, head_dim=64, num_householder=2).build(
        d_model, layer_idx=0, n_layers=1, init_device=device
    )

    x = torch.randn(2, 32, d_model, device=device, dtype=torch.bfloat16)
    cu_doc_lens = torch.tensor([0, 16, 32], dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError, match="batch size of 1"):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            module(x, cu_doc_lens=cu_doc_lens)


@requires_fla
@requires_gpu
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            KimiDeltaHouseholderConfig(n_heads=4, head_dim=64, num_householder=2), id="default"
        ),
        pytest.param(
            KimiDeltaHouseholderConfig(n_heads=4, head_dim=64, num_householder=2, n_v_heads=8),
            id="GVA",
        ),
        pytest.param(
            KimiDeltaHouseholderConfig(n_heads=4, head_dim=64, num_householder=2, expand_v=2.0),
            id="expand_v=2.0",
        ),
    ],
)
def test_kimi_delta_householder_fwd(config: KimiDeltaHouseholderConfig):
    """
    Forward smoke test at ``R = 2``: the layer must produce a finite output of the input shape.
    The backward is covered separately by :func:`test_kimi_delta_householder_backward_runs`.
    """
    pytest.importorskip("triton")
    device = "cuda"
    dtype = torch.bfloat16

    d_model, seq_len, batch_size = 256, 128, 2

    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)
    module.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=12)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

    # NOTE: the config dtype defaults to float32, so autocast is required to mix fp32 params with
    # a bf16 input. The kernel also rejects a float32 'q' outright.
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype):
            y = module(x)

    assert y.shape == (batch_size, seq_len, d_model)
    assert torch.isfinite(y.float()).all()


@requires_fla
@requires_gpu
def test_kimi_delta_householder_backward_runs():
    """
    Gradients flow through the layer via the triton backward kernel.

    This previously asserted ``NotImplementedError``; the backward is now implemented and
    validated against the ``torch`` backend (see ``probes/gpu_bwd_accept.py``).
    """
    pytest.importorskip("triton")
    device = "cuda"
    dtype = torch.bfloat16
    d_model, seq_len, batch_size = 256, 32, 1

    module = KimiDeltaHouseholderConfig(n_heads=4, head_dim=64, num_householder=2).build(
        d_model, layer_idx=0, n_layers=1, init_device=device
    )
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)
    y.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
