from test.nn.attention.attention_test import BF16_ATOL, BF16_RTOL
from typing import Any, Dict

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_full_tensor, get_rank, get_world_size
from olmo_core.nn.attention import (
    AttentionConfig,
    GatedDeltaNet2Config,
    GatedDeltaNetConfig,
)
from olmo_core.nn.attention.recurrent import GatedDeltaNet, GatedDeltaNet2
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.nn.functional import l2_normalize
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
    ],
)
def test_gated_delta_net_config_num_params(recurrent_config: GatedDeltaNetConfig):
    d_model = 512
    module = recurrent_config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in module.parameters())
    assert recurrent_config.num_params(d_model) == n_params


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
    gdn_cls: type = GatedDeltaNet,
):
    device = get_default_device()
    mesh = init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("cp",))

    gdn = gdn_cls(init_device=device.type, **gdn_kwargs)
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
# Gated DeltaNet-2 (GDN-2)       #
##################################


@requires_fla
@pytest.mark.parametrize(
    "recurrent_config",
    [
        pytest.param(GatedDeltaNet2Config(n_heads=8), id="default"),
        pytest.param(GatedDeltaNet2Config(n_heads=8, n_v_heads=16), id="GVA"),
        pytest.param(GatedDeltaNet2Config(n_heads=8, head_dim=32), id="head_dim=32"),
        pytest.param(GatedDeltaNet2Config(n_heads=8, expand_v=2.0), id="expand_v=2.0"),
        pytest.param(GatedDeltaNet2Config(n_heads=8, conv_size=8, conv_bias=True), id="conv_bias"),
        pytest.param(
            GatedDeltaNet2Config(n_heads=8, allow_neg_eigval=True), id="allow_neg_eigval=True"
        ),
    ],
)
def test_gated_delta_net_2_config_num_params(recurrent_config: GatedDeltaNet2Config):
    d_model = 512
    module = recurrent_config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # Make sure the estimated number of params matches the actual number of params.
    n_params = sum(p.numel() for p in module.parameters())
    assert recurrent_config.num_params(d_model) == n_params


@requires_fla
@pytest.mark.parametrize(
    "recurrent_config",
    [
        pytest.param(GatedDeltaNet2Config(n_heads=8), id="default"),
        pytest.param(GatedDeltaNet2Config(n_heads=8, n_v_heads=16), id="GVA"),
        pytest.param(GatedDeltaNet2Config(n_heads=8, expand_v=2.0), id="expand_v=2.0"),
    ],
)
def test_gated_delta_net_2_gate_shapes(recurrent_config: GatedDeltaNet2Config):
    """The whole point of GDN-2 is that the erase and write gates live on different axes."""
    d_model = 512
    module = recurrent_config.build(d_model, layer_idx=0, n_layers=12, init_device="meta")

    # Erase gate is channel-wise over K, write gate is channel-wise over V, and the decay
    # bias is per key channel rather than per head.
    assert module.w_b.weight.shape == (module.key_dim, d_model)
    assert module.w_w.weight.shape == (module.value_dim, d_model)
    assert module.dt_bias.shape == (module.key_dim,)
    assert module.A_log.shape == (module.n_heads,)


@requires_fla
def test_gdn2_kernel_contract_matches_paper():
    """The gates GDN-2 hands ``fla`` must mean what the paper says they mean.

    :class:`GatedDeltaNet2` passes ``g`` as an already-negative natural log-decay, ``b`` on the
    key axis and ``w`` on the value axis, and it cannot see whether the kernel agrees -- a
    release that reinterpreted any of the three would train a different model in silence rather
    than raise. So transcribe Eq. 9 of https://arxiv.org/abs/2605.22791 and compare.

    Runs on CPU: this is the reference recurrence, not the Triton path.
    """
    from fla.ops.gdn2 import naive_recurrent_gdn2

    seed_all(0)
    B, T, H, K, V = 2, 24, 4, 16, 16
    q = l2_normalize(torch.randn(B, T, H, K), dim=-1)  # the kernel does this internally
    k = l2_normalize(torch.randn(B, T, H, K), dim=-1)
    v = torch.randn(B, T, H, V)
    g = -F.softplus(torch.randn(B, T, H, K))  # log-decay: non-positive, natural log
    b = torch.rand(B, T, H, K)  # channel-wise erase gate on K
    w = torch.rand(B, T, H, V)  # channel-wise write gate on V

    # S_bar_t = Diag(exp(g_t)) S_{t-1};  r_t = S_bar_t^T (b_t * k_t)
    # S_t = S_bar_t + k_t ((w_t * v_t) - r_t)^T;  o_t = S_t^T q_t
    state = torch.zeros(B, H, K, V)
    expected = torch.zeros(B, T, H, V)
    for t in range(T):
        state = g[:, t].exp().unsqueeze(-1) * state
        read = torch.einsum("bhk,bhkv->bhv", b[:, t] * k[:, t], state)
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], (w[:, t] * v[:, t]) - read)
        expected[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t] * K**-0.5, state)

    actual, _ = naive_recurrent_gdn2(q=q, k=k, v=v, g=g, b=b, w=w)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    # Appendix A.5: tying both gates to one scalar beta must reduce to KDA.
    beta = torch.rand(B, T, H, 1)
    state = torch.zeros(B, H, K, V)
    kda = torch.zeros(B, T, H, V)
    for t in range(T):
        state = g[:, t].exp().unsqueeze(-1) * state
        resid = beta[:, t] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state))
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], resid)
        kda[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t] * K**-0.5, state)

    tied, _ = naive_recurrent_gdn2(
        q=q, k=k, v=v, g=g, b=beta.expand(B, T, H, K), w=beta.expand(B, T, H, V)
    )
    torch.testing.assert_close(tied, kda, rtol=1e-4, atol=1e-4)


@requires_fla
@requires_gpu
def test_gated_delta_net_2_fwd_bwd():
    device = "cuda"
    dtype = torch.bfloat16

    d_model, seq_len, batch_size = 256, 32, 2

    config = GatedDeltaNet2Config(n_heads=8)
    module = config.build(d_model, layer_idx=0, n_layers=12, init_device=device)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    with torch.autocast(device_type=device, dtype=dtype):
        y = module(x)
        assert y.shape == x.shape

        loss = y.sum()
        loss.backward()
    assert x.grad is not None
    # Both gate projections must actually take gradient, or one of the two decoupled gates is
    # silently dead and GDN-2 has degenerated into something else.
    assert module.w_b.weight.grad is not None and module.w_b.weight.grad.abs().sum() > 0
    assert module.w_w.weight.grad is not None and module.w_w.weight.grad.abs().sum() > 0


@requires_fla
def test_gated_delta_net_2_num_flops_per_token():
    d_model, n_heads, seq_len = 256, 2, 8192

    gdn2 = GatedDeltaNet2Config(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )
    attn = AttentionConfig(n_heads=n_heads).build(
        d_model, layer_idx=0, n_layers=1, init_device="meta"
    )

    # At long sequence lengths, recurrent layers use fewer FLOPs than quadratic attention.
    gdn2_flops = gdn2.num_flops_per_token(seq_len)
    attn_flops = attn.num_flops_per_token(seq_len)  # type: ignore
    assert 0 < gdn2_flops < attn_flops


@requires_multi_gpu
@requires_fla
def test_context_parallel_gdn2_ulysses(tmp_path):
    seed_all(0)
    device = get_default_device()

    # n_heads must be divisible by CP degree (world_size=2).
    gdn_kwargs: Dict[str, Any] = {"d_model": 128, "n_heads": 8}
    gdn = GatedDeltaNet2(init_device=device.type, **gdn_kwargs)

    bs, seq_len = 2, 64
    x = torch.randn(bs, seq_len, gdn_kwargs["d_model"], device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        y = gdn(x)

    outputs_path = tmp_path / "gdn2_y.pt"
    torch.save(y, outputs_path)
    inputs_path = tmp_path / "gdn2_x.pt"
    torch.save(x, inputs_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_model_and_optim_state(checkpoint_dir, gdn)

    run_distributed_test(
        _run_context_parallel_gdn_ulysses,
        backend="nccl",
        start_method="spawn",
        func_args=(checkpoint_dir, inputs_path, outputs_path, gdn_kwargs, GatedDeltaNet2),
    )
