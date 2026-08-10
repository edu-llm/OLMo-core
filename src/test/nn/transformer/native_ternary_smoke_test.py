"""Deferred reduced-Maple full-model smoke test for native packed TWN."""

import pytest
import torch
from torch.distributed._tensor import init_device_mesh
from torch.distributed.fsdp import fully_shard

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.moe.mlp import MoEMLP
from olmo_core.nn.quantization import QuantBackend, QuantConfig, QuantLinear
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.ops.ternary import native_packed_status
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.utils import get_default_device

requires_native_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and native_packed_status()["available"]),
    reason=f"native CUDA/Triton backend unavailable: {native_packed_status()['reason']}",
)


@pytest.mark.gpu
@requires_native_cuda
def test_reduced_maple_native_packed_full_model_forward_backward():
    """Exercise attention and capacity-MoE packed paths together in a real model."""
    quant = QuantConfig(
        enabled=True,
        backend=QuantBackend.native_packed,
        fallback_to_fake_quant=False,
    )
    config = TransformerConfig._maple_config(
        vocab_size=256,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=1,
        head_dim=32,
        num_experts=8,
        top_k=2,
        expert_hidden_size=32,
        quant=quant,
        dtype=DType.bfloat16,
        attn_backend=AttentionBackendName.torch,
        sliding_window=None,
    )
    model = config.build(init_device="meta")
    model.init_weights(
        device=torch.device("cuda"),
        max_seq_len=16,
        max_local_microbatch_size=32,
    )
    input_ids = torch.randint(0, 256, (2, 16), device="cuda")
    logits = model(input_ids=input_ids)
    assert logits.shape == (2, 16, 256)
    logits.float().square().mean().backward()
    assert all(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )


def _run_native_packed_ep_and_fsdp_smoke() -> None:
    device = get_default_device()
    quant = QuantConfig(
        enabled=True,
        backend=QuantBackend.native_packed,
        fallback_to_fake_quant=False,
    )

    ep_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("ep",))
    mlp = MoEMLP(
        d_model=32,
        hidden_size=16,
        num_experts=4,
        dtype=torch.bfloat16,
        init_device=device.type,
        quant=quant,
    )
    mlp.apply_ep(ep_mesh)
    local_x = torch.randn(2, 5, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    mlp(local_x).sum().backward()
    assert local_x.grad is not None

    fsdp_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
    layer = QuantLinear(
        32,
        16,
        backend=QuantBackend.native_packed,
        fallback_to_fake_quant=False,
        dtype=torch.bfloat16,
        device=device,
    )
    fully_shard(layer, mesh=fsdp_mesh)
    optim = torch.optim.SGD(layer.parameters(), lr=0.5)
    probe = torch.randn(8, 32, device=device, dtype=torch.bfloat16)
    outputs = []
    cache_misses = []
    for _ in range(2):
        optim.zero_grad(set_to_none=True)
        output = layer(probe)
        outputs.append(output.detach().clone())
        output.float().square().mean().backward()
        optim.step()
        cache_misses.append(layer._native_pack_cache.misses)
    assert cache_misses[1] > cache_misses[0]
    assert not torch.equal(outputs[0], outputs[1])


@pytest.mark.gpu
@requires_native_cuda
@requires_multi_gpu
def test_native_packed_ep_and_fsdp_cache_smoke():
    run_distributed_test(
        _run_native_packed_ep_and_fsdp_smoke,
        backend="nccl",
        start_method="spawn",
        world_size=2,
    )
