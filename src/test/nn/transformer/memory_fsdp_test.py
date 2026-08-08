import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import fully_shard

import olmo_core.nn.memory.counterfactual as counterfactual_module
from olmo_core.nn.memory import LngramConfig
from olmo_core.nn.utils import selective_checkpointing_context_fn
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.testing.utils import requires_gpu
from olmo_core.utils import get_default_device


class _ZeroConv(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _run_lngram_with_fsdp2() -> None:
    device = get_default_device()
    memory = LngramConfig(memory_dim=61, require_triton=True).build(
        8,
        init_device=device.type,
        dtype=torch.bfloat16,
    )
    memory.conv = _ZeroConv()
    wrapped = checkpoint_wrapper(
        memory,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=False,
    )
    wrapped.compile(fullgraph=False)
    fully_shard(wrapped)

    dispatches: list[bool] = []
    original = counterfactual_module.try_counterfactual_grad_z

    def track_dispatch(*args, **kwargs):
        result = original(*args, **kwargs)
        dispatches.append(result is not None)
        return result

    counterfactual_module.try_counterfactual_grad_z = track_dispatch
    try:
        x = torch.randn(
            2,
            4,
            8,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        wrapped(x).float().square().mean().backward()
    finally:
        counterfactual_module.try_counterfactual_grad_z = original

    assert dispatches == [True]
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@requires_multi_gpu
def test_lngram_triton_with_fsdp2() -> None:
    run_distributed_test(
        _run_lngram_with_fsdp2,
        backend="nccl",
        start_method="spawn",
    )


@requires_gpu
def test_lngram_triton_with_compile_and_activation_checkpointing() -> None:
    memory = LngramConfig(memory_dim=61, require_triton=True).build(
        8,
        init_device="cuda",
        dtype=torch.bfloat16,
    )
    memory.conv = _ZeroConv()
    wrapped = checkpoint_wrapper(
        memory,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=False,
    )
    wrapped.compile(fullgraph=False)

    x = torch.randn(
        2,
        4,
        8,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    wrapped(x).float().square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
