import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Replicate

from olmo_core.nn.transformer.block import (
    MoEEngramReorderedNormTransformerBlock,
    MoELngramReorderedNormTransformerBlock,
    MoEReorderedNormTransformerBlock,
)


class _TinyAttention(nn.Module):
    flops_per_token = 11

    def __init__(self, d_model: int, init_device: str):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=init_device))
        self.last_input: torch.Tensor | None = None
        self.last_kwargs: dict | None = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self.last_input = x.detach().clone()
        self.last_kwargs = kwargs
        return x * self.weight

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        return self.flops_per_token


class _TinySequenceMixerConfig:
    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str,
        cache,
    ) -> _TinyAttention:
        del layer_idx, n_layers, cache
        return _TinyAttention(d_model, init_device)


class _TinyMoE(nn.Module):
    flops_per_token = 13

    def __init__(self, d_model: int, init_device: str):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=init_device))

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        del loss_div_factor
        return x * self.weight

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        return self.flops_per_token


class _TinyMoEConfig:
    def build(
        self,
        *,
        d_model: int,
        n_layers: int,
        init_device: str,
        cache,
    ) -> _TinyMoE:
        del n_layers, cache
        return _TinyMoE(d_model, init_device)


class _IdentityNormConfig:
    def build(self, d_model: int, *, init_device: str) -> nn.Identity:
        del d_model, init_device
        return nn.Identity()


class _TinyEngram(nn.Module):
    flops_per_token = 17

    def __init__(self, d_model: int, init_device: str):
        super().__init__()
        self.weight = nn.Parameter(torch.full((d_model,), 0.5, device=init_device))
        self.last_input: torch.Tensor | None = None
        self.last_hash_indices: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, *, hash_indices: torch.Tensor | None = None) -> torch.Tensor:
        self.last_input = x.detach().clone()
        self.last_hash_indices = hash_indices
        return x * self.weight

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        return self.flops_per_token


class _TinyLngram(nn.Module):
    flops_per_token = 19

    def __init__(self, d_model: int, init_device: str):
        super().__init__()
        self.weight = nn.Parameter(torch.full((d_model,), 0.25, device=init_device))
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return x * self.weight

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        return self.flops_per_token


class _TinyMemoryConfig:
    def __init__(self, module_cls: type[_TinyEngram] | type[_TinyLngram]):
        self.module_cls = module_cls
        self.build_args: tuple[int, str] | None = None

    def build(self, d_model: int, *, init_device: str = "cpu") -> nn.Module:
        self.build_args = (d_model, init_device)
        return self.module_cls(d_model, init_device)


def _build_block(
    block_cls: type[MoEReorderedNormTransformerBlock],
    memory_config: _TinyMemoryConfig,
    *,
    init_device: str = "cpu",
) -> MoEReorderedNormTransformerBlock:
    return block_cls(
        d_model=4,
        block_idx=0,
        n_layers=1,
        sequence_mixer=_TinySequenceMixerConfig(),
        feed_forward_moe=_TinyMoEConfig(),
        layer_norm=_IdentityNormConfig(),
        init_device=init_device,
        memory=memory_config,
    )


@pytest.mark.parametrize(
    ("block_cls", "memory_module_cls"),
    [
        (MoEEngramReorderedNormTransformerBlock, _TinyEngram),
        (MoELngramReorderedNormTransformerBlock, _TinyLngram),
    ],
)
def test_memory_blocks_build_matching_memory_on_requested_device(
    block_cls,
    memory_module_cls,
):
    memory_config = _TinyMemoryConfig(memory_module_cls)

    block = _build_block(block_cls, memory_config, init_device="meta")

    assert MoEReorderedNormTransformerBlock in block_cls.__bases__
    assert memory_config.build_args == (4, "meta")
    assert block.memory.weight.device.type == "meta"
    assert all(parameter.device.type == "meta" for parameter in block.parameters())


def test_engram_residual_precedes_attention_and_hash_indices_do_not_leak():
    memory_config = _TinyMemoryConfig(_TinyEngram)
    block = _build_block(MoEEngramReorderedNormTransformerBlock, memory_config)
    x = torch.randn(2, 3, 4, requires_grad=True)
    hash_indices = torch.randint(0, 8, (2, 3, 2))

    output = block(x, engram_hash_indices=hash_indices, attention_marker=True)

    torch.testing.assert_close(block.memory.last_input, x.detach())
    assert block.memory.last_hash_indices is hash_indices
    torch.testing.assert_close(block.attention.last_input, 1.5 * x.detach())
    assert block.attention.last_kwargs == {"attention_marker": True}
    output.square().mean().backward()
    assert x.grad is not None
    assert block.memory.weight.grad is not None
    assert block.attention.weight.grad is not None
    assert block.feed_forward_moe.weight.grad is not None


def test_lngram_residual_precedes_attention_without_token_kwargs():
    memory_config = _TinyMemoryConfig(_TinyLngram)
    block = _build_block(MoELngramReorderedNormTransformerBlock, memory_config)
    x = torch.randn(2, 3, 4, requires_grad=True)

    output = block(x)

    torch.testing.assert_close(block.memory.last_input, x.detach())
    torch.testing.assert_close(block.attention.last_input, 1.25 * x.detach())
    assert block.attention.last_kwargs == {}
    output.square().mean().backward()
    assert x.grad is not None
    assert block.memory.weight.grad is not None
    assert block.attention.weight.grad is not None
    assert block.feed_forward_moe.weight.grad is not None


@pytest.mark.parametrize(
    ("block_cls", "memory_module_cls", "memory_flops"),
    [
        (MoEEngramReorderedNormTransformerBlock, _TinyEngram, 17),
        (MoELngramReorderedNormTransformerBlock, _TinyLngram, 19),
    ],
)
def test_memory_blocks_add_memory_flops(
    block_cls,
    memory_module_cls,
    memory_flops: int,
):
    block = _build_block(
        block_cls,
        _TinyMemoryConfig(memory_module_cls),
    )

    assert block.num_flops_per_token(seq_len=7) == 11 + 13 + memory_flops


@pytest.mark.parametrize(
    ("block_cls", "memory_module_cls", "variant_name"),
    [
        (MoEEngramReorderedNormTransformerBlock, _TinyEngram, "Engram"),
        (MoELngramReorderedNormTransformerBlock, _TinyLngram, "Lngram"),
    ],
)
def test_memory_blocks_reject_unsupported_parallelism(
    block_cls,
    memory_module_cls,
    variant_name: str,
):
    block = _build_block(
        block_cls,
        _TinyMemoryConfig(memory_module_cls),
    )

    with pytest.raises(NotImplementedError, match=f"TP.*{variant_name}"):
        block.apply_tp(None, input_layout=Replicate())
    with pytest.raises(NotImplementedError, match=f"CP.*{variant_name}"):
        block.apply_cp(None)
    with pytest.raises(NotImplementedError, match=f"PP.*{variant_name}"):
        block.apply_pp(None)
