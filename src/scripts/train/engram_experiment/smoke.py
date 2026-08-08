"""Tiny, deterministic CPU forward/backward smokes for the three experiment arms."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import MethodType
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.memory import EngramConfig, LngramConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType

ARMS = ("base", "engram", "lngram")
VOCAB_SIZE = 64
MODEL_DIM = 32
NUM_LAYERS = 6
NUM_HEADS = 4
NUM_EXPERTS = 4
EXPERT_HIDDEN_SIZE = 32
SEQUENCE_LENGTH = 4
SEED = 12536


@dataclass(frozen=True)
class SmokeResult:
    arm: str
    device: str
    loss: float
    finite_nonzero_gradients: int


def _install_cpu_moe_fallback(model) -> None:
    """Replace only the unavailable optional CUDA expert kernels with equivalent torch ops."""

    def experts_forward(
        experts,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        del batch_size_per_expert
        mlp = experts.mlp
        flat_x = x.reshape(-1, x.shape[-1])
        indices = expert_indices.reshape(flat_x.shape[0], -1).long()
        weights = expert_weights.reshape(flat_x.shape[0], -1)
        w1 = mlp.w1.reshape(mlp.num_experts, mlp.d_model, mlp.hidden_size)
        w2 = mlp.w2.reshape(mlp.num_experts, mlp.hidden_size, mlp.d_model)
        w3 = mlp.w3.reshape(mlp.num_experts, mlp.d_model, mlp.hidden_size)
        hidden = F.silu(torch.einsum("nd,nkdh->nkh", flat_x, w1[indices]))
        hidden = hidden * torch.einsum("nd,nkdh->nkh", flat_x, w3[indices])
        selected = torch.einsum("nkh,nkhd->nkd", hidden, w2[indices])
        output = (selected * weights.unsqueeze(-1)).sum(dim=1)
        return output.reshape_as(x)

    for block in model.blocks.values():
        block.feed_forward_moe.experts.forward = MethodType(
            experts_forward, block.feed_forward_moe.experts
        )


def _base_config() -> TransformerConfig:
    config = TransformerConfig.smallmoe(
        vocab_size=VOCAB_SIZE,
        d_model=MODEL_DIM,
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
        hidden_size_multiple_of=16,
        attn_backend=AttentionBackendName.torch,
    )
    config.block.sequence_mixer.backend = AttentionBackendName.torch
    assert config.block.feed_forward_moe is not None
    config.block.feed_forward_moe.num_experts = NUM_EXPERTS
    config.block.feed_forward_moe.hidden_size = EXPERT_HIDDEN_SIZE
    return config


def build_smoke_config(arm: str) -> TransformerConfig:
    """Build a downscaled config with the selected production arm's block structure."""

    if arm not in ARMS:
        raise ValueError(f"unknown smoke arm {arm!r}; expected one of {ARMS}")

    config = _base_config()
    if arm == "base":
        return config

    config.block_overrides = {}
    for layer_idx in (1, config.n_layers // 2 - 1):
        block = config.block.copy()
        if arm == "engram":
            block.name = TransformerBlockType.moe_engram_reordered_norm
            block.memory = EngramConfig(
                orders=(2, 3),
                num_hash_heads=2,
                table_sizes=(17, 19),
                embedding_dim=4,
                vocab_size=VOCAB_SIZE,
                tokenizer_compression=False,
                conv_dilation=3,
            )
        else:
            block.name = TransformerBlockType.moe_lngram_reordered_norm
            block.memory = LngramConfig(
                orders=(2, 3),
                bits_per_route=4,
                memory_dim=2,
                conv_dilation=3,
            )
        config.block_overrides[layer_idx] = block
    return config


def run_smoke(arm: str) -> SmokeResult:
    """Run one random-token CPU forward and backward without training infrastructure."""

    torch.manual_seed(SEED)
    device = torch.device("cpu")
    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (1, SEQUENCE_LENGTH),
        dtype=torch.long,
        device=device,
    )

    model = build_smoke_config(arm).build(init_device=device.type)
    model.init_weights(
        max_seq_len=SEQUENCE_LENGTH,
        device=device,
    )
    _install_cpu_moe_fallback(model)
    logits = model(input_ids)
    loss = logits.float().square().mean()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError(f"{arm} produced a non-finite scalar loss")
    loss.backward()

    gradient_count = sum(
        1
        for parameter in model.parameters()
        if parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(torch.count_nonzero(parameter.grad))
    )
    if gradient_count == 0:
        raise RuntimeError(f"{arm} produced no finite nonzero gradients")
    if arm != "base":
        memory_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if ".memory." in name and parameter.grad is not None
        ]
        if not any(
            bool(torch.isfinite(gradient).all()) and bool(torch.count_nonzero(gradient))
            for gradient in memory_gradients
        ):
            raise RuntimeError(f"{arm} produced no finite nonzero memory gradients")

    return SmokeResult(
        arm=arm,
        device=device.type,
        loss=float(loss.detach()),
        finite_nonzero_gradients=gradient_count,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", nargs="?", choices=("all", *ARMS), default="all")
    opts = parser.parse_args(argv)

    selected = ARMS if opts.arm == "all" else (opts.arm,)
    for arm in selected:
        result = run_smoke(arm)
        print(
            f"PASS {result.arm} loss={result.loss:.6f} "
            f"gradients={result.finite_nonzero_gradients}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
