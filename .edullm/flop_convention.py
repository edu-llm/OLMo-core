"""Print the FLOP/token this model is measured with, under both conventions in circulation.

Runs on CPU with meta-device parameters, so it needs a torch and nothing else. The point is
that the number is read out of the same ``num_flops_per_token`` the speed monitor divides by,
rather than transcribed from a table.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402

from olmo_core.data import TokenizerConfig  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_TOKENS = 4_194_304
WORLD = 64
# `speed_monitor.py` for an H100 SXM in half precision: int(1979e12 * 0.5).
PEAK_FLOPS = int(1979e12 * 0.5)


def build():
    tokenizer = TokenizerConfig.dolma2()
    config = TransformerConfig.llama_like_moe(
        vocab_size=tokenizer.padded_vocab_size(),
        d_model=2048,
        n_layers=16,
        n_heads=16,
        num_experts=32,
        top_k=4,
        expert_hidden_size=2048,
        shared_expert_hidden_size=None,
        dropless=True,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        reordered_norm=True,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
    )
    with torch.device("meta"):
        model = config.build()
    return tokenizer, model


def main() -> None:
    tokenizer, model = build()

    total = sum(p.numel() for p in model.parameters())
    olmo = model.num_flops_per_token(SEQUENCE_LENGTH)

    # The causal discount, applied to exactly the term that lacks one. OLMo-core counts the two
    # score matmuls at 12*H*D*S; a causal count halves them, because half of the score matrix is
    # masked and never computed by a fused kernel.
    n_heads, head_dim, n_layers = 16, 2048 // 16, 16
    score_term = 12 * n_heads * head_dim * SEQUENCE_LENGTH * n_layers
    causal = olmo - score_term // 2

    print(f"padded vocab                 {tokenizer.padded_vocab_size():,}")
    print(f"total parameters             {total:,}")
    print()
    print(f"OLMo-core num_flops_per_token   {olmo:,}  = {olmo / 1e9:.4f} GFLOP/token")
    print(f"  of which attention scores     {score_term:,}  = {score_term / 1e9:.4f} GFLOP/token")
    print(f"causal-discounted               {causal:,}  = {causal / 1e9:.4f} GFLOP/token")
    print(f"  ratio                         {olmo / causal:.4f}")
    print()

    per_step = olmo * GLOBAL_BATCH_TOKENS
    per_rank = per_step / WORLD
    print(f"FLOP per step (global)       {per_step:.4e}")
    print(f"FLOP per step per rank       {per_rank:.4e}")
    print(f"peak FLOP/s per H100 SXM     {PEAK_FLOPS:.4e}")
    print(f"step time at 100% MFU        {per_rank / PEAK_FLOPS:.4f} s")
    print()
    print("  MFU (9.29 conv)   s/step   11,921 steps")
    for mfu in (20, 25, 27.2, 30, 35, 40, 45):
        step = per_rank / PEAK_FLOPS / (mfu / 100)
        print(f"    {mfu:5.1f}%          {step:6.3f}    {step * 11921 / 3600:6.2f} h")


if __name__ == "__main__":
    main()
