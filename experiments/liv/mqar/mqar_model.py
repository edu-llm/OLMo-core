"""A scaled-down ``L0`` for MQAR calibration.

Same *topology* as the frozen 350M arm -- gated short convolutions with attention at a few
layers -- at a size where a run costs minutes instead of GPU-days. Calibration only needs the
difficulty knob to bite on the baseline; it does not need the headline model.

WHY NOT JUST REUSE THE FULL ARM BUILDER: MQAR uses a synthetic 8k vocabulary and 64-1024 token
sequences. At d=1024 with a 65,536 vocabulary the embedding alone dwarfs the mixer, so a
difficulty calibrated there would be measuring the embedding table. Zoology's own grid is 2
layers, 1 head, d_model in {64,...,512}, and that is the regime the bimodality was characterized
in.

The mixer is the *same* ``ShortConv`` class the real arms use, so a calibration that depends on
the operator being right will fail if the operator is wrong.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.nn.attention.short_conv import ShortConv


class _Attention(nn.Module):
    """Minimal causal multi-head attention, for the attention slots in the hybrid.

    Deliberately plain: no RoPE, no GQA, no KV cache. The point of these layers in a
    calibration model is to provide the global routing a short conv cannot, not to reproduce
    the production attention stack.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v)
        )
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(b, t, -1))


class _SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MQARHybrid(nn.Module):
    """
    A small LFM2-shaped hybrid: gated short conv everywhere except ``attention_layers``.

    :param vocab_size: Input and output vocabulary (MQAR uses one shared space).
    :param d_model: Hidden size.
    :param n_layers: Total layers.
    :param attention_layers: Indices that use attention; all others use :class:`ShortConv`.
    :param n_heads: Attention heads.
    :param kernel_size: Short-conv taps.
    :param gate_structure: ``"dense"``, ``"lowrank"``, or ``"grouped"``.
    :param gate_rank: Bottleneck rank when ``gate_structure="lowrank"``.
    :param gate_groups: Block count when ``gate_structure="grouped"``.
    :param ffn_mult: SwiGLU hidden size as a multiple of ``d_model``.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        attention_layers: Tuple[int, ...] = (2,),
        n_heads: int = 1,
        kernel_size: int = 3,
        gate_structure: str = "dense",
        gate_rank: int | None = None,
        gate_groups: int | None = None,
        ffn_mult: int = 2,
    ):
        super().__init__()
        self.attention_layers = set(attention_layers)
        self.embed = nn.Embedding(vocab_size, d_model)

        self.mixer_norms = nn.ModuleList(nn.RMSNorm(d_model) for _ in range(n_layers))
        self.ffn_norms = nn.ModuleList(nn.RMSNorm(d_model) for _ in range(n_layers))
        self.ffns = nn.ModuleList(_SwiGLU(d_model, ffn_mult * d_model) for _ in range(n_layers))

        mixers: list[nn.Module] = []
        for i in range(n_layers):
            if i in self.attention_layers:
                mixers.append(_Attention(d_model, n_heads))
            else:
                mixers.append(
                    ShortConv(
                        d_model=d_model,
                        kernel_size=kernel_size,
                        gate_structure=gate_structure,  # type: ignore[arg-type]
                        gate_rank=gate_rank,
                        gate_groups=gate_groups,
                        use_fla=False,  # plain nn.Conv1d: correct operator, runs anywhere
                    )
                )
        self.mixers = nn.ModuleList(mixers)

        self.out_norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """:param tokens: ``[batch, seq_len]`` int64. :returns: logits ``[batch, seq, vocab]``."""
        x = self.embed(tokens)
        for mixer, mnorm, ffn, fnorm in zip(
            self.mixers, self.mixer_norms, self.ffns, self.ffn_norms
        ):
            x = x + mixer(mnorm(x))
            x = x + ffn(fnorm(x))
        return self.head(self.out_norm(x))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
