"""Minimal probe model: a stack of linear-attention mixers with a token head.

Deliberately tiny (~1-3M params). The FFN is **optional and off by default**.

Why the FFN is optional rather than absent
------------------------------------------
With ``ffn_dim=None`` the model has no MLP at all, so measured length
generalization is attributable to the sequence mixer's recurrent state rather
than to depth or MLP capacity. That is the right shape for the original
state-tracking probes and remains the default, so existing arms are unchanged.

But an FFN-free model cannot express the ``R1-P`` control, which is defined as
"R1 with the DP2 parameter delta spent **only** in FFN width" at identical
``d_model``, heads, state dimensions and depth. Without an FFN there is nowhere
to put the delta except ``d_model``, and widening ``d_model`` changes the
recurrent state size -- which is exactly the variable the comparison is meant to
hold fixed. So the FFN exists to make the capacity control *possible*, not
because the probes need MLP capacity: an arm that enables it is knowingly
trading the "no MLP capacity" attribution for a matched-parameter comparison,
and both arms in such a comparison must enable it.

Parameter accounting for that control lives in :meth:`ProbeModel.parameter_ledger`
and :func:`solve_ffn_dim`.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLUFeedForward(nn.Module):
    """A bias-free SwiGLU MLP: ``down(silu(gate(x)) * up(x))``.

    Parameter count is exactly ``3 * d_model * ffn_dim`` -- linear in ``ffn_dim``,
    which is what makes :func:`solve_ffn_dim` a closed-form solve rather than a
    search.

    :param d_model: Hidden size in and out.
    :param ffn_dim: Inner width. Both projections into the MLP and the projection
        back out use this width.
    """

    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        if ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be positive, got {ffn_dim}")
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: ``[..., d_model]``. :returns: ``[..., d_model]``."""
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ProbeModel(nn.Module):
    """Embedding -> N x (norm + mixer + residual [+ norm + FFN + residual]) -> linear head.

    :param mixer_factory: Callable ``(d_model, layer_idx) -> nn.Module`` returning a
        module whose ``forward(x)`` maps ``[B, T, d_model] -> [B, T, d_model]``.
    :param in_vocab: Input vocabulary size.
    :param out_vocab: Output (target) vocabulary size.
    :param d_model: Hidden size.
    :param n_layers: Number of mixer layers.
    :param ffn_dim: Inner width of a per-layer residual :class:`SwiGLUFeedForward`.
        ``None`` (the default) omits the FFN entirely, reproducing the pre-FFN
        model exactly -- same modules, same parameter names, same init stream.
    """

    def __init__(
        self,
        mixer_factory,
        *,
        in_vocab: int,
        out_vocab: int,
        d_model: int = 256,
        n_layers: int = 3,
        ffn_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.embed = nn.Embedding(in_vocab, d_model)
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_layers)])
        self.mixers = nn.ModuleList([mixer_factory(d_model, i) for i in range(n_layers)])
        if ffn_dim is None:
            self.ffn_norms = None
            self.ffns = None
        else:
            self.ffn_norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_layers)])
            self.ffns = nn.ModuleList([SwiGLUFeedForward(d_model, ffn_dim) for _ in range(n_layers)])
        self.out_norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, out_vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """:param tokens: ``[B, T]`` int64. :returns: logits ``[B, T, out_vocab]``."""
        x = self.embed(tokens)
        for i, (norm, mixer) in enumerate(zip(self.norms, self.mixers)):
            x = x + mixer(norm(x))
            if self.ffns is not None:
                assert self.ffn_norms is not None
                x = x + self.ffns[i](self.ffn_norms[i](x))
        return self.head(self.out_norm(x))

    def parameter_ledger(self) -> dict[str, int]:
        """Break the parameter count down by role.

        ``non_embedding`` excludes **both** the input embedding table and the
        output head. Both scale with the task vocabulary, which differs across the
        probe panel (2 for parity, 120 for s5_words, 897 for mqar_d128), so a
        parameter budget that included them would not be comparable across tasks
        and could not be matched across arms at fixed geometry.

        :returns: mapping with keys ``total``, ``embedding``, ``head``,
            ``non_embedding``, ``mixers``, ``ffns``, ``norms``.
        """
        def count(module: Optional[nn.Module]) -> int:
            return 0 if module is None else sum(p.numel() for p in module.parameters())

        embedding = count(self.embed)
        head = count(self.head)
        mixers = count(self.mixers)
        ffns = count(self.ffns)
        norms = count(self.norms) + count(self.ffn_norms) + count(self.out_norm)
        total = sum(p.numel() for p in self.parameters())
        return {
            "total": total,
            "embedding": embedding,
            "head": head,
            "non_embedding": total - embedding - head,
            "mixers": mixers,
            "ffns": ffns,
            "norms": norms,
        }


def ffn_param_cost(d_model: int, n_layers: int, ffn_dim: int) -> int:
    """Non-embedding parameters added by enabling the FFN at ``ffn_dim``.

    Counts the three SwiGLU projections plus the extra pre-FFN
    :class:`torch.nn.RMSNorm` weight, per layer.

    :param d_model: Hidden size.
    :param n_layers: Number of layers.
    :param ffn_dim: Inner FFN width.
    :returns: Added parameter count.
    """
    return n_layers * (3 * d_model * ffn_dim + d_model)


def solve_ffn_dim(
    target_non_embedding: int,
    base_non_embedding: int,
    *,
    d_model: int,
    n_layers: int,
) -> int:
    """Solve the FFN width that brings ``base_non_embedding`` closest to a target.

    The FFN's parameter cost is exactly affine in ``ffn_dim`` (see
    :func:`ffn_param_cost`), so this is a closed-form solve and a round -- no
    search. Integer quantization of the width means the target is generally not
    hit exactly; the caller must build the model and check the realized mismatch
    against its own tolerance (the runbook's is 0.5% of the target).

    :param target_non_embedding: Non-embedding parameter count to match, e.g. the
        DP2-strict arm's.
    :param base_non_embedding: Non-embedding parameter count of the *same* model
        built with ``ffn_dim=None``.
    :param d_model: Hidden size.
    :param n_layers: Number of layers.
    :returns: The solved ``ffn_dim``, at least 1.
    :raises ValueError: If the target is below what the FFN-free model already
        costs, so no positive width can reach it.
    """
    slack = target_non_embedding - base_non_embedding - n_layers * d_model
    if slack <= 0:
        raise ValueError(
            f"target_non_embedding={target_non_embedding} is not above the FFN-free "
            f"cost {base_non_embedding} plus {n_layers * d_model} for the extra norms; "
            "no positive ffn_dim can match it"
        )
    return max(1, round(slack / (3 * d_model * n_layers)))
