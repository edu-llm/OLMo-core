"""Decoder-only transformer, with the loss normalisation the paired design needs.

## The fix this file exists for

The previous generation computed
`F.cross_entropy(..., ignore_index=-100)` with the default
`reduction="mean"`, which averages over **surviving** targets. Masking a
fraction *f* of a document's targets therefore multiplied every remaining
target's weight by `1/(1-f)` -- 1.331 inside fact documents at the measured
24.89% mask rate. The project's own results catalogue states the consequence in
bold: *"every split/masked arm in this repository was trained under a different
effective objective than its dense twin, over and above the masking itself...
They are not valid arm contrasts, and no dense-minus-split difference below may
be reported as a treatment effect."*

Here, loss is **summed** over supervised positions and divided by a **fixed
divisor** supplied by the caller -- the same constant for every arm. Each
supervised token then carries identical weight in both arms, and the only
experimental difference is *whether a position is supervised at all*, which is
the intended manipulation.

Note this is a third convention, not the same as either previous one:

    v1 (used for every reported result):  sum / n_surviving   -- arm-asymmetric
    v2 (written, never used):             token-weighted mean -- still arm-asymmetric
    here:                                 sum / fixed_divisor -- arm-symmetric

`test_loss_normalization.py` asserts the property directly: with the same
supervised positions, dense and split gradients are numerically identical
regardless of how many *other* positions are masked.

## FLOP accounting

`flops_per_token()` includes the attention term, which `6ND` omits. At
d_model=768, n_layer=12 that term is 18.2% of per-token FLOPs at ctx=1024 and
30.7% at ctx=2048, so `6ND` is not defensible at this scale and does not cancel
between arms whose sequence composition differs. Convention, stated because the
literature is split: embedding *parameters* are excluded from `n_params_nonembed`
but the output head's FLOPs are counted, since the head is a real matmul per
token. Kaplan excludes embedding FLOPs; Chinchilla includes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_INDEX = -100


@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    ctx: int = 1024
    vocab_size: int = 50304
    rope_base: float = 10000.0
    tie_embeddings: bool = False


PRESETS: dict[str, GPTConfig] = {
    # ctx is stated explicitly everywhere. The previous tree set ctx=1024 for
    # d160m in one file while the paper said 2048; being explicit removes the
    # chance of that recurring.
    "toy": GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=128, vocab_size=512),
    "d8m": GPTConfig(n_layer=6, n_head=6, d_model=192, ctx=1024),
    "d40m": GPTConfig(n_layer=8, n_head=8, d_model=512, ctx=1024),
    "d160m": GPTConfig(n_layer=12, n_head=12, d_model=768, ctx=1024),
    "d160m_ctx2048": GPTConfig(n_layer=12, n_head=12, d_model=768, ctx=2048),
}


def _mlp_hidden(d_model: int) -> int:
    return 64 * math.ceil((8 * d_model / 3) / 64)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def _rope_tables(d_head: int, ctx: int, base: float, device=None):
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(ctx, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x, cos, sin, offset: int = 0):
    # x: (B, H, T, Dh)
    T = x.shape[-2]
    c = cos[offset : offset + T].unsqueeze(0).unsqueeze(0)
    s = sin[offset : offset + T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = (B, T, self.n_head, self.d_head)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)

        offset = 0 if cache is None else cache[0].shape[-2]
        q = _apply_rope(q, cos, sin, offset)
        k = _apply_rope(k, cos, sin, offset)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=-2)
            v = torch.cat([cache[1], v], dim=-2)
        new_cache = (k, v)

        # Causal only when there is more than one query position; a single-token
        # decode step must attend to the whole cached prefix.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), new_cache


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        hidden = _mlp_hidden(cfg.d_model)
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None):
        h, new_cache = self.attn(self.ln1(x), cos, sin, cache)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        cos, sin = _rope_tables(cfg.d_model // cfg.n_head, cfg.ctx, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        idx,
        targets=None,
        target_weights=None,
        loss_divisor: float | None = None,
    ):
        """Forward pass; returns `(logits, loss)`.

        `loss_divisor` is **required** whenever `targets` is given. It is the
        fixed constant both arms divide by -- normally the global batch token
        count (`accum * micro_bs * ctx`). Passing `None` is an error rather than
        a default, because the default was the bug: silently normalising by the
        surviving-target count is what made the arms non-comparable.
        """
        B, T = idx.shape
        x = self.wte(idx)
        for block in self.blocks:
            x, _ = block(x, self.rope_cos, self.rope_sin, None)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        if loss_divisor is None:
            raise ValueError(
                "loss_divisor is required. Use the global batch token count, "
                "identical across arms; see the module docstring."
            )

        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            targets.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        if target_weights is not None:
            per_token = per_token * target_weights.reshape(-1).to(per_token.dtype)
        loss = per_token.sum() / float(loss_divisor)
        return logits, loss

    @torch.no_grad()
    def forward_step(self, idx, cache=None):
        """Incremental decode step. `cache` is a list of per-block (k, v)."""
        x = self.wte(idx)
        caches = cache if cache is not None else [None] * len(self.blocks)
        new_caches = []
        for block, c in zip(self.blocks, caches):
            x, nc = block(x, self.rope_cos, self.rope_sin, c)
            new_caches.append(nc)
        x = self.ln_f(x)
        return self.lm_head(x), new_caches

    # ------------------------------------------------------------- accounting

    def param_report(self) -> dict:
        """Parameter breakdown, because the embedding share is large and decisive.

        At `d40m` with a 50,304-token vocabulary, embedding + head is **67% of all
        parameters** (51.5M of 77.2M). A capacity claim quoted against *total*
        parameters is therefore wrong by about 3x. Every bits-per-parameter
        statement must use `blocks` (or `nonembed`), and must say which.

        The previous line's own design review made this point about a different
        configuration -- "for an experiment about non-embedding capacity, spending
        half the model on an embedding table is self-defeating" -- and recommended
        a smaller domain vocabulary. That remains the better fix if the corpus
        stays purely synthetic; the 50,304 GPT-2 vocabulary is kept here only so a
        natural-language slice can be mixed in later without retokenising.
        """
        cfg = self.cfg
        emb = self.wte.weight.numel()
        head = 0 if cfg.tie_embeddings else self.lm_head.weight.numel()
        blocks = sum(
            p.numel() for n, p in self.named_parameters() if n.startswith("blocks.")
        )
        total = self.n_params(include_embeddings=True)
        return {
            "total": total,
            "blocks": blocks,
            "embedding": emb,
            "head": head,
            "nonembed": total - emb,
            "embedding_plus_head_share": (emb + head) / total,
            "vocab_size": cfg.vocab_size,
            "tied": cfg.tie_embeddings,
            "capacity_basis_note": (
                "quote bits/param against 'blocks' or 'nonembed', never 'total'"
            ),
        }

    def capacity_bits(self, bits_per_param: float = 1.0) -> dict:
        """Storage ceiling on the defensible basis, with the exposure caveat.

        `bits_per_param` is a *band*, not a constant: 2 bits/param is measured at
        ~1000 exposures per fact and drops to ~1 bit/param at ~100 exposures.
        Junk dilution cuts it further -- with 1/8 useful content at 100 exposures
        the useful-knowledge ratio degrades by up to 20x, recovering to ~1.3x only
        by 1000 exposures, and a per-source domain token is what buys most of that
        back. So report a band and state the exposure count, never a single number.
        """
        rep = self.param_report()
        return {
            "basis_blocks_bits": rep["blocks"] * bits_per_param,
            "basis_nonembed_bits": rep["nonembed"] * bits_per_param,
            "basis_total_bits_DO_NOT_USE": rep["total"] * bits_per_param,
            "bits_per_param_assumed": bits_per_param,
            "caveat": (
                "2 bits/param is measured at ~1000 exposures/fact; ~1 bit/param at "
                "~100. Report exposures per fact alongside any occupancy figure."
            ),
        }

    def n_params(self, include_embeddings: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if include_embeddings:
            return total
        emb = self.wte.weight.numel()
        if self.cfg.tie_embeddings:
            return total - emb
        return total - emb

    def flops_per_token(self, ctx: int | None = None) -> dict:
        """Per-token training FLOPs, with the attention term kept.

        `6ND` omits the term proportional to context length:

            C_per_token = 6N + 12 * n_layer * d_model * ctx

        **Both parameter conventions are reported, because the attention share
        depends on which you use and the literature is split.**

        * `N_with_head` counts transformer blocks **plus the output head**, and
          excludes the input embedding. This is the default here and the one to
          report: the head is a real matmul on every token, and omitting it is
          known to undercount FLOPs badly at small scale (by ~90% at 5M params)
          and to inflate fitted scaling exponents by >0.1.
        * `N_kaplan` is `12 * n_layer * d_model^2`, which excludes the head. It
          is reported only so numbers quoted against Kaplan's convention can be
          reconciled.

        For d160m the attention share is 13.2% (ctx=1024) / 23.4% (ctx=2048)
        under `N_with_head`, and 18.2% / 30.7% under `N_kaplan`. Either way
        `6ND` alone is not defensible at this scale, and the error does not
        cancel between arms whose sequence composition differs.
        """
        cfg = self.cfg
        ctx = ctx or cfg.ctx
        n_with_head = self.n_params(include_embeddings=False)
        n_kaplan = 12 * cfg.n_layer * cfg.d_model**2
        attn = 12 * cfg.n_layer * cfg.d_model * ctx

        def _pack(n: int) -> dict:
            dense = 6 * n
            return {
                "n_params": n,
                "dense_6N": dense,
                "total": dense + attn,
                "attention_share": attn / (dense + attn),
                "naive_6ND_relative_error": attn / dense,
            }

        return {
            "ctx": ctx,
            "attention": attn,
            "with_head": _pack(n_with_head),
            "kaplan": _pack(n_kaplan),
            # Convenience aliases for the reported convention.
            "total": 6 * n_with_head + attn,
            "attention_share": attn / (6 * n_with_head + attn),
            "n_params_total": self.n_params(True),
        }


def build_model(preset: str, **overrides) -> GPT:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
    cfg = GPTConfig(**{**PRESETS[preset].__dict__, **overrides})
    return GPT(cfg)
