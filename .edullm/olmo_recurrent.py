"""Recurrent-depth OLMo3, as a sibling module that does not modify OLMo-core.

A standard decoder applies N blocks once. This applies the stack in three groups: a
prelude that runs once and produces an encoding ``e``, a recurrent core that runs T
times carrying a latent ``h``, and a coda that runs once on the final ``h``. Depth
becomes a runtime knob rather than a parameter count, so a checkpoint trained at T=4
can be evaluated at T=8 with no new weights.

The whole recurrence costs ``2*d^2 + 4*d`` parameters, which is 2,101,248 at d=1024, or
0.4% of the 474M ``olmo3_370M`` total. Everything else -- width, head count, FFN size,
norms, RoPE, sliding-window pattern, vocabulary -- is inherited from ``olmo3_370M``
unchanged, so the two are comparable at matched width and matched parameters.

WHY THE BLOCK DICT IS LEFT EXACTLY AS OLMo-core BUILT IT. ``Transformer.blocks`` is a
flat ``nn.ModuleDict`` keyed by the stringified layer index, and three separate places
depend on that shape: ``forward`` and ``get_rope_buffers`` both call ``int(key)``, and
``apply_activation_checkpointing`` re-registers every block under its ``enumerate()``
position rather than its actual key. Splitting the stack into three ``ModuleList``
attributes would therefore require overriding ``apply_pp``, ``apply_tp``, ``apply_cp``,
``apply_compile``, ``apply_fsdp``, ``apply_activation_checkpointing``, ``init_weights``,
``get_rope_buffers`` and ``num_flops_per_token``, and each of those would then have to
be kept in step with upstream forever. So all sixteen blocks stay in ``self.blocks``
under keys "0".."15", the split is three integers, and only ``forward`` is overridden.
Every distributed and compilation path is untouched upstream code.

WHY THE RESIDUAL SCALE IS A BUILD-TIME CONSTANT. A weight-tied loop needs a stronger
residual scale than a plain stack. In a stack the per-layer updates are close to
uncorrelated, residual energy grows as O(N), and 1/sqrt(N) holds it. Around a loop the
same weights see similar inputs every iteration, the cross terms no longer cancel,
energy grows as O(N^2), and the scale has to be 1/N. ``residual_epsilon`` factors the
two effects apart: 1/N for the within-loop correlation and 1/sqrt(L) for the L
independent layers inside one iteration.

OLMo-core already has the hook for this. ``TransformerBlockConfig`` carries
``attention_residual_alpha`` and ``feed_forward_residual_alpha``, and ``ResidualStream``
applies them as ``torch.add(residual, sublayer_out, alpha=alpha)`` -- so alpha scales the
branch and not the skip, which is exactly the eps in the recurrence literature. Setting
it through ``block_overrides`` on the twelve recurrent layers only means no block class
is subclassed and no forward is patched. It also pins eps to ``max_loops`` rather than
tracking a per-step sampled T, which is the deliberate choice: a residual scale that
changes between optimizer steps changes the function being optimized, and 1/N at the
deepest budget bounds every shallower path a fortiori.

WHAT WAS DELIBERATELY LEFT OUT. The reference implementation this is ported from also
shares attention K/V across loop iterations, projecting them once from ``e`` so the
looped blocks cross-attend to the frozen prelude encoding. That needs a ``build_cache``
entry point inside the attention module, which would mean modifying OLMo-core, so it is
not here. Without it the recurrent blocks self-attend over their own evolving ``h``,
which is the Huginn-style recurrence and is the honest minimal port. ``e`` still re-enters
at every iteration through the adapter and through the LTI input gain.

Ported from the mythos-rdt reference implementation: ``mythos/recurrence.py`` for the
LTI injection, ``mythos/layers.py`` for the residual scale, ``mythos/model.py`` for the
loop driver and the truncated-BPTT window.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.nn.transformer import model as _upstream_model
from olmo_core.nn.transformer.init import init_linear
from olmo_core.train.callbacks import Callback
from olmo_core.utils import mark_dynamic, move_to_device

log = logging.getLogger(__name__)

__all__ = [
    "RecurrentTransformer",
    "RecurrentTransformerConfig",
    "RecurrentDepthCallback",
    "StableLTIInjection",
    "residual_epsilon",
    "install",
]


ResidualMode = Literal["factored", "one_over_n", "one_over_sqrt_n", "none"]


def residual_epsilon(
    n_loops: int,
    n_layers: int,
    lam: float = 1.0,
    mode: ResidualMode = "factored",
) -> float:
    """The per-iteration residual scale for a looped group of ``n_layers`` blocks.

    ``n_loops`` is the loop BUDGET rather than the depth any particular token runs, so
    that the scale bounds the deepest path and stays fixed across a run.

    ``factored`` is ``lam / (N * sqrt(L))``. ``one_over_n`` drops the layer term and suits
    a single-layer loop. ``one_over_sqrt_n`` is the rule that holds for an untied stack and
    provably fails for a tied loop; it is here so the failure can be reproduced rather than
    argued about. ``none`` is a standard residual, for the ablation.
    """
    n = max(int(n_loops), 1)
    layers = max(int(n_layers), 1)
    if mode == "none":
        return 1.0
    if mode == "one_over_sqrt_n":
        return 1.0 / math.sqrt(n)
    if mode == "one_over_n":
        return 1.0 / n
    if mode == "factored":
        return lam / (n * math.sqrt(layers))
    raise OLMoConfigurationError(f"unknown residual mode {mode!r}")


class StableLTIInjection(nn.Module):
    """A diagonal linear carry whose spectral radius cannot leave (0, 1).

    ``A_bar = exp(dt * A_cont)`` with ``A_cont = -exp(theta_A)`` strictly negative and
    ``dt = softplus(theta_dt)`` strictly positive, so the linear part of the recurrence is
    a contraction at any T and for any value the parameters take. There is no clamp and no
    penalty term keeping it there: the parameterization makes the unstable region
    unreachable. ``margin`` caps it further at ``1 - margin`` so a long loop has a strict
    spectral gap rather than being merely marginally stable, and it is a plain float, so it
    costs nothing.

    ``B_bar`` is the exact zero-order-hold input gain rather than the Euler gain ``dt * B``.

    The discretization runs in float32 and is cast back by the caller. Under bf16 autocast
    the difference matters: ``exp`` of a bf16 product has around three decimal digits, and
    the whole point of this module is that the eigenvalues are known exactly.

    3*d parameters. Ported from ``mythos/recurrence.py``.
    """

    def __init__(self, d: int, margin: float = 0.0, init_device: str = "cpu") -> None:
        super().__init__()
        self.d = d
        self.margin = float(margin)
        self.theta_A = nn.Parameter(torch.empty(d, dtype=torch.float32, device=init_device))
        self.theta_dt = nn.Parameter(torch.empty(d, dtype=torch.float32, device=init_device))
        self.B_cont = nn.Parameter(torch.empty(d, dtype=torch.float32, device=init_device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Named so that `Transformer.init_weights`, which calls `reset_parameters` on every
        # submodule that has one after `to_empty`, initializes this module too.
        nn.init.normal_(self.theta_A, mean=0.0, std=0.5)
        # softplus^-1(0.1), so dt starts near 0.1 and A_bar near exp(-0.1 * exp(theta_A)).
        nn.init.constant_(self.theta_dt, math.log(math.expm1(0.1)))
        nn.init.ones_(self.B_cont)

    def discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a_cont = -torch.exp(self.theta_A.float())
        dt = F.softplus(self.theta_dt.float())
        a_bar = torch.exp(dt * a_cont)
        if self.margin > 0.0:
            a_bar = (1.0 - self.margin) * a_bar
        b_bar = (a_bar - 1.0) / a_cont * self.B_cont.float()
        return a_bar, b_bar

    @torch.no_grad()
    def spectral_radius(self) -> float:
        a_bar, _ = self.discretize()
        return float(a_bar.max())

    def extra_repr(self) -> str:
        return f"d={self.d}, margin={self.margin}"


class RecurrentTransformer(Transformer):
    """``Transformer`` with the middle of the stack applied T times.

    Only ``forward`` and ``init_weights`` are overridden. ``self.blocks`` keeps the flat,
    integer-keyed shape upstream builds, so every ``apply_*`` method is inherited unchanged.
    """

    def __init__(
        self,
        *,
        n_prelude: int,
        n_coda: int,
        default_n_loops: int,
        min_loops: int,
        max_loops: int,
        backprop_depth: Optional[int],
        spectral_margin: float,
        recurrent_norm: LayerNormConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)

        n_layers = self.n_layers
        n_recurrent = n_layers - n_prelude - n_coda
        if n_recurrent < 1:
            raise OLMoConfigurationError(
                f"n_prelude ({n_prelude}) + n_coda ({n_coda}) leaves {n_recurrent} layers for "
                f"the recurrent core out of {n_layers}; it needs at least one."
            )

        self.n_prelude = n_prelude
        self.n_recurrent = n_recurrent
        self.n_coda = n_coda
        self.min_loops = min_loops
        self.max_loops = max_loops
        self.backprop_depth = backprop_depth

        # The depth this forward pass will run. A plain attribute rather than a forward
        # argument because there is no channel for one: `split_batch` raises on any batch
        # value that is not a tensor or a list, and `Transformer.forward` discards kwargs it
        # does not recognize. `RecurrentDepthCallback` writes here between steps.
        self.n_loops = default_n_loops

        d = self.d_model
        init_device = str(self.init_device)
        self.norm_e = recurrent_norm.build(d, init_device=init_device)
        self.adapter = nn.Linear(2 * d, d, bias=False, dtype=self.dtype, device=init_device)
        self.injection = StableLTIInjection(d, margin=spectral_margin, init_device=init_device)

        # `Transformer.__init__` forces both cached properties on its last two lines, before
        # these three modules existed, so the cached totals are short by 2*d^2 + 4*d. Drop
        # the cache entries and take them again.
        self.__dict__.pop("num_params", None)
        self.__dict__.pop("num_non_embedding_params", None)
        self.num_params
        self.num_non_embedding_params

    @property
    def prelude_range(self) -> range:
        return range(0, self.n_prelude)

    @property
    def recurrent_range(self) -> range:
        return range(self.n_prelude, self.n_prelude + self.n_recurrent)

    @property
    def coda_range(self) -> range:
        return range(self.n_prelude + self.n_recurrent, self.n_layers)

    def _run(
        self,
        h: torch.Tensor,
        indices: range,
        all_block_kwargs: Dict[str, Any],
        per_block_kwargs: Dict[int, Dict[str, Any]],
    ) -> torch.Tensor:
        for block_idx in indices:
            block = self.blocks[str(block_idx)]
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            h = block(h, **all_block_kwargs, **per_block_kwargs.get(block_idx, {}))
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        input_embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        n_loops: Optional[int] = None,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """Prelude once, recurrent core ``n_loops`` times, coda once.

        ``n_loops`` defaults to ``self.n_loops``. It is accepted here so that an evaluation
        can ask for a depth directly; the training path cannot reach it and uses the
        attribute.
        """
        if input_embeddings is not None and self._cp_load_balancer is not None:
            raise RuntimeError(
                "`input_embeddings` is not supported with context parallelism: `_prepare_inputs` "
                "shards `input_ids`/`labels`/RoPE while `input_embeddings` stays full-size, which "
                "would misalign the hidden states."
            )

        # Same one-shot unshifted-labels check upstream does. Kept rather than dropped: it
        # catches a bug that collapses the loss without raising anything.
        if not _upstream_model.CHECKED_LABELS_FOR_SHIFT and labels is not None:
            _upstream_model.CHECKED_LABELS_FOR_SHIFT = True
            if (
                labels.shape == input_ids.shape
                and labels.device == input_ids.device
                and torch.equal(labels, input_ids)
            ):
                log.warning(
                    "`labels` is identical to `input_ids`, so the labels were never shifted. "
                    "The model is being asked to predict a token its own context already "
                    "contains, and the loss will collapse without any error being raised."
                )

        (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        ) = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if input_embeddings is not None:
            h = move_to_device(input_embeddings, self.device)
        else:
            h = self.embeddings(input_ids) if self.embeddings is not None else input_ids
            if self.embeddings is not None and self.embed_scale is not None:
                h = h * self.embed_scale
            if self.embedding_norm is not None:
                h = self.embedding_norm(h)

        # ---- Prelude: run once, produce the encoding every iteration reads. ----
        h = self._run(h, self.prelude_range, all_block_kwargs, per_block_kwargs)
        e = self.norm_e(h)

        # ---- Recurrent core. ----
        depth = int(n_loops if n_loops is not None else self.n_loops)
        depth = max(depth, 1)
        a_bar, b_bar = self.injection.discretize()

        # Only the last `backprop_depth` iterations carry gradient; earlier ones run under
        # no_grad and are detached, which bounds activation memory at O(backprop_depth)
        # instead of O(T). None means the whole loop is differentiated.
        first_grad_step = 0 if self.backprop_depth is None else max(0, depth - self.backprop_depth)

        state = torch.zeros_like(e)
        for step in range(depth):
            differentiable = step >= first_grad_step
            with torch.enable_grad() if (
                differentiable and torch.is_grad_enabled()
            ) else torch.no_grad():
                # `e` re-enters here on every iteration, not just the first.
                injected = self.adapter(torch.cat([state, e], dim=-1))
                updated = self._run(
                    injected, self.recurrent_range, all_block_kwargs, per_block_kwargs
                )
                # What the block group added, before the linear carry.
                delta = updated - injected
                # The contraction stays in fp32 whatever the activation dtype is.
                carry = (a_bar * state.float() + b_bar * e.float()).to(delta.dtype)
                state = carry + delta
            if not differentiable:
                state = state.detach()

        # ---- Coda: run once on the final latent. ----
        h = self._run(state, self.coda_range, all_block_kwargs, per_block_kwargs)

        if self.lm_head is not None:
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
                if labels is not None:
                    mark_dynamic(labels, (0, 1), strict=False)
            if labels is not None:
                lm_head_kwargs["labels"] = labels
            return self.lm_head(h, **lm_head_kwargs)
        return h

    @torch.no_grad()
    def init_weights(self, **kwargs) -> torch.Generator:
        """Upstream init, then the adapter, which upstream does not know about.

        ``super()`` has to run rather than be reimplemented: it calls ``to_empty``, which
        allocates fresh storage, and then re-ties the LM head to the embeddings. ``norm_e``
        and the LTI parameters are already covered, because upstream calls
        ``reset_parameters`` on every submodule that has one.
        """
        generator = super().init_weights(**kwargs)
        init_linear(self.adapter, std=self.init_std, generator=generator)
        return generator

    def num_flops_per_token(self, seq_len: int) -> int:
        """Idealized forward FLOPs, counting the recurrent group ``n_loops`` times.

        Without this the trainer's throughput and MFU numbers would describe a sixteen-block
        model, so a recurrent run would look several times more efficient than it is.
        """
        flops = 0
        for block_idx in list(self.prelude_range) + list(self.coda_range):
            flops += self.blocks[str(block_idx)].num_flops_per_token(seq_len)
        recurrent = sum(
            self.blocks[str(block_idx)].num_flops_per_token(seq_len)
            for block_idx in self.recurrent_range
        )
        flops += max(int(self.n_loops), 1) * recurrent
        if self.lm_head is not None:
            flops += self.lm_head.num_flops_per_token(seq_len)
        return flops


@dataclass
class RecurrentTransformerConfig(TransformerConfig):
    """``TransformerConfig`` that builds a :class:`RecurrentTransformer`.

    A subclass rather than a new ``TransformerType``, because ``TransformerConfig.build``
    is a hard-coded if/elif over a ``StrEnum`` with no registration hook. Overriding
    ``build`` is the supported way in.

    It round-trips through ``ConfigSaverCallback`` because ``Config.as_dict`` writes a
    ``_CLASS_`` field holding ``module.ClassName`` and ``from_dict`` resolves it with
    ``importlib``. That only works if this module is importable by the same dotted path
    when the config is read back, which is why the runner adds this directory to
    ``sys.path`` under a stable name rather than running it as ``__main__``.
    """

    n_prelude: int = 2
    """Blocks that run once, before the loop, to produce the encoding ``e``."""

    n_coda: int = 2
    """Blocks that run once, after the loop, before the LM head."""

    default_n_loops: int = 4
    """T used unless something sets ``model.n_loops`` at runtime."""

    min_loops: int = 1
    """Floor for a depth schedule or an adaptive-compute policy."""

    max_loops: int = 4
    """Ceiling for a depth schedule, and the N the residual scale is computed against."""

    backprop_depth: Optional[int] = None
    """Iterations at the end of the loop that carry gradient. None differentiates all of them."""

    residual_lambda: float = 1.0
    """Numerator of the residual scale."""

    residual_mode: str = "factored"
    """One of factored, one_over_n, one_over_sqrt_n, none. See :func:`residual_epsilon`."""

    spectral_margin: float = 0.02
    """Caps the LTI spectral radius at ``1 - margin`` so a long loop contracts strictly."""

    @property
    def n_recurrent_layers(self) -> int:
        """Derived, never stored, so it cannot disagree with ``n_layers``."""
        return self.n_layers - self.n_prelude - self.n_coda

    @property
    def residual_alpha(self) -> float:
        """The residual scale the recurrent blocks are built with."""
        return residual_epsilon(
            self.max_loops,
            self.n_recurrent_layers,
            lam=self.residual_lambda,
            mode=self.residual_mode,  # type: ignore[arg-type]
        )

    def _validate(self) -> None:
        if self.n_recurrent_layers < 1:
            raise OLMoConfigurationError(
                f"n_prelude ({self.n_prelude}) + n_coda ({self.n_coda}) leaves "
                f"{self.n_recurrent_layers} layers for the recurrent core out of "
                f"{self.n_layers}; it needs at least one."
            )
        if not 1 <= self.min_loops <= self.default_n_loops <= self.max_loops:
            raise OLMoConfigurationError(
                f"loop bounds must satisfy 1 <= min_loops <= default_n_loops <= max_loops, got "
                f"{self.min_loops}, {self.default_n_loops}, {self.max_loops}."
            )
        if self.backprop_depth is not None and self.backprop_depth < 1:
            raise OLMoConfigurationError(
                f"backprop_depth must be at least 1 when set, got {self.backprop_depth}."
            )
        if isinstance(self.block, dict):
            raise OLMoConfigurationError(
                "a per-key block mapping is not supported here; the recurrent split addresses "
                "blocks by integer index."
            )

    def apply_recurrent_residual_alpha(self) -> "RecurrentTransformerConfig":
        """Scale the residual branch of the recurrent blocks, and only those.

        Done through ``block_overrides`` so that no block class is subclassed and no forward
        is patched. Prelude and coda keep alpha 1.0, so they are bit-identical to the
        baseline. Called by :meth:`build`, and separately by the factory so that a config
        printed by a dry run already shows the alphas it will train with.

        Note the argument is not routed through ``llama_like``: that helper reads
        ``block_overrides`` out of kwargs without popping it and then passes both it and
        ``**kwargs`` on, so supplying it that way raises a duplicate-keyword ``TypeError``.
        """
        self._validate()
        alpha = self.residual_alpha
        overrides = dict(self.block_overrides or {})
        for block_idx in range(self.n_prelude, self.n_prelude + self.n_recurrent_layers):
            recurrent_block = self.block.copy()
            recurrent_block.attention_residual_alpha = alpha
            recurrent_block.feed_forward_residual_alpha = alpha
            overrides[block_idx] = recurrent_block
        self.block_overrides = overrides
        return self

    @property
    def num_params(self) -> int:
        """Upstream total plus the recurrence, which is ``2*d^2 + 4*d``.

        ``norm_e`` is ``d``, the adapter is ``2*d*d`` with no bias, and the LTI carries three
        vectors of length ``d``. The twelve recurrent blocks are counted once each by
        upstream and are reused across iterations rather than duplicated, so looping adds
        no parameters at all -- which is the point.
        """
        d = self.d_model
        extra = 2 * d * d + 3 * d
        norm_e = self.block.layer_norm.num_params(d) if self.block.layer_norm is not None else d
        return super().num_params + extra + norm_e

    def build(self, *, init_device: str = "cpu") -> RecurrentTransformer:
        self._validate()
        self.apply_recurrent_residual_alpha()

        log.info(
            "Building recurrent transformer with %d total params, %d non-embedding params: "
            "prelude %d, recurrent %d looped T=%d (max %d), coda %d, residual alpha %.5g",
            self.num_params,
            self.num_non_embedding_params,
            self.n_prelude,
            self.n_recurrent_layers,
            self.default_n_loops,
            self.max_loops,
            self.n_coda,
            self.residual_alpha,
        )

        if self.block.layer_norm is None:
            raise OLMoConfigurationError(
                "the block config carries no layer_norm, so there is nothing to normalize the "
                "prelude encoding with."
            )

        model = RecurrentTransformer(
            n_prelude=self.n_prelude,
            n_coda=self.n_coda,
            default_n_loops=self.default_n_loops,
            min_loops=self.min_loops,
            max_loops=self.max_loops,
            backprop_depth=self.backprop_depth,
            spectral_margin=self.spectral_margin,
            recurrent_norm=self.block.layer_norm,
            d_model=self.d_model,
            vocab_size=self.vocab_size,
            n_layers=self.n_layers,
            block=self.block,
            embedding_norm=self.embedding_norm,
            lm_head=self.lm_head,
            dtype=self.dtype.as_pt(),
            init_method=self.init_method,
            init_device=init_device,
            init_seed=self.init_seed,
            init_std=self.init_std,
            embedding_init_std=self.embedding_init_std,
            block_overrides=self.block_overrides,
            block_pattern=self.block_pattern,
            embed_scale=self.embed_scale,
            tie_word_embeddings=self.tie_word_embeddings,
        )

        # Same freezing semantics as upstream `build`.
        if self.freeze_params:
            for name, param in model.named_parameters():
                for pattern in self.freeze_params:
                    if fnmatch(name, pattern):
                        param.requires_grad = False
                        log.info("Param '%s' will be frozen", name)
                        break

        log.info(
            "Built recurrent model with %d total, %d non-embedding, %d trainable params",
            model.num_params,
            model.num_non_embedding_params,
            model.num_trainable_params,
        )
        return model

    @classmethod
    def recurrent_olmo3_370M(cls, vocab_size: int, **kwargs) -> "RecurrentTransformerConfig":
        """``olmo3_370M`` with the middle twelve of its sixteen blocks looped.

        Everything ``olmo3_370M`` sets is inherited: d_model 1024, 16 layers, 16 heads, FFN
        4096, reordered-norm blocks, QK-norm, RoPE theta 500k, the sliding-window pattern
        and the flash-2 backend. ``llama_like`` ends in ``return cls(...)``, so calling the
        inherited factory on this subclass yields this subclass with the whole recipe intact
        and the recurrence fields carried down as kwargs.
        """
        recurrence = dict(
            n_prelude=kwargs.pop("n_prelude", 2),
            n_coda=kwargs.pop("n_coda", 2),
            default_n_loops=kwargs.pop("default_n_loops", 4),
            min_loops=kwargs.pop("min_loops", 1),
            max_loops=kwargs.pop("max_loops", 4),
            backprop_depth=kwargs.pop("backprop_depth", None),
            residual_lambda=kwargs.pop("residual_lambda", 1.0),
            residual_mode=kwargs.pop("residual_mode", "factored"),
            spectral_margin=kwargs.pop("spectral_margin", 0.02),
        )
        config = cls.olmo3_370M(vocab_size=vocab_size, **recurrence, **kwargs)
        assert isinstance(config, RecurrentTransformerConfig)
        return config.apply_recurrent_residual_alpha()


@dataclass
class RecurrentDepthCallback(Callback):
    """Vary T between optimizer steps: shallow for the first stretch, then sampled.

    OFF BY DEFAULT, AND THE DEFAULT IS THE RIGHT FIRST RUN. A fixed T is deterministic,
    keeps every microbatch in an accumulation window at the same depth, and gives
    ``torch.compile`` one shape to specialize on. Varying it is a second experiment rather
    than a better version of the first.

    Two constraints shape the design. It hooks ``pre_step`` rather than anything
    per-microbatch, because the microbatches inside one optimizer step have to share a depth
    or the accumulated gradient is a sum over different functions. And it writes to the model
    rather than into the batch, because ``split_batch`` raises on any batch value that is not
    a tensor or a list, so a scalar in the batch dict would fail the moment gradient
    accumulation is on.

    The depth is drawn from the step number rather than from callback state, so a resumed run
    replays the identical sequence with nothing to checkpoint. All four fields are scalars so
    the schedule survives ``as_config_dict`` and lands in the saved config.
    """

    min_depth: int = 1
    max_depth: int = 4
    shallow_fraction: float = 0.7
    """Fraction of the run held at ``min_depth`` before any deeper step is drawn."""
    seed: int = 6198

    def depth_for_step(self, step: int, total_steps: Optional[int]) -> int:
        """The depth this step runs at. Pure, so a test can read the schedule off it."""
        if self.max_depth <= self.min_depth:
            return self.min_depth
        # An unbounded run (max_duration in tokens with no step count) has no fraction to
        # compare against, so treat every step as past the shallow stretch.
        if total_steps is not None and step < self.shallow_fraction * total_steps:
            return self.min_depth
        span = self.max_depth - self.min_depth + 1
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + step)
        return self.min_depth + int(torch.randint(span, (1,), generator=generator))

    def pre_step(self, batch: Dict[str, Any]) -> None:
        del batch
        model = self.trainer.train_module.model
        if not isinstance(model, RecurrentTransformer):
            return
        model.n_loops = self.depth_for_step(self.trainer.global_step, self.trainer.max_steps)


def install() -> None:
    """Make the recurrent factories reachable through ``--model-factory``.

    The platform's runner resolves a model with ``getattr(TransformerConfig, name, None)``.
    Attaching the factory to that class is therefore the whole integration, and it is why
    none of ``train_on_corpus.py`` has to be copied: its corpus resolution, refusal codes,
    checkpoint repair and W&B reporting stay the reviewed originals.

    ``cls`` inside the attached classmethod is ``TransformerConfig``, so the body names
    ``RecurrentTransformerConfig`` explicitly rather than relying on it.
    """

    def _factory(cls, vocab_size: int, **kwargs) -> RecurrentTransformerConfig:
        del cls
        return RecurrentTransformerConfig.recurrent_olmo3_370M(vocab_size=vocab_size, **kwargs)

    TransformerConfig.recurrent_olmo3_370M = classmethod(_factory)  # type: ignore[attr-defined]
