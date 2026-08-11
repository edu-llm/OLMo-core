"""
Ternary weight quantization-aware training (QAT): TWN threshold + straight-through estimator.

This is the quantizer Maple-Preview was trained with, reverse-engineered from the released
artifacts. It is **TWN** (Li et al. 2016, arXiv:1605.04711), *not* BitNet b1.58, and that
distinction is a measured finding rather than a preference:

===================================  =========================  ==================
scheme                               predicted zero fraction    verdict
===================================  =========================  ==================
BitNet b1.58 (absmean, round-clip)   31.0%                      **ruled out**
TWN, ``delta = 0.7 * mean|W|``       42.4%                      **matches**
===================================  =========================  ==================

Observed in the released weights: **38.7% (layer 0) rising to 42.9% (layer 23)**. TWN's 42.4%
lands inside that; BitNet's 31.0% is 8-12 points away. Maple's own blog *cites* BitNet, and
"correcting" this module to b1.58 would silently build a different model. Three independent
lines converge on TWN: this arithmetic, Bonsai's explicit TWN init, and DeepGrove's own MLX
converter using threshold ``0.7 * mean|w|`` with the docstring "matching the quantizer the
model was trained with".

Cost model, stated because it is routinely gotten backwards
-----------------------------------------------------------
**Ternary QAT is not cheaper to train.** The forward matmul consumes a dequantized
``alpha * sign(W) * 1[|W| > delta]`` tensor in the *compute* dtype, the backward runs full
precision through the straight-through estimator, and the latent master weights are full size.
There is no memory saving and no arithmetic saving during training; ``num_flops_per_token`` is
deliberately unchanged. Ternary's win is at *inference*, which is out of scope for this work.
Anyone budgeting ternary as a training-time saving is wrong, and that error would show up as a
bogus expectation in the X4a MFU comparison.

What stays full precision
-------------------------
Embeddings, the LM head, **the router**, and all norms. The router carve-out is load-bearing,
not stylistic: routing is *discrete*, so quantizing the router changes *which experts fire*,
not merely how accurately they are weighted. :func:`audit_quantization` asserts the carve-outs
on a built model so a silently-dropped one cannot produce a wrong experiment that trains happily.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..exceptions import OLMoConfigurationError

__all__ = [
    "TWN_DELTA_FACTOR",
    "TWN_GAUSSIAN_ZERO_FRACTION",
    "BITNET_B158_GAUSSIAN_ZERO_FRACTION",
    "QuantConfig",
    "QuantLinear",
    "twn_threshold_and_scale",
    "twn_quantize",
    "twn_quantize_ste",
    "TWNQuantCache",
    "reset_twn_quant_caches",
    "audit_quantization",
    "assert_no_float8_conflict",
    "QuantAuditEntry",
]


TWN_DELTA_FACTOR: float = 0.7
"""
The TWN threshold constant: ``delta = TWN_DELTA_FACTOR * mean|W|`` per output row.

Confidence on the family is high; on this exact constant, medium-high (see
``maple/plan/maple-preview-recipe-hypotheses.md`` section 3). **Do not tune it.** It is a
faithfulness parameter, and the zero-fraction match is the only evidence pinning it.
"""

TWN_GAUSSIAN_ZERO_FRACTION: float = 0.423510
"""
Zero fraction TWN produces on Gaussian latents, closed form and dimension-free.

``delta = 0.7 * E|W| = 0.7 * sigma * sqrt(2/pi) = 0.5585192 * sigma``, so the zero fraction is
``erf(0.5585192 / sqrt(2)) = 0.4235110``. This is the discriminating assertion: a build that
produces 0.31 has been "corrected" to BitNet b1.58.

Note: ``maple/plan/maple-preview-recipe-hypotheses.md`` section 3 quotes this as 42.4%, which is
this number rounded. It is written out to six places here so the assertion band cannot drift.
"""

BITNET_B158_GAUSSIAN_ZERO_FRACTION: float = 0.310064
"""
What BitNet b1.58 would produce on the same latents.

b1.58 rounds ``W / mean|W|`` to the nearest integer, so its zero band is ``|W| < 0.5 * mean|W|``
-- an effective threshold factor of 0.5 against TWN's 0.7. That gives
``erf(0.3989423 / sqrt(2)) = 0.3100643``, matching the 31.0% in the plan document. Recorded so a
test can assert we are *not* here.
"""


def _resolve_in_dim(w: torch.Tensor, in_dim: Optional[int]) -> int:
    """
    Resolve an omitted ``in_dim``, or refuse to guess.

    ``-1`` is the correct axis for a 2-D :class:`torch.nn.Linear` weight, so omitting it there is
    convenient and safe. For a stacked expert weight it is the **wrong** answer four times out of
    six -- e.g. ``DroplessMoEMLP.w2`` needs ``1`` while its identically-shaped ``w1``/``w3`` need
    ``2`` -- and the error is silent: both orientations are shape-legal, no exception fires, and
    the TWN zero-fraction assertion cannot tell them apart (both land near 0.42). So for
    ``ndim > 2`` the caller has to say which axis it means.

    Passing ``-1`` explicitly on a stacked weight is allowed: the guard is against *silence*, not
    against the value.
    """
    if in_dim is not None:
        return in_dim
    if w.ndim > 2:
        raise ValueError(
            f"in_dim must be given explicitly for a {w.ndim}-D weight of shape "
            f"{tuple(w.shape)}. Omitting it defaults to -1, which is right for a 2-D nn.Linear "
            "weight and WRONG for most stacked expert weights: DroplessMoEMLP.w2 needs "
            "in_dim=1 while its identically-shaped w1/w3 need in_dim=2. Reducing over the wrong "
            "axis yields a per-input-row alpha -- a different quantizer that trains without "
            "error. See the orientation table in twn_quantize's docstring."
        )
    return -1


def twn_threshold_and_scale(
    w: torch.Tensor, *, in_dim: Optional[int] = None, delta_factor: float = TWN_DELTA_FACTOR
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the TWN per-output-row threshold ``delta`` and scale ``alpha``.

    .. math::
        \\delta_r = 0.7 \\cdot \\mathrm{mean}_j |W_{rj}| \\qquad
        \\alpha_r = \\mathrm{mean}\\{|W_{rj}| : |W_{rj}| > \\delta_r\\}

    Statistics are accumulated in **float32** regardless of the dtype of ``w``. This is not
    cosmetic: under ``--param-dtype bfloat16`` the weight seen in the forward pass is a bf16
    copy with 8 mantissa bits, and a bf16 sum over 1024+ elements loses enough precision to
    move ``delta`` across the comparison boundary for weights sitting near it. Which weights
    become zero is the quantizer's whole identity, so the reduction is done in fp32 and only
    the result is cast back.

    :param w: The latent weight.
    :param in_dim: The single dimension holding *input* features, i.e. the axis reduced over.
        For an :class:`torch.nn.Linear` weight of shape ``(out_features, in_features)`` this
        is ``-1``, which is the default. For a weight with more than 2 dimensions it is
        **required**, because the default is right for a 2-D weight and wrong for four of the
        six stacked expert weights -- see :func:`twn_quantize`.

    :returns: ``(delta, alpha)``, both float32 and both keeping ``in_dim`` as a size-1 axis so
        they broadcast against ``w``.

    :raises ValueError: if ``w`` has more than 2 dimensions and ``in_dim`` was left at its
        default. Omitting it there is silently the wrong quantizer, so it must be stated.
    """
    in_dim = _resolve_in_dim(w, in_dim)
    w32 = w.detach().to(torch.float32)
    absw = w32.abs()
    delta = delta_factor * absw.mean(dim=in_dim, keepdim=True)
    mask = absw > delta
    # A row survives its own threshold unless it is identically zero: delta = 0.7 * mean|W| is
    # strictly below max|W| whenever any element is nonzero. The clamp only guards the
    # all-zeros row, where it correctly yields alpha = 0 (and hence W_q = 0).
    count = mask.sum(dim=in_dim, keepdim=True).clamp_(min=1)
    alpha = (absw * mask).sum(dim=in_dim, keepdim=True) / count
    return delta, alpha


def twn_quantize(
    w: torch.Tensor, *, in_dim: Optional[int] = None, delta_factor: float = TWN_DELTA_FACTOR
) -> torch.Tensor:
    """
    Ternarize ``w`` with the TWN rule: ``alpha * sign(W) * 1[|W| > delta]``, per output row.

    Every output row of the result holds **at most three distinct values**: ``{-alpha_r, 0,
    +alpha_r}``. The scale is *folded into the values* rather than kept as a separate
    parameter -- the released Maple tensor count (18,651) contains no scale tensors, which is
    what ruled out a learned per-row scale.

    On ``in_dim`` and stacked expert weights -- **read this before reusing the function.** The
    axis reduced over must be the one the forward pass treats as input features, and OLMo-core's
    expert weights are not consistently oriented:

    ==============================  ==========================  ========  =========
    tensor                          shape as used in forward    ``in_dim``  note
    ==============================  ==========================  ========  =========
    ``nn.Linear.weight``            ``(out, in)``               ``-1``    standard
    ``MoEMLP.w1`` / ``.w3``         ``(E, d_model, hidden)``    ``1``     ``bmm(x, w)``
    ``MoEMLP.w2``                   ``(E, hidden, d_model)``    ``1``     ``bmm(x, w)``
    ``DroplessMoEMLP.w1`` / ``.w3`` ``(E, hidden, d_model)``    ``2``     ``trans_b=True``
    ``DroplessMoEMLP.w2``           ``(E, hidden, d_model)``    ``1``     ``trans_b=False``
    ==============================  ==========================  ========  =========

    The dropless ``w2`` differs from its own ``w1``/``w3`` because ``gmm`` is called without
    ``trans_b``. Reducing over the wrong axis yields a per-*input*-row alpha: a different
    quantizer that trains perfectly happily and is not the one under test.

    :param w: The latent weight.
    :param in_dim: The input-feature axis. See the table above.
    :param delta_factor: The threshold constant. ``0.7`` is TWN's, derived by minimizing
        reconstruction error rather than loss, and it decides which ~42% of the weights are
        zero -- see :func:`gaussian_zero_fraction` for the closed form relating the two.
    """
    in_dim = _resolve_in_dim(w, in_dim)

    # The fused kernel is the same arithmetic in three streaming reads instead of a dozen
    # whole-tensor temporaries, and it declines rather than raises when it cannot apply, so the
    # definition below stays reachable as the reference. Measured 3.3x to 18.4x faster than it.
    #
    # It is not, however, bitwise identical to the reference on all inputs. It sums each row in
    # a different order, so `delta` can differ in its last bits, and a weight lying within about
    # one bf16 ulp of its row threshold can be classified either way. Measured at up to ~1.7e-5
    # of elements on adversarial draws, deterministic for given data, and always a zero/nonzero
    # tie rather than a sign inversion. That is a small fraction of one step's natural ternary
    # flip rate (~3e-4), so it is immaterial to training -- but it does mean a run with the
    # kernel and a run without are not bit-reproducible against each other.
    from ..kernels import fused_twn_quantize

    fused = fused_twn_quantize(w, in_dim=in_dim, delta_factor=delta_factor)
    if fused is not None:
        return fused

    delta, alpha = twn_threshold_and_scale(w, in_dim=in_dim, delta_factor=delta_factor)
    w32 = w.detach().to(torch.float32)
    q = torch.sign(w32) * (w32.abs() > delta) * alpha
    return q.to(w.dtype)


class _TWNQuantizeSTE(torch.autograd.Function):
    """
    TWN forward, straight-through-estimator backward.

    The backward is the **plain identity** STE: ``dL/dW_latent = dL/dW_q``. Two things this
    deliberately does not do, both of which would be defensible and neither of which is what
    the evidence supports:

    * It does not propagate ``d(alpha)/dW``. ``alpha`` is a function of ``W``, so the true
      Jacobian has extra terms; STE's entire premise is to ignore them and pretend
      ``W_q ~= W``. Including them turns this into a different (and much noisier) estimator.
    * It does not clip. ``rung4b_v2.py`` uses a clipped-identity STE
      (``mask = |w| <= scale``), inherited from 1-bit BitNet practice. Clipping is a
      *regularizer choice*, and there is no evidence Maple used one. Identity keeps gradient
      flowing to every latent weight, including the ones currently quantized to zero -- which
      is what lets a weight cross back over the threshold during training. A clipped STE that
      zeroes the gradient of sub-threshold weights would freeze ~42% of the network.
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, in_dim: int, delta_factor: float) -> torch.Tensor:  # type: ignore[override]
        del ctx
        return twn_quantize(w, in_dim=in_dim, delta_factor=delta_factor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        del ctx
        # Identity STE: the gradient of the quantized weight is the gradient of the latent
        # weight. `None` for the non-tensor `in_dim` and `delta_factor` arguments.
        return grad_output, None, None


def twn_quantize_ste(
    w: torch.Tensor, *, in_dim: Optional[int] = None, delta_factor: float = TWN_DELTA_FACTOR
) -> torch.Tensor:
    """
    :func:`twn_quantize` in the forward direction, identity straight-through in the backward.

    Use this anywhere a quantized weight feeds a matmul during training. See
    :func:`twn_quantize` for the ``in_dim`` orientation table -- getting it wrong builds a
    different quantizer without erroring.
    """
    return _TWNQuantizeSTE.apply(  # type: ignore[no-any-return]
        w, _resolve_in_dim(w, in_dim), delta_factor
    )


class _CachedSTE(torch.autograd.Function):
    """
    Return an already-quantized weight, routing the gradient to the latent weight.

    Identical backward to :class:`_TWNQuantizeSTE`. The forward does no elementwise work at
    all: ``q`` was computed on an earlier microbatch and is handed straight back, so a cache
    hit costs one autograd node instead of a full pass over the weight.
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, q: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        del ctx, w
        # `view_as` rather than returning `q` itself: autograd attaches output metadata to
        # whatever forward returns, and `q` outlives this call inside the cache.
        return q.view_as(q)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        del ctx
        return grad_output, None


class TWNQuantCache:
    """
    Memo of one quantized weight, valid until the latent weight it came from changes.

    ``twn_quantize`` is a scan over the whole weight -- abs, a mean reduction for ``delta``, a
    comparison, a masked sum for ``alpha``, a sign, and a multiply -- and ``MoEMLP.forward``
    and :class:`QuantLinear.forward` call it inline on every forward. Under gradient
    accumulation that repeats the identical scan once per microbatch, because the latent
    weights do not move until the optimizer steps. At M20 the ternarized weights total ~19.6B
    elements, which measured ~630ms of pure scan per forward when scaled to H100 bandwidth, so
    the repeat is worth removing.

    Validity is keyed on the latent tensor's identity, its version counter, and the address of
    its storage. The version counter alone covers the in-place update an optimizer performs, so
    an entry survives exactly the window in which the weight is unchanged and no explicit "new
    step" signal is needed. It does not cover ``param.data = <fresh tensor>``, which keeps the
    Parameter object and restarts the counter at zero -- the collision against an entry stored
    at zero is silent, and ``Module._apply`` (``.to()``, ``.cuda()``, a dtype cast) swaps
    storage exactly that way. The storage address closes that hole.

    **The entry holds a reference to the latent tensor and to its storage.** That is what makes
    the address half of the key sound -- an address cannot be recycled while a live reference
    pins it -- and it is free when the tensor is a persistent parameter. Under FSDP2 the
    unsharded weight is transient and pinning it would defeat resharding, which is why
    :attr:`QuantConfig.cache_quantized_weight` defaults to off and has to be turned on
    deliberately.
    """

    __slots__ = ("_source", "_version", "_storage", "_data_ptr", "_quantized", "hits", "misses")

    def __init__(self) -> None:
        self._source: Optional[torch.Tensor] = None
        self._version: Optional[int] = None
        self._storage: Optional[torch.UntypedStorage] = None
        self._data_ptr: Optional[int] = None
        self._quantized: Optional[torch.Tensor] = None
        # Lifetime counters, deliberately not reset by `clear()`. An entry pins the latent
        # tensor, so a cache that never hits is worse than no cache -- and that is exactly what
        # happens under FSDP2, where the weight the forward sees is a fresh all-gather buffer on
        # every unshard and the key can never match. Counting makes that visible instead of
        # leaving it to be inferred from a throughput number that did not move.
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        """Drop the entry and release the tensors it holds."""
        self._source = None
        self._version = None
        self._storage = None
        self._data_ptr = None
        self._quantized = None

    def quantize(
        self, w: torch.Tensor, *, in_dim: int, delta_factor: float = TWN_DELTA_FACTOR
    ) -> torch.Tensor:
        """
        Return the quantized ``w`` with an identity-STE backward, reusing the memo if valid.

        :param w: The latent weight.
        :param in_dim: The axis the forward pass treats as input features. See
            :func:`twn_quantize`.
        :param delta_factor: The threshold constant. See :func:`twn_quantize`.

        :returns: The quantized weight, differentiable back to ``w``.
        """
        if (
            self._quantized is not None
            and self._source is w
            and self._version == w._version
            and self._data_ptr == w.data_ptr()
        ):
            self.hits += 1
            return _CachedSTE.apply(w, self._quantized)  # type: ignore[no-any-return]

        self.misses += 1

        # Release a stale entry before allocating its replacement. At M20 scale the cached
        # quantized weights nearly fill the remaining H100 memory; keeping each old tensor
        # alive until after its replacement is produced creates a transient double buffer
        # and can OOM while refreshing an otherwise viable cache.
        self.clear()
        quantized = twn_quantize(w, in_dim=in_dim, delta_factor=delta_factor)
        self._source = w
        self._version = w._version
        self._storage = w.untyped_storage()
        self._data_ptr = w.data_ptr()
        self._quantized = quantized
        return _CachedSTE.apply(w, quantized)  # type: ignore[no-any-return]


def reset_twn_quant_caches(module: nn.Module) -> int:
    """
    Clear every :class:`TWNQuantCache` reachable from ``module``.

    Only needed to release the memory an entry holds -- correctness does not depend on it,
    since entries invalidate themselves on the latent weight's version counter.

    :param module: The root module to walk.

    :returns: How many caches were cleared.
    """
    cleared = 0
    for submodule in module.modules():
        for attribute in vars(submodule).values():
            if isinstance(attribute, TWNQuantCache):
                attribute.clear()
                cleared += 1
    return cleared


@dataclass
class QuantConfig(Config):
    """
    Configuration for ternary QAT on a module's matmul weights.

    Three states, and the distinction between the last two is the whole point:

    * ``quant=None`` on the owning config -- stock behavior, plain :class:`torch.nn.Linear`.
    * ``QuantConfig(enabled=False)`` -- :class:`QuantLinear` is built but bypassed. Forward is
      **bitwise identical** to :class:`torch.nn.Linear`. This is the *control arm* of the
      paired bf16-vs-ternary comparison: same module graph, same parameter names, same
      state-dict keys, quantizer off. Using stock ``nn.Linear`` for the control instead would
      make it a comparison between two different models.
    * ``QuantConfig(enabled=True)`` -- the ternary arm.
    """

    enabled: bool = True
    """
    Whether the quantizer actually fires. ``False`` gives an exact-equality control arm.
    """
    delta_factor: float = TWN_DELTA_FACTOR
    """
    The TWN threshold constant, ``delta_r = delta_factor * mean_j |W_rj|``.

    ``0.7`` is TWN's own value, obtained by minimizing ``||W - alpha*W_t||^2`` -- reconstruction
    error, not loss -- and it places ~42% of the weights at zero. BitNet's round-to-nearest rule
    is an effective factor of ``0.5`` and lands at ~31%. Exposed because it decides the fate of
    every weight in the model and the right value is an empirical question, not a settled one;
    :func:`gaussian_zero_fraction` gives the closed form for a Gaussian latent.
    """
    cache_quantized_weight: bool = False
    """
    Reuse the quantized weight across microbatches until the optimizer moves the latent one.

    The quantized values are a pure function of the latent weight, so recomputing them for each
    microbatch of a gradient-accumulation window returns the same numbers every time. Turning
    this on makes the result bitwise identical while paying the scan once per optimizer step
    instead of once per microbatch. See :class:`TWNQuantCache` for how validity is tracked.

    Off by default because the entry holds the latent tensor alive. That costs nothing for a
    persistent parameter, but under FSDP2 the unsharded weight is transient and pinning it
    would defeat resharding.
    """


class QuantLinear(nn.Linear):
    """
    Ternary-QAT drop-in for :class:`torch.nn.Linear`. The latent weight stays full precision.

    Implemented as a *subclass* of :class:`torch.nn.Linear` rather than a fresh
    :class:`torch.nn.Module`, which buys several properties that a standalone module would
    have had to re-earn one at a time:

    * ``enabled=False`` is **exactly** ``nn.Linear`` -- the same ``F.linear(x, self.weight,
      self.bias)`` call, not a reimplementation that happens to agree.
    * The latent weight is ``self.weight`` and the bias is ``self.bias``, so **state dicts are
      identical** between the bf16 and ternary arms. Checkpoints interoperate; the paired
      comparison can share an init.
    * ``isinstance(m, nn.Linear)`` still holds, so ``init_linear``, the tensor-parallel
      wrappers, and ``normalize_matrices`` keep working untouched.
    * FSDP2 sees the same parameter structure and shards it the same way.

    :param in_features: Input dimensionality.
    :param out_features: Output dimensionality.
    :param bias: Include a bias. The bias is **never** quantized -- it is not a matmul.
    :param enabled: Whether to quantize. ``False`` is an exact ``nn.Linear``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        enabled: bool = True,
        cache_quantized_weight: bool = False,
        delta_factor: float = TWN_DELTA_FACTOR,
        device: Any = None,
        dtype: Any = None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.quant_enabled = enabled
        self.delta_factor = delta_factor
        self.quant_cache = TWNQuantCache() if cache_quantized_weight else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            # Exactly nn.Linear.forward. Do not "optimize" this into a shared code path with
            # the quantized branch -- the exact-equality property is what makes the control
            # arm a control.
            return F.linear(x, self.weight, self.bias)
        if self.quant_cache is not None:
            quantized = self.quant_cache.quantize(
                self.weight, in_dim=-1, delta_factor=self.delta_factor
            )
        else:
            quantized = twn_quantize_ste(
                self.weight, in_dim=-1, delta_factor=self.delta_factor
            )
        return F.linear(x, quantized, self.bias)

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, quant_enabled={self.quant_enabled}, "
            f"quant_cached={self.quant_cache is not None}"
        )

    def assert_no_tensor_parallel(self) -> None:
        """
        Refuse tensor parallelism on a quantized linear.

        TWN reduces over the input-feature axis. Under ``colwise_parallel`` the weight is
        sharded on ``out_features``, so each rank owns whole rows and the per-row statistics
        are still correct. Under ``rowwise_parallel`` it is sharded on ``in_features``, so
        ``mean|W|`` would be taken over only the local shard -- a per-row alpha computed from a
        fraction of the row. That is a silently different quantizer, so this raises rather
        than warns. Maple at 1.5B active needs expert parallelism, not tensor parallelism.
        """
        if self.quant_enabled:
            raise NotImplementedError(
                "tensor parallelism is not supported with ternary QAT: a row-wise-sharded "
                "weight would compute the TWN threshold over a fraction of each output row, "
                "which is a different quantizer. Use expert parallelism instead."
            )


def assert_no_float8_conflict(model: nn.Module) -> None:
    """
    Refuse float8 conversion on a model that has ternary QAT enabled.

    This guards a **silent** failure that follows directly from ``QuantLinear`` subclassing
    ``nn.Linear``. ``Float8Config.apply_float8_linear`` filters candidates with
    ``isinstance(m, nn.Linear)``, which a ``QuantLinear`` satisfies -- so
    ``convert_to_float8_training`` would **replace** every quantized projection with a
    ``Float8Linear``, discarding the quantizer entirely. The result is a run that reports itself
    as the ternary arm, trains happily, and is actually fp8: exactly the "wrong experiment that
    trains happily" failure mode the carve-out audit exists to prevent, arriving by a different
    door.

    The subclassing is still the right call -- it is what buys bitwise-exact ``enabled=False``,
    a shared state dict, and free compatibility with ``init_linear`` and the parallel wrappers.
    This is the one place that inheritance cuts the wrong way, so it gets an explicit guard
    rather than a comment.

    Not currently reachable in our run path (``.edullm/train_on_corpus.py`` sets no
    ``float8_config``), but it is one config field away from being reachable.

    :raises OLMoConfigurationError: if any enabled :class:`QuantLinear` is present.
    """
    offenders = [
        fqn for fqn, m in model.named_modules() if isinstance(m, QuantLinear) and m.quant_enabled
    ]
    if offenders:
        raise OLMoConfigurationError(
            f"float8 conversion would silently replace {len(offenders)} ternary-QAT "
            f"QuantLinear module(s) with Float8Linear, discarding the quantizer -- the run "
            f"would report as the ternary arm while actually training in fp8. Disable one or "
            f"the other. First offenders: {offenders[:4]}"
        )


# ===================================================================================
# Carve-out audit
# ===================================================================================


@dataclass
class QuantAuditEntry:
    """One module's audit result."""

    fqn: str
    kind: str
    quantized: bool
    numel: int


#: Substrings of a fully-qualified module/parameter name that identify a tensor which
#: **must** stay full precision. Not stylistic -- see the module docstring on the router, and
#: MoTE (arXiv:2506.14435), which measured a ternary shared expert at 48.2 against 57.3 for
#: BF16 and had to keep it in BF16.
FULL_PRECISION_NAME_MARKERS: Tuple[str, ...] = (
    "embeddings",
    "embedding",
    "lm_head",
    "router",
    "norm",
)


def _is_stacked_expert_mlp(module: nn.Module) -> bool:
    """
    Is this a stacked-expert MLP (``MoEMLP`` / ``DroplessMoEMLP``)?

    Identified structurally rather than by import, to avoid a circular import between
    ``nn.quantization`` and ``nn.moe.mlp``. The discriminating test is that ``w1``/``w2`` are
    raw ``nn.Parameter``s: a dense ``FeedForward`` also carries ``.quant``, ``.w1`` and ``.w2``
    but holds them as ``nn.Linear`` submodules.
    """
    if not all(hasattr(module, a) for a in ("quant", "w1", "w2", "w3")):
        return False
    return all(isinstance(getattr(module, a), nn.Parameter) for a in ("w1", "w2", "w3"))


def audit_quantization(model: nn.Module) -> Dict[str, Any]:
    """
    Walk a built model and report exactly which matmuls are ternary and which are not.

    Asserts magnitudes, not existence: it raises if any tensor matching
    :data:`FULL_PRECISION_NAME_MARKERS` is quantized, and it returns the counts so a caller
    can assert that the quantized set is non-empty and covers the tensors it is supposed to.
    A carve-out that gets silently dropped produces a wrong experiment that trains happily,
    which is the failure mode this exists to catch.

    :returns: A dict with ``entries``, ``num_quantized``, ``num_full_precision``,
        ``quantized_numel`` and ``full_precision_numel``.
    """
    entries: List[QuantAuditEntry] = []
    violations: List[str] = []

    for fqn, module in model.named_modules():
        quantized: Optional[bool] = None
        kind = type(module).__name__

        if isinstance(module, QuantLinear):
            quantized = bool(module.quant_enabled)
        elif isinstance(module, nn.Linear):
            quantized = False
        elif _is_stacked_expert_mlp(module):
            # Stacked-expert MLP (MoEMLP / DroplessMoEMLP): weights are raw Parameters, not
            # Linear submodules, so `isinstance(m, nn.Linear)` cannot see them and they have to
            # be identified structurally. The `nn.Parameter` test is what distinguishes them
            # from a dense `FeedForward`, which also has `.quant`, `.w1` and `.w2` but holds
            # them as `nn.Linear` submodules -- and whose projections are therefore already
            # counted individually by the `QuantLinear` branch above. Without that test a dense
            # FFN would be double-counted, inflating `num_quantized`.
            quant = getattr(module, "quant", None)
            quantized = bool(quant is not None and getattr(quant, "enabled", False))
        elif isinstance(module, nn.Embedding):
            quantized = False

        if quantized is None:
            continue

        numel = sum(p.numel() for p in module.parameters(recurse=False))
        entries.append(QuantAuditEntry(fqn=fqn, kind=kind, quantized=quantized, numel=numel))

        if quantized:
            lowered = fqn.lower()
            for marker in FULL_PRECISION_NAME_MARKERS:
                if marker in lowered:
                    violations.append(f"{fqn} ({kind}) matched full-precision marker '{marker}'")
                    break

    # The router holds its weight as a bare Parameter, not an nn.Linear, so the loop above
    # cannot see it. Check the parameter names directly as a second, independent net.
    for pname, _ in model.named_parameters():
        lowered = pname.lower()
        if "router" not in lowered:
            continue
        owner = pname.rsplit(".", 1)[0]
        mod = dict(model.named_modules()).get(owner)
        if isinstance(mod, QuantLinear) and mod.quant_enabled:
            violations.append(f"{pname} is a router parameter on an enabled QuantLinear")

    if violations:
        raise OLMoConfigurationError(
            "ternary carve-out violated -- these must stay full precision:\n  "
            + "\n  ".join(violations)
        )

    quantized_entries = [e for e in entries if e.quantized]
    return {
        "entries": entries,
        "num_quantized": len(quantized_entries),
        "num_full_precision": len(entries) - len(quantized_entries),
        "quantized_numel": sum(e.numel for e in quantized_entries),
        "full_precision_numel": sum(e.numel for e in entries if not e.quantized),
    }


def gaussian_zero_fraction(delta_factor: float = TWN_DELTA_FACTOR) -> float:
    """
    Closed-form zero fraction this quantizer produces on Gaussian latents.

    ``delta = f * E|W| = f * sigma * sqrt(2/pi)``, so the fraction below threshold is
    ``2 * Phi(f * sqrt(2/pi)) - 1``. Dimension-free and independent of ``sigma``, which is
    what makes it a usable assertion band on any real weight matrix. Returns 0.4237 at
    ``f = 0.7``.
    """
    z = delta_factor * math.sqrt(2.0 / math.pi)
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    return math.erf(z / math.sqrt(2.0))
