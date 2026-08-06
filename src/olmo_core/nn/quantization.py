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
    "audit_quantization",
    "QuantAuditEntry",
]


TWN_DELTA_FACTOR: float = 0.7
"""
The TWN threshold constant: ``delta = TWN_DELTA_FACTOR * mean|W|`` per output row.

Confidence on the family is high; on this exact constant, medium-high (see
``maple/plan/maple-preview-recipe-hypotheses.md`` section 3). **Do not tune it.** It is a
faithfulness parameter, and the zero-fraction match is the only evidence pinning it.
"""

TWN_GAUSSIAN_ZERO_FRACTION: float = 0.4237
"""
Zero fraction TWN produces on Gaussian latents, closed form and dimension-free.

``delta = 0.7 * E|W| = 0.7 * sigma * sqrt(2/pi) = 0.5585 * sigma``, so the zero fraction is
``2 * Phi(0.5585) - 1 = 0.4237``. This is the discriminating assertion: a build that produces
0.31 has been "corrected" to BitNet b1.58.
"""

BITNET_B158_GAUSSIAN_ZERO_FRACTION: float = 0.3100
"""
What BitNet b1.58 would produce on the same latents: ``2 * Phi(0.399) - 1 = 0.310``. Recorded
so a test can assert we are *not* here.
"""


def twn_threshold_and_scale(
    w: torch.Tensor, *, in_dim: int = -1
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
        is ``-1``. For stacked expert weights it depends on the orientation the forward pass
        uses -- see :func:`twn_quantize`.

    :returns: ``(delta, alpha)``, both float32 and both keeping ``in_dim`` as a size-1 axis so
        they broadcast against ``w``.
    """
    w32 = w.detach().to(torch.float32)
    absw = w32.abs()
    delta = TWN_DELTA_FACTOR * absw.mean(dim=in_dim, keepdim=True)
    mask = absw > delta
    # A row survives its own threshold unless it is identically zero: delta = 0.7 * mean|W| is
    # strictly below max|W| whenever any element is nonzero. The clamp only guards the
    # all-zeros row, where it correctly yields alpha = 0 (and hence W_q = 0).
    count = mask.sum(dim=in_dim, keepdim=True).clamp_(min=1)
    alpha = (absw * mask).sum(dim=in_dim, keepdim=True) / count
    return delta, alpha


def twn_quantize(w: torch.Tensor, *, in_dim: int = -1) -> torch.Tensor:
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
    """
    delta, alpha = twn_threshold_and_scale(w, in_dim=in_dim)
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
    def forward(ctx, w: torch.Tensor, in_dim: int) -> torch.Tensor:  # type: ignore[override]
        del ctx
        return twn_quantize(w, in_dim=in_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        del ctx
        # Identity STE: the gradient of the quantized weight is the gradient of the latent
        # weight. `None` for the non-tensor `in_dim` argument.
        return grad_output, None


def twn_quantize_ste(w: torch.Tensor, *, in_dim: int = -1) -> torch.Tensor:
    """
    :func:`twn_quantize` in the forward direction, identity straight-through in the backward.

    Use this anywhere a quantized weight feeds a matmul during training. See
    :func:`twn_quantize` for the ``in_dim`` orientation table -- getting it wrong builds a
    different quantizer without erroring.
    """
    return _TWNQuantizeSTE.apply(w, in_dim)  # type: ignore[no-any-return]


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
        device: Any = None,
        dtype: Any = None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.quant_enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            # Exactly nn.Linear.forward. Do not "optimize" this into a shared code path with
            # the quantized branch -- the exact-equality property is what makes the control
            # arm a control.
            return F.linear(x, self.weight, self.bias)
        return F.linear(x, twn_quantize_ste(self.weight, in_dim=-1), self.bias)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, quant_enabled={self.quant_enabled}"

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
        elif hasattr(module, "quant") and hasattr(module, "w1") and hasattr(module, "w2"):
            # Stacked-expert MLP (MoEMLP / DroplessMoEMLP): weights are raw Parameters, not
            # Linear submodules, so identify it structurally.
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
