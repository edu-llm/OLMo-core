"""
Multiply-free ternary MoE decode kernels -- **INFERENCE / EXPORT PATH ONLY**.

Nothing in this module is reachable from the training path. It is imported by nothing in
``src/olmo_core/nn/`` and by nothing in ``.edullm/``. Training keeps using
:class:`~olmo_core.nn.quantization.QuantLinear` and :class:`~olmo_core.nn.moe.mlp.MoEMLP` with the
straight-through estimator, exactly as before. The precedent is the 4-bit ``lm_head``, adopted as an
**export-time transform** (``maple/agents/DECISIONS.md`` D-106 -- D-108) rather than a training
change; this is the same shape.

**EVERY PERFORMANCE STATEMENT IN THIS FILE IS UNMEASURED.** No number below was produced on
hardware by this lane. Where a measured figure is quoted it is attributed to the run that produced
it, with its device.

Why "multiply-free" is a weak premise, stated up front
-----------------------------------------------------
The identity is real::

    sum_i w_i x_i = alpha * ( sum_{w_i > 0} x_i  -  sum_{w_i < 0} x_i )

Adds and subtracts, plus one multiply by ``alpha`` per output element per expert. But on NVIDIA
hardware the multiply it removes **was never separately billed**:

1. ``a*b + c`` (``fma.rn.f32``) and ``a + c`` (``add.f32``) are *both one instruction* at the *same*
   throughput per SM per clock -- NVIDIA's own arithmetic-instruction table lists 16/32-bit FP add,
   multiply and multiply-add at one common rate. Removing the multiply therefore saves **zero**
   issue slots. INT32 add is listed at the same rate again, so an integer reformulation does not
   escape either.
2. There is **no add-only datapath** faster than the MAC datapath. Tensor cores have no add-only
   mode, so a formulation that avoids multiplies necessarily also avoids tensor cores.
3. At batch-1 decode the arithmetic is hidden behind memory anyway: a tuned gathered ternary GEMV
   already reached **747.9 GB/s = 99.76% of this device's measured achievable 749.7** on one L40S
   (``DECISIONS.md`` D-101, FarmShare job 1677635, sm_89). There is 0.24% left. Arithmetic nobody
   was waiting on cannot be removed for profit.

So the honest expectation is that the multiply-free arm **ties** the fused-multiply-add arm, and
that if it does not, the difference is a register/select-pressure artifact rather than saved
arithmetic. Both arms are therefore implemented behind one ``MULFREE`` flag over the *same* kernel,
same grid, same bytes, so the question gets an A/B rather than an assertion --
and :func:`ptx_arith_histogram` answers the separate, static question of whether the multiply-free
source *actually lowers* to multiply-free PTX or whether LLVM folds the select chain back into an
``fma``. That negative result, if it is one, is worth more than a speedup.

What this module is actually for
--------------------------------
The load-bearing win here is **not** arithmetic. It is **launch count**. Per-launch overhead was
measured at **23.39 us** (D-101, L40S), and a naive one-launch-per-(expert, matrix) decode issues
``L * k * 3`` launches per token for the expert path -- **576 at M20** (D-104's corrected figure),
which is 13.47 ms of pure dispatch and a ~74 tok/s ceiling *regardless of kernel quality*. This
module fuses to **two launches per layer**:

* :func:`fused_gathered_w13_swiglu` -- gather the top-``k`` experts, run ``w1`` and ``w3`` against
  the same activation vector, apply the SwiGLU clamp and gate, emit the activated hidden. One
  launch. Fusing ``w1`` with ``w3`` also halves the redundant ``x`` re-read that D-101 diagnosed as
  the cause of a 41% -> 87% swing in its own kernel.
* :func:`fused_gathered_w2_combine` -- gather the same experts, contract ``w2``, scale by
  ``alpha * router_weight``, and reduce over all ``k`` experts in-register. One launch, no scatter,
  no atomics, no host sync.

``w2`` consumes ``w1``/``w3``'s output, so a single launch per layer would need a grid-wide barrier
that Triton does not portably offer. Two is the floor without CUDA graphs.

**And the redundancy has to be said**: D-101 ruled CUDA graphs *mandatory*. Under graph capture the
entire decode step is one replay, so most of this launch-count reduction is **subsumed** by the
graph. What fusion still buys under a graph is fewer graph nodes and one fewer HBM round-trip for
the gate/up intermediate -- real, but much smaller than the 12x on raw launch count. Quote the 12x
against the *naive* path only, never against a graph-captured one.

Scope and non-scope
-------------------
* Batch-1 decode. At ``top_k=8`` each selected expert receives exactly one token, so this is a
  **fixed-M** problem, not a variable-M grouped GEMM -- which is why D-009's ``grouped_gemm``
  blocker does not bind here (D-101, strengthened to ``B ~ 32`` by a Poisson bound). Prefill is
  variable-M, genuinely *is* the grouped-GEMM problem, and is **not** addressed.
* ``FlashHead`` is out of scope by project rule (``maple/CLAUDE.md``) and is neither imported nor
  cited.
* The ``lm_head`` is **not** touched. It is 58.4% of M20 bytes/token and ternary on it is
  catastrophic (KL 3.53 nats, +1.099 bpb -- D-104/D-108); the adopted transform there is 4-bit.
* No attention/QKVO kernel. The dense ternary GEMV for those already exists (D-097).
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from ..exceptions import OLMoConfigurationError
from ..nn.quantization import twn_threshold_and_scale

__all__ = [
    "TERNARY_CODE_ZERO",
    "TERNARY_CODE_POS",
    "TERNARY_CODE_NEG",
    "TERNARY_CODE_ILLEGAL",
    "CODES_PER_BYTE",
    "EXPORT_RUNGS",
    "MEASURED_US_PER_LAUNCH_L40S",
    "TernaryExportSpec",
    "pack_ternary",
    "unpack_ternary_signs",
    "assert_no_illegal_codes",
    "PackedExpertBank",
    "pack_expert_bank",
    "fused_gathered_w13_swiglu",
    "fused_gathered_w2_combine",
    "fused_ternary_moe_decode",
    "reference_ternary_moe_decode",
    "ptx_arith_histogram",
]


# ===================================================================================
# The packing. 2-bit, byte-aligned, 4 codes/byte, contracted axis packed fastest.
# ===================================================================================

TERNARY_CODE_ZERO: int = 0
TERNARY_CODE_POS: int = 1
TERNARY_CODE_NEG: int = 2
TERNARY_CODE_ILLEGAL: int = 3
"""
The unused fourth state. Kept unused **deliberately**: it is a free integrity canary. A pack that
has been corrupted by a bad transpose, a truncated write, or a wrong-dtype view will very often
produce a ``3`` somewhere, and :func:`assert_no_illegal_codes` is an O(bytes) check that costs
nothing at export time and cannot be satisfied by the act of checking. See
:func:`assert_no_illegal_codes` for what it does *not* catch.
"""

CODES_PER_BYTE: int = 4

#: Per-launch dispatch overhead, **measured on L40S (sm_89)**, FarmShare job 1677635, D-101.
#: Quoted for the launch-count arithmetic only. It is **not** an A100 (sm_80) figure and must not
#: be used as one.
MEASURED_US_PER_LAUNCH_L40S: float = 23.39


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise OLMoConfigurationError(msg)


def _refuse_autograd(*tensors: Optional[torch.Tensor]) -> None:
    """
    Refuse any tensor that carries grad, because this is an inference/export kernel.

    Not decoration. The whole point of keeping this module off the training path is that a
    multiply-free forward would need a *matching custom backward* -- ``grad_x = grad_out @ W_q^T``
    is itself a ternary matmul, and ``grad_W = grad_out^T @ x`` has no ternary operand at all, so
    at least one of training's three GEMMs can never benefit. Silently running this in a graph that
    expects gradients would produce a model that trains happily and is not the one under test,
    which is the exact failure class ``nn/quantization.py``'s carve-out audit exists to prevent.
    """
    for t in tensors:
        if t is not None and t.requires_grad:
            raise OLMoConfigurationError(
                "ternary_moe is an INFERENCE/EXPORT-path kernel and has no backward. An input "
                "tensor has requires_grad=True. The training path must keep using QuantLinear / "
                "MoEMLP with twn_quantize_ste, whose STE backward is full precision. See this "
                "module's docstring."
            )


def pack_ternary(w: torch.Tensor, *, in_dim: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    TWN-ternarize ``w`` and pack it 2-bit, 4 codes/byte, with the contracted axis packed fastest.

    The quantizer is *not* reimplemented here: :func:`~olmo_core.nn.quantization.twn_threshold_and_scale`
    is called directly, so the exported artifact is ternarized by the identical rule the model was
    trained with (``delta = 0.7 * mean|W|`` per output row, ``alpha = mean|W| over survivors``).
    Reimplementing it would be the "silently a different quantizer" failure that module documents at
    length.

    ``in_dim`` is the **contracted** axis and is **required**, matching the caller contract of
    ``twn_quantize``. For OLMo-core's stacked expert weights as consumed by ``MoEMLP.forward``'s
    ``torch.bmm(x, w)``, ``in_dim=1`` for **all three** of ``w1``/``w2``/``w3`` -- ``bmm`` contracts
    ``w``'s axis ``-2`` unconditionally. Getting this wrong yields a per-*input*-row ``alpha``: a
    different quantizer that is shape-legal and does not raise.

    :param w: Latent weight, ``(..., A, B)`` with ``A`` contracted when ``in_dim=-2``.
    :param in_dim: The contracted axis.
    :returns: ``(codes, alpha, contract_len)`` where ``codes`` is ``uint8`` of shape
        ``(..., B, ceil(A/4))``, ``alpha`` is ``float32`` of shape ``(..., B)``, and
        ``contract_len`` is ``A`` -- returned rather than inferred because ``ceil(A/4)*4`` loses it
        and the kernel must mask the pad codes out via the *activation* index, not the code value.
    """
    _refuse_autograd(w)
    delta, alpha = twn_threshold_and_scale(w, in_dim=in_dim)
    w32 = w.detach().to(torch.float32)
    signs = torch.sign(w32) * (w32.abs() > delta)

    signs_t = signs.movedim(in_dim, -1).contiguous()  # (..., B, A)
    alpha_t = alpha.movedim(in_dim, -1).squeeze(-1).contiguous().to(torch.float32)  # (..., B)
    contract_len = int(signs_t.shape[-1])

    codes = torch.zeros_like(signs_t, dtype=torch.uint8)
    codes[signs_t > 0] = TERNARY_CODE_POS
    codes[signs_t < 0] = TERNARY_CODE_NEG

    pad = (-contract_len) % CODES_PER_BYTE
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad), value=TERNARY_CODE_ZERO)

    n_bytes = codes.shape[-1] // CODES_PER_BYTE
    quads = codes.reshape(*codes.shape[:-1], n_bytes, CODES_PER_BYTE).to(torch.uint8)
    packed = quads[..., 0].clone()
    for sub in range(1, CODES_PER_BYTE):
        packed |= quads[..., sub] << (2 * sub)
    return packed.contiguous(), alpha_t, contract_len


def unpack_ternary_signs(
    packed: torch.Tensor, contract_len: int, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Inverse of :func:`pack_ternary`'s code stage: ``(..., B, KB)`` uint8 -> ``(..., B, A)`` signs.

    Exact and integer-only. The round trip ``pack -> unpack`` is asserted bitwise by the
    correctness script -- a *bitwise* check, with no floating-point slack anywhere in it.
    """
    shifts = torch.arange(0, 2 * CODES_PER_BYTE, 2, device=packed.device, dtype=torch.int32)
    codes = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 3  # (..., B, KB, 4)
    codes = codes.reshape(*packed.shape[:-1], packed.shape[-1] * CODES_PER_BYTE)
    codes = codes[..., :contract_len]
    return (codes == TERNARY_CODE_POS).to(dtype) - (codes == TERNARY_CODE_NEG).to(dtype)


def assert_no_illegal_codes(packed: torch.Tensor) -> None:
    """
    Assert no 2-bit field holds the unused ``0b11`` state.

    **What this does not catch, stated so nobody mistakes it for a correctness proof:** a
    permutation of legal codes, a wrong-axis transpose that happens to stay legal, or a sign flip.
    All three are legal-code corruptions and pass this check. It is a cheap canary on *format*, not
    a check on *content* -- the content check is the bitwise integer-exact test in the correctness
    script, and the mutation check there deliberately uses a legal-code flip (1 -> 2) precisely so
    that it cannot be caught by this function.
    """
    shifts = torch.arange(0, 2 * CODES_PER_BYTE, 2, device=packed.device, dtype=torch.int32)
    codes = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 3
    n_bad = int((codes == TERNARY_CODE_ILLEGAL).sum().item())
    _require(
        n_bad == 0,
        f"{n_bad} ternary code fields hold the illegal state 0b11. The pack is corrupt or was "
        f"produced by something other than pack_ternary.",
    )


# ===================================================================================
# Rung-named entry points, so M20 and M7B are both reachable and nothing defaults on.
# ===================================================================================

#: Geometry per rung, for the **export path only**.
#:
#: This table is local rather than read from ``TransformerConfig.MAPLE_RUNGS`` because ``M7B``
#: is committed on ``agent/W-M7B/m7b-shape`` and does **not** exist on this lane's base
#: (``bcc05d6``), and this lane must not depend on an unmerged branch.
#: :func:`TernaryExportSpec.verify_against_transformer_config` closes that gap the other way: it
#: cross-checks every rung that *is* present in ``MAPLE_RUNGS``, so the check strengthens
#: automatically when W-M7B merges, without this file changing.
#:
#: ``M7B``'s row was transcribed from ``maple/plan/m7b-shape.md`` and then **independently
#: reproduced by closed-form parameter arithmetic** -- the closed form in ``maple/CLAUDE.md``
#: yields total 7,656,756,736 / active 635,491,840 / per-block 459,279,616 for this geometry,
#: matching the frozen figures exactly. Same check reproduces M20's 20,002,742,272 /
#: 1,279,369,216 / 816,320,768. Transcription is where every prior error in this ladder entered,
#: so the arithmetic is the evidence, not the transcription.
EXPORT_RUNGS: Dict[str, Dict[str, int]] = {
    "R0": dict(d_model=512, n_layers=8, num_experts=64, n_heads=4, n_kv_heads=1),
    "R1": dict(d_model=1024, n_layers=12, num_experts=64, n_heads=8, n_kv_heads=2),
    "R2": dict(d_model=1024, n_layers=12, num_experts=128, n_heads=8, n_kv_heads=2),
    "R3": dict(d_model=1024, n_layers=12, num_experts=256, n_heads=8, n_kv_heads=2),
    "E8": dict(d_model=1024, n_layers=12, num_experts=8, n_heads=8, n_kv_heads=2),
    "M20": dict(d_model=2048, n_layers=24, num_experts=256, n_heads=16, n_kv_heads=4),
    "M7B": dict(d_model=1536, n_layers=16, num_experts=256, n_heads=12, n_kv_heads=3),
}


@dataclass
class TernaryExportSpec:
    """
    One rung's geometry, for sizing the export-path kernels. Nothing here touches training.

    ``expert_hidden`` and ``top_k`` are *derived*, not stored: the ladder's frozen ratios are
    ``f_e/d = 1/4`` and ``k*f_e/d = 2.0``, which force ``f_e = d/4`` and ``k = 8``. Deriving them
    means a rung row cannot disagree with the ratios it is supposed to satisfy.
    """

    rung: str
    d_model: int
    n_layers: int
    num_experts: int
    n_heads: int
    n_kv_heads: int
    head_dim: int = 128

    @property
    def expert_hidden(self) -> int:
        return self.d_model // 4

    @property
    def top_k(self) -> int:
        return 8

    @classmethod
    def from_rung(cls, rung: str) -> "TernaryExportSpec":
        """
        Build a spec by rung name -- ``"M20"`` and ``"M7B"`` are both reachable here.

        This is the *only* entry point that names a model, and it is opt-in: importing this module
        changes no default and no training behaviour.
        """
        if rung not in EXPORT_RUNGS:
            raise OLMoConfigurationError(
                f"unknown export rung {rung!r}. Known: {sorted(EXPORT_RUNGS)}"
            )
        return cls(rung=rung, **EXPORT_RUNGS[rung])

    def __post_init__(self) -> None:
        # The D-012/D-014 assertion: attention geometry is 1.0x, not 2.0x. Kept here so an export
        # spec cannot silently describe a different model than the factory builds.
        _require(
            self.n_heads * self.head_dim == self.d_model,
            f"{self.rung}: n_heads*head_dim ({self.n_heads}*{self.head_dim}) != d_model "
            f"({self.d_model}). This is the 2.0x-vs-1.0x geometry error.",
        )
        _require(
            self.d_model % CODES_PER_BYTE == 0 and self.expert_hidden % CODES_PER_BYTE == 0,
            f"{self.rung}: d_model={self.d_model} and f_e={self.expert_hidden} must both be "
            f"multiples of {CODES_PER_BYTE} for the byte-aligned fast path.",
        )
        # NOTE ON n_kv_heads: this module has NO head handling at all -- the MoE expert path is
        # parameterised only by (d_model, expert_hidden, num_experts, top_k). So M7B's n_kv=3
        # cannot bind here, and there is nothing odd/even about it to get wrong. Recorded because
        # it was asked: for a *dense* ternary QKVO GEMV the only alignment requirement is that the
        # contracted dim be a multiple of 4, and the contracted dim there is d_model (1536 for
        # M7B), never n_kv_heads. n_kv*head_dim = 384 is itself a multiple of 4 as an *output*
        # extent, and output extents are masked per-block anyway, so odd or prime output extents
        # are legal. No n_kv parity assumption exists in this file.

    def verify_against_transformer_config(self) -> str:
        """
        Cross-check this row against ``TransformerConfig.MAPLE_RUNGS``, if that rung exists there.

        Returns a human-readable verdict rather than raising on absence, because ``M7B`` is
        legitimately absent from this lane's base commit. Raises only on **disagreement**, which is
        the case that would mean the export kernel is sized for a different model than the factory
        builds.
        """
        from ..nn.transformer.config import TransformerConfig  # lazy: avoids a circular import

        table = TransformerConfig.MAPLE_RUNGS
        if self.rung not in table:
            return (
                f"{self.rung}: ABSENT from TransformerConfig.MAPLE_RUNGS on this commit -- "
                f"not cross-checked. Expected for M7B on base bcc05d6."
            )
        ref = table[self.rung]
        mine = EXPORT_RUNGS[self.rung]
        diffs = {k: (mine.get(k), v) for k, v in ref.items() if mine.get(k) != v}
        _require(
            not diffs,
            f"{self.rung}: EXPORT_RUNGS disagrees with TransformerConfig.MAPLE_RUNGS on "
            f"{diffs}. The export kernel would be sized for a different model.",
        )
        return f"{self.rung}: agrees with TransformerConfig.MAPLE_RUNGS on {sorted(ref)}"

    # -- launch accounting -------------------------------------------------------------

    def launch_counts(self) -> Dict[str, float]:
        """
        Launch counts per token for the **expert path**, naive vs this module's fusion.

        Formulas printed rather than asserted, so the arithmetic is auditable:

        * ``naive       = n_layers * top_k * 3``   (one launch per (expert, matrix))
        * ``per_matrix  = n_layers * 3``           (fuse over experts only)
        * ``fused       = n_layers * 2``           (this module: w1+w3+SwiGLU, then w2+combine)

        ``fused`` cannot go to 1 per layer without a grid-wide barrier, because ``w2`` consumes
        ``w1``/``w3``'s output.

        The us/ceiling columns use the **L40S-measured** 23.39 us/launch (D-101) and are therefore
        L40S figures. **A CUDA-graph-captured decode is one replay, so these ceilings describe the
        naive and fused *ungraphed* paths only.**
        """
        L, k = self.n_layers, self.top_k
        naive = L * k * 3
        per_matrix = L * 3
        fused = L * 2
        us = MEASURED_US_PER_LAUNCH_L40S
        return {
            "naive_launches": naive,
            "per_matrix_launches": per_matrix,
            "fused_launches": fused,
            "reduction_vs_naive": naive / fused,
            "naive_dispatch_us": naive * us,
            "fused_dispatch_us": fused * us,
            "naive_dispatch_tok_s_ceiling": 1e6 / (naive * us),
            "fused_dispatch_tok_s_ceiling": 1e6 / (fused * us),
        }

    def expert_path_bytes_per_token(self) -> Dict[str, float]:
        """
        Strict expert-path bytes/token, packed and bf16, with the formula.

        ``per_layer_packed = k * ( 2*f_e*ceil(d/4) + d*ceil(f_e/4) )`` weight bytes
        ``               + k * ( 2*f_e + d ) * 4``                     alpha bytes (fp32)

        The alpha term is why the effective format is **2.0312 bit/weight at M20 and 2.0417 at
        M7B** rather than exactly 2.00 (alpha overhead 1.54% and 2.04% respectively), and it is
        counted rather than rounded away. Note this is *tighter* than the 2.25 bit/weight
        convention DeepGrove uses, and close to -- but not identical to -- the ~2.01-2.1 the prior
        dense probe reported for its own shapes. The nominal packed-vs-bf16 ratio is therefore
        **7.88x at M20 / 7.84x at M7B, not 8.00x**.
        """
        d, fe, k, L = self.d_model, self.expert_hidden, self.top_k, self.n_layers
        kb13 = -(-d // CODES_PER_BYTE)
        kb2 = -(-fe // CODES_PER_BYTE)
        w_packed = k * (2 * fe * kb13 + d * kb2)
        a_bytes = k * (2 * fe + d) * 4
        w_bf16 = k * (2 * fe * d + d * fe) * 2
        n_weights = k * (2 * fe * d + d * fe)
        return {
            "per_layer_packed_bytes": w_packed + a_bytes,
            "per_layer_bf16_bytes": w_bf16,
            "per_token_packed_bytes": (w_packed + a_bytes) * L,
            "per_token_bf16_bytes": w_bf16 * L,
            "alpha_overhead_frac": a_bytes / (w_packed + a_bytes),
            "effective_bits_per_weight": (w_packed + a_bytes) * 8 / n_weights,
            "nominal_packed_speedup": w_bf16 / (w_packed + a_bytes),
        }

    def arith_mix(self) -> Dict[str, float]:
        """
        Add/sub vs multiply counts per token for the expert path, in the multiply-free form.

        This is the accounting that makes "multiply-free" a checkable claim rather than a slogan:

        * ``adds  = L * k * ( 2*f_e*d + d*f_e )``  -- one add or sub per nonzero-or-zero code
        * ``muls  = L * k * ( 2*f_e + 2*d )``      -- alpha once per (expert, output element) in
          w1/w3, and ``alpha * router_weight`` then that product once per (expert, output element)
          in w2

        Note the honest correction to the brief's phrasing: alpha is applied **once per (expert,
        output element)**, not once per output element. It cannot be once per output element in the
        ``w2`` combine, because ``k=8`` experts each contribute to the same output element with a
        *different* per-row alpha and a *different* router weight, and eight distinct scales cannot
        be folded into one multiply. The ratio is ``adds/muls = 3*f_e*d / (2*f_e + 2*d)``, which is
        **614.4 at M20 and 460.8 at M7B** -- i.e. 0.3*d, *not* ~d, and I state the computed figure
        rather than the convenient approximation. Either way the arithmetic is **99.84% adds**, so
        "multiply-free" is an accurate description of the instruction mix. That it is accurate does
        not make it profitable: see this module's header on why the removed multiply was never
        separately billed.

        A conventional ``2 * MACs`` FLOP count is reported alongside, because that is the
        denominator any MFU number would use and it must be printed to be auditable.
        """
        d, fe, k, L = self.d_model, self.expert_hidden, self.top_k, self.n_layers
        macs = L * k * (2 * fe * d + d * fe)
        adds = macs
        muls = L * k * (2 * fe + 2 * d)
        return {
            "macs_per_token": macs,
            "conventional_flops_per_token_2xmacs": 2 * macs,
            "mulfree_adds_per_token": adds,
            "mulfree_muls_per_token": muls,
            "adds_per_mul": adds / muls,
        }


# ===================================================================================
# Triton kernels. Imported lazily so this module is importable on a CPU-only box.
# ===================================================================================


def _triton():
    try:
        import triton  # type: ignore
        import triton.language as tl  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise OLMoConfigurationError(
            "the ternary MoE decode kernels need Triton. It is present in the platform image and "
            "in FarmShare's shared venv (triton 3.4.0); it is not needed to import this module."
        ) from e
    return triton, tl


_KERNELS: Dict[str, object] = {}


def _build_kernels():
    """
    JIT-define both kernels once. Defined inside a function so ``import triton`` is lazy.
    """
    if "w13" in _KERNELS:
        return _KERNELS
    triton, tl = _triton()

    @triton.jit
    def _w13_swiglu(
        x_ptr,  # (R, D) activation, bf16 or fp32
        codes_ptr,  # (R_BANK, E, 2*FE, KB) uint8, contracted axis packed fastest
        alpha_ptr,  # (R_BANK, E, 2*FE) fp32
        idx_ptr,  # (R, K_TOP) int32 -- the gathered expert ids
        h_ptr,  # (R, K_TOP, FE) fp32 out
        D: tl.constexpr,
        FE: tl.constexpr,
        KB: tl.constexpr,
        E: tl.constexpr,
        K_TOP: tl.constexpr,
        BANK_STRIDE: tl.constexpr,  # elements to advance per r; 0 => cache-resident control arm
        ALPHA_BANK_STRIDE: tl.constexpr,
        X_STRIDE: tl.constexpr,  # 0 => reuse one activation vector
        SWIGLU_LIMIT: tl.constexpr,
        HAS_LIMIT: tl.constexpr,
        MULFREE: tl.constexpr,
        RPP: tl.constexpr,
        BLOCK_KB: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        slot = tl.program_id(1)
        pid_row = tl.program_id(2)

        eid = tl.load(idx_ptr + pid_r * K_TOP + slot).to(tl.int64)
        wbase = pid_r.to(tl.int64) * BANK_STRIDE + eid * (2 * FE * KB)
        abase = pid_r.to(tl.int64) * ALPHA_BANK_STRIDE + eid * (2 * FE)
        xbase = pid_r.to(tl.int64) * X_STRIDE

        rows = pid_row * RPP + tl.arange(0, RPP)
        rmask = rows < FE
        g_row = rows.to(tl.int64)
        u_row = (rows + FE).to(tl.int64)

        acc_g = tl.zeros((RPP,), dtype=tl.float32)
        acc_u = tl.zeros((RPP,), dtype=tl.float32)

        for kb0 in range(0, KB, BLOCK_KB):
            offs = kb0 + tl.arange(0, BLOCK_KB)
            kmask = offs < KB
            byte_g = tl.load(
                codes_ptr + wbase + g_row[:, None] * KB + offs[None, :].to(tl.int64),
                mask=rmask[:, None] & kmask[None, :],
                other=0,
            ).to(tl.int32)
            byte_u = tl.load(
                codes_ptr + wbase + u_row[:, None] * KB + offs[None, :].to(tl.int64),
                mask=rmask[:, None] & kmask[None, :],
                other=0,
            ).to(tl.int32)
            for sub in tl.static_range(4):
                xi = offs * 4 + sub
                xv = tl.load(x_ptr + xbase + xi, mask=xi < D, other=0.0).to(tl.float32)
                cg = (byte_g >> (2 * sub)) & 3
                cu = (byte_u >> (2 * sub)) & 3
                if MULFREE:
                    # No multiply: select x, -x, or 0. Negation is a sign-bit flip.
                    tg = tl.where(cg == 1, xv[None, :], tl.where(cg == 2, -xv[None, :], 0.0))
                    tu = tl.where(cu == 1, xv[None, :], tl.where(cu == 2, -xv[None, :], 0.0))
                else:
                    # Control arm: materialise +-1.0 and multiply-accumulate.
                    wg = tl.where(cg == 1, 1.0, tl.where(cg == 2, -1.0, 0.0))
                    wu = tl.where(cu == 1, 1.0, tl.where(cu == 2, -1.0, 0.0))
                    tg = xv[None, :] * wg
                    tu = xv[None, :] * wu
                acc_g += tl.sum(tg, axis=1)
                acc_u += tl.sum(tu, axis=1)

        # alpha applied ONCE per output element, at the end, per output row.
        a_g = tl.load(alpha_ptr + abase + g_row, mask=rmask, other=0.0)
        a_u = tl.load(alpha_ptr + abase + u_row, mask=rmask, other=0.0)
        gate = acc_g * a_g
        up = acc_u * a_u
        if HAS_LIMIT:
            gate = tl.minimum(gate, SWIGLU_LIMIT)
            up = tl.minimum(tl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)
        h = (gate * (1.0 / (1.0 + tl.exp(-gate)))) * up
        tl.store(h_ptr + (pid_r.to(tl.int64) * K_TOP + slot) * FE + g_row, h, mask=rmask)

    @triton.jit
    def _w2_combine(
        h_ptr,  # (R, K_TOP, FE) fp32
        codes_ptr,  # (R_BANK, E, D, KB) uint8
        alpha_ptr,  # (R_BANK, E, D) fp32
        idx_ptr,  # (R, K_TOP) int32
        rw_ptr,  # (R, K_TOP) fp32 router weights
        y_ptr,  # (R, D) fp32 out
        D: tl.constexpr,
        FE: tl.constexpr,
        KB: tl.constexpr,
        E: tl.constexpr,
        K_TOP: tl.constexpr,
        BANK_STRIDE: tl.constexpr,
        ALPHA_BANK_STRIDE: tl.constexpr,
        MULFREE: tl.constexpr,
        RPP: tl.constexpr,
        BLOCK_KB: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_row = tl.program_id(1)

        rows = pid_row * RPP + tl.arange(0, RPP)
        rmask = rows < D
        j = rows.to(tl.int64)
        acc = tl.zeros((RPP,), dtype=tl.float32)

        for slot in tl.static_range(K_TOP):
            eid = tl.load(idx_ptr + pid_r * K_TOP + slot).to(tl.int64)
            rw = tl.load(rw_ptr + pid_r * K_TOP + slot)
            wbase = pid_r.to(tl.int64) * BANK_STRIDE + eid * (D * KB)
            abase = pid_r.to(tl.int64) * ALPHA_BANK_STRIDE + eid * D
            hbase = (pid_r.to(tl.int64) * K_TOP + slot) * FE

            part = tl.zeros((RPP,), dtype=tl.float32)
            for kb0 in range(0, KB, BLOCK_KB):
                offs = kb0 + tl.arange(0, BLOCK_KB)
                kmask = offs < KB
                byte = tl.load(
                    codes_ptr + wbase + j[:, None] * KB + offs[None, :].to(tl.int64),
                    mask=rmask[:, None] & kmask[None, :],
                    other=0,
                ).to(tl.int32)
                for sub in tl.static_range(4):
                    hi = offs * 4 + sub
                    hv = tl.load(h_ptr + hbase + hi, mask=hi < FE, other=0.0).to(tl.float32)
                    c = (byte >> (2 * sub)) & 3
                    if MULFREE:
                        t = tl.where(c == 1, hv[None, :], tl.where(c == 2, -hv[None, :], 0.0))
                    else:
                        wv = tl.where(c == 1, 1.0, tl.where(c == 2, -1.0, 0.0))
                        t = hv[None, :] * wv
                    part += tl.sum(t, axis=1)

            a = tl.load(alpha_ptr + abase + j, mask=rmask, other=0.0)
            # Two multiplies per (expert, output element): fold the router weight into alpha, then
            # apply once. Cannot be one multiply per output element -- see arith_mix's docstring.
            acc += (rw * a) * part

        tl.store(y_ptr + pid_r.to(tl.int64) * D + j, acc, mask=rmask)

    _KERNELS["w13"] = _w13_swiglu
    _KERNELS["w2"] = _w2_combine
    _KERNELS["triton"] = triton
    return _KERNELS


# ===================================================================================
# Python-side launchers
# ===================================================================================


@dataclass
class PackedExpertBank:
    """
    A layer's full expert bank, packed for export. ``E`` experts, all of them resident.

    Batch-1 decode *touches* only ``top_k`` experts, but the whole bank is *resident*, and that is
    what makes real decode HBM-bound: at M20 one layer's packed bank is 201 MB against an L40S L2
    of 100.7 MB, so it cannot be cache-resident. A benchmark that sizes its buffer to the touched
    set instead of the resident set is the cache trap that produced a physically impossible 4.77x
    in this project once already (D-107).
    """

    w13_codes: torch.Tensor  # (R_BANK, E, 2*FE, KB13) uint8
    w13_alpha: torch.Tensor  # (R_BANK, E, 2*FE) fp32
    w2_codes: torch.Tensor  # (R_BANK, E, D, KB2) uint8
    w2_alpha: torch.Tensor  # (R_BANK, E, D) fp32
    d_model: int
    expert_hidden: int
    num_experts: int
    n_replicas: int = 1

    @property
    def resident_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.w13_codes, self.w13_alpha, self.w2_codes, self.w2_alpha)
        )


def pack_expert_bank(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    *,
    n_replicas: int = 1,
) -> PackedExpertBank:
    """
    Pack one layer's ``MoEMLP`` expert weights into a :class:`PackedExpertBank`.

    Shapes follow ``MoEMLP.forward`` after its ``.view``: ``w1``/``w3`` are ``(E, d, f_e)`` and
    ``w2`` is ``(E, f_e, d)``, and **all three use ``in_dim=1``** because ``torch.bmm(x, w)``
    contracts ``w``'s axis ``-2`` unconditionally.

    ``w1`` and ``w3`` are stacked on the *output* axis into a single ``(E, 2*f_e, KB)`` tensor --
    gate rows ``[0, f_e)``, up rows ``[f_e, 2*f_e)`` -- so one launch reads both against one load
    of the activation vector.

    :param n_replicas: Tile the bank this many times. Benchmark-only knob: it is how the harness
        pushes the working set past a multiple of L2 without pretending a smaller model. It costs
        real memory (``Tensor.repeat`` copies -- it is tile, not ``expand``), which is the point.
    """
    _refuse_autograd(w1, w2, w3)
    E = w1.shape[0]
    _require(
        w2.shape[0] == E and w3.shape[0] == E,
        f"expert count mismatch: {w1.shape[0]}, {w2.shape[0]}, {w3.shape[0]}",
    )
    d, fe = int(w1.shape[1]), int(w1.shape[2])
    _require(
        tuple(w3.shape) == (E, d, fe) and tuple(w2.shape) == (E, fe, d),
        f"expected w1/w3 (E,d,f_e)=({E},{d},{fe}) and w2 (E,f_e,d)=({E},{fe},{d}); got "
        f"w2={tuple(w2.shape)} w3={tuple(w3.shape)}",
    )

    c1, a1, len1 = pack_ternary(w1, in_dim=1)  # (E, fe, KB13), (E, fe)
    c3, a3, len3 = pack_ternary(w3, in_dim=1)
    c2, a2, len2 = pack_ternary(w2, in_dim=1)  # (E, d, KB2), (E, d)
    _require(len1 == d and len3 == d and len2 == fe, "packer contracted the wrong axis")

    w13_codes = torch.cat([c1, c3], dim=1).unsqueeze(0)  # (1, E, 2fe, KB13)
    w13_alpha = torch.cat([a1, a3], dim=1).unsqueeze(0)  # (1, E, 2fe)
    w2_codes = c2.unsqueeze(0)
    w2_alpha = a2.unsqueeze(0)

    if n_replicas > 1:
        w13_codes = w13_codes.repeat(n_replicas, 1, 1, 1).contiguous()
        w13_alpha = w13_alpha.repeat(n_replicas, 1, 1).contiguous()
        w2_codes = w2_codes.repeat(n_replicas, 1, 1, 1).contiguous()
        w2_alpha = w2_alpha.repeat(n_replicas, 1, 1).contiguous()

    return PackedExpertBank(
        w13_codes=w13_codes.contiguous(),
        w13_alpha=w13_alpha.contiguous(),
        w2_codes=w2_codes.contiguous(),
        w2_alpha=w2_alpha.contiguous(),
        d_model=d,
        expert_hidden=fe,
        num_experts=E,
        n_replicas=n_replicas,
    )


def _pow2_at_most(n: int) -> int:
    return 1 << max(0, int(math.floor(math.log2(max(1, n)))))


def fused_gathered_w13_swiglu(
    x: torch.Tensor,
    bank: PackedExpertBank,
    expert_idx: torch.Tensor,
    *,
    multiply_free: bool = True,
    swiglu_limit: Optional[float] = 7.0,
    rows_per_prog: int = 8,
    block_kb: Optional[int] = None,
    num_warps: int = 2,
    bank_stride_replicas: bool = True,
    reuse_x: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    One launch: gather ``top_k`` experts, run ``w1`` and ``w3``, clamp, SwiGLU, emit hidden.

    :param x: ``(R, d)``. ``R`` is the number of *independent decode steps* batched into one
        dispatch -- a benchmark-amortisation axis, **not** a token batch. Real decode uses ``R=1``;
        the harness uses ``R>1`` to keep the launch-count trap (D-097) from swallowing the
        bandwidth signal.
    :param expert_idx: ``(R, top_k)`` int32.
    :param multiply_free: ``True`` selects the add/sub/select accumulate; ``False`` the
        multiply-accumulate control. Same kernel, same grid, same bytes.
    :param rows_per_prog: Output rows per program. D-101 measured ``rpp=8, num_warps=2`` optimal
        **on L40S** and expects the point to move on other devices; it is a knob, not a constant.
    """
    _refuse_autograd(x)
    triton = _build_kernels()["triton"]
    kern = _KERNELS["w13"]
    R, d = int(x.shape[0]), int(x.shape[1])
    E = bank.num_experts
    fe = bank.expert_hidden
    kb = int(bank.w13_codes.shape[-1])
    k_top = int(expert_idx.shape[1])
    _require(d == bank.d_model, f"x has d={d}, bank has d={bank.d_model}")
    _require(expert_idx.shape[0] == R, "expert_idx must have one row per decode step")
    _require(expert_idx.dtype == torch.int32, "expert_idx must be int32")

    if out is None:
        out = torch.empty((R, k_top, fe), device=x.device, dtype=torch.float32)
    bkb = _pow2_at_most(kb) if block_kb is None else block_kb
    rpp = _pow2_at_most(rows_per_prog)

    per_bank13 = E * 2 * fe * kb
    per_alpha13 = E * 2 * fe
    kern[(R, k_top, -(-fe // rpp))](
        x,
        bank.w13_codes,
        bank.w13_alpha,
        expert_idx,
        out,
        D=d,
        FE=fe,
        KB=kb,
        E=E,
        K_TOP=k_top,
        BANK_STRIDE=(per_bank13 if bank_stride_replicas and bank.n_replicas > 1 else 0),
        ALPHA_BANK_STRIDE=(per_alpha13 if bank_stride_replicas and bank.n_replicas > 1 else 0),
        X_STRIDE=(0 if reuse_x else d),
        SWIGLU_LIMIT=(0.0 if swiglu_limit is None else float(swiglu_limit)),
        HAS_LIMIT=swiglu_limit is not None,
        MULFREE=multiply_free,
        RPP=rpp,
        BLOCK_KB=bkb,
        num_warps=num_warps,
    )
    del triton
    return out


def fused_gathered_w2_combine(
    h: torch.Tensor,
    bank: PackedExpertBank,
    expert_idx: torch.Tensor,
    router_w: torch.Tensor,
    *,
    multiply_free: bool = True,
    rows_per_prog: int = 8,
    block_kb: Optional[int] = None,
    num_warps: int = 2,
    bank_stride_replicas: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    One launch: gather the same ``top_k`` experts, contract ``w2``, scale, reduce over experts.

    The reduction over experts happens **in-register**, so there is no scatter kernel, no
    ``index_add``, no atomics, and no host sync -- each program owns a disjoint block of output
    rows.
    """
    _refuse_autograd(h, router_w)
    triton = _build_kernels()["triton"]
    kern = _KERNELS["w2"]
    R, k_top, fe = (int(v) for v in h.shape)
    d = bank.d_model
    E = bank.num_experts
    kb = int(bank.w2_codes.shape[-1])
    _require(fe == bank.expert_hidden, f"h has f_e={fe}, bank has {bank.expert_hidden}")
    _require(expert_idx.dtype == torch.int32, "expert_idx must be int32")
    _require(router_w.dtype == torch.float32, "router_w must be float32")

    if out is None:
        out = torch.empty((R, d), device=h.device, dtype=torch.float32)
    bkb = _pow2_at_most(kb) if block_kb is None else block_kb
    rpp = _pow2_at_most(rows_per_prog)

    per_bank2 = E * d * kb
    per_alpha2 = E * d
    kern[(R, -(-d // rpp))](
        h,
        bank.w2_codes,
        bank.w2_alpha,
        expert_idx,
        router_w,
        out,
        D=d,
        FE=fe,
        KB=kb,
        E=E,
        K_TOP=k_top,
        BANK_STRIDE=(per_bank2 if bank_stride_replicas and bank.n_replicas > 1 else 0),
        ALPHA_BANK_STRIDE=(per_alpha2 if bank_stride_replicas and bank.n_replicas > 1 else 0),
        MULFREE=multiply_free,
        RPP=rpp,
        BLOCK_KB=bkb,
        num_warps=num_warps,
    )
    del triton
    return out


def fused_ternary_moe_decode(
    x: torch.Tensor,
    bank: PackedExpertBank,
    expert_idx: torch.Tensor,
    router_w: torch.Tensor,
    *,
    multiply_free: bool = True,
    swiglu_limit: Optional[float] = 7.0,
    rows_per_prog: int = 8,
    num_warps: int = 2,
    **kwargs,
) -> torch.Tensor:
    """
    One layer's MoE expert path at batch-1 decode, in **two** launches.

    Returns ``(R, d)`` float32. Router weights must already be normalised -- Maple's
    ``norm_topk_prob`` is OLMo-core's ``normalize_expert_weights``, whose stock default is measured
    broken (gate mass 0.161 vs 1.000, a 6.2x error; ``maple/CLAUDE.md``). This kernel does **not**
    normalise for you, because silently renormalising would hide exactly that bug.
    """
    h = fused_gathered_w13_swiglu(
        x,
        bank,
        expert_idx,
        multiply_free=multiply_free,
        swiglu_limit=swiglu_limit,
        rows_per_prog=rows_per_prog,
        num_warps=num_warps,
        **kwargs,
    )
    return fused_gathered_w2_combine(
        h,
        bank,
        expert_idx,
        router_w,
        multiply_free=multiply_free,
        rows_per_prog=rows_per_prog,
        num_warps=num_warps,
        **kwargs,
    )


# ===================================================================================
# References, for the correctness script
# ===================================================================================


def reference_ternary_moe_decode(
    x: torch.Tensor,
    bank: PackedExpertBank,
    expert_idx: torch.Tensor,
    router_w: torch.Tensor,
    *,
    swiglu_limit: Optional[float] = 7.0,
    dtype: torch.dtype = torch.float32,
    distribute_alpha: bool = False,
    replica: int = 0,
) -> torch.Tensor:
    """
    Torch reference. Two variants, and the distinction is the whole tolerance argument.

    ``distribute_alpha=False`` (**the matched reference**) computes ``alpha * (signs @ x)``: the
    signs are exactly ``+-1``, so every product is exact and the *only* difference from the kernel
    is summation **order**. That makes the tolerance a pure summation-order bound, derivable from
    the block structure, with no second effect mixed in.

    ``distribute_alpha=True`` (**the naive dequantised reference**) computes ``(alpha*signs) @ x``,
    i.e. it materialises a dense tensor holding ``+-alpha`` and matmuls -- which is what a naive
    dequantise-then-matmul implementation does, and what the training forward does today. This
    rounds ``alpha*x_i`` once per term instead of once per output, so it is *less* accurate than
    the kernel, and its disagreement with the kernel is **not** evidence of a kernel bug. Any
    tolerance quoted against it must be looser, and separately justified.

    :param dtype: Accumulate dtype. Pass ``torch.float64`` for ground truth: with ``K <= 2^11``
        and ``+-1`` coefficients, an fp64 sum is exact to ~1e-16 relative, so it is a legitimate
        oracle for scoring *both* the kernel and the fp32 reference.
    """
    d, fe = bank.d_model, bank.expert_hidden
    s13 = unpack_ternary_signs(bank.w13_codes[replica], d, dtype=dtype)  # (E, 2fe, d)
    s2 = unpack_ternary_signs(bank.w2_codes[replica], fe, dtype=dtype)  # (E, d, fe)
    a13 = bank.w13_alpha[replica].to(dtype)
    a2 = bank.w2_alpha[replica].to(dtype)

    R = int(x.shape[0])
    out = torch.zeros((R, d), device=x.device, dtype=dtype)
    for r in range(R):
        xr = x[r].to(dtype)
        acc = torch.zeros((d,), device=x.device, dtype=dtype)
        for s in range(int(expert_idx.shape[1])):
            e = int(expert_idx[r, s].item())
            if distribute_alpha:
                w13 = s13[e] * a13[e].unsqueeze(-1)
                gate_up = w13 @ xr
            else:
                gate_up = (s13[e] @ xr) * a13[e]
            gate, up = gate_up[:fe], gate_up[fe:]
            if swiglu_limit is not None:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
            hv = torch.nn.functional.silu(gate) * up
            if distribute_alpha:
                contrib = (s2[e] * a2[e].unsqueeze(-1)) @ hv
            else:
                contrib = (s2[e] @ hv) * a2[e]
            acc = acc + router_w[r, s].to(dtype) * contrib
        out[r] = acc
    return out


# ===================================================================================
# The static question: does "multiply-free" actually lower to multiply-free?
# ===================================================================================


def ptx_arith_histogram(kernel_name: str = "w13") -> Dict[str, Dict[str, int]]:
    """
    Count arithmetic instructions in the generated PTX for both arms of the same kernel.

    This is the decisive test of the brief's own hypothesis that *"on real hardware a
    'multiply-free' formulation often lowers to the same tensor-core path anyway."* It is **static**
    -- it needs a compile, not a timing run, so it cannot be confounded by cache residency, clock
    drift, or dispatch overhead. If ``fma.rn.f32`` and ``mul.f32`` counts are equal between the
    ``MULFREE=True`` and ``MULFREE=False`` arms, then LLVM folded the select chain back into a
    multiply and *"multiply-free" is a source-level fiction on this toolchain* -- which is a real
    and useful negative result.

    Requires a compiled kernel, so it must run where Triton can compile for the target arch --
    i.e. in the GPU job, not on the laptop.
    """
    counts: Dict[str, Dict[str, int]] = {}
    want = (
        "fma.rn.f32",
        "mul.f32",
        "add.f32",
        "sub.f32",
        "selp.f32",
        "neg.f32",
        "mma.",
        "wgmma.",
        "ld.global",
    )
    kern = _KERNELS.get(kernel_name)
    if kern is None:
        raise OLMoConfigurationError(
            f"kernel {kernel_name!r} has not been compiled yet -- run it once first so Triton "
            "populates its cache, then call this."
        )
    for sig, cfunc in getattr(kern, "cache", {}).get(0, {}).items():
        ptx = getattr(cfunc, "asm", {}).get("ptx")
        if ptx is None:
            continue
        counts[str(sig)] = {w: ptx.count(w) for w in want}
    return counts
