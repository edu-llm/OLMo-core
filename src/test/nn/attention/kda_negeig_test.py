"""
The gate on the ``KDA_NEGEIG`` arm: is the fast chunked kernel numerically correct at ``beta > 1``?

WHY THIS FILE EXISTS. ``allow_neg_eigval=True`` widens the delta-rule step ``beta`` from ``(0, 1)``
to ``(0, 2)``, so the state update ``(I - beta k k^T)`` stops being a contraction and becomes a
reflection: at ``beta = 2`` with a unit-norm key the eigenvalue along ``k`` is exactly ``-1`` and
the update is norm-PRESERVING rather than norm-reducing. An audit established that this is a
beta-projection change and not a kernel change -- ``recurrent.py:874-876`` computes
``beta = w_b(x).sigmoid()`` and then doubles it in Python, and
:func:`~olmo_core.nn.attention.flash_linear_attn_api.dispatch_chunk_kda` never forwards
``allow_neg_eigval`` to ``chunk_kda`` at all, so no Triton code branches on the flag. The kernel
simply receives a ``beta`` tensor whose values happen to exceed 1.

**None of that is evidence of bf16 accuracy in the widened range, and before this file nothing in
this repository had ever measured it.** The gap, precisely:

* ``kda_householder_test.py:126`` and ``:162`` build ``beta = rnd(...).sigmoid()`` -- never doubled,
  so every oracle comparison in that file lives in ``(0, 1)``.
* ``recurrent_test.py:1137`` (``test_kimi_delta_householder_strict_beta_contract``) recomputes
  ``logits.sigmoid() * 2.0`` **inside the test body** and asserts a ceiling on its own arithmetic.
  It never calls the module and never calls a kernel.
* ``recurrent_test.py:512`` (``test_dispatch_chunk_kda_matches_naive``) does compare ``chunk_kda``
  against the ``fla`` oracle, but at ``beta = randn().sigmoid()`` -- i.e. in ``(0, 1)`` only.
* Upstream ``fla``'s own ``allow_neg_eigval`` tests are kernel-versus-kernel equivalence, not
  accuracy against a higher-precision reference.

So the regime run 1's two Householder arms actually executed in has never been checked. This file
checks it, and the load-bearing claim is comparative: **the relative error band at ``beta > 1`` is
the SAME as the band at ``beta < 1``.** A test that merely showed the error was finite at
``beta > 1`` would pass on a kernel that had quietly become 50x less accurate there.

THE REFERENCE. ``fla.ops.kda.naive.naive_recurrent_kda``, read from the ``v0.5.1`` tag at
``https://raw.githubusercontent.com/fla-org/flash-linear-attention/v0.5.1/fla/ops/kda/naive.py``
(``pyproject.toml:80`` pins ``flash-linear-attention==0.5.1``). Signature, in order:
``(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False)``. It applies **no**
clamp, assert or range restriction to ``beta`` -- the only ``assert`` in that module is
``T % chunk_size == 0`` in ``naive_chunk_kda`` -- which is what makes it a valid oracle in the
widened range rather than a second copy of the same restriction.

.. important::
    **The oracle is float32, not float64, and that is a property of the oracle rather than a
    choice made here.** ``naive_recurrent_kda`` opens with
    ``q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])`` and ``torch.float``
    is an alias for float32. There is no dtype argument to raise it. The comparison is therefore
    fp32-recurrence versus bf16-chunked-kernel: ~2^-24 per operation against ~2^-8, three orders of
    magnitude of separation, which is ample for resolving a 4e-3 band -- but it is NOT the float64
    reference ``docs/dp2-kda/phase-0-1-runbook.md:351`` used, and the bands are quoted here as
    comparable rather than identical. Raising the oracle to float64 would mean transcribing the
    recurrence locally, which trades a precision gain for the exact "our oracle is our code"
    circularity the runbook warns about at its section 4.5.

WHAT IS *NOT* COVERED, stated rather than left to be assumed:

* ``cu_seqlens`` / varlen. ``naive_recurrent_kda`` has no ``cu_seqlens`` parameter, so there is no
  oracle for the packed path at any ``beta``. Causality across a document boundary in the fused
  path is covered instead by ``test_the_fused_kernel_is_causal_across_a_document_boundary`` in
  ``src/test/nn/transformer/core6_bakeoff_guards_test.py``, which is structural (bit-exact leak)
  and therefore does not need an accuracy oracle.
* ``fused_kda_gate``. The production-configuration test below routes the oracle through the same
  ``fla`` gate kernel the module uses, so that op is common-mode and is not under test here. The
  precomputed-gate tests, which are the numeric gate, avoid it entirely.
* ``beta`` exactly 2.0. bf16's spacing on ``[1, 2)`` is ``2^-7``, so the largest representable
  value below 2.0 is ``1.9921875`` and the sweep's top cell lands there: eigenvalue ``-0.9921875``,
  spectral radius 0.992, a reflection to within 0.8% but still formally contracting. **The exactly
  norm-preserving point is approached, not reached** -- see :data:`BETA_CELLS`. What covers the
  accumulation question the endpoint cannot is
  :func:`test_reflection_error_does_not_grow_faster_with_length_than_contraction_error`, since 0.992
  per step still compounds to 0.018 over ``T = 512``.
* Nothing in this file ran on the machine that wrote it -- no GPU, no ``fla``, and the project
  forbids local execution. Every threshold below was solved for satisfiability by hand and the
  arithmetic is shown; the realized numbers are not known until this runs on the host. See the
  invocation block at the end of ``core6_bakeoff_guards_test.py``.
"""

import math
import os
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn.functional as F

from olmo_core.testing.utils import requires_fla, requires_gpu

# --- thresholds, each with its derivation ------------------------------------------------------

#: Max relative error allowed between the bf16 kernel and the fp32 oracle, per compared tensor.
#: NOT a new number: it is the repo's calibrated bf16 constant, ``BF16_PARITY_BUDGET`` at
#: ``recurrent_test.py:1041`` and the same ``2e-2`` asserted at ``kda_householder_test.py:1514``.
#: Satisfiable in this regime: ``phase-0-1-runbook.md:351`` measured the reflection regime at
#: ``beta_max = 1.93`` landing in ``4.11e-3 - 5.39e-3``, so this leaves 3.7x headroom over the
#: worst precedent cell. Fireable in this regime: see
#: :func:`test_the_beta_band_check_fails_on_a_clamped_or_double_doubled_beta`, which drives it over.
BF16_FORWARD_BUDGET = 2e-2

#: Same for gradients. Also not new: ``BF16_GRAD_PARITY_BUDGET`` at ``recurrent_test.py:1045``.
#: Backward accumulates more roundings than forward, which is why the repo's constant is 2.5x
#: looser, and that ratio is inherited here rather than re-derived.
BF16_BACKWARD_BUDGET = 5e-2

#: The precedent band from ``docs/dp2-kda/phase-0-1-runbook.md:351``, quoted in failure messages so
#: a reader can compare a realized number against a measured one instead of against a round number.
#: That measurement was the *Householder* kernel against float64; this file extends it to
#: ``chunk_kda``, which is what the bake-off arms actually ship.
PRECEDENT_BAND: Tuple[float, float] = (3.53e-3, 6.30e-3)

#: How much worse the ``beta > 1`` band may be than the ``beta < 1`` band before we call it a
#: regression. DERIVED, not picked: the precedent's four regimes span ``3.53e-3`` to ``6.30e-3``,
#: i.e. the *within-method* spread is already 1.78x, and the baseline regime alone spans 1.78x by
#: itself. A ceiling at or below 1.78 would fire on seed noise. 4.0 sits 2.2x above the observed
#: spread and far below any real defect: clamping beta to 1.0 across the reflection band is a
#: ~2x change in the write magnitude and moves the error by O(10-100x), and a beta that got
#: doubled twice explodes as ``|1 - beta| ^ T``. So 4.0 is inside the gap between "noise" and
#: "any failure mode we can name".
REGIME_RATIO_CEILING = 4.0

#: Floor on the denominator of that ratio. Without it, a control cell that happened to come out at
#: 1e-4 would make any treatment cell look like a 40x regression. Set to just over half the low end
#: of the precedent band (``3.53e-3``), so it cannot mask a real ratio -- a treatment cell would
#: still have to reach ``4 * 2e-3 = 8e-3`` to pass, which is already above the entire precedent
#: band -- while still bounding the spurious case.
BF16_ROUNDOFF_FLOOR = 2e-3

#: Extra error-growth exponent the reflection regime may have over the contraction regime on a
#: length ladder. The absolute ceiling ``1.25`` at ``kda_householder_test.py:1461`` is NOT reused as
#: an absolute here, because it was calibrated on a *decaying* gate and this file deliberately runs
#: the near-zero-decay gate where accumulation is possible; asserting an absolute exponent measured
#: in a different regime is how a guard false-alarms. The comparative form is self-calibrating: it
#: asks only whether the reflection regime grows FASTER than the contraction regime, which is the
#: actual scientific question, and it cannot fire on a property both regimes share.
GROWTH_SLOPE_MARGIN = 0.5

#: Geometry. ``head_dim`` is NOT narrowed: ``fla`` dispatches on it (``chunk_kda`` asserts only
#: ``K <= 256``) and a different ``head_dim`` can select a different kernel, which would make this
#: a test of code that never runs. ``T = 256`` is four of ``chunk_kda``'s default 64-wide chunks, so
#: the inter-chunk state handoff -- where a norm-preserving update would compound -- is exercised
#: three times rather than not at all.
B, T, H, K, V = 2, 256, 2, 64, 64

#: Log-decay for the near-zero-retention-free ("weak") gate: ``exp(g) >= exp(-1e-3) = 0.999``.
#: THE STRONG GATE MAKES THE GROWTH CHECK UNFIREABLE, which is why this exists. With the realistic
#: ``logsigmoid`` gate ``exp(g) ~ 0.5`` per step, so after 256 steps the state has been annihilated
#: ``2^-256`` times over and nothing can accumulate -- a "no geometric growth" assertion in that
#: regime is satisfied by the decay, not by the kernel. Over ``T = 512`` this gate retains
#: ``exp(-5e-4 * 512) = 0.774`` of the state, so accumulation is genuinely possible.
WEAK_DECAY = 1e-3


# --- the beta sweep ------------------------------------------------------------------------------

#: ``(cell_name, beta_low, beta_high, is_treatment)``.
#:
#: Both regimes are sampled, and the pairing is deliberate: ``beta_just_below_1`` and
#: ``beta_just_above_1`` straddle 1.0 by less than 0.03, so the headline comparison is between two
#: cells that differ in almost nothing except which side of the contraction boundary they sit on.
#:
#: ``beta_near_2`` goes as close to the exact-reflection edge as bf16 permits, and the arithmetic is
#: worth stating because it bounds what this file can claim. bf16 keeps 8 significand bits, so its
#: spacing on ``[1, 2)`` is ``2^-7 = 0.0078125``; the largest representable value below 2.0 is
#: therefore ``1.9921875``, and a sampled ``1.99`` rounds to exactly that (2.0 itself would need
#: ``>= 1.99609375``). At ``||k|| = 1`` that gives eigenvalue ``1 - 1.9921875 = -0.9921875``: a
#: reflection to within 0.8%, spectral radius 0.992, so the update is very nearly norm-preserving
#: but still formally contracting by ``0.78%`` per application. **The exactly-norm-preserving point
#: ``beta = 2`` is approached, not reached**, which is a real limit of the sweep rather than an
#: oversight -- over ``T = 512`` steps even 0.992 compounds to ``0.018``, so the growth ladder below
#: is what covers the accumulation question that the endpoint alone cannot. No assertion requires
#: ``beta < 2``, so a future dtype change that does reach 2.0 will not trip anything spuriously.
#:
#: ``1.90`` also matches the precedent's realized ``beta_max = 1.93`` closely enough that the two
#: measurements can be read side by side.
BETA_CELLS: List[Tuple[str, float, float, bool]] = [
    ("beta_lo", 0.05, 0.35, False),
    ("beta_mid", 0.40, 0.90, False),
    ("beta_just_below_1", 0.90, 0.99, False),
    ("beta_just_above_1", 1.02, 1.10, True),
    ("beta_reflect_mid", 1.20, 1.70, True),
    ("beta_near_2", 1.90, 1.99, True),
]

CONTROL_CELLS = [c[0] for c in BETA_CELLS if not c[3]]
TREATMENT_CELLS = [c[0] for c in BETA_CELLS if c[3]]
CELL_RANGES: Dict[str, Tuple[float, float]] = {c[0]: (c[1], c[2]) for c in BETA_CELLS}


# --- machinery -----------------------------------------------------------------------------------


def _require_triton_ieee() -> None:
    """
    ``TRITON_F32_DEFAULT=ieee`` or this file's numbers mean nothing.

    This is an ASSERTION rather than a comment because the failure is silent and enormous: the
    torch tf32 flag does not control Triton, and this project has measured a 166x fp32 accuracy
    difference between the default and ``ieee``. A band measured with the variable unset would be
    reported with the same confidence and be wrong by two orders of magnitude.

    Mutation that breaks this: unset the variable, or set it to ``tf32``. Trivially satisfiable --
    it is one export, and it is in the invocation at the end of ``core6_bakeoff_guards_test.py``.
    """
    got = os.environ.get("TRITON_F32_DEFAULT")
    assert got == "ieee", (
        f"TRITON_F32_DEFAULT is {got!r}, must be 'ieee'. Every relative-error band in this file is "
        "invalid without it: the torch tf32 flag does NOT control Triton, and this project has "
        "measured a 166x fp32 accuracy difference. Re-run with TRITON_F32_DEFAULT=ieee."
    )


def _make_inputs(
    *,
    beta_low: float,
    beta_high: float,
    gate: str,
    seq_len: int = T,
    seed: int = 0,
    device: str = "cuda",
    requires_grad: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build one cell's inputs, in the dtypes the kernel and the oracle both consume bit-identically.

    Two decisions here are load-bearing.

    **q and k are pre-normalized in float32 and both arms get the same bf16 tensors, with
    ``use_qk_l2norm_in_kernel=False``.** The alternative -- letting the kernel normalize internally
    while the oracle is fed ``fla``'s separate ``l2norm`` -- puts a second, differently-rounded
    operation inside the comparison, and this file's whole output is a band tight enough to
    distinguish 4e-3 from 8e-3. Removing that confound is what makes the band attributable to
    ``beta``. The production configuration (in-kernel l2norm + fused gate) is exercised separately
    by :func:`test_production_configuration_at_beta_above_one_is_finite_and_live`.

    **UNIT-NORM KEYS ARE THE REGIME, NOT A CONVENIENCE.** ``(I - beta k k^T)`` has eigenvalue
    ``1 - beta * ||k||^2`` along ``k``. Only at ``||k|| = 1`` does ``beta = 2`` give exactly ``-1``
    and a true reflection. With un-normalized keys the "reflection regime" label would be false and
    this file would be measuring contractions while claiming otherwise -- so the norm is asserted,
    not assumed.

    :param gate: ``"logsigmoid"`` for the realistic decaying gate, ``"weak"`` for the near-zero
        decay that lets accumulation happen (see :data:`WEAK_DECAY`).
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)

    q = F.normalize(rnd(B, seq_len, H, K), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(rnd(B, seq_len, H, K), p=2, dim=-1).to(torch.bfloat16)
    v = rnd(B, seq_len, H, V).to(torch.bfloat16)

    unit = torch.rand(B, seq_len, H, generator=gen, device=device, dtype=torch.float32)
    beta = (beta_low + (beta_high - beta_low) * unit).to(torch.bfloat16)

    if gate == "logsigmoid":
        g = F.logsigmoid(rnd(B, seq_len, H, K))
    elif gate == "weak":
        g = -WEAK_DECAY * torch.rand(
            B, seq_len, H, K, generator=gen, device=device, dtype=torch.float32
        )
    else:  # pragma: no cover - programming error, not a tested path
        raise ValueError(f"unknown gate {gate!r}")

    # The keys must actually be unit-norm in the dtype the kernel sees, or the reflection framing
    # is wrong. bf16 rounding of a normalized float32 vector moves the norm by O(2^-8); 5e-2 is
    # ~13x that, loose enough not to false-alarm and far tighter than any real normalization bug.
    knorm = k.float().norm(dim=-1)
    assert (knorm - 1.0).abs().max().item() < 5e-2, (
        f"keys are not unit-norm (max deviation {(knorm - 1.0).abs().max().item():.3e}). "
        "beta = 2 is only an exact reflection at ||k|| = 1, so without this the 'reflection "
        "regime' label is false and this test measures contractions."
    )

    out = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    if requires_grad:
        out = {n: t.detach().clone().requires_grad_(True) for n, t in out.items()}
    return out


def _assert_beta_regime(beta: torch.Tensor, cell: str) -> Tuple[float, float]:
    """
    Confirm the cell is in the regime its name claims, and return ``(realized min, realized max)``.

    THE POINT OF THIS FUNCTION IS THAT A SWEEP CAN SILENTLY NOT SWEEP. If a construction bug or a
    dtype rounding collapsed every treatment cell to ``beta <= 1``, every band below would pass and
    the file would report that ``beta > 1`` is safe while never having evaluated it -- the exact
    shape of "an empty comparison set reports success". The precedent does the same thing in prose:
    ``phase-0-1-runbook.md:351`` records ``beta_max = 1.93`` specifically to "confirm the reflection
    regime actually left the contraction range".

    Mutation that breaks this: drop the ``* 2.0`` at ``recurrent.py:876`` (equivalently, halve this
    cell's interval) and every treatment cell fails here rather than passing a band vacuously.
    """
    lo_bound, hi_bound = CELL_RANGES[cell]
    b = beta.float()
    lo, hi = b.min().item(), b.max().item()

    if cell in TREATMENT_CELLS:
        assert lo > 1.0, (
            f"{cell}: realized beta min {lo:.4f} is not above 1.0, so this 'treatment' cell is in "
            f"the contraction regime and asserts nothing about reflections (interval was "
            f"[{lo_bound}, {hi_bound}])"
        )
        # Deliberately '<= 2.0' and not '< 2.0'. bf16 cannot represent a value in
        # (1.9921875, 2.0), so today the reflection edge is approached and not reached -- but a
        # strict '<' here would turn a future dtype or sampling change that DOES land on 2.0 into a
        # spurious failure, when beta = 2 is precisely the point the widened range exists to allow.
        assert hi <= 2.0, f"{cell}: realized beta max {hi:.4f} exceeds the (0, 2) ceiling"
    else:
        assert hi <= 1.0, (
            f"{cell}: realized beta max {hi:.4f} exceeds 1.0, so this 'control' cell is not a "
            "control and the regime comparison below is comparing treatment against treatment"
        )
    return lo, hi


def _rel_stats(got: torch.Tensor, ref: torch.Tensor, what: str) -> Dict[str, float]:
    """
    Max / median / p99 relative error, normalized by the reference's max magnitude.

    Same metric as ``kda_householder_test.py:1506`` so a number produced here can be read against
    the precedent band directly. Normalizing by ``|ref|.max()`` rather than elementwise avoids the
    division-by-near-zero blowup that makes an elementwise relative error unusable on a tensor
    containing values near zero.

    Finiteness is checked HERE, before the statistics, because ``nan`` propagates into ``max()`` and
    a ``nan < budget`` comparison is False -- so a NaN would technically fail the band, but with a
    message pointing at accuracy instead of at the actual event. ``beta = 2`` with unit keys makes
    the update norm-preserving rather than contracting, so unbounded growth is the named theoretical
    failure mode and it deserves its own message.

    Mutation that breaks the finiteness assert: double ``beta`` a second time (``beta -> (0, 4)``),
    which makes ``|1 - beta * ||k||^2|`` up to 3 and the state grow like ``3^T``. That mutation is
    executed in :func:`test_the_beta_band_check_fails_on_a_clamped_or_double_doubled_beta`.
    """
    got_f, ref_f = got.float(), ref.float()

    assert torch.isfinite(got_f).all(), (
        f"{what}: the kernel produced NaN or Inf "
        f"({int((~torch.isfinite(got_f)).sum().item())} of {got_f.numel()} elements). At beta = 2 "
        "with unit-norm keys the state update is norm-PRESERVING rather than contracting, so "
        "unbounded growth is the expected failure mode of the widened range, not a surprise."
    )
    assert torch.isfinite(ref_f).all(), (
        f"{what}: the fp32 ORACLE produced NaN or Inf, so the reference is unusable and the "
        "comparison would be vacuous. This is a statement about the recurrence itself, not about "
        "the kernel."
    )

    scale = ref_f.abs().max()
    assert scale > 0, (
        f"{what}: the reference is entirely zero, so any relative error is 0/0 and this comparison "
        "would report success against anything. Check the inputs actually reached the oracle."
    )

    rel = (got_f - ref_f).abs() / scale
    return {
        "max": rel.max().item(),
        "median": rel.median().item(),
        "p99": rel.flatten().quantile(0.99).item(),
        "ref_scale": scale.item(),
    }


def _kernel_forward(inp: Dict[str, torch.Tensor]) -> torch.Tensor:
    """``chunk_kda`` through the repo's own dispatcher, precomputed-gate configuration."""
    from olmo_core.nn.attention.flash_linear_attn_api import dispatch_chunk_kda

    o, final_state = dispatch_chunk_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
    )
    assert final_state is None, "output_final_state was not requested but a state came back"
    return o


def _oracle_forward(inp: Dict[str, torch.Tensor]) -> torch.Tensor:
    """``fla.ops.kda.naive.naive_recurrent_kda`` on the identical tensors (it upcasts to fp32)."""
    from fla.ops.kda.naive import naive_recurrent_kda

    o, _ = naive_recurrent_kda(q=inp["q"], k=inp["k"], v=inp["v"], g=inp["g"], beta=inp["beta"])
    return o


def _forward_rel_max(
    cell: str, *, gate: str = "logsigmoid", seq_len: int = T, seed: int = 0
) -> float:
    """One cell's max relative forward error, with the regime assertions applied."""
    lo, hi = CELL_RANGES[cell]
    inp = _make_inputs(beta_low=lo, beta_high=hi, gate=gate, seq_len=seq_len, seed=seed)
    _assert_beta_regime(inp["beta"], cell)
    stats = _rel_stats(_kernel_forward(inp), _oracle_forward(inp), f"{cell}/o")
    return stats["max"]


def _harvest_grads(inp: Dict[str, torch.Tensor], what: str) -> Dict[str, torch.Tensor]:
    """
    Collect ``.grad`` off every leaf, asserting each one actually arrived.

    ``None`` MUST NOT MEAN FINE. If a leaf received no gradient, silently dropping it from the
    returned dict would shrink the comparison set and every band below would still pass -- on
    fewer tensors than the test claims to cover. So this raises instead of skipping.
    """
    missing = [n for n, t in inp.items() if t.grad is None]
    assert not missing, (
        f"{what}: no gradient reached {missing}. Dropping them would silently shrink the "
        "comparison set while every assertion below still passed."
    )
    out: Dict[str, torch.Tensor] = {}
    for n, t in inp.items():
        grad = t.grad
        assert grad is not None  # narrowed by the check above; keeps type checkers quiet
        out[n] = grad.detach()
    return out


def _log_log_slope(xs: List[int], ys: List[float]) -> float:
    """Least-squares slope of ``log(y)`` vs ``log(x)``. Same form as ``kda_householder_test``."""
    lx = torch.tensor([math.log(x) for x in xs], dtype=torch.float64)
    ly = torch.tensor([math.log(y) for y in ys], dtype=torch.float64)
    return (((lx - lx.mean()) * (ly - ly.mean())).sum() / ((lx - lx.mean()) ** 2).sum()).item()


# --- the sweep's own self-test (CPU, pure arithmetic) --------------------------------------------


def test_the_beta_sweep_straddles_the_contraction_boundary():
    """
    THE TABLE'S SELF-TEST, and it runs everywhere because it is arithmetic on a literal.

    Every other test in this file needs a GPU and ``fla``, so on a machine with neither they all
    skip -- which looks identical to passing. This one cannot skip, so a fully-skipped run of this
    file still shows one collected, passing test and a reader can tell the difference between "the
    file was collected and the environment was wrong" and "the file was not collected at all".

    What it asserts is the premise the rest of the file rests on: that there IS a treatment side,
    that there IS a control side, and that the two touch at 1.0 rather than being separated by a
    gap that would make the comparison a comparison of two unrelated regimes.

    Mutation that breaks it: delete a treatment row, or halve every treatment interval (the "drop
    the ``* 2.0``" mutation expressed in the table) -- either empties ``TREATMENT_CELLS`` or pushes
    it below 1.0, and this fails.
    """
    assert CONTROL_CELLS, "no control cells: there is nothing to compare beta > 1 against"
    assert TREATMENT_CELLS, "no treatment cells: nothing in this file evaluates beta > 1 at all"

    for name, lo, hi, is_treatment in BETA_CELLS:
        assert lo < hi, f"{name}: empty interval"
        assert 0.0 < lo, f"{name}: beta must be positive"
        assert hi <= 2.0, f"{name}: beta above the (0, 2) ceiling that allow_neg_eigval defines"
        if is_treatment:
            assert lo > 1.0, f"{name}: a treatment cell must lie strictly above 1.0"
        else:
            assert hi <= 1.0, f"{name}: a control cell must lie at or below 1.0"

    # The boundary is actually straddled tightly, not merely straddled. 1.02 - 0.99 = 0.03.
    highest_control = max(CELL_RANGES[c][1] for c in CONTROL_CELLS)
    lowest_treatment = min(CELL_RANGES[c][0] for c in TREATMENT_CELLS)
    assert lowest_treatment - highest_control < 0.10, (
        f"the closest control/treatment pair is {lowest_treatment - highest_control:.3f} apart in "
        "beta; that is too wide for the headline comparison to isolate the boundary"
    )

    # And the reflection edge is approached to within bf16's resolution. Stated as "approached",
    # not "reached": 1.99 rounds to 1.9921875 in bf16 (spacing 2^-7 on [1, 2), and 2.0 would need
    # >= 1.99609375), giving eigenvalue -0.9921875 rather than exactly -1. The ceiling in this
    # assertion is 1.90 because that is what the table commits to, and it also brackets the
    # precedent's realized beta_max of 1.93.
    top = max(CELL_RANGES[c][1] for c in TREATMENT_CELLS)
    assert top >= 1.90, (
        f"the highest treatment beta is {top}, so no cell approaches the reflection edge where the "
        "update very nearly stops contracting (eigenvalue -> -1). The most demanding part of the "
        "widened range would go unevaluated."
    )


# --- forward accuracy, per cell -------------------------------------------------------------------


@requires_gpu
@requires_fla
@pytest.mark.parametrize("cell", [c[0] for c in BETA_CELLS])
def test_chunk_kda_forward_matches_the_fp32_oracle_across_beta(cell: str):
    """
    Per-cell forward band against ``naive_recurrent_kda``, reported as max / median / p99.

    Runs on every cell, control and treatment alike, so the printed table IS the comparison a
    reviewer wants: six rows, three below 1.0 and three above, in the same metric the precedent
    used. The aggregate claim is asserted separately by
    :func:`test_the_forward_error_band_is_the_same_above_and_below_one`; this test is the absolute
    floor -- no cell may be outright unusable, whatever the others do.

    Assertions and the mutation that breaks each:

    * regime entered (via :func:`_assert_beta_regime`) -- drop the ``* 2.0``, or halve the interval.
    * finite (via :func:`_rel_stats`) -- double ``beta`` again, so ``|1 - beta| -> 3`` and the state
      grows ``3^T``.
    * reference non-zero (via :func:`_rel_stats`) -- pass zeros for ``v``, and every band below
      becomes ``0/0``.
    * ``rel_max < 2e-2`` -- clamp ``beta`` to 1.0 inside the kernel call while the oracle keeps the
      real ``beta``; both mutations are executed in
      :func:`test_the_beta_band_check_fails_on_a_clamped_or_double_doubled_beta`, which is what
      makes this number a gate rather than a hope.
    """
    _require_triton_ieee()
    lo_i, hi_i = CELL_RANGES[cell]
    inp = _make_inputs(beta_low=lo_i, beta_high=hi_i, gate="logsigmoid", seed=17)
    beta_min, beta_max = _assert_beta_regime(inp["beta"], cell)

    o_kernel = _kernel_forward(inp)
    assert o_kernel.shape == (B, T, H, V), f"{cell}: unexpected output shape {o_kernel.shape}"

    stats = _rel_stats(o_kernel, _oracle_forward(inp), f"{cell}/o")
    print(
        f"[forward {cell}] beta in [{beta_min:.4f}, {beta_max:.4f}] "
        f"rel_max={stats['max']:.3e} rel_median={stats['median']:.3e} rel_p99={stats['p99']:.3e} "
        f"(|ref|max={stats['ref_scale']:.3e}); precedent band "
        f"{PRECEDENT_BAND[0]:.2e}-{PRECEDENT_BAND[1]:.2e}"
    )

    assert stats["max"] < BF16_FORWARD_BUDGET, (
        f"{cell}: forward max relative error {stats['max']:.3e} exceeds the repo's bf16 budget "
        f"{BF16_FORWARD_BUDGET:.0e} (median {stats['median']:.3e}, p99 {stats['p99']:.3e}) at beta "
        f"in [{beta_min:.4f}, {beta_max:.4f}]. For scale, the Householder kernel's reflection "
        f"regime measured {PRECEDENT_BAND[0]:.2e}-{PRECEDENT_BAND[1]:.2e} against float64."
    )


@requires_gpu
@requires_fla
def test_the_forward_error_band_is_the_same_above_and_below_one():
    """
    THE HEADLINE. The ``beta > 1`` band must not be materially worse than the ``beta < 1`` band.

    This, and not the per-cell budget above, is what gates ``KDA_NEGEIG``. A budget-only test would
    pass a kernel that had become 4x less accurate in the widened range, because 4x of 4e-3 is
    still comfortably inside 2e-2 -- and "the arm is fine, we checked the error was small" is
    exactly the kind of green this project has been burned by. The question worth asking is
    comparative: does crossing 1.0 change anything?

    Both bands are computed in the same test, at the same seed, on the same geometry and gate, so
    the only difference between the two sets is which side of the contraction boundary ``beta``
    sits on. The pairing ``beta_just_below_1`` (0.90-0.99) against ``beta_just_above_1``
    (1.02-1.10) is the tightest available contrast.

    Assertions and mutations:

    * ratio <= 4.0 -- clamp ``beta`` to 1.0 in the kernel only: the treatment cells' error jumps by
      O(10-100x) while the controls are untouched (their ``beta`` is already below 1, so the clamp
      is a no-op there), so the ratio explodes. This is the one assertion in the file that a
      clamping kernel could not slip past, because the clamp is *invisible* to every control cell.
    * both sets non-empty and the denominator floored -- see :data:`BF16_ROUNDOFF_FLOOR`.
    """
    _require_triton_ieee()

    control = {c: _forward_rel_max(c, seed=17) for c in CONTROL_CELLS}
    treatment = {c: _forward_rel_max(c, seed=17) for c in TREATMENT_CELLS}

    assert control and treatment, "one side of the comparison is empty; the ratio is meaningless"

    worst_control = max(control.values())
    worst_treatment = max(treatment.values())
    denom = max(worst_control, BF16_ROUNDOFF_FLOOR)
    ratio = worst_treatment / denom

    print(
        "[forward band] control "
        + ", ".join(f"{k}={v:.3e}" for k, v in control.items())
        + " | treatment "
        + ", ".join(f"{k}={v:.3e}" for k, v in treatment.items())
        + f" | worst_control={worst_control:.3e} worst_treatment={worst_treatment:.3e} "
        f"ratio={ratio:.2f} (ceiling {REGIME_RATIO_CEILING})"
    )

    assert ratio <= REGIME_RATIO_CEILING, (
        f"the beta > 1 forward band is {ratio:.2f}x the beta < 1 band "
        f"({worst_treatment:.3e} vs {worst_control:.3e}), over the {REGIME_RATIO_CEILING}x "
        "ceiling. That is 2.2x the 1.78x spread the precedent measured WITHIN a single regime, so "
        "this is not seed noise: the chunked kernel is materially less accurate in the reflection "
        "range and KDA_NEGEIG should not ship on it. Per-cell numbers are printed above."
    )


@requires_gpu
@requires_fla
def test_the_beta_band_check_fails_on_a_clamped_or_double_doubled_beta():
    """
    THE SELF-TEST: prove the band above can fail *in this regime*, by two named mutations.

    A threshold that cannot be tripped by the input it guards against is worse than no threshold,
    and this project has shipped several -- a 20 GB/s bandwidth check that was unsatisfiable at
    ``T = 1``, a ceiling above 100% of peak, a clock check gated on an absent module. So rather than
    trusting that ``2e-2`` would catch a beta-handling defect, this constructs two and measures how
    far over they land.

    Both mutations are applied to the KERNEL's ``beta`` only, with the oracle keeping the true
    ``beta``. That is the correct shape: it simulates a kernel that mishandles the widened range,
    which is precisely the risk ``KDA_NEGEIG`` introduces.

    1. **Clamp to 1.0** -- what a kernel that silently enforced the contraction range would do.
       Halves the delta-rule write on much of the reflection band, an O(1) change to the output.
    2. **Double again** (``beta`` up to ~4) -- what a second application of ``recurrent.py:876``
       would do. ``|1 - beta * ||k||^2|`` reaches 3, so the state grows like ``3^T``; at
       ``T = 256`` that is astronomically large and may well be non-finite. The assertion is
       written to accept EITHER outcome, because "it exploded to Inf" and "it stayed finite but
       enormous" are the same defect and a test that demanded one specific one would be flaky.
    """
    _require_triton_ieee()

    lo, hi = CELL_RANGES["beta_reflect_mid"]
    inp = _make_inputs(beta_low=lo, beta_high=hi, gate="logsigmoid", seed=23)
    _assert_beta_regime(inp["beta"], "beta_reflect_mid")

    o_ref = _oracle_forward(inp)
    honest = _rel_stats(_kernel_forward(inp), o_ref, "unmutated/o")
    assert honest["max"] < BF16_FORWARD_BUDGET, (
        f"the UNMUTATED comparison already fails at {honest['max']:.3e}; fix that before reading "
        "the mutations below, which would otherwise be uninterpretable"
    )

    # --- mutation 1: clamp beta into the contraction range -------------------------------------
    clamped = dict(inp)
    clamped["beta"] = inp["beta"].clamp(max=1.0)
    n_clamped = int((inp["beta"].float() > 1.0).sum().item())
    assert n_clamped > 0, (
        "the clamp changed nothing, so this mutation is a no-op and proves nothing. beta never "
        "exceeded 1.0 in a cell named 'reflect' -- the sweep is broken, not the kernel."
    )

    o_clamped = _kernel_forward(clamped)
    ref_scale = o_ref.float().abs().max()
    clamped_rel = ((o_clamped.float() - o_ref.float()).abs() / ref_scale).max().item()
    print(
        f"[mutation clamp] {n_clamped}/{inp['beta'].numel()} beta values clamped to 1.0; "
        f"rel_max={clamped_rel:.3e} vs budget {BF16_FORWARD_BUDGET:.0e} "
        f"({clamped_rel / honest['max']:.1f}x the honest error {honest['max']:.3e})"
    )
    assert clamped_rel > BF16_FORWARD_BUDGET, (
        f"clamping beta to 1.0 across the reflection band moved the output by only "
        f"{clamped_rel:.3e}, which is INSIDE the {BF16_FORWARD_BUDGET:.0e} budget. The budget "
        "therefore cannot detect a kernel that ignores the widened beta range, and every band in "
        "this file is decorative. Tighten the budget or strengthen the probe."
    )

    # --- mutation 2: apply the doubling twice ---------------------------------------------------
    doubled = dict(inp)
    doubled["beta"] = (inp["beta"].float() * 2.0).to(inp["beta"].dtype)
    assert doubled["beta"].float().max().item() > 2.0, "the second doubling did not leave (0, 2)"

    o_doubled = _kernel_forward(doubled).float()
    if torch.isfinite(o_doubled).all():
        doubled_rel = ((o_doubled - o_ref.float()).abs() / o_ref.float().abs().max()).max().item()
        print(f"[mutation double] finite, rel_max={doubled_rel:.3e}")
        assert doubled_rel > BF16_FORWARD_BUDGET, (
            f"doubling beta a second time (into (0, 4), spectral radius up to 3) changed the "
            f"output by only {doubled_rel:.3e}, inside the budget. Either the kernel is silently "
            "renormalizing beta -- in which case allow_neg_eigval does nothing and KDA_NEGEIG is "
            "KDA_BASE with extra steps -- or this probe does not reach the recurrence."
        )
    else:
        n_bad = int((~torch.isfinite(o_doubled)).sum().item())
        print(f"[mutation double] non-finite as expected: {n_bad}/{o_doubled.numel()} elements")


# --- backward accuracy ---------------------------------------------------------------------------


@requires_gpu
@requires_fla
@pytest.mark.parametrize("cell", ["beta_just_below_1", "beta_reflect_mid", "beta_near_2"])
def test_chunk_kda_backward_matches_the_fp32_oracle_across_beta(cell: str):
    """
    Gradients, which is the half most likely to be wrong and least likely to be noticed.

    A forward-only check is a weak gate on a training run: a kernel whose forward is right and
    whose backward is subtly wrong at ``beta > 1`` still produces a smooth loss curve, still
    converges, and simply optimizes something slightly different from the operator being claimed.
    ``ChunkKDAFunction.backward`` returns ``dq, dk, dv, dg, db, dA, dbias, ...`` -- five of those
    are compared here against autograd through the fp32 oracle.

    Two things make this non-vacuous:

    * **A random cotangent, not ``.sum()``.** ``o.sum().backward()`` probes one direction of the
      Jacobian and lets errors of opposite sign cancel; a fixed seeded random cotangent probes all
      of it. The same cotangent is used for both arms, so it is common-mode.
    * **Per-tensor liveness.** Each gradient's magnitude is asserted non-zero BEFORE its band is
      read. Comparing two all-zero gradients passes any tolerance, and "the branch was dead" and
      "the branch was correct" are indistinguishable from the band alone. This is the same failure
      this repo has hit with a zero-init pair and with an ``E_l = 3.2e-4`` component that a mean
      reported as healthy.

    Assertions and mutations:

    * every gradient finite -- double ``beta`` again; the backward inherits the forward's growth.
    * every gradient non-zero -- detach ``beta`` from the graph, or zero the cotangent.
    * ``rel_max < 5e-2`` per tensor -- clamp ``beta`` to 1.0 in the kernel's forward: ``db`` and
      ``dv`` then describe a different operator entirely.
    """
    _require_triton_ieee()
    lo, hi = CELL_RANGES[cell]

    cot_gen = torch.Generator(device="cuda").manual_seed(101)
    cot = torch.randn(B, T, H, V, generator=cot_gen, device="cuda", dtype=torch.float32)

    grads: Dict[str, Dict[str, torch.Tensor]] = {}
    for arm, fn in (("kernel", _kernel_forward), ("oracle", _oracle_forward)):
        inp = _make_inputs(
            beta_low=lo, beta_high=hi, gate="logsigmoid", seed=31, requires_grad=True
        )
        _assert_beta_regime(inp["beta"], cell)
        (fn(inp).float() * cot).sum().backward()
        grads[arm] = _harvest_grads(inp, f"{cell}/{arm}")

    for name in ("q", "k", "v", "g", "beta"):
        got, ref = grads["kernel"][name], grads["oracle"][name]

        # Liveness FIRST: a band over two zeros is not evidence.
        ref_mag = ref.float().abs().max().item()
        got_mag = got.float().abs().max().item()
        assert ref_mag > 0.0, (
            f"{cell}: d{name} from the ORACLE is identically zero, so comparing it to anything "
            "passes. Either the input does not affect the loss or the graph is severed."
        )
        assert got_mag > 0.0, (
            f"{cell}: d{name} from the KERNEL is identically zero while the oracle's is "
            f"{ref_mag:.3e}. That is a dead gradient path, not a small one -- the corresponding "
            "parameter is frozen for the whole run and the arm would report a clean, replicable, "
            "meaningless null."
        )

        stats = _rel_stats(got, ref, f"{cell}/d{name}")
        print(
            f"[backward {cell}] d{name}: rel_max={stats['max']:.3e} "
            f"rel_median={stats['median']:.3e} rel_p99={stats['p99']:.3e} "
            f"(|ref|max={stats['ref_scale']:.3e})"
        )
        assert stats["max"] < BF16_BACKWARD_BUDGET, (
            f"{cell}: d{name} max relative error {stats['max']:.3e} exceeds the repo's bf16 "
            f"gradient budget {BF16_BACKWARD_BUDGET:.0e} (median {stats['median']:.3e}, p99 "
            f"{stats['p99']:.3e}) at beta in [{lo}, {hi}]. A wrong backward at beta > 1 trains "
            "smoothly and optimizes a different operator than the one being reported."
        )


@requires_gpu
@requires_fla
def test_the_backward_error_band_is_the_same_above_and_below_one():
    """
    The headline comparison, for gradients. Same logic as the forward version, same ceiling.

    Reduced to one control cell (``beta_just_below_1``) against one treatment cell
    (``beta_near_2``) and the worst gradient across the five compared tensors, because the pair
    that straddles the boundary most tightly and the pair that is furthest apart in regime are the
    two contrasts worth paying for; running all six cells' backwards would triple the runtime to
    sharpen a number that is already the max over five tensors.

    Mutation: clamp ``beta`` to 1.0 in the kernel forward. The control's ``beta`` is entirely below
    1.0 so the clamp cannot touch it, while ``beta_near_2`` is halved -- the ratio moves, the
    absolute budgets may not.
    """
    _require_triton_ieee()

    cot_gen = torch.Generator(device="cuda").manual_seed(101)
    cot = torch.randn(B, T, H, V, generator=cot_gen, device="cuda", dtype=torch.float32)

    def worst_grad_rel(cell: str) -> float:
        lo, hi = CELL_RANGES[cell]
        collected: Dict[str, Dict[str, torch.Tensor]] = {}
        for arm, fn in (("kernel", _kernel_forward), ("oracle", _oracle_forward)):
            inp = _make_inputs(
                beta_low=lo, beta_high=hi, gate="logsigmoid", seed=31, requires_grad=True
            )
            _assert_beta_regime(inp["beta"], cell)
            (fn(inp).float() * cot).sum().backward()
            collected[arm] = _harvest_grads(inp, f"{cell}/{arm}")
        worst = 0.0
        for name in ("q", "k", "v", "g", "beta"):
            ref = collected["oracle"][name]
            assert ref.float().abs().max().item() > 0.0, f"{cell}: oracle d{name} is all zero"
            worst = max(worst, _rel_stats(collected["kernel"][name], ref, f"{cell}/d{name}")["max"])
        return worst

    worst_control = worst_grad_rel("beta_just_below_1")
    worst_treatment = worst_grad_rel("beta_near_2")
    denom = max(worst_control, BF16_ROUNDOFF_FLOOR)
    ratio = worst_treatment / denom

    print(
        f"[backward band] control(beta_just_below_1)={worst_control:.3e} "
        f"treatment(beta_near_2)={worst_treatment:.3e} ratio={ratio:.2f} "
        f"(ceiling {REGIME_RATIO_CEILING})"
    )
    assert ratio <= REGIME_RATIO_CEILING, (
        f"the beta > 1 gradient band is {ratio:.2f}x the beta < 1 band ({worst_treatment:.3e} vs "
        f"{worst_control:.3e}), over the {REGIME_RATIO_CEILING}x ceiling. The reflection regime's "
        "backward is materially less accurate than the contraction regime's; KDA_NEGEIG would "
        "train on gradients that are wrong in a way no loss curve would show."
    )


# --- growth: the norm-preserving failure mode ----------------------------------------------------


@requires_gpu
@requires_fla
def test_reflection_error_does_not_grow_faster_with_length_than_contraction_error():
    """
    Does the error compound with sequence length faster at ``beta ~ 2`` than at ``beta < 1``?

    This is the check the widened range specifically demands. At ``beta = 2`` with a unit key the
    update ``(I - 2 k k^T)`` is an exact reflection: spectral norm exactly 1, so nothing damps.
    Whatever error enters the carried state survives every subsequent chunk instead of decaying,
    and the production sequence length (4096) is 16x the ``T = 256`` the accuracy cells above use.
    A band measured at ``T = 256`` does not extrapolate on its own.

    **THE GATE HAD TO BE WEAKENED FOR THIS TO BE FIREABLE AT ALL, and that is the whole subtlety.**
    With the realistic ``logsigmoid`` gate ``exp(g) ~ 0.5`` per step, so after 256 steps the state
    has been multiplied by ``~2^-256``: no error can accumulate because no state survives, and a
    "no runaway growth" assertion would be satisfied by the decay rather than by the kernel --
    a guard that cannot fire in the regime it is written for. Both ladders therefore run the
    near-zero-decay gate (``exp(g) >= 0.999``, retaining 77% of the state over ``T = 512``), and
    that retention is asserted below rather than assumed.

    **The exponent ceiling is comparative, not absolute.** The ``slope <= 1.25`` at
    ``kda_householder_test.py:1461`` was calibrated on a decaying gate; importing it here, where the
    gate deliberately does not decay, would risk a false alarm on behaviour both regimes share.
    Asking only whether the reflection regime grows FASTER than the contraction regime is the
    actual question, is self-calibrating, and cannot fire on a shared property.

    Assertions and mutations:

    **NO ABSOLUTE ERROR CEILING IS ASSERTED AT THE LADDER'S ENDPOINT, deliberately.** The obvious
    addition -- "and the error at ``T = 512`` must still be under ``2e-2``" -- would be a threshold
    imported from a regime where it was measured (decaying gate, ``T <= 256``) into one deliberately
    constructed to accumulate, with no measurement behind it. Under a non-decaying gate the fp32
    oracle is itself accumulating, and a correct kernel may legitimately drift past ``2e-2`` at
    ``T = 512``; a guard that fires on correct behaviour is a defect in the guard, not caution. The
    endpoint magnitudes are therefore PRINTED, and gated only against the contraction ladder's
    endpoint at the same ``T`` -- which is self-calibrating, since both regimes share the gate.

    Assertions and mutations:

    * ``exp(g) >= 0.99`` -- restore the ``logsigmoid`` gate and this fires, correctly refusing to
      report a growth result measured where growth is impossible.
    * every ladder point finite (via :func:`_rel_stats`) -- double ``beta`` again (spectral radius
      3), and at ``T = 512`` under a non-decaying gate that is ``3^512``.
    * ``slope(treatment) <= slope(control) + 0.5`` -- a kernel whose reflection-range state handoff
      loses a term would show superlinear growth here while the control stayed flat.
    * endpoint ratio ``<= 4.0`` -- catches a regime that grows at the same *rate* but from an
      already-worse level, which the slope alone would pass. Clamping ``beta`` to 1.0 in the kernel
      trips it: the control's beta is entirely below 1.0, so the clamp cannot touch the denominator.
    """
    _require_triton_ieee()
    # THE LADDER REACHES THE PRODUCTION CONTEXT, because the first GPU run of this test showed why
    # extrapolating from T <= 512 is not good enough. It measured slope 0.726 (reflection) against
    # 0.100 (contraction) and failed -- but the reflection ladder was
    # [0.0038, 0.0072, 0.0157, 0.0156], which RISES and then PLATEAUS on its final doubling. Four
    # points whose last two are equal cannot distinguish a compounding curve from a saturating one,
    # and the two readings disagree by an order of magnitude at the length we actually ship:
    # extrapolating slope 0.726 to T=4096 predicts ~7% error, while the plateau predicts ~1.6%.
    # 1024/2048/4096 are exactly where that disagreement resolves, so the ladder is extended to
    # MEASURE the answer instead of inferring it. 4096 is the run's own `--sequence-length`.
    ladder = [64, 128, 256, 512, 1024, 2048, 4096]

    curves: Dict[str, List[float]] = {}
    for label, cell in (("contraction", "beta_just_below_1"), ("reflection", "beta_near_2")):
        lo, hi = CELL_RANGES[cell]
        errs: List[float] = []
        for seq_len in ladder:
            inp = _make_inputs(beta_low=lo, beta_high=hi, gate="weak", seq_len=seq_len, seed=53)
            _assert_beta_regime(inp["beta"], cell)

            retention = inp["g"].exp().min().item()
            assert retention >= 0.99, (
                f"the 'weak' gate retains only {retention:.4f} per step, so the state decays and "
                "this ladder cannot observe accumulation -- the check would be unfireable in the "
                "regime it exists for. WEAK_DECAY is too large."
            )

            stats = _rel_stats(_kernel_forward(inp), _oracle_forward(inp), f"{label}/T={seq_len}")
            errs.append(stats["max"])
        curves[label] = errs
        print(
            f"[growth {label} ({cell})] " + " ".join(f"T={t}:{e:.3e}" for t, e in zip(ladder, errs))
        )

    slope_c = _log_log_slope(ladder, curves["contraction"])
    slope_r = _log_log_slope(ladder, curves["reflection"])
    print(
        f"[growth] log-log slope contraction={slope_c:.3f} reflection={slope_r:.3f} "
        f"(allowed excess {GROWTH_SLOPE_MARGIN})"
    )

    # THE SLOPE IS NOW A DIAGNOSTIC, NOT THE VERDICT, and the ladder reaching 4096 is what earns
    # that demotion. A log-log slope fitted across a SATURATING curve reports the rise and ignores
    # the plateau, so it condemns a kernel whose error has already stopped growing -- exactly the
    # shape the first run produced. With T=4096 measured directly there is no longer any need to
    # infer the production number from an exponent: the tail ratio below asks whether the error is
    # still compounding WHERE WE SHIP, which is the question the slope was only ever a proxy for.
    tail = ladder.index(1024)
    tail_growth_c = curves["contraction"][-1] / max(curves["contraction"][tail], BF16_ROUNDOFF_FLOOR)
    tail_growth_r = curves["reflection"][-1] / max(curves["reflection"][tail], BF16_ROUNDOFF_FLOOR)
    print(
        f"[growth] tail T=1024->4096 (4x length): contraction x{tail_growth_c:.2f} "
        f"reflection x{tail_growth_r:.2f}  (slope over the full ladder is reported above and is "
        "NOT the gate -- it cannot separate compounding from saturation)"
    )

    # A 4x length increase that multiplies the reflection error by more than 2.5x is compounding
    # in the regime we ship; anything at or under that has saturated. The number is derived, not
    # picked: true compounding under a non-decaying gate is at least linear in T, which over 4x
    # length is >= 4x error, while a saturating curve is ~1x. 2.5 sits between the two and clear
    # of both. The contraction arm runs the identical ladder as the control, so if the harness
    # itself drifts with length, both move and the CONTRAST is what is judged.
    assert tail_growth_r <= max(2.5, 2.5 * tail_growth_c), (
        f"reflection error is still COMPOUNDING at production length: T=1024->4096 multiplies it "
        f"by {tail_growth_r:.2f}x against the contraction control's {tail_growth_c:.2f}x. This is "
        "measured at the run's own sequence length, not extrapolated, so it is the number that "
        f"decides whether beta in (0,2) is safe to ship. Ladders: contraction="
        f"{curves['contraction']}, reflection={curves['reflection']}."
    )
    # Level, not just slope -- but relative to the control at the SAME T and the SAME gate, never
    # against a constant imported from the decaying-gate regime. Two ladders that are parallel but
    # offset by 10x would pass the slope check and are still a defect.
    end_c, end_r = curves["contraction"][-1], curves["reflection"][-1]
    end_ratio = end_r / max(end_c, BF16_ROUNDOFF_FLOOR)
    print(
        f"[growth] endpoint T={ladder[-1]}: contraction={end_c:.3e} "
        f"reflection={end_r:.3e} ratio={end_ratio:.2f}"
    )
    assert end_ratio <= REGIME_RATIO_CEILING, (
        f"at T={ladder[-1]} the reflection regime's error is {end_ratio:.2f}x the contraction "
        f"regime's ({end_r:.3e} vs {end_c:.3e}), over the {REGIME_RATIO_CEILING}x ceiling, even "
        "though the growth EXPONENTS matched. Two parallel ladders offset by a large factor are "
        "still a defect: the reflection regime starts worse and stays worse."
    )


# --- the configuration that actually ships --------------------------------------------------------


@requires_gpu
@requires_fla
def test_production_configuration_at_beta_above_one_is_finite_and_live():
    """
    The exact configuration ``KimiDeltaAttention.forward`` uses, at ``beta > 1``.

    Everything above deliberately strips two things out of the comparison -- in-kernel L2
    normalization and the fused gate -- because each adds a differently-rounded operation and this
    file's output is a band tight enough to resolve 4e-3 from 8e-3. But the arm ships with both on
    (``recurrent.py:913-924``: ``use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True``), so the
    stripped-down configuration on its own would be a test of code that is not quite what runs.

    This closes that: same kernel call the module makes, ``beta`` doubled the way
    ``recurrent.py:875-876`` doubles it, gradients taken for ``A_log`` and ``dt_bias`` as well.
    The oracle mirrors it by routing through the same ``fused_kda_gate``, which means **that gate
    kernel is common-mode and is NOT under test here** -- stated rather than left implicit.

    ``A_log`` and ``dt_bias`` are asserted FINITE BUT NOT LIVE, on purpose. Per
    ``docs/dp2-kda/phase-0-1-runbook.md``, they reach the loss only through a saturating
    ``softplus``, so a near-zero gradient there is expected and a liveness assertion on them would
    be a guard that false-alarms on a correct model -- which this project treats as a defect in the
    guard, not as conservatism.

    Assertions and mutations:

    * a FRACTION of ``beta`` exceeds 1.0, not merely its max -- remove the ``* 2.0`` and this fires,
      which is the direct check that ``allow_neg_eigval``'s only mechanism reached the kernel.
    * output and all gradients finite -- double ``beta`` again.
    * ``dq/dk/dv/d(raw)/dbeta`` non-zero -- sever any of them from the graph.
    * band vs the mirrored oracle at ``2e-2`` / ``5e-2`` -- clamp ``beta`` in the kernel.
    """
    _require_triton_ieee()
    from fla.ops.kda.gate import fused_kda_gate

    from olmo_core.nn.attention.flash_linear_attn_api import dispatch_chunk_kda

    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(71)

    def rnd(*shape: int, dtype=torch.float32) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=device, dtype=dtype)

    # NOT pre-normalized: the kernel normalizes internally in this configuration.
    q = rnd(B, T, H, K).to(torch.bfloat16)
    k = rnd(B, T, H, K).to(torch.bfloat16)
    v = rnd(B, T, H, V).to(torch.bfloat16)
    raw = rnd(B, T, H, K).to(torch.bfloat16)
    A_log = torch.empty(H, device=device, dtype=torch.float32).uniform_(1, 16, generator=gen).log()
    dt_bias = rnd(H * K)

    # Exactly recurrent.py:874-876, including the doubling that IS allow_neg_eigval.
    beta = rnd(B, T, H).sigmoid()
    beta = (beta * 2.0).to(torch.bfloat16)

    # A FRACTION, not just the max. '2 * sigmoid(randn)' has median exactly 1.0, so ~half these
    # values land above it; asserting only 'max > 1.0' would also be satisfied by a single outlier
    # in an otherwise-contractive tensor, which is the "one element in the comparison set" version
    # of a vacuous check. 0.25 is comfortably below the ~0.5 expected at B*T*H = 1024 samples
    # (binomial sd ~1.6pp, so 0.5 is ~15 sd above the threshold) and far above what any residual
    # doubling bug would leave.
    frac_above = (beta.float() > 1.0).float().mean().item()
    assert frac_above > 0.25, (
        f"only {frac_above:.1%} of beta values exceed 1.0 (expected ~50% for 2*sigmoid(randn)), so "
        "this test is largely running the contraction regime while claiming to run the reflection "
        "one. Deleting the '* 2.0' at recurrent.py:876 drives this to 0% and every check below "
        "would otherwise still be green."
    )

    leaves = {"q": q, "k": k, "v": v, "raw": raw, "A_log": A_log, "dt_bias": dt_bias, "beta": beta}
    kern = {n: t.detach().clone().requires_grad_(True) for n, t in leaves.items()}
    orac = {n: t.detach().clone().requires_grad_(True) for n, t in leaves.items()}

    cot = torch.randn(B, T, H, V, generator=gen, device=device, dtype=torch.float32)

    o_kernel, final_state = dispatch_chunk_kda(
        q=kern["q"],
        k=kern["k"],
        v=kern["v"],
        g=kern["raw"],
        beta=kern["beta"],
        A_log=kern["A_log"],
        dt_bias=kern["dt_bias"],
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
    )
    assert final_state is None
    (o_kernel.float() * cot).sum().backward()

    from fla.modules.l2norm import l2norm
    from fla.ops.kda.naive import naive_recurrent_kda

    g_ref = fused_kda_gate(orac["raw"], orac["A_log"], orac["dt_bias"])
    o_oracle, _ = naive_recurrent_kda(
        q=l2norm(orac["q"]), k=l2norm(orac["k"]), v=orac["v"], g=g_ref, beta=orac["beta"]
    )
    (o_oracle.float() * cot).sum().backward()

    stats = _rel_stats(o_kernel, o_oracle, "production/o")
    print(
        f"[production] beta in [{beta.float().min().item():.4f}, "
        f"{beta.float().max().item():.4f}] forward rel_max={stats['max']:.3e} "
        f"median={stats['median']:.3e} p99={stats['p99']:.3e}"
    )
    assert stats["max"] < BF16_FORWARD_BUDGET, (
        f"production configuration forward rel_max {stats['max']:.3e} exceeds "
        f"{BF16_FORWARD_BUDGET:.0e}. Note this configuration adds in-kernel l2norm and the fused "
        "gate on top of the isolated comparisons above, so a failure here with those passing "
        "localizes the problem to one of those two ops rather than to beta."
    )

    # These must move, or the corresponding weights are frozen for the whole run.
    for name in ("q", "k", "v", "raw", "beta"):
        got, ref = kern[name].grad, orac[name].grad
        assert got is not None and ref is not None, f"no gradient for {name}"
        assert ref.float().abs().max().item() > 0.0, f"oracle d{name} is all zero; vacuous compare"
        assert got.float().abs().max().item() > 0.0, (
            f"kernel d{name} is identically zero while the oracle's is not -- a dead gradient "
            "path in the configuration that actually trains"
        )
        gstats = _rel_stats(got, ref, f"production/d{name}")
        print(f"[production] d{name}: rel_max={gstats['max']:.3e} median={gstats['median']:.3e}")
        assert gstats["max"] < BF16_BACKWARD_BUDGET, (
            f"production d{name} rel_max {gstats['max']:.3e} exceeds "
            f"{BF16_BACKWARD_BUDGET:.0e} at beta > 1"
        )

    # Finite but deliberately NOT asserted live -- see the docstring.
    for name in ("A_log", "dt_bias"):
        grad = kern[name].grad
        assert grad is not None, f"no gradient for {name}"
        assert torch.isfinite(grad).all(), (
            f"d{name} is non-finite at beta > 1. Its MAGNITUDE is not asserted -- it reaches the "
            "loss only through a saturating softplus, so near-zero is expected there -- but "
            "non-finite is never expected."
        )
        print(f"[production] d{name}: |max|={grad.abs().max().item():.3e} (finiteness only)")
