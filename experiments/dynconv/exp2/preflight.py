"""Pre-flight assertion suite for Exp-2. SPEC §6, checks 1-13, adapted to d=128 / vocab 256 / W swept.

Owner: sub-agent A. Binding spec: ``docs/dynconv-review/build/exp2/SPEC.md``.
Primary source: ``docs/dynconv-review/R7-redteam.md`` §3.

THE DESIGN RULE
---------------
Every check asserts a **magnitude predictable from theory**, not the existence of a thing. The
repo scar is ``green-that-means-nothing``: ``p.grad is not None`` passed under uninitialized
weights; ``loss ~ ln(vocab)`` did not. Five separate harness bugs in this repo have shipped an
exit code of 0.

Each check returns a :class:`CheckResult` with ``name / passed / expected / actual / tolerance``,
so the results go into the design doc as a table rather than as prose.

THREE THINGS TO UNDERSTAND BEFORE READING A GREEN RESULT
--------------------------------------------------------
1. **Check 3 (``alpha = 0`` reproduces the static path) is NECESSARY AND NOT SUFFICIENT.** A
   mechanism wired to nothing passes it *trivially and perfectly* -- the arm simply IS the static
   arm. It must always be read together with check 7, which counts generator modules and their
   layer indices. This is the single most important structural point in the suite.
2. **``||V.grad|| == 0`` at step 0 is CORRECT, not a failure.** With ``U = 0`` the LoRA chain rule
   gives ``V`` no gradient until ``U`` moves. What must be true is ``||V.grad|| > 0`` **after one
   optimizer step** -- check 5b, which is the one that actually catches the dead-branch bug.
3. **``E_l == 0`` at step 0 is also CORRECT** (``U = 0`` means ``Delta_w == 0`` identically). The
   engagement floor is a claim about training, so check 9 takes optimizer steps before asserting.

SPEC §6.5 ADDITIONS
-------------------
* **Check 8 is a HARD GATE** (``severity="gate"``), not a reportable line. A missing BOS measured
  **2.4-3.8 nats** on LFM2-350M -- ~100x the effect Exp-2 is chasing -- and it fails silently.
  :func:`require_absolute_loss_in_band` **refuses to emit a between-arm delta** when any gate
  failed, because a delta on a broken absolute is a different experiment, not a small error.
* **Check 8c** asserts a BOS/sentinel at position 0 of every sequence, when one is declared.
* **Check 16** asserts ``use_fla`` parity and logs the **realised** conv backend per arm, so a
  fused treatment can never be compared against an unfused baseline.
* **Every result carries a ``device=... dtype=...`` label**, auto-populated from the ambient
  :class:`labelled` context so a new check cannot forget it.

Run: ``python3 preflight.py`` for the full grid, or ``--fast`` for W in {2,3}.
**Execution venue is FarmShare (`rice-02`), not the local laptop.** Never queue anything that
could contend for `oat-06` (Exp-1 job 1676346).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.nn.attention.short_conv import ShortConv

try:  # pragma: no cover - environment-dependent
    from olmo_core.nn.attention.flash_linear_attn_api import has_fla as _has_fla
except Exception:  # pragma: no cover - fla is absent in most environments

    def _has_fla() -> bool:
        return False

from arms import (
    ARMS,
    D_MODEL,
    HYBRID_ATTENTION_LAYERS,
    N_LAYERS,
    RANK,
    TOPOLOGIES,
    VOCAB_SIZE,
    WIDTHS,
    ArmNotDefined,
    ArmSpec,
    Attention,
    MQARModel,
    arm_grid,
    build_arm,
    expected_param_count,
)
from dynamic_conv import (
    DynamicFilterGen,
    DynamicQKVConv,
    DynamicShortConv,
    bf16_dead_zone_probe,
    depthwise_causal_conv_static,
    dyn_param_names,
    engagement_report,
    gen_param_count,
    iter_generators,
    named_shared_params,
    reset_permutations,
    set_alpha_override,
    split_param_groups,
    static_realizability_residual,
)

__all__ = [
    "CheckResult",
    "run_preflight",
    "results_table",
    "summarize",
    "LN_VOCAB",
    "BF16_HALF_ULP",
    "ENGAGEMENT_ABORT",
    "AbsoluteLossOutOfBand",
    "require_absolute_loss_in_band",
    "device_dtype_label",
    "labelled",
    "resolved_backend",
    "check_16_backend_parity",
]


LN_VOCAB = math.log(VOCAB_SIZE)
"""``ln 256 = 5.545177``. Init loss must land in ``[5.5452, 5.7952]``: below ``ln V`` is
*impossible* for an untrained model and indicates a broken loss (a vocab-axis mean, or label
leakage). The one-sided +0.25 band is empirical -- the in-tree measured step-0 loss was 11.72
against ``ln(100,352) = 11.52``, i.e. +0.20."""

BF16_HALF_ULP = 2.0**-8
"""``3.90625e-3``, bf16's exact half-ulp at 1.0. The identity-tap init puts the current-token tap
at exactly 1.0, so a filter perturbation below this **rounds away entirely** in bf16. That is
where the 1e-3 engagement abort floor comes from: it is a physical threshold, not taste."""

ENGAGEMENT_ABORT = 1e-3
ENGAGEMENT_TARGET = 1e-2


# ---------------------------------------------------------------------------------------------


def device_dtype_label(model: Optional[nn.Module] = None) -> str:
    """``device=cpu dtype=torch.float32``-style label, attached to EVERY result.

    SPEC §6.5: label device and dtype on every number. A residual, an engagement ratio or a loss
    is not portable across devices or precisions -- the fp32/bf16 gap in check 3 is four orders of
    magnitude -- so an unlabelled number invites a reader to compare a CPU fp32 figure against a
    FarmShare GPU bf16 one and conclude something false.
    """
    if model is None:
        return f"device=cpu dtype={torch.get_default_dtype()}"
    try:
        p = next(model.parameters())
        return f"device={p.device} dtype={p.dtype}"
    except StopIteration:  # pragma: no cover - defensive
        return f"device=? dtype={torch.get_default_dtype()}"


_ACTIVE_LABEL: List[str] = [device_dtype_label()]
"""Ambient device/dtype label, used to auto-populate :attr:`CheckResult.device_dtype`.

A stack, not a scalar, so nesting restores correctly. Auto-population rather than 27 hand-written
call sites: SPEC §6.5 requires a label on *every* number, and an opt-in field is one a future
check will forget -- silently producing exactly the unlabelled number the rule exists to prevent.
"""


class labelled:
    """Context manager setting the ambient device/dtype label from a model (or a literal)."""

    def __init__(self, source: Union[nn.Module, str, None] = None):
        self.label = source if isinstance(source, str) else device_dtype_label(source)

    def __enter__(self) -> str:
        _ACTIVE_LABEL.append(self.label)
        return self.label

    def __exit__(self, *exc) -> None:
        _ACTIVE_LABEL.pop()


Severity = str
"""One of:

* ``"gate"`` -- **HARD ABORT.** A failure invalidates every downstream number, so no between-arm
  delta may be emitted at all. Check 8 is a gate (SPEC §6.5).
* ``"fail"`` -- blocks the run for this cell.
* ``"warn"`` -- flags without blocking.
* ``"info"`` -- documents and never blocks (check 4).
"""


@dataclass
class CheckResult:
    check: str
    name: str
    cell: str
    passed: bool
    expected: str
    actual: str
    tolerance: str = ""
    severity: Severity = "fail"
    note: str = ""
    device_dtype: str = ""
    """Populated on EVERY result. Never report a number from this suite without it.

    Defaults from the ambient :class:`labelled` context, so a new check cannot omit it."""

    def __post_init__(self) -> None:
        if not self.device_dtype:
            self.device_dtype = _ACTIVE_LABEL[-1]

    @property
    def blocking(self) -> bool:
        return (not self.passed) and self.severity in ("fail", "gate")

    @property
    def gating(self) -> bool:
        """A failed HARD GATE. Distinct from :attr:`blocking`: a blocking failure invalidates
        *this cell*, a gating failure invalidates every **between-arm delta** computed from the
        run, because the absolute is out of band and a delta on a broken absolute is a different
        experiment."""
        return (not self.passed) and self.severity == "gate"

    def __str__(self) -> str:
        if self.passed:
            mark = "PASS"
        elif self.severity == "info":
            mark = "INFO"
        elif self.severity == "gate":
            mark = "GATE-ABORT"
        else:
            mark = "FAIL"
        return (
            f"[{mark}] {self.check:>4} {self.name:<34} {self.cell:<18} "
            f"expected={self.expected}  actual={self.actual}"
            + (f"  tol={self.tolerance}" if self.tolerance else "")
            + (f"  [{self.device_dtype}]" if self.device_dtype else "")
            + (f"  ({self.note})" if self.note else "")
        )


def _rand_batch(
    spec: ArmSpec, batch: int = 4, seq: int = 64, seed: int = 1234
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A deterministic random-token batch: ``(tokens, labels)`` with next-token labels.

    Uniform tokens on purpose. Init loss must equal ``ln(vocab)`` for an untrained model on
    *any* input distribution, and using structured data would let a lucky correlation move it.
    Sub-agent B's real MQAR batches can be substituted via the ``batch_fn`` argument of every
    check that consumes data; nothing here imports the harness.
    """
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(0, spec.vocab_size, (batch, seq), generator=g)
    return toks[:, :-1].contiguous(), toks[:, 1:].contiguous()


BatchFn = Callable[[ArmSpec], Tuple[torch.Tensor, torch.Tensor]]


def _loss(model: MQARModel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def _first_gen(model: MQARModel) -> Optional[Tuple[str, DynamicFilterGen]]:
    gens = iter_generators(model)
    return gens[0] if gens else None


# =============================================================================================
# Check 1 -- exact parameter shapes of the generator
# =============================================================================================


def check_01_generator_numel(spec: ArmSpec, model: MQARModel) -> List[CheckResult]:
    """``numel(V) == d*R`` and ``numel(U) == R * n_streams * W * d``, exact, no tolerance.

    The ``W`` factor in ``U`` is the one that gets dropped. A generator wired ``R -> d`` instead of
    ``R -> W*d`` is off by a factor of ``W`` and **still trains**, so nothing downstream notices.
    """
    out: List[CheckResult] = []
    gens = iter_generators(model)
    if not gens:
        # An arm that DECLARES dynamic layers but built none is the silent-no-op trap. It must not
        # be excused just because there is nothing left to measure -- "no generator to check" is
        # exactly what the trap produces.
        declares = len(spec.dynamic_layers) > 0
        return [
            CheckResult(
                "1",
                "numel(V), numel(U)",
                spec.cell,
                passed=not declares,
                expected=(
                    f"{len(spec.dynamic_layers)} generators" if declares else "no generator (S1)"
                ),
                actual="0 generators",
                note=(
                    "arm declares dynamic layers but built NONE -- wired to nothing"
                    if declares
                    else "S1 has no generator by design"
                ),
            )
        ]
    for name, g in gens:
        exp_v = spec.d_model * spec.rank
        exp_u = spec.rank * g.n_streams * spec.width * spec.d_model
        got_v, got_u = g.V.weight.numel(), g.U.weight.numel()
        out.append(
            CheckResult(
                "1",
                "numel(V), numel(U)",
                f"{spec.cell}/{name.split('.')[1]}",
                passed=(got_v == exp_v and got_u == exp_u),
                expected=f"V={exp_v} U={exp_u}",
                actual=f"V={got_v} U={got_u}",
                tolerance="exact",
            )
        )
    return out


# =============================================================================================
# Check 2 -- alpha is learnable, nonzero, in the optimizer, and NOT weight-decayed
# =============================================================================================


def check_02_alpha_optimizer(spec: ArmSpec, model: MQARModel) -> List[CheckResult]:
    """A parameter created *after* the optimizer is built never updates while every other check
    passes. Identity-membership in ``optim.param_groups`` is the only way to see that.

    Also asserts ``{V, U, alpha}`` are in a ``weight_decay = 0`` group. With ``U`` starting at
    exactly 0 and ``alpha`` a bare scalar, decay is a race the mechanism can lose (R7 FN3), and
    the loser looks exactly like "the mechanism does not help".
    """
    gens = iter_generators(model)
    if not gens:
        return []
    groups = split_param_groups(model, weight_decay=0.1)
    opt = torch.optim.AdamW(groups, lr=1e-3)
    in_opt = {id(p) for grp in opt.param_groups for p in grp["params"]}
    decayed = {id(p) for grp in opt.param_groups if grp["weight_decay"] != 0.0 for p in grp["params"]}

    out: List[CheckResult] = []
    for name, g in gens:
        short = name.split(".")[1]
        checks = {
            "requires_grad": bool(g.alpha.requires_grad),
            "nonzero": float(g.alpha.detach()) != 0.0,
            "in_optimizer": id(g.alpha) in in_opt,
            "V_in_optimizer": id(g.V.weight) in in_opt,
            "U_in_optimizer": id(g.U.weight) in in_opt,
            "alpha_not_decayed": id(g.alpha) not in decayed,
            "V_not_decayed": id(g.V.weight) not in decayed,
            "U_not_decayed": id(g.U.weight) not in decayed,
        }
        failed = [k for k, v in checks.items() if not v]
        out.append(
            CheckResult(
                "2",
                "alpha learnable/in optim/no wd",
                f"{spec.cell}/{short}",
                passed=not failed,
                expected="all 8 sub-checks true",
                actual=("all true" if not failed else f"failed: {failed}"),
                note=f"alpha={float(g.alpha.detach()):.4g}",
            )
        )
    return out


# =============================================================================================
# Check 3 -- alpha=0 reproduces the static path
# =============================================================================================


def check_03_alpha_zero_equiv(
    spec: ArmSpec, seed: int = 0, batch_fn: BatchFn = _rand_batch, tol: float = 1e-6
) -> List[CheckResult]:
    """fp32 primary, ``rel_err < 1e-6``. Run TWICE, at two filter states.

    **NECESSARY AND NOT SUFFICIENT.** An arm whose mechanism is wired to nothing passes this
    perfectly -- it *is* the static arm. Always read alongside check 7 (module count and layer
    indices). This pairing is what turns an uninterpretable null into a result.

    **3a, at the identity-tap init, is EXPECTED to be exactly 0.0 -- and that is the reason 3b
    exists.** At init the static filter is ``a = [0, ..., 0, 1]``, so both conv paths degenerate to
    a pass-through of the current token and agree bitwise no matter how they are implemented. A
    suite that only ran 3a would report a perfect 0.000e+00 for every cell while **never exercising
    the tolerance at all** -- a green that means nothing, since a genuinely divergent operator would
    also read 0 at init.

    **3b re-runs the comparison with the static filter randomized**, which is the state training
    actually reaches. There the fp32 residual is real -- MEASURED ~6.8e-8 relative -- because the
    dynamic kernel accumulates the ``W`` taps in a different order from ``nn.Conv1d`` and fp32
    addition is not associative. That is two orders inside the 1e-6 gate.

    Bitwise equality at 3b will NOT hold and must not be chased by tightening the tolerance. A
    *bf16* failure at <= 2e-2 is numerics; above 2e-2 it means the two arms' conv paths are not the
    same operator, which is itself a finding.
    """
    if spec.arm == "S1":
        return []
    static_spec = ArmSpec(
        arm="S1",
        topology=spec.topology,
        width=spec.width,
        d_model=spec.d_model,
        n_layers=spec.n_layers,
        vocab_size=spec.vocab_size,
        rank=spec.rank,
        init_method=spec.init_method,
        init_std=spec.init_std,
    )
    x, _ = batch_fn(spec)
    out: List[CheckResult] = []

    for tag, randomize in (("3a", False), ("3b", True)):
        dyn = build_arm(spec, seed=seed, strict=False)
        stat = build_arm(static_spec, seed=seed)
        if randomize:
            # Same random static filter in both arms, so only the KERNEL differs.
            #
            # S3's filters live in the ATTENTION blocks as a (3, d, W) parameter, while S1 has
            # (d, W) ShortConv filters in the LIV blocks -- so they cannot be zipped positionally.
            # Match by NAME, and skip any tensor the reference arm does not have: for S3 that
            # leaves the LIV filters (shared, so still randomized in both) and correctly excludes
            # the Q/K/V filters, which have no S1 counterpart and are already the identity in both
            # arms at alpha=0.
            g = torch.Generator().manual_seed(4242)
            ref_filters = dict(stat.static_filters())
            with torch.no_grad():
                for name, pd in dyn.static_filters():
                    ps = ref_filters.get(name)
                    if ps is None or ps.shape != pd.shape:
                        continue
                    val = torch.empty_like(pd).normal_(0.0, 0.5, generator=g)
                    pd.copy_(val)
                    ps.copy_(val)
        n = set_alpha_override(dyn, 0.0)
        with torch.no_grad():
            yd, ys = dyn(x), stat(x)
        set_alpha_override(dyn, None)
        rel = float((yd - ys).norm() / ys.norm())
        out.append(
            CheckResult(
                tag,
                "alpha=0 == static path (fp32)"
                + (" [random filter]" if randomize else " [identity init]"),
                spec.cell,
                passed=(rel < tol),
                expected="rel_err < 1e-6",
                actual=f"{rel:.3e}",
                tolerance="1e-6",
                note=(
                    f"{n} generators forced to 0; NECESSARY NOT SUFFICIENT, pair with check 7"
                    + (
                        "; exercises the tolerance -- 0.0 here would mean the two kernels are the "
                        "same code and 3a proved nothing"
                        if randomize
                        else "; 0.0 is EXPECTED (identity tap is a pass-through), so 3a alone is "
                        "vacuous -- see 3b"
                    )
                ),
            )
        )
    return out


# =============================================================================================
# Check 4 -- bf16 tap dead zone, characterised. Documents; never blocks.
# =============================================================================================


def check_04_bf16_dead_zone(spec: ArmSpec) -> List[CheckResult]:
    out: List[CheckResult] = []
    for mag in (1e-4, 1e-3, 3.9e-3, 1e-2):
        p = bf16_dead_zone_probe(spec.d_model, spec.width, mag)
        dead = bool(p["current_tap_unchanged"])
        expect_dead = mag < BF16_HALF_ULP
        out.append(
            CheckResult(
                "4",
                f"bf16 dead zone @ |dw|={mag:g}",
                spec.cell,
                passed=(dead == expect_dead),
                expected=f"current tap {'unchanged' if expect_dead else 'moves'}",
                actual=f"current tap {'unchanged' if dead else 'moves'}",
                severity="info",
                note=f"half-ulp at 1.0 = {BF16_HALF_ULP:g}; history rel err {p['history_rel_err']:.2e}",
            )
        )
    return out


# =============================================================================================
# Checks 5 and 5b -- gradient magnitudes, and the dead-branch trap
# =============================================================================================


def check_05_grad_magnitudes(
    spec: ArmSpec, seed: int = 0, batch_fn: BatchFn = _rand_batch
) -> List[CheckResult]:
    """``||U.grad|| / ||out_proj.grad||`` in the SAME block, in ``[1e-4, 1e2]``.

    Comparing inside one block holds architecture and depth fixed. The ratio bound is the
    load-bearing part: a gradient twelve orders of magnitude below its neighbours' is functionally
    zero, and ``grad is not None`` passes.
    """
    model = build_arm(spec, seed=seed, strict=False)
    gens = iter_generators(model)
    if not gens:
        return []
    x, y = batch_fn(spec)
    model.zero_grad(set_to_none=True)
    _loss(model, x, y).backward()

    out: List[CheckResult] = []
    for name, g in gens:
        blk_i = int(name.split(".")[1])
        blk = model.blocks[blk_i]
        mixer = blk.sequence_mixer
        if isinstance(mixer, DynamicShortConv):
            ref, ref_name = mixer.out_proj.weight, "out_proj"
        elif isinstance(mixer, Attention):
            ref, ref_name = mixer.out.weight, "attn.out"
        else:  # pragma: no cover
            continue
        gu = g.U.weight.grad
        gr = ref.grad
        if gu is None or gr is None or float(gr.norm()) == 0.0:
            out.append(
                CheckResult(
                    "5",
                    "grad ratio U vs out_proj",
                    f"{spec.cell}/L{blk_i}",
                    passed=False,
                    expected="both grads present and nonzero",
                    actual=f"U.grad={'None' if gu is None else float(gu.norm()):.3g} "
                    f"{ref_name}.grad={'None' if gr is None else float(gr.norm()):.3g}",
                )
            )
            continue
        ratio = float(gu.norm() / gr.norm())
        out.append(
            CheckResult(
                "5",
                "grad ratio U vs out_proj",
                f"{spec.cell}/L{blk_i}",
                passed=(1e-4 <= ratio <= 1e2),
                expected="ratio in [1e-4, 1e2]",
                actual=f"{ratio:.3e}",
                note=f"||U.grad||={float(gu.norm()):.3e} ||{ref_name}.grad||={float(gr.norm()):.3e}",
            )
        )
    return out


def check_05b_dead_branch(
    spec: ArmSpec, seed: int = 0, batch_fn: BatchFn = _rand_batch, lr: float = 1e-2
) -> List[CheckResult]:
    """**THE CHECK THAT MATTERS.** Two halves, and both must hold:

    * at step 0, ``||V.grad|| == 0`` -- this is *correct and expected*, the LoRA chain rule with
      ``U = 0``;
    * after **one optimizer step**, ``||V.grad|| > 0``.

    If ``V``'s gradient is still exactly zero after one step, ``Delta_w`` is unreachable for the
    entire run and the arm is the baseline carrying dead weight. It trains stably, every arm ties,
    and it reads as a clean replicable negative -- the most expensive possible failure, because it
    looks like science. This is the difference between a five-minute fix and a $6,100 post-mortem.

    Catches all three fatal init variants: ``U = 0 AND alpha = 0`` (the exact saddle),
    ``alpha = 0`` fixed, and ``alpha`` accidentally a float rather than a Parameter.
    """
    model = build_arm(spec, seed=seed, strict=False)
    gens = iter_generators(model)
    if not gens:
        return []
    x, y = batch_fn(spec)
    opt = torch.optim.AdamW(split_param_groups(model), lr=lr)

    model.zero_grad(set_to_none=True)
    _loss(model, x, y).backward()
    step0 = {
        n: (
            float(g.V.weight.grad.norm()) if g.V.weight.grad is not None else None,
            float(g.U.weight.grad.norm()) if g.U.weight.grad is not None else None,
            float(g.alpha.grad.norm()) if g.alpha.grad is not None else None,
        )
        for n, g in gens
    }
    opt.step()

    model.zero_grad(set_to_none=True)
    _loss(model, x, y).backward()
    step1 = {
        n: (
            float(g.V.weight.grad.norm()) if g.V.weight.grad is not None else None,
            float(g.U.weight.grad.norm()) if g.U.weight.grad is not None else None,
        )
        for n, g in gens
    }

    out: List[CheckResult] = []
    for name, _g in gens:
        short = name.split(".")[1]
        v0, u0, a0 = step0[name]
        v1, u1 = step1[name]
        ok_v0 = v0 == 0.0
        ok_u0 = u0 is not None and u0 > 0.0
        ok_v1 = v1 is not None and v1 > 0.0
        out.append(
            CheckResult(
                "5b",
                "||V.grad||==0 @0, >0 after 1 step",
                f"{spec.cell}/L{short}",
                passed=(ok_v0 and ok_u0 and ok_v1),
                expected="step0: ||V.grad||==0 and ||U.grad||>0; step1: ||V.grad||>0",
                actual=f"step0 V={v0!r} U={u0!r} alpha={a0!r}; step1 V={v1!r} U={u1!r}",
                note=(
                    "DEAD BRANCH -- Delta_w unreachable for the whole run"
                    if not (ok_u0 and ok_v1)
                    else ""
                ),
            )
        )
    return out


# =============================================================================================
# Check 6 -- shared parameters bit-identical across arms at the same seed
# =============================================================================================


def check_06_paired_seeding(
    spec: ArmSpec, seed: int = 0, reference_arm: str = "S1"
) -> List[CheckResult]:
    """``torch.equal``, not ``allclose``. If this fails, "paired initialization seeds" is FALSE
    and the paired power analysis is void (R7 FP1).

    Fails under a single sequential RNG stream, because the stream diverges at the first tensor an
    arm does not share (S4's ``V`` / ``U`` / ``alpha``) and every subsequent draw is misaligned --
    while each individual model still looks perfectly well-initialized.
    """
    if spec.arm == reference_arm:
        return []
    ref_spec = ArmSpec(
        arm=reference_arm,  # type: ignore[arg-type]
        topology=spec.topology,
        width=spec.width,
        d_model=spec.d_model,
        n_layers=spec.n_layers,
        vocab_size=spec.vocab_size,
        rank=spec.rank,
        init_method=spec.init_method,
        init_std=spec.init_std,
        seeding=spec.seeding,
    )
    a = build_arm(ref_spec, seed=seed, strict=False)
    b = build_arm(spec, seed=seed, strict=False)
    sa, sb = dict(a.named_parameters()), dict(b.named_parameters())
    shared = list(named_shared_params(a, b))
    mismatched = [n for n in shared if not torch.equal(sa[n].detach(), sb[n].detach())]
    return [
        CheckResult(
            "6",
            f"shared params bit-identical vs {reference_arm}",
            spec.cell,
            passed=not mismatched,
            expected=f"all {len(shared)} shared tensors torch.equal",
            actual=(
                "all equal"
                if not mismatched
                else f"{len(mismatched)}/{len(shared)} differ, first={mismatched[0]}"
            ),
            tolerance="exact (torch.equal, NOT allclose)",
            note="pairing false => power analysis void",
        )
    ]


# =============================================================================================
# Check 7 -- param counts AND module counts AND layer indices
# =============================================================================================


def check_07_counts(spec: ArmSpec, model: MQARModel) -> List[CheckResult]:
    """**The silent-no-op catcher, and the one people skip.**

    An exact *total* can hide two offsetting errors -- the in-tree 350M reconciliation caught two
    geometry omissions (+67,108,864 from untied embeddings defaulting on, and a +768 residual from
    missing per-head QK-norm) only because it reconciled components. So this reports:

    * 7a the analytic total against the built module,
    * 7b every component independently,
    * 7c the count of ``DynamicFilterGen`` modules,
    * 7d their layer INDICES.

    7c/7d are what make check 3 interpretable: a mechanism wired to ``block.attention`` instead of
    ``block.sequence_mixer`` passes check 3 perfectly and reports 0 dynamic modules here, **while
    the forward pass still succeeds**.
    """
    parts = expected_param_count(spec)
    out = [
        CheckResult(
            "7a",
            "total params == analytic",
            spec.cell,
            passed=(model.n_params == parts["total"]),
            expected=f"{parts['total']}",
            actual=f"{model.n_params}",
            tolerance="exact",
        )
    ]

    # 7b -- component reconciliation, measured off the module tree.
    measured: Dict[str, int] = {
        "embed": model.embed.weight.numel(),
        "head": model.head.weight.numel(),
        "norms": model.out_norm.weight.numel()
        + sum(b.mixer_norm.weight.numel() + b.ffn_norm.weight.numel() for b in model.blocks),
        "ffn": sum(p.numel() for b in model.blocks for p in b.ffn.parameters()),
    }
    liv, attn, dyn_liv, dyn_qkv = 0, 0, 0, 0
    for b in model.blocks:
        m = b.sequence_mixer
        if isinstance(m, DynamicShortConv):
            dyn_liv += sum(p.numel() for p in m.dyn.parameters())
            liv += sum(p.numel() for n, p in m.named_parameters() if not n.startswith("dyn."))
        elif isinstance(m, Attention):
            attn += m.qkv.weight.numel() + m.out.weight.numel()
            if m.qkv_conv is not None:
                dyn_qkv += sum(p.numel() for p in m.qkv_conv.parameters())
        else:
            liv += sum(p.numel() for p in m.parameters())
    measured.update(
        {"liv_mixers": liv, "attn_mixers": attn, "dyn_liv_gen": dyn_liv, "dyn_qkv_gen": dyn_qkv}
    )
    bad = {k: (parts[k], measured[k]) for k in measured if parts[k] != measured[k]}
    out.append(
        CheckResult(
            "7b",
            "component reconciliation",
            spec.cell,
            passed=not bad,
            expected="every component == analytic",
            actual=("all match" if not bad else f"{bad}"),
            tolerance="exact",
        )
    )

    exp_layers = spec.dynamic_layers
    got_layers = model.dynamic_module_layers()
    out.append(
        CheckResult(
            "7c",
            "n DynamicFilterGen modules",
            spec.cell,
            passed=(model.n_dynamic_modules() == len(exp_layers)),
            expected=f"{len(exp_layers)}",
            actual=f"{model.n_dynamic_modules()}",
            tolerance="exact",
            note="0 here with check 3 green == the silent-no-op trap",
        )
    )
    out.append(
        CheckResult(
            "7d",
            "dynamic layer INDICES",
            spec.cell,
            passed=(got_layers == exp_layers),
            expected=f"{list(exp_layers)}",
            actual=f"{list(got_layers)}",
            tolerance="exact",
        )
    )
    return out


# =============================================================================================
# Check 8 -- init loss in [ln V, ln V + 0.25], for EVERY arm
# =============================================================================================


class AbsoluteLossOutOfBand(RuntimeError):
    """Raised by :func:`require_absolute_loss_in_band` when a between-arm delta is requested on
    top of an out-of-band absolute. See :func:`check_08_init_loss`."""


def check_08_init_loss(
    spec: ArmSpec,
    seed: int = 0,
    batch_fn: BatchFn = _rand_batch,
    reference_loss: Optional[float] = None,
    bos_token: Optional[int] = None,
) -> List[CheckResult]:
    """``[5.5452, 5.7952]`` at vocab 256. **HARD GATE (SPEC §6.5) -- severity="gate".**

    Has caught uninitialized weights ~4x in this repo (926 and ~900 against 11.52; 908.8; 994.7)
    -- cases where ``cfg.build()`` allocated without initializing, and the model forwarded,
    backpropagated and produced finite gradients on every parameter. Exit code 0 caught none of
    them. A value **below** ``ln V`` is impossible for an untrained model and means the loss is
    broken (a vocab-axis mean, or label leakage).

    **Why this is now a hard abort rather than a reportable line.** The Exp-0 team measured that a
    missing BOS puts LFM2-350M **2.4-3.8 nats** off -- roughly 100x the effect Exp-2 is chasing --
    and it fails *silently*: the run trains, the curve looks plausible, and every between-arm
    delta is computed on top of a broken absolute. A delta on a broken absolute is not a small
    error, it is a different experiment. So an out-of-band absolute must stop the run, not
    annotate it. Enforced downstream by :func:`require_absolute_loss_in_band`, which **refuses to
    emit a delta at all**.

    8c asserts the BOS/sentinel is at position 0 of **every** sequence in the batch, when the
    harness supplies one. Skipped, and reported as skipped, when ``bos_token is None`` -- absence
    of a sentinel is a legitimate configuration, but it must be a stated one rather than an
    unnoticed one.

    Also: **if S4's step-0 loss differs from S1's by more than 0.01, the ``alpha = 0`` equivalence
    is broken in a way check 3 did not see** -- e.g. an unseeded ``reset_parameters`` on V/U.
    """
    model = build_arm(spec, seed=seed, strict=False)
    label = device_dtype_label(model)
    x, y = batch_fn(spec)
    with torch.no_grad():
        loss = float(_loss(model, x, y))
    in_band = LN_VOCAB <= loss <= LN_VOCAB + 0.25
    out = [
        CheckResult(
            "8a",
            "init loss in [lnV, lnV+0.25]",
            spec.cell,
            passed=in_band,
            expected=f"[{LN_VOCAB:.4f}, {LN_VOCAB + 0.25:.4f}]",
            actual=f"{loss:.6f}",
            severity="gate",
            device_dtype=label,
            note=(
                f"ln(vocab={spec.vocab_size}) = {LN_VOCAB:.6f}"
                + (
                    ""
                    if in_band
                    else "; HARD ABORT -- no between-arm delta may be emitted from this run"
                )
            ),
        )
    ]

    if bos_token is not None:
        col0 = x[:, 0]
        ok = bool((col0 == bos_token).all())
        out.append(
            CheckResult(
                "8c",
                "BOS at position 0 of every sequence",
                spec.cell,
                passed=ok,
                expected=f"x[:, 0] == {bos_token} for all {x.shape[0]} sequences",
                actual=(
                    "all present"
                    if ok
                    else f"{int((col0 != bos_token).sum())}/{x.shape[0]} sequences missing it"
                ),
                severity="gate",
                device_dtype=label,
                note="a missing BOS measured 2.4-3.8 nats on LFM2-350M, ~100x the target effect",
            )
        )
    else:
        out.append(
            CheckResult(
                "8c",
                "BOS at position 0 of every sequence",
                spec.cell,
                passed=True,
                expected="bos_token declared, or explicitly none",
                actual="no BOS declared -- check SKIPPED",
                severity="info",
                device_dtype=label,
                note="state this in the design doc; an undeclared sentinel is the failure mode",
            )
        )

    if reference_loss is not None:
        d = abs(loss - reference_loss)
        out.append(
            CheckResult(
                "8b",
                "init loss == S1's within 0.01",
                spec.cell,
                passed=(d <= 0.01),
                expected=f"|loss - {reference_loss:.6f}| <= 0.01",
                actual=f"{d:.3e}",
                tolerance="0.01",
                device_dtype=label,
                note="a gap here means alpha=0 equivalence is broken invisibly to check 3",
            )
        )
    return out


def require_absolute_loss_in_band(
    results: Sequence[CheckResult],
    *,
    what: str = "between-arm delta",
) -> None:
    """**Refuse to emit a between-arm delta when the absolute init loss is out of band.**

    Call this immediately before computing any arm-vs-arm contrast. It raises rather than warns,
    because the whole hazard is that the broken case looks fine: the run trains, the curve is
    plausible, and the delta is a real number computed on a wrong absolute.

    :raises AbsoluteLossOutOfBand: if any ``severity="gate"`` check failed.
    """
    failed = [r for r in results if r.gating]
    if not failed:
        return
    detail = "\n".join(f"  {r}" for r in failed)
    raise AbsoluteLossOutOfBand(
        f"REFUSING to emit a {what}: {len(failed)} hard gate(s) failed.\n{detail}\n"
        "A delta computed on top of an out-of-band absolute is a different experiment, not a "
        "small error. Fix the absolute (init, BOS, vocab, loss reduction) and re-run."
    )


# =============================================================================================
# Checks 9, 9b, 10 -- engagement, ablate-at-eval, and the weight-decay signature
# =============================================================================================


@dataclass
class TrainTrace:
    losses: List[float] = field(default_factory=list)
    u_norms: Dict[str, List[float]] = field(default_factory=dict)
    engagement: Dict[str, List[float]] = field(default_factory=dict)
    grad_norms: List[float] = field(default_factory=list)


def _short_train(
    model: MQARModel,
    spec: ArmSpec,
    steps: int,
    batch_fn: BatchFn,
    lr: float = 3e-3,
    weight_decay: float = 0.1,
    decay_dynamic: bool = False,
) -> TrainTrace:
    """A few CPU optimizer steps, tracing per-layer ``||U||`` and ``E_l``.

    ``decay_dynamic=True`` puts ``{V, U, alpha}`` INTO the decay group -- the R7 FN3 failure --
    so check 10's negative control can produce the signature rather than assume it.
    """
    if decay_dynamic:
        groups: List[Dict[str, object]] = [
            {"params": list(model.parameters()), "weight_decay": weight_decay}
        ]
    else:
        groups = split_param_groups(model, weight_decay=weight_decay)
    opt = torch.optim.AdamW(groups, lr=lr)
    tr = TrainTrace()
    for step in range(steps):
        x, y = batch_fn(spec) if steps == 1 else _rand_batch(spec, seed=1234 + step)
        model.zero_grad(set_to_none=True)
        loss = _loss(model, x, y)
        loss.backward()
        gn = float(
            torch.sqrt(
                sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)
            )
        )
        tr.grad_norms.append(gn)
        opt.step()
        tr.losses.append(float(loss.detach()))
        for st in engagement_report(model):
            tr.u_norms.setdefault(st.name, []).append(st.u_norm)
            tr.engagement.setdefault(st.name, []).append(st.engagement)
    return tr


def check_09_engagement(
    spec: ArmSpec,
    seed: int = 0,
    steps: int = 30,
    batch_fn: BatchFn = _rand_batch,
    lr: float = 3e-3,
) -> List[CheckResult]:
    """Per-layer ``E_l = ||alpha*Delta_w||_F / ||a||_F``. **ABORT below 1e-3.**

    The floor is physical: ``2^-8 = 3.90625e-3`` is bf16's half-ulp at 1.0, which is the size of
    the current-token tap, so below ~1e-3 the perturbation provably cannot move the dominant tap
    and the arm is the static arm carrying fossils.

    **Reported per layer, never averaged.** The failure mode is layer-dependent -- depth-scaled
    ``out_proj`` init means late layers start smaller, so a mean over 6 layers can sit above the
    floor while most layers are dead.

    ``E_l == 0`` at step 0 is CORRECT (``U = 0`` gives ``Delta_w == 0`` identically), so this
    check takes optimizer steps first. It also reports ``input_dependence``, which ``E_l`` cannot
    see: a ``U`` that learns a position-*constant* offset drives ``E_l`` up while the filter is no
    longer input-dependent at all -- something ``a`` could have absorbed for free.
    """
    model = build_arm(spec, seed=seed, strict=False)
    if not iter_generators(model):
        return []
    with torch.no_grad():
        x, y = batch_fn(spec)
        _loss(model, x, y)
    e0 = {st.name: st.engagement for st in engagement_report(model)}
    tr = _short_train(model, spec, steps=steps, batch_fn=batch_fn, lr=lr)
    final = engagement_report(model)

    out: List[CheckResult] = []
    for st in final:
        short = st.name.split(".")[1]
        out.append(
            CheckResult(
                "9",
                f"engagement E_l after {steps} steps",
                f"{spec.cell}/L{short}",
                passed=(st.engagement >= ENGAGEMENT_ABORT),
                expected=f">= {ENGAGEMENT_ABORT:g} (abort floor); >= {ENGAGEMENT_TARGET:g} desired",
                actual=f"{st.engagement:.4e}",
                severity="fail",
                note=(
                    f"E_0={e0.get(st.name, float('nan')):.1e} (0 is CORRECT at init); "
                    f"input_dep={st.input_dependence:.3f}; alpha={st.alpha:.4f}; "
                    f"||U||={st.u_norm:.3e}"
                    + ("" if st.engagement >= ENGAGEMENT_TARGET else "; below 1e-2 target")
                ),
            )
        )
    out.append(
        CheckResult(
            "9c",
            "E_l reported per layer, not averaged",
            spec.cell,
            passed=(len(final) == len(spec.dynamic_layers)),
            expected=f"{len(spec.dynamic_layers)} per-layer values",
            actual=f"{len(final)} values",
            note="a mean over layers can sit above the floor while most layers are dead",
        )
    )
    return out


def check_09b_ablate_at_eval(
    spec: ArmSpec,
    seed: int = 0,
    steps: int = 30,
    batch_fn: BatchFn = _rand_batch,
    lr: float = 3e-3,
) -> List[CheckResult]:
    """``Delta_loss = loss(alpha=0) - loss(alpha_hat)``.

    No other single measurement separates "bug" from "redundant" from "harmful":

    * ``Delta_loss ~ 0`` with ``E_l`` large => the mechanism is learned and genuinely redundant
      with the B/C gates. A real, publishable negative.
    * ``Delta_loss > 0.01`` => load-bearing.
    * ``Delta_loss < 0`` => zeroing alpha *improves* loss, i.e. the mechanism is actively harmful,
      which is a third, different result.

    Reported, not gated: at 30 CPU steps on random tokens the magnitude is not meaningful. The
    *plumbing* is what is being verified here.
    """
    model = build_arm(spec, seed=seed, strict=False)
    if not iter_generators(model):
        return []
    _short_train(model, spec, steps=steps, batch_fn=batch_fn, lr=lr)
    x, y = batch_fn(spec)

    # Rewind the permutation stream before each forward. For arm S2 the permutation is REDRAWN
    # EVERY FORWARD by design, so two forwards of the same batch legitimately differ -- measured
    # at 3.48e-05 relative on S2-hybrid-W3, which is the control working, not a bug. Without the
    # rewind the reversibility assertion flaps on S2 alone, which would be the worst kind of
    # flake: it would look like the control arm is the broken one.
    def _eval() -> float:
        reset_permutations(model)
        with torch.no_grad():
            return float(_loss(model, x, y))

    l_hat = _eval()
    set_alpha_override(model, 0.0)
    l_zero = _eval()
    set_alpha_override(model, None)
    l_back = _eval()
    return [
        CheckResult(
            "9b",
            "ablate-at-eval delta_loss",
            spec.cell,
            passed=(abs(l_back - l_hat) < 1e-9),
            expected="override is reversible (loss returns exactly)",
            actual=f"delta_loss={l_zero - l_hat:+.5f}; restore residual={abs(l_back - l_hat):.2e}",
            severity="info",
            note=(
                ">0.01 load-bearing / ~0 redundant / <0 harmful. "
                f"loss(alpha_hat)={l_hat:.5f} loss(alpha=0)={l_zero:.5f}"
            ),
        )
    ]


def check_10_u_norm_trend(
    spec: ArmSpec,
    seed: int = 0,
    steps: int = 40,
    batch_fn: BatchFn = _rand_batch,
    lr: float = 3e-3,
    decay_dynamic: bool = False,
) -> List[CheckResult]:
    """``||U||`` must not be monotonically decreasing after the first 5% of training.

    A monotone decrease is the weight-decay signature: decay is winning and ``{V, U, alpha}`` must
    be out of the decay group. Rising-then-falling ``||U||`` with ``E_l`` falling is the
    unambiguous version.
    """
    model = build_arm(spec, seed=seed, strict=False)
    if not iter_generators(model):
        return []
    tr = _short_train(
        model, spec, steps=steps, batch_fn=batch_fn, lr=lr, decay_dynamic=decay_dynamic
    )
    start = max(1, int(0.05 * steps))
    out: List[CheckResult] = []
    for name, seq in tr.u_norms.items():
        tail = seq[start:]
        mono_down = len(tail) > 2 and all(b <= a for a, b in zip(tail, tail[1:]))
        out.append(
            CheckResult(
                "10",
                "||U|| not monotone-decreasing",
                f"{spec.cell}/L{name.split('.')[1]}",
                passed=not mono_down,
                expected="not monotone decreasing after first 5%",
                actual=("monotone DOWN" if mono_down else "not monotone down"),
                note=f"||U||: {tail[0]:.3e} -> {tail[-1]:.3e}"
                + ("  (weight decay winning)" if mono_down else ""),
            )
        )
    return out


# =============================================================================================
# Check 11 -- grad-norm parity across arms
# =============================================================================================


def check_11_grad_norm_parity(
    spec: ArmSpec,
    seed: int = 0,
    steps: int = 20,
    batch_fn: BatchFn = _rand_batch,
    reference_arm: str = "S1",
) -> List[CheckResult]:
    """Median global grad norm within 2x of the reference arm's.

    A mismatch means global clipping bites one arm differently, so that arm's *other* parameters
    train at a different effective LR and the between-arm contrast is confounded by something
    other than the mechanism.
    """
    if spec.arm == reference_arm:
        return []
    ref_spec = ArmSpec(
        arm=reference_arm,  # type: ignore[arg-type]
        topology=spec.topology,
        width=spec.width,
        d_model=spec.d_model,
        vocab_size=spec.vocab_size,
        rank=spec.rank,
    )
    a = _short_train(build_arm(ref_spec, seed=seed), ref_spec, steps, batch_fn)
    b = _short_train(build_arm(spec, seed=seed, strict=False), spec, steps, batch_fn)
    ma = float(torch.tensor(a.grad_norms).median())
    mb = float(torch.tensor(b.grad_norms).median())
    ratio = mb / ma if ma > 0 else float("inf")
    return [
        CheckResult(
            "11",
            f"grad-norm parity vs {reference_arm}",
            spec.cell,
            passed=(0.5 <= ratio <= 2.0),
            expected="median global grad norm within 2x",
            actual=f"ratio={ratio:.3f}",
            tolerance="[0.5, 2.0]",
            note=f"{reference_arm}={ma:.4e} {spec.arm}={mb:.4e} over {steps} steps",
        )
    ]


# =============================================================================================
# Check 12 -- no activation in the conv path, asserted NUMERICALLY
# =============================================================================================


def check_12_no_activation(spec: ArmSpec, model: MQARModel) -> List[CheckResult]:
    """``CausalConv1d.__init__`` defaults ``activation="silu"`` in this fork
    (``olmo_core/nn/convolution.py:37``) while the released ``Lfm2ShortConv`` passes ``None``. The
    silu version is a **different operator** that trains happily, just worse -- a silent failure.

    Asserted **numerically**, not by reading a flag: the mixer's own gated-conv output is compared
    against an independent reference built from ``in_proj``, the static filter and
    ``depthwise_causal_conv_static``. A flag check only catches the activation *the flag controls*;
    this catches a silu introduced anywhere in the path.
    """
    out: List[CheckResult] = []
    for i, blk in enumerate(model.blocks):
        m = blk.sequence_mixer
        if not isinstance(m, DynamicShortConv):
            continue
        set_alpha_override(model, 0.0)
        with torch.no_grad():
            x = torch.randn(2, 16, spec.d_model, generator=torch.Generator().manual_seed(7))
            pre, post, val = m.in_proj(x)
            ref = m.out_proj(post * depthwise_causal_conv_static(pre * val, m.static_filter))
            got = m(x)
        set_alpha_override(model, None)
        rel = float((got - ref).norm() / ref.norm())
        out.append(
            CheckResult(
                "12",
                "conv path activation-free (numeric)",
                f"{spec.cell}/L{i}",
                passed=(rel < 1e-5 and m.conv_activation is None),
                expected="rel_err < 1e-5 vs activation-free reference, and flag is None",
                actual=f"rel_err={rel:.3e}, conv_activation={m.conv_activation!r}",
                tolerance="1e-5",
            )
        )
    return out


# =============================================================================================
# Check 16 -- use_fla parity and the REALISED conv backend, per arm
# =============================================================================================


def resolved_backend(conv_owner: ShortConv, device: torch.device) -> str:
    """The backend this conv will ACTUALLY dispatch to, evaluated the same way the forward does.

    ``ShortConv._conv`` takes the fused path only when ``self.use_fla and has_fla() and
    x.is_cuda`` all hold; otherwise it falls through to ``nn.Conv1d``. Reproduced here rather than
    inferred from the flag, because the flag is only one of three conjuncts.
    """
    if conv_owner.use_fla and _has_fla() and device.type == "cuda":
        return "fla_fused"
    return "nn.Conv1d"


def _conv_owners(model: MQARModel) -> List[Tuple[str, ShortConv]]:
    return [
        (f"blocks.{i}.sequence_mixer", b.sequence_mixer)
        for i, b in enumerate(model.blocks)
        if isinstance(b.sequence_mixer, ShortConv)
    ]


def check_16_backend_parity(
    spec: ArmSpec,
    model: MQARModel,
    reference: Optional[Mapping[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> List[CheckResult]:
    """``use_fla`` identical on every conv, and the REALISED backend logged and matched.

    **Why this is a guard and not a convention.** Both construction sites currently pin
    ``use_fla=False`` (``arms.py`` for the static ``ShortConv``, ``dynamic_conv.py`` for
    ``DynamicShortConv``), so no path reaches ``ShortConv.__init__``'s ``use_fla=True`` default
    today. But a hardcoded literal is a convention that the next edit can break silently:
    ``use_fla=True`` is **inert** wherever ``fla`` is not installed, because ``has_fla()`` returns
    False and every forward quietly runs ``nn.Conv1d`` anyway. So the flag can diverge between
    arms without changing a single number -- until the code runs somewhere ``fla`` *is* present,
    at which point one arm fuses and the other does not.

    A fused treatment against an unfused baseline is not a controlled comparison in either
    direction, and the bias is invisible in the loss curve. Hence three assertions:

    * 16a -- ``use_fla`` is the same value on every conv **within** an arm;
    * 16b -- the **realised** backend (the three-way conjunct, not the flag) is the same on every
      conv within an arm, and is logged;
    * 16c -- the realised backend family matches the **reference arm's**, so baseline and
      treatment cannot resolve differently.

    :param reference: ``{"use_fla": bool, "backend": str}`` from the S1 arm of the same cell.
    """
    dev = device if device is not None else next(model.parameters()).device
    label = device_dtype_label(model)
    owners = _conv_owners(model)
    out: List[CheckResult] = []
    if not owners:
        return out

    flags = {n: m.use_fla for n, m in owners}
    uniq_flags = set(flags.values())
    out.append(
        CheckResult(
            "16a",
            "use_fla identical across convs",
            spec.cell,
            passed=(len(uniq_flags) == 1),
            expected="one distinct use_fla value",
            actual=f"{sorted(uniq_flags)} over {len(owners)} convs",
            tolerance="exact",
            device_dtype=label,
            note="a hardcoded literal is a convention; this is the guard",
        )
    )

    backends = {n: resolved_backend(m, dev) for n, m in owners}
    uniq_backends = set(backends.values())
    out.append(
        CheckResult(
            "16b",
            "realised conv backend, logged",
            spec.cell,
            passed=(len(uniq_backends) == 1),
            expected="one realised backend across convs",
            actual=f"{sorted(uniq_backends)}",
            tolerance="exact",
            device_dtype=label,
            note=(
                f"has_fla()={_has_fla()}; use_fla={sorted(uniq_flags)}; "
                f"resolved per conv: {backends}"
            ),
        )
    )

    if reference is not None:
        ref_backend = reference.get("backend")
        ref_flag = reference.get("use_fla")
        got_backend = next(iter(uniq_backends)) if len(uniq_backends) == 1 else "MIXED"
        got_flag = next(iter(uniq_flags)) if len(uniq_flags) == 1 else "MIXED"
        out.append(
            CheckResult(
                "16c",
                "backend family matches reference arm",
                spec.cell,
                passed=(got_backend == ref_backend and got_flag == ref_flag),
                expected=f"backend={ref_backend} use_fla={ref_flag} (S1)",
                actual=f"backend={got_backend} use_fla={got_flag}",
                tolerance="exact",
                device_dtype=label,
                note=(
                    "a fused treatment against an unfused baseline is not a controlled "
                    "comparison in either direction"
                ),
            )
        )
    return out


# =============================================================================================
# Check 13 -- the W=2 exact-reparameterization structural check
# =============================================================================================


def check_13_w2_reparam(spec: ArmSpec) -> List[CheckResult]:
    """The verified theorem, in its cheap structural form.

    ``orch_verify_W_minus_2.py``: the static tap family ``kappa[t,k] = C_t*a_k*B_{t-k}`` has
    Jacobian rank exactly ``2T + W - 3``, so the genuinely new dynamic DOF are ``W - 2`` per
    position per channel, and **at W=2 the dynamic block is an EXACT reparameterization of the
    static block** (max log-residual 8.3e-16 -- a constructive realization, not a fit).

    Therefore: **a W=2 dynamic-vs-static difference exceeding seed noise is a bug, not a result.**
    Cheapest falsification test in the program. The lead owns the standalone script; this is the
    structural copy.
    """
    T = 12
    resid = static_realizability_residual(T, spec.width, seed=13)
    new_dof = max(0, spec.width - 2)
    if spec.width == 2:
        ok, exp = resid < 1e-9, "residual < 1e-9 (exactly realizable)"
    else:
        ok, exp = resid > 1e-6, "residual > 1e-6 (NOT static-realizable)"
    return [
        CheckResult(
            "13",
            "W=2 exact reparameterization",
            spec.cell,
            passed=ok,
            expected=exp,
            actual=f"max|log resid|={resid:.3e}",
            note=(
                f"new DOF per position = W-2 = {new_dof}"
                + (
                    "; W=2 IS A FALSIFICATION CONTROL -- any difference above seed noise is a bug"
                    if spec.width == 2
                    else ""
                )
            ),
        )
    ]


# =============================================================================================
# Driver
# =============================================================================================


def run_preflight(
    specs: Optional[Sequence[ArmSpec]] = None,
    seed: int = 0,
    *,
    fast: bool = False,
    engagement_steps: int = 30,
    trend_steps: int = 40,
    parity_steps: int = 20,
    batch_fn: BatchFn = _rand_batch,
    bos_token: Optional[int] = None,
    verbose: bool = True,
) -> List[CheckResult]:
    """Run checks 1-13 and 16 over ``specs``. Returns every :class:`CheckResult`.

    :param bos_token: the harness's BOS/sentinel id, if it uses one. Passed to check 8c. Leave
        ``None`` only when the batch genuinely has no sentinel -- the check reports the skip.
    """
    if specs is None:
        widths = (2, 3) if fast else WIDTHS
        specs = arm_grid(widths=widths)
    results: List[CheckResult] = []

    # S1 reference init losses AND reference conv backend, per (topology, W).
    # Check 8b compares against the loss; check 16c against the backend, so that baseline and
    # treatment cannot silently resolve to different conv implementations.
    ref_loss: Dict[Tuple[str, int], float] = {}
    ref_backend: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for spec in specs:
        if spec.arm != "S1":
            continue
        m = build_arm(spec, seed=seed)
        x, y = batch_fn(spec)
        with torch.no_grad():
            ref_loss[(spec.topology, spec.width)] = float(_loss(m, x, y))
        owners = _conv_owners(m)
        dev = next(m.parameters()).device
        ref_backend[(spec.topology, spec.width)] = {
            "use_fla": owners[0][1].use_fla if owners else None,
            "backend": resolved_backend(owners[0][1], dev) if owners else None,
        }

    for spec in specs:
        model = build_arm(spec, seed=seed, strict=False)
        # Every result produced inside this block inherits the arm's real device/dtype.
        with labelled(model):
            results += _run_cell(
                spec,
                model,
                seed=seed,
                batch_fn=batch_fn,
                bos_token=bos_token,
                engagement_steps=engagement_steps,
                trend_steps=trend_steps,
                parity_steps=parity_steps,
                ref_loss=ref_loss,
                ref_backend=ref_backend,
            )
        if verbose:
            print(
                f"--- {spec.cell} done "
                f"({len([r for r in results if r.blocking])} blocking so far)"
            )
    return results


def _run_cell(
    spec: ArmSpec,
    model: MQARModel,
    *,
    seed: int,
    batch_fn: BatchFn,
    bos_token: Optional[int],
    engagement_steps: int,
    trend_steps: int,
    parity_steps: int,
    ref_loss: Mapping[Tuple[str, int], float],
    ref_backend: Mapping[Tuple[str, int], Dict[str, Any]],
) -> List[CheckResult]:
    """All checks for one cell. Split out so the ambient label wraps exactly one arm."""
    results: List[CheckResult] = []
    results += check_01_generator_numel(spec, model)
    results += check_02_alpha_optimizer(spec, model)
    results += check_03_alpha_zero_equiv(spec, seed=seed, batch_fn=batch_fn)
    results += check_04_bf16_dead_zone(spec)
    results += check_05_grad_magnitudes(spec, seed=seed, batch_fn=batch_fn)
    results += check_05b_dead_branch(spec, seed=seed, batch_fn=batch_fn)
    results += check_06_paired_seeding(spec, seed=seed)
    results += check_07_counts(spec, model)
    results += check_08_init_loss(
        spec,
        seed=seed,
        batch_fn=batch_fn,
        reference_loss=(None if spec.arm == "S1" else ref_loss.get((spec.topology, spec.width))),
        bos_token=bos_token,
    )
    results += check_16_backend_parity(
        spec,
        model,
        reference=(None if spec.arm == "S1" else ref_backend.get((spec.topology, spec.width))),
    )
    results += check_09_engagement(spec, seed=seed, steps=engagement_steps, batch_fn=batch_fn)
    results += check_09b_ablate_at_eval(
        spec, seed=seed, steps=engagement_steps, batch_fn=batch_fn
    )
    results += check_10_u_norm_trend(spec, seed=seed, steps=trend_steps, batch_fn=batch_fn)
    results += check_11_grad_norm_parity(spec, seed=seed, steps=parity_steps, batch_fn=batch_fn)
    results += check_12_no_activation(spec, model)
    results += check_13_w2_reparam(spec)
    return results


def _md(s: str) -> str:
    """Escape a cell for a markdown table.

    Check 4's own name contains ``|dw|``, which silently breaks the table into extra columns. The
    design doc renders straight from this function, so the escape has to live here.
    """
    return str(s).replace("|", "\\|").replace("\n", " ")


def results_table(results: Sequence[CheckResult]) -> str:
    """Markdown, for the design doc."""
    rows = [
        "| # | check | cell | result | expected | actual | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        # An `info` check is labelled INFO whether or not it passed -- it documents rather than
        # gates, so rendering it as PASS would let a reader mistake it for a guard.
        if r.severity == "info":
            mark = f"INFO ({'ok' if r.passed else 'note'})"
        else:
            mark = "PASS" if r.passed else "**FAIL**"
        rows.append(
            f"| {_md(r.check)} | {_md(r.name)} | `{_md(r.cell)}` | {mark} | "
            f"{_md(r.expected)} | {_md(r.actual)} | {_md(r.note)} |"
        )
    return "\n".join(rows)


def summarize(results: Sequence[CheckResult]) -> str:
    blocking = [r for r in results if r.blocking]
    gating = [r for r in results if r.gating]
    info = [r for r in results if r.severity == "info"]
    lines = [
        f"{len(results)} checks: "
        f"{sum(1 for r in results if r.passed)} pass, "
        f"{len(blocking)} BLOCKING FAIL ({len(gating)} of them HARD GATES), "
        f"{sum(1 for r in results if not r.passed and r.severity not in ('fail', 'gate'))} "
        f"non-blocking, {len(info)} info",
    ]
    for r in blocking:
        lines.append(f"  {'GATE-ABORT' if r.gating else 'BLOCKING'}: {r}")
    if gating:
        lines.append("")
        lines.append(
            "  *** HARD GATE FAILED: the absolute init loss (or BOS) is out of band. NO "
            "BETWEEN-ARM DELTA MAY BE REPORTED from this run. A delta on a broken absolute is a "
            "different experiment. ***"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="W in {2,3} only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--topologies", nargs="*", default=None)
    ap.add_argument("--widths", nargs="*", type=int, default=None)
    ap.add_argument("--engagement-steps", type=int, default=30)
    ap.add_argument(
        "--bos-token",
        type=int,
        default=None,
        help="BOS/sentinel id, for check 8c. Omit ONLY if the batch genuinely has none.",
    )
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    kw = {}
    if args.arms:
        kw["arms"] = args.arms
    if args.topologies:
        kw["topologies"] = args.topologies
    widths = tuple(args.widths) if args.widths else ((2, 3) if args.fast else WIDTHS)
    specs = arm_grid(widths=widths, **kw)

    print(f"Exp-2 pre-flight: {len(specs)} cells, seed {args.seed}")
    print(f"  {device_dtype_label()}   has_fla()={_has_fla()}")
    print(f"  d_model={D_MODEL} n_layers={N_LAYERS} vocab={VOCAB_SIZE} R={RANK}")
    print(f"  ln(vocab)={LN_VOCAB:.6f}  band=[{LN_VOCAB:.4f}, {LN_VOCAB + 0.25:.4f}]  HARD GATE")
    print(f"  bos_token={args.bos_token!r}" + ("  (check 8c SKIPPED)" if args.bos_token is None else ""))
    print(f"  bf16 half-ulp at 1.0 = {BF16_HALF_ULP:g}  engagement abort floor = {ENGAGEMENT_ABORT:g}")
    print()
    results = run_preflight(
        specs,
        seed=args.seed,
        engagement_steps=args.engagement_steps,
        bos_token=args.bos_token,
        verbose=not args.quiet,
    )
    print()
    if args.markdown:
        print(results_table(results))
        print()
    else:
        for r in results:
            print(r)
        print()
    print(summarize(results))
    return 1 if any(r.blocking for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
