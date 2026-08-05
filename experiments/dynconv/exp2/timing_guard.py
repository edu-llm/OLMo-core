"""Physical-impossibility checks for the Exp-2 timing path.

WHY THIS FILE EXISTS -- a cross-team scar, not a hypothetical
-------------------------------------------------------------
Exp-1 found that ``torch._dynamo``'s eager fallback is **process-sticky**: it warns ONCE at
``recompile_limit``, then silently stops compiling for the rest of the process. 13 of 18 of their
cells were poisoned while the log still said "2 warnings", producing a **43x** timing error that
no config assertion caught. Only a per-cell PHYSICAL check found it.

The transferable lesson, and the reason this is not just a `torch._dynamo.reset()` call:

    ASSERT WHAT EXECUTED, NOT WHAT WAS CONFIGURED.

A ``backend_used`` field records the REQUEST, not the RECEIPT. A config assertion re-reads the
value you set. Neither can see a silent fallback. What CAN see it is a quantity that would have to
be physically impossible for a correct measurement -- so these checks are stated as impossibilities
and are deliberately loose: a wide band that catches a 43x error is worth far more than a tight one
that flags noise.

This is also the `empty comparison set` finding (EXP2-DESIGN.md Sec 12.4) applied to timing: a
check that re-reads configuration has an empty comparison set and reports success no matter what
ran.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# MEASURED on FarmShare wheat-04, CPU, torch 2.13.0+cpu, fp32, 16 threads, N512_D64, batch 64,
# real arms (job 1676379). Per-step, warmed up, steps-only -- NOT whole-run wall clock, per repo
# memory `measure-throughput-post-startup` (that read 3.1x LOW on a 40-step probe).
REFERENCE_S_PER_STEP: Dict[str, float] = {
    "S1": 0.9957,   # static-LIV baseline, allliv
    "S4": 1.7723,   # dynamic-LIV
    "S2": 1.8551,   # permuted-conditioning control
}

#: A correct measurement cannot be this far off the reference. Deliberately WIDE -- a shared node,
#: a different core count or another tenant easily moves this 2-3x. It is sized to catch the
#: Exp-1 class of error (43x), not to police noise.
SLOWDOWN_CEILING = 8.0
SPEEDUP_FLOOR = 4.0


def check_rate_plausible(arm: str, s_per_step: float) -> Optional[str]:
    """Is this per-step time physically possible for a correct run of ``arm``?"""
    ref = REFERENCE_S_PER_STEP.get(arm)
    if ref is None or s_per_step <= 0:
        return None
    if s_per_step > ref * SLOWDOWN_CEILING:
        return (
            f"IMPOSSIBLE: {arm} at {s_per_step:.3f} s/step is {s_per_step/ref:.1f}x the measured "
            f"reference {ref:.3f}. A silent fallback (Exp-1's process-sticky dynamo eager) looks "
            f"exactly like this."
        )
    if s_per_step * SPEEDUP_FLOOR < ref:
        return (
            f"IMPOSSIBLE: {arm} at {s_per_step:.3f} s/step is {ref/s_per_step:.1f}x FASTER than the "
            f"reference {ref:.3f}. Work is being skipped -- a no-op arm, a short-circuited kernel, "
            f"or a cell that did not run."
        )
    return None


def check_arm_ordering(rates: Dict[str, float]) -> List[str]:
    """S2 and S4 MUST be slower than S1. This is the ordering check, and it is the sharp one.

    S2/S4 add a `d -> R -> W*d` generator and a per-position filter apply that S1 does not have,
    at byte-identical parameter counts. So on identical hardware they are strictly more work.
    **A dynamic arm measuring faster than the static baseline is not a surprising result, it is a
    broken measurement** -- which is precisely how Exp-1's poisoned cells presented.

    Measured margins (FarmShare CPU, real arms): S4/S1 = 1.78x, S2/S1 = 1.86x. The 1.02 threshold
    below is far inside that, so it fires on an inversion rather than on jitter.
    """
    problems: List[str] = []
    base = rates.get("S1")
    if not base:
        return problems
    for arm in ("S2", "S4"):
        got = rates.get(arm)
        if got and got < base * 1.02:
            problems.append(
                f"IMPOSSIBLE: {arm} ({got:.3f} s/step) is not slower than S1 ({base:.3f}). {arm} "
                f"adds a generator S1 does not have at an identical parameter count, so it cannot "
                f"be faster. Measured reference margin is 1.8x. Suspect a no-op mechanism (wrong "
                f"wire slot) or a poisoned cell."
            )
    return problems


def check_recompile_health() -> List[str]:
    """Did anything silently stop compiling? Reads the RECEIPT, not the request.

    ``torch._dynamo.utils.counters`` accumulates real recompile/graph-break events. Empty is fine
    (nothing compiled); non-trivial fallback counts are the signature Exp-1 hit. Returns notes
    rather than raising, because on a pure-eager CPU run there is legitimately nothing to see --
    and per Sec 12.4 that case must be reported as SKIPPED, never as a PASS.
    """
    notes: List[str] = []
    try:
        import torch._dynamo as dynamo  # type: ignore
    except Exception:
        return ["SKIPPED: torch._dynamo unavailable (nothing compiled, nothing to check)"]
    try:
        counters = dynamo.utils.counters
    except Exception:
        return ["SKIPPED: dynamo counters unreadable"]
    frames = dict(counters.get("frames", {}))
    graph_breaks = sum(dict(counters.get("graph_break", {})).values())
    if not frames and not graph_breaks:
        return ["SKIPPED: no dynamo activity (pure eager) -- not a PASS, there was nothing to check"]
    notes.append(f"dynamo frames={frames} graph_breaks={graph_breaks}")
    ok = frames.get("ok", 0)
    total = frames.get("total", 0)
    if total and ok < total:
        notes.append(
            f"WARNING: {total - ok} of {total} frames failed to compile. Exp-1's fallback is "
            f"PROCESS-STICKY -- it warns once then stops compiling silently for the whole process."
        )
    return notes


def reset_between_cells() -> str:
    """Clear dynamo state between cells. Exp-1's fix, and it is cheap insurance even in eager.

    Their fallback persisted for the remainder of the PROCESS, so a 240-cell sweep in one process
    can have every later cell poisoned by one early limit breach.
    """
    try:
        import torch._dynamo as dynamo  # type: ignore

        dynamo.reset()
        return "torch._dynamo.reset() called"
    except Exception as exc:
        return f"dynamo reset unavailable ({type(exc).__name__}) -- fine if nothing compiles"


def audit(rates: Dict[str, float], *, strict: bool = True) -> List[str]:
    """Run every physical check. Returns problems; raises under ``strict`` if any are impossible."""
    problems: List[str] = []
    for arm, s in rates.items():
        msg = check_rate_plausible(arm, s)
        if msg:
            problems.append(msg)
    problems.extend(check_arm_ordering(rates))
    if problems and strict:
        raise RuntimeError(
            "TIMING AUDIT FAILED -- these are physical impossibilities, not slow cells:\n  "
            + "\n  ".join(problems)
        )
    return problems


if __name__ == "__main__":
    print("Reference (FarmShare wheat-04, CPU, torch 2.13.0+cpu, fp32, N512_D64, real arms):")
    for a, s in REFERENCE_S_PER_STEP.items():
        print(f"  {a}: {s:.4f} s/step -> {s*8000/60:.1f} min/cell at the calibrated 8000 steps")
    print("\nrecompile health:", "; ".join(check_recompile_health()))
    print("\n-- self-test: the checks must FIRE on impossible input --")
    for label, rates in (
        ("dynamic FASTER than static (Exp-1's signature)", {"S1": 1.0, "S4": 0.5}),
        ("43x slowdown (Exp-1's actual error)", {"S1": 1.0, "S4": 76.2}),
        ("healthy, measured", dict(REFERENCE_S_PER_STEP)),
    ):
        try:
            found = audit(rates, strict=True)
            print(f"  {label}: no problems{' (correct)' if not found else ''}")
        except RuntimeError as exc:
            first = str(exc).splitlines()[1].strip()
            print(f"  {label}: CAUGHT -> {first[:110]}")
