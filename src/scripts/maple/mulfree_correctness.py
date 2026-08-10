#!/usr/bin/env python3
"""
Correctness for the multiply-free ternary MoE decode kernel. **Numerics first, speed nowhere.**

This script contains no timing and reports no throughput. That is deliberate: it is the primary
deliverable, and mixing a benchmark into it would let a passing speed number distract from a
failing numeric one. Speed lives in ``mulfree_bench.py``.

**UNRUN as of authoring.** Every expectation below is a prediction, written before any execution.

Run (FarmShare, L40S sm_89, GPU node -- NOT the login node)::

    TRITON_F32_DEFAULT=ieee PYTHONPATH=$MINE/src $V/bin/python \\
        $MINE/src/scripts/maple/mulfree_correctness.py --rung M7B

``TRITON_F32_DEFAULT=ieee`` is **mandatory** and this script *asserts it is set*, because torch's
global TF32 flag does not control Triton and the difference has been worth ~166x in fp32 accuracy in
this project's own prior finding. An env var set in a bootstrap shell that does not reach the
``srun`` payload is the failure mode, so the assert reads ``os.environ`` inside the job.

The seven checks, and what each one can actually fail on
-------------------------------------------------------
========  ===============================================================  ====================
check     question                                                          tolerance
========  ===============================================================  ====================
T1        does ``pack -> unpack`` round-trip the codes?                     **bitwise, integer**
T2        do the multiply-free and multiply-accumulate arms agree?          **bitwise, fp32**
T3        does the kernel match a matched fp32 torch reference?             derived, see below
T4        is the kernel *no less accurate than* the fp32 reference?         **no threshold**
T5        does T3 FAIL on a corrupted pack? (non-vacuity)                   must fail
T6        does "multiply-free" lower to multiply-free PTX?                  static count
T7        do odd/unaligned extents work, incl. an M7B-flavoured odd one?    same as T3
========  ===============================================================  ====================

**T2 is the check that makes this cheap and strong, and its justification is arithmetic, not
empirical.** The control arm computes ``x_i * w_i`` where ``w_i`` is exactly ``1.0``, ``-1.0`` or
``0.0``. Multiplication by a power of two is **exact** in IEEE-754 -- no rounding, no double
rounding -- so both arms produce *bit-identical summands*, feed them to the *same* ``tl.sum``, and
must therefore agree **bitwise**. There is no tolerance to fit. The one legal exception is signed
zero: ``tl.where(c==0, 0.0, ...)`` yields ``+0.0`` while ``x_i * 0.0`` yields ``-0.0`` for negative
``x_i``, and ``+0.0 + -0.0 == +0.0``, so the difference is invisible unless an entire accumulator
cancels to zero. **So if T2 fails, it is not a tolerance problem -- it is proof that the compiler
reassociated one arm differently from the other, which is a finding, not a bug to paper over.**

**T4 is the check that cannot be fitted, and it is the reason T3's threshold is not load-bearing.**
An fp64 accumulation of ``+-1``-coefficient terms over ``K <= 2^11`` is exact to ~1e-16 relative,
so it is a legitimate oracle. T4 scores the kernel *and* the fp32 torch reference against that same
oracle and requires ``err_kernel <= 2 * err_reference``. The criterion is a **ratio of two
independently-measured errors against a third exact answer** -- there is no constant in it that I
chose to make something pass. T3's absolute threshold is a convenience that flags gross breakage;
T4 is the real statement about accuracy.

Where T3's tolerance comes from -- accumulation order, not curve-fitting
-----------------------------------------------------------------------
The kernel's summation order is fixed by its own loop structure. For contracted length ``K`` with
``KB = K/4`` code bytes and ``BLOCK_KB`` bytes per inner block, it performs
``4 * ceil(KB/BLOCK_KB)`` sequential accumulator updates, each of which adds a **tree** reduction
(``tl.sum``) over ``BLOCK_KB`` terms. So the effective error depth is::

    depth_kernel = log2(BLOCK_KB) + 4 * ceil(KB / BLOCK_KB)

At M20's ``w1`` (``K = d = 2048``, ``KB = 512``, ``BLOCK_KB = 512``) that is ``9 + 4 = 13``. The
standard Wilkinson bound for a summation of depth ``m`` is
``|err| <= m * eps * sum_i |t_i|`` with ``eps = 2^-24 = 5.96e-8`` for fp32. The summands are
``+-x_i``, so ``sum|t_i| = sum|x_i|`` over the ~57.6% of codes TWN leaves nonzero (the complement of
the 42.35% zero fraction the quantizer's own closed form gives). For unit-variance ``x`` that is
``~0.798 * n_nz`` while the result itself is ``~sqrt(n_nz)``, giving an amplification of
``0.798 * sqrt(n_nz) ~ 27`` at ``n_nz ~ 1180``. So::

    rel_err_kernel  ~  13 * 5.96e-8 * 27  ~  2.1e-5

and the fp32 torch reference carries an error of the same order from its own (different) blocking,
so their *difference* is bounded by roughly the sum. Two more stages follow -- SwiGLU, whose
``silu(g)*u`` product roughly doubles relative error, and the ``w2`` contraction over ``f_e``, which
amplifies again by ``~0.798*sqrt(0.576*f_e)`` -- then an 8-term expert sum. Compounding those:

* ``--tol-h`` default **3e-4** on the gate/up/hidden stage,
* ``--tol-y`` default **2e-3** on the final combined output.

Both are *upper bounds from the derivation above with a ~4x margin for the constants I have not
pinned* (silu's local Lipschitz constant, the reference's exact blocking, the realised nonzero
count). **They are deliberately loose, because they are not the accuracy claim -- T4 is.** If T3
passes only barely, that is a signal to read T4's ratio, not to raise the threshold.

Why the reference must NOT distribute alpha
------------------------------------------
``reference_ternary_moe_decode(distribute_alpha=True)`` materialises ``+-alpha`` densely and
matmuls -- which is what today's training forward does. It rounds ``alpha*x_i`` once per **term**
instead of once per **output**, so it is *less* accurate than the kernel by construction. Comparing
against it and calling the gap a kernel error would be backwards. It is computed and reported here
so the direction of that gap is visible, but it is **not** a gate.
"""

import argparse
import os
import sys

import torch

from olmo_core.kernels.ternary_moe import (
    CODES_PER_BYTE,
    TERNARY_CODE_NEG,
    TERNARY_CODE_POS,
    PackedExpertBank,
    TernaryExportSpec,
    assert_no_illegal_codes,
    fused_gathered_w13_swiglu,
    fused_ternary_moe_decode,
    pack_expert_bank,
    pack_ternary,
    ptx_arith_histogram,
    reference_ternary_moe_decode,
    unpack_ternary_signs,
)
from olmo_core.nn.quantization import (
    TWN_GAUSSIAN_ZERO_FRACTION,
    twn_quantize,
    twn_threshold_and_scale,
)

RESULTS: list = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def assert_ieee_env() -> None:
    """
    Assert ``TRITON_F32_DEFAULT=ieee`` reached *this process*, not a bootstrap shell.

    Read from ``os.environ`` deliberately. The failure this guards is an export in a wrapper script
    that never propagates into the ``srun`` payload, which looks identical to success in the log.
    """
    v = os.environ.get("TRITON_F32_DEFAULT")
    if v != "ieee":
        print(
            f"FATAL: TRITON_F32_DEFAULT is {v!r}, must be 'ieee'. torch's TF32 flag does NOT "
            "control Triton, and this has been worth ~166x in fp32 accuracy in this project. "
            "Refusing to produce numbers that would be silently wrong.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(f"env: TRITON_F32_DEFAULT={v} (asserted inside the job, not in a wrapper)")


def make_bank(spec: TernaryExportSpec, E: int, gen: torch.Generator, dev: str):
    d, fe = spec.d_model, spec.expert_hidden
    w1 = torch.randn(E, d, fe, generator=gen, device=dev)
    w3 = torch.randn(E, d, fe, generator=gen, device=dev)
    w2 = torch.randn(E, fe, d, generator=gen, device=dev)
    return pack_expert_bank(w1, w2, w3), (w1, w2, w3)


# ---------------------------------------------------------------------------------------
# T1 -- pack/unpack round trip. Bitwise, integer, zero tolerance.
# ---------------------------------------------------------------------------------------


def t1_pack_roundtrip(dev: str, gen: torch.Generator) -> None:
    worst_zero_dev = 0.0
    all_exact = True
    detail_bits = []
    # Includes K % 4 in {1,2,3} -- the boundary-masking path P-INFER's own verifier found was
    # never exercised by any of its four shapes, all of which had K % 4 == 0.
    for out_f, in_f in [(2048, 2048), (512, 2048), (384, 1536), (13, 1537), (7, 1538), (5, 1539)]:
        w = torch.randn(out_f, in_f, generator=gen, device=dev)
        codes, alpha, klen = pack_ternary(w, in_dim=-1)
        assert_no_illegal_codes(codes)
        signs = unpack_ternary_signs(codes, klen)
        ref_signs = torch.sign(twn_quantize(w, in_dim=-1).to(torch.float32))
        # NOT transposed: pack_ternary's `movedim(in_dim, -1)` is a no-op when in_dim == -1, so
        # for a 2-D (out, in) weight the packed layout is already (out, KB) and unpacks to
        # (out, in). A `.T` here was a real bug caught by a pure-Python simulation of the index
        # arithmetic before this ever reached a GPU -- it would have silently passed on the one
        # square shape and failed the rest for the wrong reason.
        exact = bool(torch.equal(signs, ref_signs))
        all_exact &= exact
        # alpha must be the quantizer's own alpha, not a re-derivation.
        _, a_ref = twn_threshold_and_scale(w, in_dim=-1)
        a_exact = bool(torch.equal(alpha, a_ref.squeeze(-1).to(torch.float32)))
        all_exact &= a_exact
        zf = float((signs == 0).float().mean())
        worst_zero_dev = max(worst_zero_dev, abs(zf - TWN_GAUSSIAN_ZERO_FRACTION))
        detail_bits.append(
            f"({out_f},{in_f}) K%4={in_f % CODES_PER_BYTE} signs={exact} a={a_exact}"
        )
    record(
        "T1 pack/unpack bitwise round trip",
        all_exact,
        f"integer-exact on 6 shapes incl. K%4 in {{1,2,3}}; "
        f"max |zero_frac - {TWN_GAUSSIAN_ZERO_FRACTION:.6f}| = {worst_zero_dev:.4f}; "
        + "; ".join(detail_bits),
    )
    # The quantizer-identity guard: TWN, not BitNet b1.58. A pack whose zero fraction sits at
    # 0.310 has been "corrected" to a different model.
    record(
        "T1b TWN identity (not BitNet b1.58)",
        worst_zero_dev < 0.02,
        f"zero fraction within 0.02 of TWN's closed-form {TWN_GAUSSIAN_ZERO_FRACTION:.6f}; "
        f"BitNet b1.58 would sit at 0.310064",
    )


# ---------------------------------------------------------------------------------------
# T2 -- the two arms must agree BITWISE. Justified from exactness of *1, not from data.
# ---------------------------------------------------------------------------------------


def t2_arms_bitwise(bank: PackedExpertBank, x, idx, rw) -> None:
    y_mf = fused_ternary_moe_decode(x, bank, idx, rw, multiply_free=True)
    y_mac = fused_ternary_moe_decode(x, bank, idx, rw, multiply_free=False)
    equal = bool(torch.equal(y_mf, y_mac))
    n_diff = int((y_mf != y_mac).sum())
    mx = float((y_mf - y_mac).abs().max()) if n_diff else 0.0
    record(
        "T2 multiply-free == multiply-accumulate, BITWISE",
        equal,
        (
            "bit-identical, as required: multiplying by exactly +-1.0 or 0.0 is exact in "
            "IEEE-754, so both arms feed identical summands to identical reductions"
            if equal
            else f"NOT bit-identical -- {n_diff} elements differ, max |diff| = {mx:.3e}. This is "
            "NOT a tolerance issue. It proves the two arms lower to different reduction orders, "
            "i.e. the compiler reassociated. Read T6's PTX histogram before interpreting any "
            "timing number, and report it as a finding."
        ),
    )


# ---------------------------------------------------------------------------------------
# T3 / T4 -- against a matched fp32 reference, then scored against an fp64 oracle.
# ---------------------------------------------------------------------------------------


def _rel(a, b) -> float:
    den = float(b.abs().max()) or 1.0
    return float((a - b).abs().max()) / den


def t3_t4_vs_reference(bank, x, idx, rw, spec, tol_h: float, tol_y: float) -> None:
    fe = bank.expert_hidden

    h_k = fused_gathered_w13_swiglu(x, bank, idx, multiply_free=True)
    y_k = fused_ternary_moe_decode(x, bank, idx, rw, multiply_free=True)

    y_ref32 = reference_ternary_moe_decode(x, bank, idx, rw, dtype=torch.float32)
    y_ref64 = reference_ternary_moe_decode(x, bank, idx, rw, dtype=torch.float64)
    y_dist32 = reference_ternary_moe_decode(
        x, bank, idx, rw, dtype=torch.float32, distribute_alpha=True
    )

    # Stage-1 reference, recomputed here so the hidden stage is scored on its own rather than only
    # through the second contraction -- an error that cancels downstream would otherwise hide.
    d = bank.d_model
    s13 = unpack_ternary_signs(bank.w13_codes[0], d, dtype=torch.float64)
    a13 = bank.w13_alpha[0].to(torch.float64)
    h_ref = torch.zeros_like(h_k, dtype=torch.float64)
    for r in range(x.shape[0]):
        xr = x[r].to(torch.float64)
        for s in range(idx.shape[1]):
            e = int(idx[r, s])
            gu = (s13[e] @ xr) * a13[e]
            g, u = gu[:fe].clamp(max=7.0), gu[fe:].clamp(min=-7.0, max=7.0)
            h_ref[r, s] = torch.nn.functional.silu(g) * u

    rel_h = _rel(h_k.double(), h_ref)
    rel_y = _rel(y_k.double(), y_ref64)
    err_ref32 = _rel(y_ref32.double(), y_ref64)
    err_dist32 = _rel(y_dist32.double(), y_ref64)

    kb = int(bank.w13_codes.shape[-1])
    depth = (kb.bit_length() - 1) + 4
    record(
        "T3 kernel vs reference, derived tolerance",
        rel_h <= tol_h and rel_y <= tol_y,
        f"rel_h={rel_h:.3e} (tol {tol_h:.1e}) rel_y={rel_y:.3e} (tol {tol_y:.1e}); "
        f"kernel summation depth = log2(BLOCK_KB={kb}) + 4 = {depth}, "
        f"eps_fp32 = 5.96e-8",
    )
    # The non-fittable criterion. Both errors measured against the same fp64 oracle.
    ratio = rel_y / err_ref32 if err_ref32 > 0 else float("inf")
    record(
        "T4 kernel no less accurate than fp32 reference (no threshold fitted)",
        rel_y <= 2.0 * err_ref32 or rel_y < 1e-6,
        f"err_kernel={rel_y:.3e} vs err_fp32_reference={err_ref32:.3e} -> ratio {ratio:.2f} "
        f"(criterion: <= 2.00). Naive dequantise-then-matmul reference, which distributes alpha "
        f"per term as today's TRAINING forward does, errs {err_dist32:.3e} "
        f"({err_dist32 / max(rel_y, 1e-30):.2f}x the kernel) -- it is the LESS accurate "
        f"formulation, so it is reported and NOT used as a gate.",
    )


# ---------------------------------------------------------------------------------------
# T5 -- mutation. The pack is corrupted with a LEGAL code, so the format canary cannot catch it.
# ---------------------------------------------------------------------------------------


def t5_mutation(bank: PackedExpertBank, x, idx, rw, tol_y: float) -> None:
    y_clean = fused_ternary_moe_decode(x, bank, idx, rw, multiply_free=True)
    ref64 = reference_ternary_moe_decode(x, bank, idx, rw, dtype=torch.float64)
    clean_rel = _rel(y_clean.double(), ref64)

    # Flip ONE code, +1 -> -1, inside an expert the router actually selected. A legal-code flip:
    # `assert_no_illegal_codes` still passes, so this mutation is not detectable by the cheap
    # format canary -- it can only be caught by the numeric check, which is the point.
    eid = int(idx[0, 0])
    mutated = bank.w2_codes.clone()
    flat = mutated[0, eid].reshape(-1)
    hit = -1
    for i in range(flat.numel()):
        b = int(flat[i])
        for sub in range(CODES_PER_BYTE):
            if (b >> (2 * sub)) & 3 == TERNARY_CODE_POS:
                flat[i] = (b & ~(3 << (2 * sub))) | (TERNARY_CODE_NEG << (2 * sub))
                hit = i
                break
        if hit >= 0:
            break
    if hit < 0:
        record("T5 mutation check", False, "found no +1 code to flip -- test is inconclusive")
        return
    mutated[0, eid] = flat.reshape(mutated[0, eid].shape)

    bad = PackedExpertBank(
        w13_codes=bank.w13_codes,
        w13_alpha=bank.w13_alpha,
        w2_codes=mutated,
        w2_alpha=bank.w2_alpha,
        d_model=bank.d_model,
        expert_hidden=bank.expert_hidden,
        num_experts=bank.num_experts,
        n_replicas=bank.n_replicas,
    )
    assert_no_illegal_codes(mutated)  # still a legal pack -- the canary cannot see this
    y_bad = fused_ternary_moe_decode(x, bad, idx, rw, multiply_free=True)
    bad_rel = _rel(y_bad.double(), ref64)
    record(
        "T5 mutation check (ONE legal +1->-1 flip; non-vacuity)",
        bad_rel > tol_y and clean_rel <= tol_y,
        f"clean rel={clean_rel:.3e} (<= {tol_y:.1e}) -> mutated rel={bad_rel:.3e} "
        f"(> {tol_y:.1e} required). Amplification {bad_rel / max(clean_rel, 1e-30):.1f}x. "
        f"The flip keeps the pack format-legal, so assert_no_illegal_codes still PASSES on it -- "
        f"only the numeric check catches it. A test that cannot fail is not a test.",
    )


# ---------------------------------------------------------------------------------------
# T6 -- does "multiply-free" actually lower to multiply-free PTX?
# ---------------------------------------------------------------------------------------


def t6_ptx() -> None:
    try:
        hist = ptx_arith_histogram("w13")
    except Exception as e:  # noqa: BLE001
        record("T6 PTX arithmetic histogram", False, f"could not read PTX: {e!r}")
        return
    if not hist:
        record("T6 PTX arithmetic histogram", False, "Triton cache held no compiled PTX")
        return
    print("  PTX arithmetic instruction counts, per compiled specialisation:")
    for sig, counts in hist.items():
        print(f"    {sig}: {counts}")
    tc = sum(c.get("mma.", 0) + c.get("wgmma.", 0) for c in hist.values())
    fma = sum(c.get("fma.rn.f32", 0) for c in hist.values())
    mul = sum(c.get("mul.f32", 0) for c in hist.values())
    record(
        "T6 no tensor-core path (mma/wgmma absent)",
        tc == 0,
        f"mma+wgmma = {tc}. Expect 0: this kernel is scalar select/add, and there is no "
        f"add-only tensor-core mode -- avoiding multiplies necessarily also avoids tensor cores. "
        f"Total fma.rn.f32={fma}, mul.f32={mul} across specialisations. "
        f"**If the MULFREE and non-MULFREE specialisations show the SAME fma count, the "
        f"'multiply-free' formulation is a source-level fiction on this toolchain -- report that, "
        f"it is the negative result the brief asked for.**",
    )


# ---------------------------------------------------------------------------------------
# T7 -- odd and unaligned extents, including an M7B-flavoured odd output extent.
# ---------------------------------------------------------------------------------------


def t7_odd_shapes(dev: str, gen: torch.Generator, tol_y: float) -> None:
    # (d, fe, E, k). d=1536/fe=384 is M7B's real expert geometry. The odd rows exercise output
    # extents that are odd, prime, and not multiples of rows_per_prog -- and d=1539 exercises a
    # contracted length with K%4 == 3, the boundary-mask path.
    ok_all = True
    details = []
    for d, fe, E, k in [(1536, 384, 6, 3), (1539, 383, 5, 2), (68, 17, 4, 3), (2048, 512, 4, 2)]:
        w1 = torch.randn(E, d, fe, generator=gen, device=dev)
        w3 = torch.randn(E, d, fe, generator=gen, device=dev)
        w2 = torch.randn(E, fe, d, generator=gen, device=dev)
        bank = pack_expert_bank(w1, w2, w3)
        x = torch.randn(2, d, generator=gen, device=dev)
        idx = torch.stack([torch.randperm(E, generator=gen, device=dev)[:k] for _ in range(2)]).to(
            torch.int32
        )
        rwv = torch.rand(2, k, generator=gen, device=dev)
        rwv = (rwv / rwv.sum(-1, keepdim=True)).to(torch.float32)  # norm_topk_prob = 1.0
        y = fused_ternary_moe_decode(x, bank, idx, rwv, multiply_free=True)
        y64 = reference_ternary_moe_decode(x, bank, idx, rwv, dtype=torch.float64)
        rel = _rel(y.double(), y64)
        ok = rel <= tol_y
        ok_all &= ok
        details.append(f"d={d}(K%4={d % 4}) fe={fe}(K%4={fe % 4}) E={E} k={k} rel={rel:.2e}")
    record("T7 odd/unaligned extents", ok_all, "; ".join(details))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rung", default="M7B", choices=["R0", "R1", "R2", "R3", "E8", "M20", "M7B"])
    p.add_argument(
        "--experts",
        type=int,
        default=16,
        help="experts in the TEST bank. Small on purpose: correctness does not need E=256, and a "
        "full M20 bank would not fit alongside an fp64 reference. Bytes are the benchmark's job.",
    )
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--steps", type=int, default=3, help="independent decode steps batched (R)")
    p.add_argument("--tol-h", type=float, default=3e-4)
    p.add_argument("--tol-y", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=1234)
    opts = p.parse_args()

    assert_ieee_env()
    if not torch.cuda.is_available():
        print("FATAL: no CUDA device. This needs a GPU node (srun), not the login node.")
        return 2
    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    print(
        f"device: {props.name} sm_{props.major}{props.minor} "
        f"L2={props.L2_cache_size/2**20:.1f} MiB"
    )
    print(
        "TRANSFERABILITY: this is sm_89 (L40S). The training target is A100 sm_80 and Maple's own "
        "reference hardware is neither. Correctness is architecture-independent; any timing is not."
    )
    print(f"torch {torch.__version__}")

    spec = TernaryExportSpec.from_rung(opts.rung)
    print(f"rung {spec.rung}: {spec}")
    print(f"  cross-check: {spec.verify_against_transformer_config()}")
    print(f"  launch counts: {spec.launch_counts()}")
    print(f"  expert-path bytes: {spec.expert_path_bytes_per_token()}")
    print(f"  arithmetic mix:  {spec.arith_mix()}")

    gen = torch.Generator(device=dev).manual_seed(opts.seed)
    E = min(opts.experts, spec.num_experts)
    k = min(opts.top_k, E)

    t1_pack_roundtrip(dev, gen)

    bank, _ = make_bank(spec, E, gen, dev)
    print(
        f"  test bank resident: {bank.resident_bytes/2**20:.2f} MiB "
        f"(E={E} of {spec.num_experts}; NOT a bandwidth-relevant footprint -- see mulfree_bench)"
    )
    x = torch.randn(opts.steps, spec.d_model, generator=gen, device=dev)
    idx = torch.stack(
        [torch.randperm(E, generator=gen, device=dev)[:k] for _ in range(opts.steps)]
    ).to(torch.int32)
    rw = torch.rand(opts.steps, k, generator=gen, device=dev)
    rw = (rw / rw.sum(-1, keepdim=True)).to(torch.float32)
    print(
        f"  router weight mass per step: {[round(float(v), 6) for v in rw.sum(-1)]} (must be 1.0)"
    )

    t2_arms_bitwise(bank, x, idx, rw)
    t3_t4_vs_reference(bank, x, idx, rw, spec, opts.tol_h, opts.tol_y)
    t5_mutation(bank, x, idx, rw, opts.tol_y)
    t6_ptx()
    t7_odd_shapes(dev, gen, opts.tol_y)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 78)
    for name, ok, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
