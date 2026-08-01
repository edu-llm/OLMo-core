"""
Independent closed-form parameter / FLOP ledger for the LIV arm study,
swept over candidate vocabulary sizes.

Written from first principles (NOT by importing olmo_core) so that it is an
independent check on `liv_arms.py`.  A second script, `ledger_vocab_olmo.py`,
cross-checks the parameter numbers by actually building the olmo-core configs.

Conventions replicated from olmo-core (verified by reading the source):
  * Attention.num_flops_per_token  = 6*params + 12*n_heads*head_dim*seq_len
      (no causal 1/2; params includes the per-head q/k RMSNorms)
  * FeedForward.num_flops_per_token = 6*params
  * LMHead.num_flops_per_token      = 6*params  (includes the final RMSNorm AND
      the tied vocab x d matrix, which is NOT deduped for FLOPs)
  * TransformerBlock.num_flops_per_token = attention + feed_forward ONLY
      -> the two per-block RMSNorms contribute 0 FLOPs
  * ShortConv.num_flops_per_token = M*(in_proj+out_proj params) + M*k*d + M*d
      where M = 6 in the CURRENT worktree and M = 2 in the OLDER version still
      installed on FarmShare.  Both are computed below; see SC_MULT.

Parameter count is DEDUPED (tied embeddings counted once), matching
`liv_arms._count_params`.
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- frozen geometry
N_LAYERS = 16
D_MODEL = 1024
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 64
SWIGLU_WIDTH = 4608
KERNEL = 3
ATTN_LAYERS: Tuple[int, ...] = (2, 5, 8, 10, 12, 14)
L0_PARAM_TARGET = 354_483_968

VOCABS = [
    ("50257 gpt2-raw", 50257),
    ("50304 gpt2-pad", 50304),
    ("50432 neox-20b", 50432),
    ("65536 lfm2", 65536),
]


@dataclass(frozen=True)
class Arm:
    name: str
    attention_layers: Tuple[int, ...] = ATTN_LAYERS
    kernel_size: int = KERNEL
    gate_structure: str = "dense"
    gate_rank: Optional[int] = None
    gate_groups: Optional[int] = None
    d_model: int = D_MODEL
    swiglu_width: int = SWIGLU_WIDTH
    n_kv_heads: int = N_KV_HEADS

    @property
    def n_attn(self) -> int:
        return len(self.attention_layers)

    @property
    def n_liv(self) -> int:
        return N_LAYERS - len(self.attention_layers)


# ---------------------------------------------------------------- component formulas
def mixer_params(a: Arm) -> int:
    """ShortConvConfig.num_params(d_model), transcribed."""
    d = a.d_model
    p = d * d  # value_proj (always dense)
    if a.gate_structure == "dense":
        p += 2 * d * d
    elif a.gate_structure == "lowrank":
        assert a.gate_rank
        p += d * (2 * a.gate_rank)  # shared d -> 2r down-projection
        p += 2 * a.gate_rank * d  # two r -> d up-projections
    elif a.gate_structure == "grouped":
        assert a.gate_groups
        p += 2 * (d * d // a.gate_groups)
    else:
        raise ValueError(a.gate_structure)
    p += d * d  # out_proj
    p += a.kernel_size * d  # depthwise conv, no bias
    return p


def attn_params(a: Arm) -> int:
    """q,k,v,o projections + per-head q/k RMSNorm.  Note H = n_heads*head_dim is
    INDEPENDENT of d_model, so attention is LINEAR in d, not quadratic."""
    d = a.d_model
    H = N_HEADS * HEAD_DIM
    Hkv = a.n_kv_heads * HEAD_DIM
    return d * H + 2 * (d * Hkv) + H * d + 2 * HEAD_DIM


def mlp_params(a: Arm) -> int:
    return 3 * a.d_model * a.swiglu_width


def body_params(a: Arm) -> int:
    """Everything except the vocab x d_model matrix."""
    d = a.d_model
    return (
        a.n_attn * attn_params(a)
        + a.n_liv * mixer_params(a)
        + N_LAYERS * (mlp_params(a) + 2 * d)  # two RMSNorms per block
        + d  # final RMSNorm
    )


def total_params(a: Arm, vocab: int, tied: bool = True) -> int:
    return body_params(a) + (1 if tied else 2) * vocab * a.d_model


SC_MULT_NEW = 6  # worktree short_conv.py (fwd+bwd, matches Attention)
SC_MULT_OLD = 2  # older short_conv.py still installed on FarmShare


def flops_per_token(a: Arm, vocab: int, seq_len: int, sc_mult: int = SC_MULT_NEW) -> int:
    d = a.d_model
    H = N_HEADS * HEAD_DIM
    f = 0
    f += a.n_attn * (6 * attn_params(a) + 12 * H * seq_len)
    sc_lin = mixer_params(a) - a.kernel_size * d  # in_proj + out_proj weights
    f += a.n_liv * (sc_mult * sc_lin + sc_mult * a.kernel_size * d + sc_mult * d)
    f += N_LAYERS * 6 * mlp_params(a)
    f += 6 * (vocab * d + d)  # lm_head: tied matrix (not deduped) + final norm
    return f


def attn_score_flops(a: Arm, seq_len: int, fwd_only_causal: bool = False) -> int:
    """Attention score+AV FLOPs per token.

    olmo-core convention: 12*n_heads*head_dim*T  (fwd+bwd, NO causal 1/2).
    design-doc convention: n_attn*2*2*hq*hd*(T/2)  (fwd only, WITH causal 1/2)
      -> exactly 6x smaller.
    """
    H = N_HEADS * HEAD_DIM
    if fwd_only_causal:
        return a.n_attn * 2 * 2 * H * (seq_len // 2)
    return a.n_attn * 12 * H * seq_len


# ---------------------------------------------------------------- the arms
def base_arms() -> Dict[str, Arm]:
    return {
        a.name: a
        for a in [
            Arm("L0"),
            Arm("A16-P", attention_layers=tuple(range(N_LAYERS)), swiglu_width=4820),
            Arm("F-r128", gate_structure="lowrank", gate_rank=128),
            Arm("F-r256", gate_structure="lowrank", gate_rank=256),
            Arm("G-grouped", gate_structure="grouped", gate_groups=4),
            Arm("N-narrow", d_model=976, swiglu_width=4668),
            Arm("W-k5", kernel_size=5),
            Arm("W-k9", kernel_size=9),
            Arm("W-k15", kernel_size=15),
            Arm("A-fewer3", attention_layers=(5, 10, 14)),
            Arm("Q-mqa", n_kv_heads=1),
        ]
    }


ARM_ORDER = [
    "L0", "A16-P", "F-r128", "F-r256", "G-grouped", "N-narrow",
    "W-k5", "W-k9", "W-k15", "A-fewer3", "Q-mqa",
]


# ---------------------------------------------------------------- solvers
def solve_swiglu_width(a: Arm, target: int, vocab: int, multiple_of: int = 4):
    """Params are exactly linear in the SwiGLU width: slope = 3*d*N_LAYERS."""
    d = a.d_model
    slope = 3 * d * N_LAYERS
    p0 = total_params(replace(a, swiglu_width=0), vocab)
    ideal = (target - p0) / slope
    w = max(multiple_of, int(round(ideal / multiple_of)) * multiple_of)
    return w, total_params(replace(a, swiglu_width=w), vocab)


def solve_swiglu_width_twoprobe(a: Arm, target: int, vocab: int, multiple_of: int = 4):
    """Bit-for-bit replication of liv_arms.solve_swiglu_width (two probes)."""
    w0, w1 = 1024, 8192
    p0 = total_params(replace(a, swiglu_width=w0), vocab)
    p1 = total_params(replace(a, swiglu_width=w1), vocab)
    slope = (p1 - p0) / (w1 - w0)
    ideal = w0 + (target - p0) / slope
    w = max(multiple_of, int(round(ideal / multiple_of)) * multiple_of)
    return w, total_params(replace(a, swiglu_width=w), vocab)


def solve_d_model(a: Arm, target: int, vocab: int, multiple_of: int = 16):
    """Replication of liv_arms.solve_d_model: scan the 16-multiple grid."""
    best = None
    for dm in range(multiple_of * 8, D_MODEL + multiple_of, multiple_of):
        if dm % N_HEADS != 0:
            continue
        got = total_params(replace(a, d_model=dm), vocab)
        if best is None or abs(got - target) < abs(best[1] - target):
            best = (dm, got)
    assert best
    return best


def dP_dd(a: Arm, vocab: int) -> float:
    """Analytic derivative of total params w.r.t. d_model at fixed swiglu width.

    P(d) = V*d + A*d^2 + B*d + C  with
      A = 4*n_liv
      B = V-free linear terms
    """
    d = a.d_model
    A = 4 * a.n_liv
    H = N_HEADS * HEAD_DIM
    Hkv = a.n_kv_heads * HEAD_DIM
    B = (
        vocab
        + a.n_attn * (2 * H + 2 * Hkv)
        + a.n_liv * a.kernel_size
        + N_LAYERS * 3 * a.swiglu_width
        + N_LAYERS * 2
        + 1
    )
    return 2 * A * d + B


def quadratic_coeffs(a: Arm, vocab: int):
    A = 4 * a.n_liv
    H = N_HEADS * HEAD_DIM
    Hkv = a.n_kv_heads * HEAD_DIM
    B = (
        vocab
        + a.n_attn * (2 * H + 2 * Hkv)
        + a.n_liv * a.kernel_size
        + N_LAYERS * 3 * a.swiglu_width
        + N_LAYERS * 2
        + 1
    )
    C = a.n_attn * 2 * HEAD_DIM
    return A, B, C


# ---------------------------------------------------------------- validation
def lfm2_total(d, n_layers, n_liv, n_attn, hq, hkv, hd, ff, vocab, k=3):
    """Generic closed form, for validating against released LFM2 checkpoints."""
    H, Hkv = hq * hd, hkv * hd
    attn = d * H + 2 * d * Hkv + H * d + 2 * hd
    liv = 4 * d * d + k * d
    mlp = 3 * d * ff
    return vocab * d + n_attn * attn + n_liv * liv + n_layers * (mlp + 2 * d) + d


def validate() -> List[Tuple[str, bool, str]]:
    out = []
    arms = base_arms()

    # -- unit test: the brainlift's verified LIV mixer at d=2048, k=3
    a2048 = Arm("probe", d_model=2048)
    got = mixer_params(a2048)
    out.append(("LIV mixer 4d^2+kd at d=2048 == 16,783,360", got == 16_783_360, f"{got:,}"))

    # -- unit test: LFM2-1.2B released checkpoint total (HF actual)
    got = lfm2_total(2048, 16, 10, 6, 32, 8, 64, 8192, 65536)
    out.append(("LFM2-1.2B total == 1,170,340,608 (HF actual)", got == 1_170_340_608, f"{got:,}"))

    # -- unit test: LFM2-2.6B released checkpoint total
    got = lfm2_total(2048, 32, 22, 10, 32, 8, 64, 8192, 65536)
    out.append(("LFM2-2.6B total == 2,569,272,320 (HF actual)", got == 2_569_272_320, f"{got:,}"))

    # -- L0 exact
    got = total_params(arms["L0"], 65536)
    out.append((f"L0 == {L0_PARAM_TARGET:,}", got == L0_PARAM_TARGET, f"{got:,}"))

    # -- the six published arm numbers
    published = {
        "L0": 354_483_968,
        "A16-P": 354_388_992,
        "F-r128": 338_755_328,
        "G-grouped": 338_755_328,
        "N-narrow": 338_804_528,
        "A-fewer3": 357_638_528,
        "Q-mqa": 348_978_944,
    }
    for name, want in published.items():
        got = total_params(arms[name], 65536)
        out.append((f"{name} == {want:,}", got == want, f"{got:,}"))

    # -- solver reproduction at 65536
    w, _ = solve_swiglu_width(arms["A16-P"], L0_PARAM_TARGET, 65536)
    out.append(("solve_swiglu_width(A16-P) == 4820", w == 4820, str(w)))
    w2, _ = solve_swiglu_width_twoprobe(arms["A16-P"], L0_PARAM_TARGET, 65536)
    out.append(("two-probe solver agrees", w2 == w, str(w2)))
    dm, _ = solve_d_model(arms["N-narrow"], total_params(arms["F-r128"], 65536), 65536)
    out.append(("solve_d_model(N-narrow) == 976", dm == 976, str(dm)))

    # -- FLOP ratio reproduction (old 2x ShortConv convention -> HANDOFF table)
    l0, a16 = arms["L0"], arms["A16-P"]
    r4 = flops_per_token(a16, 65536, 4096, SC_MULT_OLD) / flops_per_token(l0, 65536, 4096, SC_MULT_OLD)
    r32 = flops_per_token(a16, 65536, 32768, SC_MULT_OLD) / flops_per_token(l0, 65536, 32768, SC_MULT_OLD)
    out.append(("HANDOFF 1.297x@4K reproduced with SC_MULT=2", abs(r4 - 1.297) < 5e-4, f"{r4:.4f}"))
    out.append(("HANDOFF 1.959x@32K reproduced with SC_MULT=2", abs(r32 - 1.959) < 5e-4, f"{r32:.4f}"))
    n4 = flops_per_token(a16, 65536, 4096, SC_MULT_NEW) / flops_per_token(l0, 65536, 4096, SC_MULT_NEW)
    n32 = flops_per_token(a16, 65536, 32768, SC_MULT_NEW) / flops_per_token(l0, 65536, 32768, SC_MULT_NEW)
    out.append(("test-docstring 1.207x@4K reproduced with SC_MULT=6", abs(n4 - 1.207) < 5e-4, f"{n4:.4f}"))
    out.append(("test-docstring 1.886x@32K reproduced with SC_MULT=6", abs(n32 - 1.886) < 5e-4, f"{n32:.4f}"))

    # -- decode traffic reproduction
    W = total_params(arms["L0"], 65536) * 2
    kv = 6 * 2 * 8 * 64 * 2
    out.append(("weight read 708.9 MB reproduced (tied counted ONCE)", abs(W / 1e6 - 708.9) < 0.1, f"{W/1e6:.3f} MB"))
    out.append(("KV share @4K == 6.6%", abs(100 * kv * 4096 / (W + kv * 4096) - 6.6) < 0.05,
                f"{100*kv*4096/(W+kv*4096):.3f}%"))
    out.append(("KV share @32K == 36.2%", abs(100 * kv * 32768 / (W + kv * 32768) - 36.2) < 0.05,
                f"{100*kv*32768/(W+kv*32768):.3f}%"))
    out.append(("crossover T == 57,690 (only if 708.9 rounded first)",
                abs(708.9e6 / kv - 57690) < 1, f"exact={W/kv:,.1f}  rounded={708.9e6/kv:,.1f}"))
    return out


# ---------------------------------------------------------------- reporting
def sep(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


def resolved_arms(vocab: int) -> Dict[str, Arm]:
    """Re-solve the two derived arms against the new vocab."""
    arms = base_arms()
    l0_target = total_params(arms["L0"], vocab)
    w, _ = solve_swiglu_width(arms["A16-P"], l0_target, vocab)
    arms["A16-P*"] = replace(arms["A16-P"], name="A16-P*", swiglu_width=w)
    f_target = total_params(arms["F-r128"], vocab)
    stage1 = replace(arms["N-narrow"], swiglu_width=SWIGLU_WIDTH)
    dm, _ = solve_d_model(stage1, f_target, vocab)
    w2, _ = solve_swiglu_width(replace(stage1, d_model=dm), f_target, vocab)
    arms["N-narrow*"] = replace(arms["N-narrow"], name="N-narrow*", d_model=dm, swiglu_width=w2)
    return arms


def main():
    sep("SECTION 0 -- VALIDATION")
    ok = True
    for label, passed, detail in validate():
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label:<62} got {detail}")
    print(f"\n  overall: {'ALL PASS' if ok else 'FAILURES PRESENT'}")

    sep("SECTION 1 -- PARAM TABLE PER VOCAB (frozen literals, no re-solve)")
    arms = base_arms()
    hdr = f"{'arm':<11}{'body':>13}"
    for lab, v in VOCABS:
        hdr += f"{lab.split()[0]:>13}"
    print(hdr)
    print("-" * len(hdr))
    for n in ARM_ORDER:
        a = arms[n]
        row = f"{n:<11}{body_params(a):>13,}"
        for _, v in VOCABS:
            row += f"{total_params(a, v):>13,}"
        print(row)

    print("\n  ratio vs L0 at each vocab (frozen literals):")
    hdr = f"{'arm':<11}"
    for lab, v in VOCABS:
        hdr += f"{lab.split()[0]:>13}"
    print("  " + hdr)
    for n in ARM_ORDER:
        a = arms[n]
        row = f"{n:<11}"
        for _, v in VOCABS:
            row += f"{total_params(a, v)/total_params(arms['L0'], v):>13.5f}"
        print("  " + row)

    print("\n  embedding share of total (%):")
    for lab, v in VOCABS:
        a = arms["L0"]
        print(f"    V={lab:<16} emb={v*a.d_model:>12,}  total={total_params(a,v):>13,}  "
              f"share={100*v*a.d_model/total_params(a,v):>6.2f}%   untied total="
              f"{total_params(a,v,tied=False):>13,}")

    sep("SECTION 2 -- RE-SOLVED DERIVED ARMS")
    for lab, v in VOCABS:
        arms = base_arms()
        l0t = total_params(arms["L0"], v)
        ft = total_params(arms["F-r128"], v)
        print(f"\n  V = {lab}")
        print(f"    L0 target                     {l0t:,}")
        print(f"    F-r128 target                 {ft:,}")

        # frozen literals
        a16_frozen = total_params(arms["A16-P"], v)
        nn_frozen = total_params(arms["N-narrow"], v)
        print(f"    A16-P  frozen w=4820          {a16_frozen:,}   resid "
              f"{a16_frozen-l0t:+,} ({100*(a16_frozen-l0t)/l0t:+.4f}%)")
        print(f"    N-narrow frozen d=976 w=4668  {nn_frozen:,}   resid "
              f"{nn_frozen-ft:+,} ({100*(nn_frozen-ft)/ft:+.4f}%)")

        # re-solved
        w, p = solve_swiglu_width(arms["A16-P"], l0t, v)
        print(f"    A16-P  RESOLVED w={w:<5}        {p:,}   resid {p-l0t:+,} "
              f"({100*(p-l0t)/l0t:+.4f}%)")

        stage1 = replace(arms["N-narrow"], swiglu_width=SWIGLU_WIDTH)
        dm, p1 = solve_d_model(stage1, ft, v)
        print(f"      stage-1 d_model grid solve (w=4608): d={dm}  -> {p1:,}  resid "
              f"{p1-ft:+,} ({100*(p1-ft)/ft:+.4f}%)")
        w2, p2 = solve_swiglu_width(replace(stage1, d_model=dm), ft, v)
        print(f"    N-narrow RESOLVED d={dm} w={w2:<5} {p2:,}   resid {p2-ft:+,} "
              f"({100*(p2-ft)/ft:+.4f}%)")
        # also reproduce liv_arms' own (circular) d solve which keeps w=4668
        dm_c, _ = solve_d_model(arms["N-narrow"], ft, v)
        print(f"      (liv_arms-style d solve holding w=4668: d={dm_c})")

    sep("SECTION 3 -- N-narrow ANALYTIC: dP/dd AND THE d CUT REQUIRED")
    arms = base_arms()
    print("  P(d) = V*d + A*d^2 + B*d + C ;  A = 4*n_liv = 40 ;  "
          "B = V-independent linear terms + V")
    print(f"  {'V':>8} {'A':>5} {'B':>10} {'dP/dd@1024':>12} {'gap to free':>13} "
          f"{'ideal dd':>10} {'ideal d':>9} {'grid d':>8}")
    for _, v in VOCABS:
        A, B, C = quadratic_coeffs(arms["N-narrow"], v)
        # note: N-narrow's stage-1 uses w=4608
        st = replace(arms["N-narrow"], swiglu_width=SWIGLU_WIDTH, d_model=D_MODEL)
        A, B, C = quadratic_coeffs(st, v)
        deriv = dP_dd(st, v)
        gap = total_params(arms["L0"], v) - total_params(arms["F-r128"], v)
        # exact quadratic solve: A d^2 + (B) d + C = target
        target = total_params(arms["F-r128"], v)
        disc = B * B - 4 * A * (C - target)
        d_exact = (-B + disc ** 0.5) / (2 * A)
        dm, _ = solve_d_model(st, target, v)
        print(f"  {v:>8} {A:>5} {B:>10,} {deriv:>12,.0f} {gap:>13,} "
              f"{d_exact-1024:>10.2f} {d_exact:>9.2f} {dm:>8}")
    print("\n  gap-to-free (L0 - F-r128) is IDENTICAL at every vocab -> the vocab term")
    print("  cancels exactly in the difference.  Only the MARGINAL COST of d changes.")

    sep("SECTION 4 -- FLOPS PER TOKEN (SC_MULT=6, current worktree code)")
    for lab, v in VOCABS:
        arms = resolved_arms(v)
        l0 = arms["L0"]
        print(f"\n  V = {lab}")
        hdr = (f"    {'arm':<11}{'flops@4K':>16}{'vs L0':>9}"
               f"{'flops@32K':>16}{'vs L0':>9}")
        print(hdr)
        for n in ARM_ORDER + ["A16-P*", "N-narrow*"]:
            a = arms[n]
            f4 = flops_per_token(a, v, 4096)
            f32 = flops_per_token(a, v, 32768)
            b4 = flops_per_token(l0, v, 4096)
            b32 = flops_per_token(l0, v, 32768)
            print(f"    {n:<11}{f4:>16,}{f4/b4:>8.3f}x{f32:>16,}{f32/b32:>8.3f}x")

    sep("SECTION 4b -- FLOPS PER TOKEN (SC_MULT=2, the OLD convention the HANDOFF table used)")
    for lab, v in VOCABS:
        arms = base_arms()
        l0 = arms["L0"]
        print(f"\n  V = {lab}")
        for n in ARM_ORDER:
            a = arms[n]
            f4 = flops_per_token(a, v, 4096, SC_MULT_OLD)
            f32 = flops_per_token(a, v, 32768, SC_MULT_OLD)
            b4 = flops_per_token(l0, v, 4096, SC_MULT_OLD)
            b32 = flops_per_token(l0, v, 32768, SC_MULT_OLD)
            print(f"    {n:<11}{f4:>16,}{f4/b4:>8.3f}x{f32:>16,}{f32/b32:>8.3f}x")

    sep("SECTION 5 -- ATTENTION-SCORE SHARE OF 6ND (design doc s4 table)")
    print("  design-doc convention: numerator = n_attn*2*2*hq*hd*(T/2) (FWD ONLY, causal),")
    print("  denominator = 6*N*1 with N = total params.  Mixed conventions -- see notes.")
    for lab, v in VOCABS:
        arms = base_arms()
        N = total_params(arms["L0"], v)
        print(f"\n    V = {lab}   N = {N:,}   6N = {6*N:,}")
        for T in (4096, 16384, 32768):
            l0s = attn_score_flops(arms["L0"], T, fwd_only_causal=True)
            a16s = attn_score_flops(arms["A16-P"], T, fwd_only_causal=True)
            print(f"      T={T:>6,}  L0 {100*l0s/(6*N):>6.2f}%   A16-P {100*a16s/(6*N):>6.2f}%"
                  f"   diff {100*(a16s-l0s)/(6*N):>6.2f}%")
        print("      [consistent fwd+bwd, no causal 1/2 -- olmo-core's own convention]")
        for T in (4096, 16384, 32768):
            l0s = attn_score_flops(arms["L0"], T)
            a16s = attn_score_flops(arms["A16-P"], T)
            print(f"      T={T:>6,}  L0 {100*l0s/(6*N):>6.2f}%   A16-P {100*a16s/(6*N):>6.2f}%"
                  f"   diff {100*(a16s-l0s)/(6*N):>6.2f}%")

    sep("SECTION 6 -- PARAMETER SHARES ('the mixer is X% of the model')")
    for lab, v in VOCABS:
        a = base_arms()["L0"]
        tot = total_params(a, v)
        emb = v * a.d_model
        liv = a.n_liv * mixer_params(a)
        att = a.n_attn * attn_params(a)
        mlp = N_LAYERS * mlp_params(a)
        nrm = N_LAYERS * 2 * a.d_model + a.d_model
        print(f"\n  V = {lab}   total {tot:,}   body {body_params(a):,}")
        for nm, x in (("embeddings", emb), ("10 LIV mixers", liv), ("6 GQA mixers", att),
                      ("16 MLPs", mlp), ("norms", nrm)):
            print(f"      {nm:<16}{x:>13,}   {100*x/tot:>6.2f}% of total   "
                  f"{100*x/body_params(a) if nm!='embeddings' else float('nan'):>6.2f}% of body")
        print(f"      depthwise kernel {a.kernel_size*a.d_model:>8,} = "
              f"{100*a.kernel_size*a.d_model/mixer_params(a):.4f}% of one LIV mixer, "
              f"{100*a.n_liv*a.kernel_size*a.d_model/tot:.5f}% of total")

    print("\n  -- the same shares at the 1.2B geometry (d=2048), for the doc's s5.1 block --")
    tot12 = lfm2_total(2048, 16, 10, 6, 32, 8, 64, 8192, 65536)
    emb12 = 65536 * 2048
    liv12 = 10 * (4 * 2048 ** 2 + 3 * 2048)
    att12 = 6 * (2048 * 2048 + 2 * 2048 * 512 + 2048 * 2048 + 128)
    mlp12 = 16 * 3 * 2048 * 8192
    for nm, x in (("embeddings", emb12), ("10 LIV mixers", liv12),
                  ("6 GQA mixers", att12), ("16 MLPs", mlp12)):
        print(f"      {nm:<16}{x:>14,}   {100*x/tot12:>6.2f}%")

    sep("SECTION 7 -- DECODE TRAFFIC (bf16, 2 bytes/weight)")
    print("  weight read/token = DEDUPED total params * 2   (the tied V x d matrix is read")
    print("  exactly once, for the LM head; the embedding lookup is a single d-vector row).")
    for lab, v in VOCABS:
        arms = base_arms()
        for nm in ("L0", "A16-P"):
            a = arms[nm]
            W = total_params(a, v) * 2
            kv = a.n_attn * 2 * a.n_kv_heads * HEAD_DIM * 2
            print(f"\n    V={lab}  arm={nm}")
            print(f"      weight read       {W/1e6:>10.2f} MB   ({W:,} B)")
            print(f"      KV bytes/token    {kv:>10,} B = {kv/1024:.2f} KiB")
            print(f"      crossover T       {W/kv:>10,.0f} tokens")
            for T in (4096, 32768):
                print(f"      KV share @{T:>6,}  {100*kv*T/(W+kv*T):>9.2f}%")
    print("\n    sanity: the 'read the tied matrix twice' variant would give:")
    for lab, v in VOCABS:
        a = base_arms()["L0"]
        W2 = total_params(a, v, tied=False) * 2
        print(f"      V={lab}  {W2/1e6:>10.2f} MB  (NOT what the doc used)")

    sep("SECTION 8 -- VOCAB-INVARIANCE OF ARM DIFFERENCES")
    print("  For any two arms with the SAME d_model, total_params(arm, V) = body(arm) + V*d,")
    print("  so the difference cancels V exactly.  Verified numerically:")
    arms = base_arms()
    pairs = [("L0", "F-r128"), ("L0", "F-r256"), ("L0", "G-grouped"), ("L0", "W-k5"),
             ("L0", "W-k9"), ("L0", "W-k15"), ("L0", "A-fewer3"), ("L0", "Q-mqa"),
             ("F-r128", "G-grouped"), ("L0", "A16-P"), ("F-r128", "N-narrow")]
    hdr = f"    {'pair':<24}"
    for lab, v in VOCABS:
        hdr += f"{lab.split()[0]:>14}"
    print(hdr)
    for x, y in pairs:
        row = f"    {x+' - '+y:<24}"
        for _, v in VOCABS:
            row += f"{total_params(arms[x],v)-total_params(arms[y],v):>14,}"
        same = len({total_params(arms[x], v) - total_params(arms[y], v) for _, v in VOCABS}) == 1
        print(row + ("   INVARIANT" if same else "   *** VARIES ***"))


if __name__ == "__main__":
    main()


# =========================================================================================
# SECTION 9-12 -- N-narrow honesty, doc-percentage audit, decode crossovers, misc.
# Run with:  python ledger_vocab.py --extra
# =========================================================================================
def extra():
    sep("SECTION 9 -- IS N-narrow A WEAKER OR STRONGER CONTROL AT LARGE VOCAB?")
    print("  'Honest' test: at MATCHED TOTAL params, how much extra NON-EMBEDDING ('body')")
    print("  capacity does N-narrow carry vs F-r128?  Body is the capacity that computes;")
    print("  embedding width is inert lexical capacity.")
    arms0 = base_arms()
    print("")
    print(f"  {'V':>8} {'N-narrow (d,w)':>18} {'total':>13} {'body':>13} "
          f"{'F-r128 body':>13} {'body excess':>12} {'%':>9}")
    for _, v in VOCABS:
        a = resolved_arms(v)["N-narrow*"]
        f = arms0["F-r128"]
        tn, bn, bf = total_params(a, v), body_params(a), body_params(f)
        lbl = "d=" + str(a.d_model) + " w=" + str(a.swiglu_width)
        print(f"  {v:>8} {lbl:>18} {tn:>13,} {bn:>13,} {bf:>13,} "
              f"{bn-bf:>+12,} {100*(bn-bf)/bf:>+8.3f}%")
    print("")
    print("  Where does N-narrow's sacrifice come from? (stage-1: d 1024->976 at w=4608)")
    print(f"  {'V':>8} {'freed total':>13} {'from emb':>13} {'emb share':>10} "
          f"{'from body':>13} {'dP/dd@1024':>11} {'ideal dd':>9} {'ideal d':>9}")
    for _, v in VOCABS:
        st1024 = replace(arms0["N-narrow"], d_model=D_MODEL, swiglu_width=SWIGLU_WIDTH)
        st976 = replace(st1024, d_model=976)
        freed = total_params(st1024, v) - total_params(st976, v)
        emb_part = v * (1024 - 976)
        A, B, C = quadratic_coeffs(st1024, v)
        target = total_params(arms0["F-r128"], v)
        disc = B * B - 4 * A * (C - target)
        d_exact = (-B + disc ** 0.5) / (2 * A)
        print(f"  {v:>8} {freed:>13,} {emb_part:>13,} {100*emb_part/freed:>9.2f}% "
              f"{freed-emb_part:>13,} {dP_dd(st1024, v):>11,.0f} "
              f"{1024-d_exact:>9.2f} {d_exact:>9.2f}")

    sep("SECTION 10 -- DOC PERCENTAGE AUDIT")
    a = base_arms()["L0"]
    for _, v in (VOCABS[0], VOCABS[3]):
        tot = total_params(a, v)
        print("")
        print(f"  V={v}:")
        print(f"    embeddings share of total    {100*v*a.d_model/tot:>7.2f}%")
        print(f"    10 LIV mixers of total       {100*a.n_liv*mixer_params(a)/tot:>7.2f}%")
        print(f"    10 LIV mixers of BODY        "
              f"{100*a.n_liv*mixer_params(a)/body_params(a):>7.2f}%")
        print(f"    16 MLPs of total             {100*N_LAYERS*mlp_params(a)/tot:>7.2f}%")
        print(f"    6 GQA mixers of total        {100*a.n_attn*attn_params(a)/tot:>7.2f}%")
    print("")
    print("  doc s3.3 claim: 'a 32k vocab would cut embeddings to 10.2%'")
    for vv in (32000, 32768):
        t = body_params(a) + vv * a.d_model
        print(f"    V={vv}: {100*vv*a.d_model/t:>6.2f}%  (total {t:,})")
    print("")
    print("  doc s3.3 claim: 'at d=768 the same vocab would be ~34% of the model'")
    for vv in (65536, 50257):
        b = replace(a, d_model=768)
        t = total_params(b, vv)
        print(f"    d=768 V={vv}: emb share {100*vv*768/t:>6.2f}%  (total {t:,}, "
              f"body {body_params(b):,})  [swiglu held at 4608]")
    print("")
    print("  P1 ceiling: whole-model weight cut from r=128 gates, L0 -> F-r128")
    for _, v in VOCABS:
        z = base_arms()
        cut = total_params(z["L0"], v) - total_params(z["F-r128"], v)
        print(f"    V={v:<7} cut {cut:,} = {100*cut/total_params(z['L0'], v):.3f}% of "
              f"total, {100*cut/body_params(z['L0']):.3f}% of body")

    sep("SECTION 11 -- DECODE TRAFFIC: TOPOLOGY WIN AND ITS CROSSOVERS")
    for _, v in VOCABS:
        z = base_arms()
        WL = total_params(z["L0"], v) * 2
        WA = total_params(z["A16-P"], v) * 2
        kvL, kvA = 6 * 2 * 8 * 64 * 2, 16 * 2 * 8 * 64 * 2
        T10 = 0.10 * WA / ((kvA - kvL) - 0.10 * kvA)
        print("")
        print(f"  V={v}   W(L0)={WL/1e6:.2f} MB   W(A16-P)={WA/1e6:.2f} MB")
        print(f"    10% end-to-end decode-traffic win at T = {T10:,.0f}")
        print(f"    {'T':>8} {'allGQA MB':>11} {'hybrid MB':>11} {'saving MB':>10} {'%':>7}")
        for T in (2048, 4096, 8192, 16384, 32768):
            ag = (WA + kvA * T) / 1e6
            hy = (WL + kvL * T) / 1e6
            print(f"    {T:>8,} {ag:>11.1f} {hy:>11.1f} {ag-hy:>10.1f} "
                  f"{100*(ag-hy)/ag:>6.1f}%")

    sep("SECTION 12 -- MISC CHECKS")
    for vv in (50257, 50304, 50432, 65536):
        print(f"  V={vv}:  mod64={vv%64:<4} mod128={vv%128:<4} mod256={vv%256:<4} "
              f"mod512={vv%512}")
    print(f"  50257 up to multiple of  64 -> {-(-50257//64)*64}")
    print(f"  50257 up to multiple of 128 -> {-(-50257//128)*128}")
    print(f"  50257 up to multiple of 256 -> {-(-50257//256)*256}")
    a = base_arms()["L0"]
    print("")
    print(f"  norm derivation: L0 body must be {body_params(a):,}")
    d = a.d_model
    for nb in (1, 2, 3):
        for fin in (0, 1):
            t = (a.n_attn * attn_params(a) + a.n_liv * mixer_params(a)
                 + N_LAYERS * (mlp_params(a) + nb * d) + fin * d + 65536 * d)
            flag = "  <== L0 TARGET" if t == L0_PARAM_TARGET else ""
            print(f"    {nb} norm(s)/block + {fin} final norm -> {t:,}{flag}")


if __name__ == "__main__":
    import sys
    if "--extra" in sys.argv:
        extra()
