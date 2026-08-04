# Impl 3×5 — results

James's Impl-3 per-token loss weighting on Impl 5's self-distilled targets. Four arms, all
trained 2026-08-04 in one job, all against **D4** as the baseline.

**All three axes are in.** The control is confirmed three independent ways; matched pedagogy
quality is established with CIs; the forgetting axis shows a real KL reduction that does not
convert into a resolvable retention gain. What follows is what is measured, and nothing else.

| run | what |
|---|---|
| `run_019fcaae-3e79-70f8-908a-952c04a4d459` | training + ped_nll, all 4 arms, commit `c2aef0cc` |
| `run_019fca79-…` | gate; everything passed, then died on a missing `aws` CLI in the last line |
| `run_019fcd96-…` | math axis on `gpu-4xl40s` — **cancelled**, no `g6e.12xlarge` capacity in any AZ |
| `run_019fce3c-…` | math axis on `gpu-1xa10g` — **cancelled**, driver bug put 4 arms on 1 GPU, 2 OOM'd |
| `run_019fce5a-…` | math axis, fixed driver (`0b652500`) — **SUCCEEDED**, all 4 arms × 11 checkpoints |

## The control, which is what licenses reading anything else

James's `common/weighting.py` is **not** in `impl3_handoff.tar.gz`, so the objective was
reimplemented from IMPL3_HANDOFF §4.1 against the spec. `bT451` (T→∞) is what prices that
reimplementation, and it passed twice:

| | |
|---|---|
| `ped_nll` @923 | **0.906021** vs D4's **0.9059** — |diff| **0.000121**, 16× inside the ~0.002 tolerance |
| judge generation | **byte-identical to D4 on all 100 problems** (paired contrast `+0.000 [+0.000, +0.000]`) |

The second is stronger than intended. With ESS 0.9999 the multipliers deviate ≤0.4% and greedy
argmax never flipped, so at T→∞ the arm is not merely close to D4, it *is* D4.

## Weighting actually applied (acceptance check W7, real corpus)

No b arm collapsed — the three conditions are genuinely distinct pressures.

| arm | variant | T | **ESS** | m p50 | m p99 / max | m min |
|---|---|---|---|---|---|---|
| `bT1` | b | 1 | **0.633** | 1.00 | 2.02 | 8.6e-31 |
| `bT2` | b | 2 | **0.761** | 1.15 | 1.63 | 1.1e-15 |
| `aT8` | a | 8 | **0.894** | 1.18 | 1.28 | 1.9e-3 |
| `bT451` | b | 451 | **0.9999** | 1.00 | 1.00 | 0.86 |

Signal: variant a median 0.360 / 1.4826·MAD 0.533; variant b 0.151 / 0.216.

**Mechanism worth stating precisely:** multipliers are bounded above (~2.0 at T=1) but have an
enormous lower tail. Low T works almost entirely by **suppressing high-KL tokens**, not by
amplifying low-KL ones — a sharper description than "concentrates weight on tokens the base
finds easy".

Mechanics all clean: 22/22 grid checkpoints per arm, 3692 weighted batches each (923 × 4, so no
batch trained unweighted), mean m = 1.000000 everywhere, `loss_denom=global` throughout.

## New-task fit — ped_nll on held-out gold (base 1.4158)

| step | bT451 | bT2 | bT1 | aT8 |
|---|---|---|---|---|
| 32 | 1.0359 | 1.0677 | 1.0959 | 1.1066 |
| 128 | 0.9615 | 1.0020 | 1.0378 | 1.0320 |
| 512 | 0.9163 | 0.9419 | 0.9727 | 0.9852 |
| **923** | **0.9060** | **0.9279** | **0.9554** | **0.9788** |

Monotone in ESS: more suppression, worse gold fit. This is the *cost* side of the trade and is
not a result on its own.

### A hint, explicitly not a claim

The ped_nll cost of reweighting is systematically lower here than James measured on gold:

| config | cost vs D4 (distilled) | his cost vs SFT (gold) | ratio |
|---|---|---|---|
| b-T1 | +0.0495 | +0.0620 | 0.80 |
| b-T2 | +0.0220 | +0.0390 | 0.57 |
| a-T8 | +0.0729 | +0.1040 | 0.70 |

20–43% cheaper, consistent with the composition hypothesis — the targets are already ~37%
base-model text, so the multipliers have less distance to fight. **But this compares across
corpora *and* pipelines** (his baseline 0.862 was gold-trained/gold-evaluated; ours 0.9059 is
distill-trained/gold-evaluated). Settling it needs his three configs re-run on gold through this
pipeline — 3 runs, completes the 2×2. Until then it is a hint.

## Pedagogy quality — blind judge, n=100, 3 repeats, 0 errors

`gpt-5.6-sol`, +SI, 100 **distinct** problems, blinding verified by the harness.

**Anchors pass**, which is what makes the batch comparable to the published table:

| | measured | published |
|---|---|---|
| `B_raw_SI` | 0.705 [0.660, 0.751] | cell B ≈ 0.70 ✓ |
| `impl4_A1` | 0.887 [0.869, 0.905] | cell D ≈ 0.86 ✓ |

**All three conditions are null against D4:**

| contrast | Δ | 95% CI |
|---|---|---|
| `bT2` − D4 | −0.000 | [−0.013, +0.013] |
| `bT1` − D4 | −0.003 | [−0.020, +0.012] |
| `aT8` − D4 | −0.005 | [−0.023, +0.011] |
| A1 − D4 | +0.018 | [−0.002, +0.040] |

Arms sit at 0.864–0.869 against D4's 0.869, CI half-widths ~±0.02 — tighter than the power table
predicted, because contrasts are paired on problems.

**This null is the enabling condition, not a disappointment.** Impl 5's Definition of Done is
reduced forgetting *at matched pedagogy quality*, and matched pedagogy quality is now established
with CIs. A forgetting difference would therefore be readable as a real result rather than
"forgot less because it taught less" — the ambiguity Impl 5's own BUILD.md predicted and could
not resolve.

Caveats: `Tone` is 0.50 for every arm (everything scored Neutral; dilutes all OVERALLs equally),
and this is one judge family.

**A correction to an earlier plan:** `bT451` was meant to give a within-batch judge noise floor.
It cannot — its outputs are byte-identical to D4's, so the zero measures determinism, not judge
precision. No within-batch noise floor exists; the external reference is ~0.010 seed-to-seed
in the +SI condition.

## Not done

- **A larger math probe.** 250 GSM8K items resolve ~12 points; the retention gaps here are
  0.8–8.4. This is the binding constraint on the Definition of Done, and it is a power problem,
  not a compute problem — another training run would not help.
- **The 2×2.** See the hint above: settling whether reweighting composes with distillation, or
  would have helped on gold anyway, needs his three configs re-run on gold through this pipeline.
- **A κ-validated judge.** Still one judge family; `tutor-eval-suite` remains the missing piece.

## Forgetting — GSM8K retention and KL (run `run_019fce5a-…`, 2026-08-04)

11 checkpoints per arm, `gpu-1xa10g`, arms serialised. **Base measured in this run: 0.676 bare /
0.652 hint** — use these, not the published 0.672/0.672, since generation is hardware-sensitive
and D4's figures came off an L40S while these are A10G.

| @923 | bare | hint | Δbare vs base | Δhint vs base | KL(SI) | deflect |
|---|---|---|---|---|---|---|
| base | 0.676 | 0.652 | — | — | 0.000 | 0.000 |
| `aT8` | **0.684** | 0.620 | **+0.8** | −3.2 | 0.138 | 0.000 |
| `bT2` | 0.624 | 0.600 | −5.2 | −5.2 | 0.150 | 0.000 |
| `bT1` | 0.620 | **0.640** | −5.6 | **−1.2** | **0.134** | 0.000 |
| `bT451` *(control)* | 0.600 | 0.588 | −7.6 | −6.4 | 0.155 | 0.012 |
| D4 *(published)* | 0.612 | 0.572 | — | — | 0.155 | 0.004 / 0.012 |
| A1 *(gold SFT)* | 0.456 | 0.216 | — | — | 0.790 | 0.148 / 0.516 |

### (a) The control reproduces D4. Third confirmation — the question is closed.

`bT451` vs D4: **KL 0.155 vs 0.155**, **hinted deflect 0.012 vs 0.012** — both exact. Bare −1.2,
hint +1.6, inside noise. It also reproduced D4's *KL trajectory shape*, not just its endpoint:
peak **0.220 at step 16** then a monotone decline (0.220 → 0.197 → 0.164 → 0.145 → 0.159 →
0.155), which is the signature Impl 5 documented and which distinguishes D4 from A1's monotone
rise.

Together with `ped_nll` matching to **1.2e-4** and judge generation being **byte-identical**,
the reimplementation of IMPL3_HANDOFF §4.1 is confirmed on all three axes. James's
`common/weighting.py` is not in the handoff bundle, so this was rebuilt from the spec; it is
now verified as faithful.

### (b) All three arms move down-left — but only the KL half is resolvable.

Against the control, measured in the same run on the same hardware:

| | Δbare | Δhint | ΔKL |
|---|---|---|---|
| `aT8` | +8.4 | +3.2 | **−0.017** |
| `bT2` | +2.4 | +1.2 | **−0.005** |
| `bT1` | +2.0 | +5.2 | **−0.021** |

**Direction is consistent: 3/3 arms retain more at lower KL.** But a 250-item GSM8K probe
resolves ~12 points, and we watched `bT2`'s bare swing 0.644 → 0.704 across adjacent checkpoints
at near-zero drift. **None of the retention gaps is individually resolvable.** The KL reductions
are: KL is a continuous mean over 64 contexts, far more precise than an accuracy probe.

So the honest verdict is **split**. The reweighting demonstrably does what it is designed to do —
it reduces drift from base, most in the arm that suppresses hardest (`bT1`, ESS 0.633, KL 0.134).
That drift reduction **did not convert into a measurable retention gain**, because D4 had already
recovered most of the forgetting: gold SFT drops to 0.456/0.216, D4 sits at 0.600/0.588, and
there is only ~7.6 points of bare forgetting left to remove. `aT8` removes essentially all of it
(+0.8 vs base, i.e. no bare forgetting at all) — but 6.8 points below a 12-point resolution
floor cannot be called a result.

**Impl 5's Definition of Done is therefore NOT met with confidence.** Matched pedagogy quality is
established (judge, n=100, CIs), and the arms trend the right way, but the forgetting half is
under-powered at this probe size. Resolving it needs a larger math probe, not another training
run — the power calculation, not the compute, is the binding constraint.

### (c) No refusal confound anywhere, and a mechanism worth naming.

`deflect = 0.000` for all three conditions at **every one of 33 checkpoints**. Bare and hint move
together rather than diverging. This is the opposite of Impl 4's A3 mirage, where a hinted
advantage turned out to be pure later-onset Socratic refusal — none of that is present here, so
the hinted numbers measure math skill.

Notably the **control** does show refusal (0.012 hinted, matching D4) while the reweighted arms
show none. Suppressing high-KL tokens appears to suppress the tutor persona's bleed into
non-tutoring contexts. That is a sharper description of what the weighting does than "stays
closer to base," and it is the clearest mechanistic finding in the run.
