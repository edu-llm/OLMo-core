# Impl 3×5 — results

James's Impl-3 per-token loss weighting on Impl 5's self-distilled targets. Four arms, all
trained 2026-08-04 in one job, all against **D4** as the baseline.

**Two of three axes are in. The forgetting axis has not run yet, so there is no answer yet to
whether the combination bought anything.** What follows is what is measured, and nothing else.

| run | what |
|---|---|
| `run_019fcaae-3e79-70f8-908a-952c04a4d459` | training + ped_nll, all 4 arms, commit `c2aef0cc` |
| `run_019fca79-…` | gate; everything passed, then died on a missing `aws` CLI in the last line |
| `run_019fcd96-…` | math axis on `gpu-4xl40s` — **cancelled**, no `g6e.12xlarge` capacity in any AZ |

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

- **The forgetting axis.** Payload ready at `gpu-4xa10g`; `gpu-4xl40s` was abandoned after
  `InsufficientInstanceCapacity` in all four AZs it can use. Note `gpu-1xl40s` is **not** a
  fallback — same g6e family, same four AZs. The g5/g6/g4dn families get five, including
  `us-east-1f`.
- **The 2×2.** See the hint above.
- **A κ-validated judge.** Still one judge family; `tutor-eval-suite` remains the missing piece.

## Honest expectation

`bT451` landed byte-identically on D4 and the three conditions are pedagogically
indistinguishable from it. The forgetting axis may well show little movement too. If so the
result is still clean and worth reporting: **on targets already pulled toward the base model,
James's reweighting has little left to remove** — and the ESS numbers explain the mechanism.
That is a real finding about *when* the two methods compose, not a failed experiment.
