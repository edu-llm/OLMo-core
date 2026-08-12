# Measurement rules

Short, because each rule exists for one specific failure that already happened.
Full derivation and citations:
`../../memory-split/docs/2026-08-12-n-hop-design-and-evidence-audit.md`.

## Ordering

**The calibration must be cheaper than the grid, and it must run first.** The
previous line implemented a bracketing gate that correctly refused its primary
endpoint -- after 32 cells had trained, because a module constant made the depth
sweep inexpressible. `scripts/calibrate_nhop.py` trains nothing and exits 1 on an
unusable endpoint. Gate the pipeline on it.

Seven endpoints in that programme failed on instrumentation rather than science.
Assume the eighth will too, and check first.

## Every accuracy carries four numbers

`n`, `chance`, `best_constant`, and `unparseable_rate`. Not chance alone --
the best-constant floor is what a claim has to clear, and the two differ a lot: on
the n-hop endpoint chance is 0.28% while the floor is 0.7-0.9%, and on a 40%-yes
question family a constant "no" scores 60%, so two arms at 59.5% and 40.5% are the
two constant policies rather than a 19-point effect.

If `z_vs_chance` is strongly negative on a balanced task, the endpoint is broken,
not the model. A deduction scorer once read 0.369 on a balanced 750/750 set --
**10.1 SE below chance**, which no model policy can produce.

## One scoring mode per table

State it. The previous headline two-hop table mixed exact-match-after-`Answer:`
for two rows with substring-anywhere-in-the-continuation for the third, on
different sample sizes. Since a trace restates its value several times, the
lenient mode credits a correct trace with a wrong final answer.

## Depth

Overlay the *pⁿ* curve on every depth plot. A per-hop reliability difference
produces a gap that grows with depth arithmetically -- 13.3pp at depth 1 to 34.7pp
at depth 5 for p=0.93 vs 0.999. The previous two-hop result *is* that curve
(7.81pp predicted, 7.4pp observed). The reasoning quantity is **conditional
per-hop accuracy**: accuracy at hop *k* given hops 1..*k*-1 correct.

Hold the surface form fixed across depths and vary it within each depth. A 3-hop
probe scoring 0.00% is not evidence about depth if the 3-hop phrasing was never
trained.

## Arms

One token stream, four loss-weight sidecars. Never render arms separately: the
streams then differ in length, the arms see different exposure counts (measured:
0.497x for biographies), and iso-token and iso-exposure become mutually
exclusive.

Loss is `sum / fixed_divisor` with the divisor identical across arms. `mean` over
surviving targets inflates each remaining token by `1/(1-f)` -- 1.331 at a 24.89%
mask rate -- and makes every arm contrast invalid.

Ship **both** equal-mass controls. Contiguous matches span structure and misses
difficulty by 23-35%; scattered matches difficulty to 5-10% and gives up
contiguity. They have orthogonal confounds, so bracket the effect between them.

## Divergence

JSD primary: bounded in [0, ln 2], so a saturated value is interpretable. Report
both KL directions with the direction named. Report **distributions over
positions**, not means -- drift concentrates in a small fraction of positions that
a mean hides.

Before claiming H2, measure the **seed floor**: KL between two same-arm runs
differing only in seed is ~0.1 bits/byte, and the old "<= 0.08 nats outside fact
positions" plausibly sits under it. `metrics.seed_floor_report` makes the
comparison explicit.

## Compute

`C = 6ND + 12 * n_layer * d_model * ctx` per token. Bare `6ND` is off by 13-23%
at d160m (23-31% under Kaplan's head-excluded *N*; state which convention).

Report tokens-, steps- **and** FLOPs-to-threshold. A claim surviving on only one
axis is not a compute claim. And refuse unbracketed crossings: if the first
evaluated point is already above threshold the ratio is a censored lower bound,
which is what "10-15x fewer tokens" was.

Then report inference cost honestly. Training and prefill run at 40-60% MFU;
sequential generation at ~1%. The split arm's overhead is generated tokens and its
saving is training compute, so those FLOPs are not interchangeable.

## Statistics

Minimum defensible design, cheapest first:

1. **Free**: relative std of the last ~30 checkpoints of each existing run
   (R² = 0.82-0.95 against true seed noise). Every run already carries this.
2. **Free**: published 160M init-only and data-order-only variants for a seed-SD
   estimate at this scale.
3. **3 runs**: three seeds of the baseline arm only; score single-seed arms as
   `z = delta / sd_baseline`.
4. **3 per arm**: paired bootstrap over seeds *and* examples.

Randomise init **and** data order in every seed; init alone converges to the
equivalent of two ideal runs. Pre-register the MDE
(`calibration.required_n_for_mde`) and never compute post-hoc power -- at n=750
against the best available anchor, power was 28-29%, so a null there is
uninterpretable.
