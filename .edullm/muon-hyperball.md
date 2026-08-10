# MuonH (Hyperball) vs MuonW on a 370M-active MoE

Two arms: `olmo2_370M_moe × {muon_w, muon_h}`.

The dense specs (`run-dense-*.yaml`, `run-smoke-dense-*.yaml`) are committed and unused. They
are kept rather than deleted because `olmo2_370M` is the config the MoE one is parameter-matched
*against*, so the dense pair is the obvious follow-up if the MoE result needs a dense
counterpart to be interpretable. Nothing in the current experiment reads them.

## What is being tested

Hyperball ([arXiv 2606.16899](https://arxiv.org/abs/2606.16899)) fixes each constrained weight
matrix's Frobenius norm at `R = ||W_0||_F` and normalizes the update to unit Frobenius norm, so
one step moves exactly `η_t · R` before a radial projection back onto the sphere. Constrained
matrices take **no weight decay** — the constraint replaces it. MuonH is that wrapper with
Muon's `msign(M_t)` as the base update.

`src/olmo_core/optim/hyperball.py` implements both arms as one optimizer with a `constraint`
switch, so they share the identical momentum and Newton–Schulz path and the comparison isolates
the wrapper. The dion-backed `MuonConfig` is untouched.

## The three things that would invalidate this comparison

1. **The run must finish its learning-rate decay.** The paper's own finding is that Hyperball
   "starts slightly worse but overtakes WD as the learning rate decays". A run truncated before
   the cosine schedule completes is not a shorter version of this experiment — it is an
   experiment biased toward MuonW. Every arm gets a schedule that completes inside its step
   budget; no arm is stopped early and compared to one that was not.

2. **`--learning-rate` is not the same quantity across arms.** For `muon_w` it is scaled per
   matrix by `adjust_lr` (Moonlight's `0.2·√max(d_in,d_out)`). For `muon_h` it is a *relative*
   step size — dimensionless, `||ΔW||/||W|| ≈ η`. A single LR shared between the arms compares
   nothing, so each arm carries its own.

   **Neither value is tuned, and this is the weakest part of the experiment as it stands.**
   `muon_w=0.02` / `muon_h=0.01` are starting points, not optima. The paper sweeps per scale on a
   √2 grid and reports the *best* LR per arm, because that is the only comparison that is about
   the optimizer rather than about two arbitrary points. A single pair of untuned rates can favour
   either arm, so read a first result as "does this run at all and does the constraint hold",
   and treat a loss difference as provisional until each arm has been swept.

3. **`--init-method fan_in` on every arm.** `R` is measured from `W_0`, so the initializer sets
   the absolute step length. The paper uses `std = 1/√d_in`, which is `fan_in` here;
   `llama_like` and so every `olmo2_*` factory default to `normal` (std 0.02). Both arms must
   use the same one, and `fan_in` is the one the method was designed around.

## Why the MoE arm needed library work

OLMo-core stores expert weights with the expert dimension folded into rows — `w1` is
`(num_experts · d_model, hidden_size)`. That is 32 independent matrices in one 2D tensor, so
orthogonalizing it whole mixes experts and computes the radius and `adjust_lr` from the stacked
shape. The `block_rows` param-group option makes `msign`, both Frobenius norms, the radius and
`adjust_lr` run per expert; `default_group_overrides` derives it from the owning module's
`num_experts`, for **both** arms. This is an extension — the paper says nothing about MoE.

Convenient consequence: FSDP shards those tensors on the expert-major dimension, so at any
world size dividing the expert count every block is already rank-local and the step needs no
communication. Dense attention/MLP matrices are all-gathered instead.

## Parameter matching

`olmo2_370M_moe` is matched on **active** parameters, not total:

| | total | non-embedding | active/token |
|---|---|---|---|
| `olmo2_370M` | 474.0M | 371.3M | 474.0M |
| `olmo2_370M_moe` | 1,078.5M | 975.8M | 373.9M |

32 experts × hidden 512, `top_k=4`, no shared MLP. Everything else — `d_model`, depth, heads,
QK-norm, RoPE theta, reordered-norm blocks — is held equal to the dense config.

## What lands in W&B

The platform supplies the project (`EDULLM_WANDB_PROJECT`) and puts the experiment slug in
`WANDB_RUN_GROUP`, so every arm of one `--experiment` groups on its own. The run *name* is the
platform run id, which is what `edullm status` and the lineage record use and which says nothing
about the arm — so the arm goes on as **tags**: `muon_h` / `muon_w`, the model factory,
`init-fan_in`, and `lr-<value>`. Filter or group on those.

Alongside the usual loss and throughput, `MuonMetricsCallback` logs every step:

| metric | reads as |
|---|---|
| `optim/radius_relative_drift_max` | `max\|‖W_b‖_F / R_b − 1\|`. **MuonH only.** Should sit at the fp32 accumulation floor for the whole run. |
| `optim/matrix_norm_{mean,min,max}` | Frobenius norms over constrained blocks. Pinned on MuonH by construction; free to move on MuonW, and where it settles is what Hyperball pins. |

**Check the drift metric before reading any loss curve.** If it climbs, the constraint stopped
holding — a shard boundary splitting an expert, a resume that recovered the wrong radius — and
the arm is no longer testing Hyperball. The run still trains and still reports a loss, and it
will probably lose; that reads as a result about the method and is a result about a bug. Both
are invisible in the loss alone, which is the entire reason this metric is logged.

These are rank-local: a sharded matrix contributes its own slice, so the drift is reduced with
`max` (the worst rank is the one worth seeing) and the norms with `mean`.

## Running it

```bash
for arm in moe-muonh moe-muonw; do
  edullm submit --spec .edullm/run-$arm.yaml --experiment muonh-370m \
    --dataset olmo-150b-dolma2-v1 --team scratch --compute gpu-8xa100 \
    --hours 7 --attempts 1
done
```

Always pass `--hours` and `--attempts`. The profile's defaults are its maximums (24 h × 2), which
price past the automatic threshold and park the run for a lead instead of starting it. Read the
real figures out of `check --json` (`cost`, `approval_class`) — never from this file.

**`check`'s `approval_class` is not the gate.** Measured 2026-08-08: `check` reported
`automatic` for both arms and the submission still parked at `run-approval-lead`. The deferred
checks are why — `image_scan_findings_unreviewed` is decided after `check` has already answered,
and a freshly built image has no reviewed findings. So budget an approval cycle per submission
regardless of what `check` said, and read the real state from `edullm status --json` (`gate`,
`admitted`, `you_can_release`).

The `run-smoke-*.yaml` specs are 120-step single-GPU versions. They measure tokens/s and prove
the path runs; they do **not** rank the optimizers, for the decay reason above. A cheap thing to
run first if the sharded path has changed — every distributed test in
`src/test/optim/hyperball_test.py` is `requires_multi_gpu` and does not execute on a CPU box, so
the first real exercise of the FSDP gather path is a GPU run.

## Result, 2026-08-08 — SUPERSEDED, read "Second grid" first

> **The 0.042-nat MuonH win reported in this section did not survive tuning.** With both arms at
> their own optimal learning rate the gap is **0.0003 nats**, which is 0.15 of the within-window
> noise: MuonH and MuonW are indistinguishable at this scale. See
> "Second grid — tuned, the two optimizers are indistinguishable" below.
>
> This section is kept because everything in it *except the cross-arm comparison* still stands and
> is still the evidence: the constraint held at the fp32 floor, the arms were controlled to an
> identical first loss, and the crossover/erosion shape is real. It is the interpretation of the
> gap as a property of the optimizer that was wrong, and it was wrong for the reason this document
> already named — the control was under-tuned. Left in place rather than rewritten so that the
> correction is legible instead of invisible.

Both arms ran to completion on `gpu-8xa100`, 7630/7630 steps, 4,000,317,440 tokens each.

| | MuonH | MuonW |
|---|---|---|
| run id | `run_019fdfa2-52d2-708d-ace7-5cb8f54f441c` | `run_019fdfa3-6f0a-7070-8f1c-89eb4bfb6905` |
| learning rate | 0.01 (relative) | 0.02 (`adjust_lr`-scaled) |
| CE loss, mean of last 800 steps | **2.8561** | 2.8978 |
| PPL at that loss | **17.39** | 18.13 |
| MFU (actual avg) | 27.82% | 28.04% |
| wall clock | 3 h 16 m 44 s | 3 h 15 m 14 s |
| `optim/matrix_norm_mean`, first → peak → last | 26.947 → 26.947 → 26.947 | 26.946 → **242.6** (step ~1350) → 117.879 |
| `optim/radius_relative_drift_max` | 1.19e-07 | absent, as designed |

**The constraint held, so the loss comparison means something.** Over all 7630 steps the drift
never exceeded **1.788e-07**, and **zero** steps were above the 1e-5 threshold. It does not trend:
it oscillates between 1.192e-07 and 1.788e-07 — exactly 1.0 and 1.5 × 2⁻²³ — for the whole run,
which is the quantization of an fp32 ratio either side of 1 and not a drift at all. The maximum is
attained repeatedly, in the last bucket of steps as much as the first, which is the point. `matrix_norm_mean` is identical to four decimals at the
first and last step. This is the check the section above says to make before reading any loss
curve, and it passes. The MuonW arm has no drift metric at all, which is the correct behaviour
for an arm with no radius rather than a missing log.

The two arms started from an identical loss — 12.002816200256348 on both, to every digit — which
confirms the shared initializer, seed and data order. MFU within 0.2 points and load-balancing
loss within 1% mean neither arm was throughput- or routing-advantaged.

**MuonH wins by 0.042 nats (≈4.1% perplexity).** Averaged over the last 800 steps rather than
read off the final step, because single-step MoE CE loss is noisy by several hundredths.

### The overtake happens, but not where the paper says it does

Decile means of CE loss, and the difference:

| steps | LR (group 0) | MuonH | MuonW | H − W |
|---|---|---|---|---|
| 0–762 | 0.01000 | 5.4274 | 4.8746 | **+0.5528** |
| 763–1525 | 0.00976 | 3.6052 | 3.5637 | +0.0415 |
| 1526–2288 | 0.00905 | 3.4000 | 3.4047 | −0.0047 |
| 2289–3051 | 0.00796 | 3.2848 | 3.3147 | −0.0299 |
| 3052–3814 | 0.00660 | 3.1869 | 3.2358 | −0.0489 |
| 3815–4577 | 0.00513 | 3.1015 | 3.1652 | −0.0637 |
| 4578–5340 | 0.00369 | 3.0199 | 3.0903 | **−0.0704** |
| 5341–6103 | 0.00245 | 2.9494 | 3.0162 | −0.0668 |
| 6104–6866 | 0.00154 | 2.8927 | 2.9469 | −0.0542 |
| 6867–7629 | 0.00106 | 2.8557 | 2.8971 | −0.0414 |

The qualitative shape the paper predicts — starts worse, ends better — reproduces. Two details
of it do not, and both are worth more than the headline number:

1. **"Starts *slightly* worse" understates it.** MuonH is 0.55 nats behind over the first decile
   and 1.21 nats behind at step 50. That is not a small early penalty, it is a different regime;
   the radial projection is discarding most of the progress a large early step would make while
   the norm is still at its initial value.

2. **The crossover is early, and not attributable to decay.** The 100-step mean of H−W turns
   negative for good at **step 1818** — 24% of the way in, with the LR still at 93% of peak.
   The cosine schedule has barely moved. And the advantage *peaks* at −0.0710 around step 5306
   (LR 0.0031) and then **erodes by 42%** across the final decay, to −0.0414. The paper's stated
   mechanism is that Hyperball "overtakes WD as the learning rate decays", which implies the gap
   widens as the LR falls. Here it narrows. Whatever MuonH is winning on, this run does not
   support decay as the cause.

### A hypothesis for the erosion, from the norm trajectory

`optim/matrix_norm_mean` on the MuonW arm is not the monotone climb it would be easy to assume.
It overshoots to **242.6 by step ~1350** — 9.0× MuonH's pinned 26.947, while the LR is still near
peak — and then falls back monotonically for the remaining 82% of the run, ending at 117.9, or
4.4×. Under decoupled weight decay the equilibrium norm is set by the ratio of step size to decay
strength, so it tracks a target that the cosine schedule is dragging downward; the constraint arm
has no such transient because `R` is fixed at `‖W_0‖_F` from step 0 (constant to twelve digits
across the whole run — 26.946675515 to 26.946676034).

So over the second half the two arms' weight geometries are *converging*, and the loss gap narrows
over exactly the same stretch: −0.0704 at the widest, −0.0414 at the end, while the norm ratio goes
9.0× → 4.4×. That is a coherent story — MuonW spends the decay phase approaching the norm scale
MuonH was pinned to from the start, and gets some of MuonH's advantage back as it arrives.

> **REFUTED by the LR sweep — see "The norm-scale hypothesis does not survive this grid".** MuonW's
> best learning rate leaves its norm at 95.34, still 3.5× `R`, and it matches MuonH's loss there.
> The two series moving together here was the correlation it was labelled as, and nothing more.

**It is a hypothesis and not a finding.** Two series moving together on one seed is a correlation,
and the causal test is not in this data: it needs a MuonW arm swept over `weight_decay` so the
equilibrium norm is set deliberately rather than incidentally, and ideally a MuonH arm whose `R` is
scaled away from `‖W_0‖_F`. If it survives that, the interesting claim is not "Hyperball wins" but
"the norm scale is the active variable and Hyperball is one way to set it" — which would also
explain why the crossover lands at step 1818 with the LR barely moved, since by then MuonW is
already far from the norm it started at.

### What this result does not establish

**Neither learning rate is tuned, and that is enough on its own to keep this provisional.**
0.042 nats is a small gap and the two LRs are different quantities picked as starting points, not
optima. The paper sweeps a √2 grid per arm and compares each at its own best; an untuned pair can
favour either side, and a 4% perplexity difference is well inside what one grid step could move.
Read this as "the implementation is correct, the constraint holds, and the method is not worse" —
not as a ranking.

Also outstanding: **one seed per arm**, so the gap has no error bar; **train loss only**, no
held-out evaluation (the arms saw one epoch of a 150B corpus, so this is near-validation, but it
is not validation); and **no dense counterpart** — `run-dense-*.yaml` is committed and unrun, and
the MoE result is not automatically a dense result given that per-expert blocking is the part of
this that needed new library code.

The obvious next experiment is a √2 LR sweep on both arms at this scale, which is what would turn
this into a comparison of optimizers rather than of two points. It is specified below.

## Tuning the learning rate

The sweep that makes the result above a claim about optimizers rather than about two points. Specs:
`run-sweep-{muonh,muonw}-*.yaml`.

**Three points per arm, and only four of the six are new runs.** The finished runs at `muon_h=0.01`
and `muon_w=0.02` are already points on this grid — same corpus, seed, batch, microbatch, dtype,
initializer, steps and warmup — so they are the centres and the sweep only adds the brackets:

| arm | lower (÷√2) | centre | upper (×√2) |
|---|---|---|---|
| `muon_h` | **0.00707107** | 0.01 *(done)* | **0.01414214** |
| `muon_w` | **0.01414214** | 0.02 *(done)* | **0.02828427** |

Reusing the finished runs is only legitimate because every non-LR parameter is identical, which is
checked rather than asserted: each sweep spec's command normalises to its baseline's command exactly
when the swept LR is substituted back. Change any other flag in one of these files and the centre
stops being a grid point, which quietly costs two more runs.

**`muon_h`'s upper bracket and `muon_w`'s lower bracket are both 0.01414214, and this means
nothing.** They are different quantities — MuonW's is scaled per matrix by `adjust_lr`, MuonH's is a
dimensionless relative step size. Neither run can stand in for the other. Called out in both spec
headers because it is exactly the kind of thing that looks like a free saving.

**The horizon is not shortened, and that is the expensive decision.** A cheaper sweep would locate
each optimum at ~1B tokens and confirm at 4B. Two reasons not to. The comparison is defined at a
*completed* cosine decay — the whole finding concerns what happens as the LR decays — so a 1B-token
point cannot be compared against the finished 4B runs without discarding them and re-running the
centres too. And the LR optimum moves with the horizon, so a short sweep optimises the wrong problem:
it hands back a best-LR for 1B tokens and invites exactly the "untuned" objection the sweep exists to
close.

**Read it as a bracket test, not a curve fit.** Three points per arm can only say "the centre beats
both neighbours" or "an edge wins". If an edge wins, the grid has not bracketed that arm's optimum
and it needs one more run beyond that edge before any cross-arm number is quoted — a fifth and
possibly sixth run, and not optional. Only once both arms are bracketed does "MuonH beats MuonW by X
at each arm's own optimum" mean anything.

**`--adamw-learning-rate 8.2e-4` is held fixed at every point.** Only the muon-group LR is swept;
scaling both together sweeps two variables and locates neither.

Cost and approvals: read them from `edullm check --json` per spec, never from this file. Budget one
human approval cycle per submission regardless of what `approval_class` says, for the reason in
"Running it" above.

### First grid, 2026-08-09 — MuonH brackets, MuonW does not

All four bracket runs completed 7630/7630. Mean CE over the last 800 steps, the same statistic the
headline result uses:

| arm | LR | last-800 CE | PPL | |
|---|---|---|---|---|
| `muon_h` | 0.00707107 | 2.8639 | 17.529 | |
| `muon_h` | **0.01** | **2.8561** | **17.393** | **best — centre, bracketed** |
| `muon_h` | 0.01414214 | 2.8704 | 17.644 | |
| `muon_w` | 0.01414214 | **2.8780** | **17.778** | **best — LOWER EDGE, not bracketed** |
| `muon_w` | 0.02 | 2.8978 | 18.134 | |
| `muon_w` | 0.02828427 | 2.9312 | 18.749 | |

**MuonH is bracketed at 0.01.** The centre beats both neighbours, so 0.01 is its optimum to within a
√2 step, and the original run was — by luck — already tuned. The invariant held on all three MuonH
points: drift max 1.788e-07, 1.788e-07 and 2.384e-07, all still integer multiples of 2⁻²³ and all
four orders below the 1e-5 threshold.

**MuonW is not bracketed, and the trend is steeply monotone downward** — 2.9312 → 2.8978 → 2.8780 as
the LR falls, with no sign of turning. Its optimum is at or below 0.01414214 and this grid says
nothing about where. Extension runs at 0.01 and 0.00707107 are submitted together rather than
serially, because the trend makes it likely both are needed and a serial round costs another 3h20m
to learn what two parallel runs learn at once.

**Do not quote a cross-arm number yet.** For the record, comparing each arm at its best *so far*
gives MuonH ahead by 0.0219 nats (2.8561 vs 2.8780) — roughly **half** the 0.042 in the headline
result above, which compared MuonH at its optimum against MuonW at a point now known to be
mistuned. That 0.0219 is an upper bound on MuonH's advantage and will only shrink if MuonW improves
further below 0.01414214. Whether anything survives is exactly what the extension runs decide.

### Second grid — tuned, the two optimizers are indistinguishable

Eight runs now, all 7630/7630. Mean CE over the last 800 steps:

| arm | LR | last-800 CE | PPL | ‖W‖_F last | |
|---|---|---|---|---|---|
| `muon_h` | 0.00707107 | 2.8639 | 17.529 | 26.95 | |
| `muon_h` | **0.01** | **2.8561** | **17.393** | 26.95 | **optimum — interior, bracketed** |
| `muon_h` | 0.01414214 | 2.8704 | 17.644 | 26.95 | |
| `muon_w` | **0.00707107** | **2.8564** | **17.399** | 95.34 | **best — still the LOWER EDGE** |
| `muon_w` | 0.01 | 2.8601 | 17.464 | 102.75 | |
| `muon_w` | 0.01414214 | 2.8780 | 17.778 | 109.85 | |
| `muon_w` | 0.02 | 2.8978 | 18.134 | 117.88 | |
| `muon_w` | 0.02828427 | 2.9312 | 18.749 | 127.06 | |

**At each arm's own optimum the gap is −0.0003 nats — 2.8561 against 2.8564, +0.03% perplexity.**
The within-window standard error of that difference is ±0.0022, so |gap|/sem = 0.15. **MuonH and
MuonW are indistinguishable at this scale.** The honest summary of the whole experiment is that
Hyperball matches decoupled weight decay here; it does not beat it.

The 0.042 nats in the result above was almost entirely an under-tuned control:

| comparison | gap |
|---|---|
| both arms untuned (the original pair) | 0.042 nats |
| MuonH tuned vs MuonW mid-sweep | 0.0219 |
| **both arms at their optima** | **0.0003** |

That is what the "neither LR is tuned" caveat was worth, and it was load-bearing rather than
boilerplate: the effect shrank by two orders of magnitude once the control had a fair learning rate.

**MuonW is still not bracketed, and the rule still applies.** Its optimum has been at the lower edge
twice now. `run-sweep-muonw-0.005.yaml` is the next step down. Note what that run can and cannot do:
it cannot change the conclusion above, because any further MuonW improvement moves MuonW *ahead* —
it can only erase a MuonH advantage, never create one. It runs to say where MuonW's optimum actually
is. The decrements are flattening (0.0334, 0.0198, 0.0179, 0.0037 per √2 step down), so expect
little movement; if 0.005 also wins, "MuonW keeps improving as its LR falls" is itself the finding
and wants a different experiment, not a ninth run of this one.

### The norm-scale hypothesis does not survive this grid

The hypothesis recorded above — that the norm scale is the active variable and Hyperball is one way
to set it — predicts MuonW does best where its equilibrium norm approaches MuonH's pinned
`R = 26.95`. The grid contradicts it. MuonW's final norm does fall monotonically with LR (127.06 →
95.34 across the five points), but its *best* point sits at **95.34, still 3.5× R**, and matches
MuonH's loss there. The two arms reach the same loss at weight scales differing by three and a half.

So the norm scale is **not** what the arms were differing on, and the earlier correlation between a
closing norm ratio and a closing loss gap was the coincidence it was labelled as. Recorded as
refuted rather than deleted, because the refutation is the more useful half: whatever the constraint
is doing at this scale, it is not buying a better weight scale, and it is not buying lower loss.

### A defect this run exposed

`optim/radius_relative_drift_max` reached W&B and nothing else. `ConsoleLoggerCallback` prints an
allowlist and the drift matched none of its patterns, so the single number that decides whether a
MuonH arm is testing Hyperball at all was absent from `edullm logs` — the channel the platform
actually gives you for a running job. Mid-flight, that log showed fifty lines of throughput, load
imbalance and LR, and no drift.

That is the same failure `MuonMetricsCallback` exists to prevent, displaced one layer out: from
the log stream alone, a Hyperball result and a Hyperball bug were still indistinguishable. Fixed
by configuring the console logger explicitly and extending the default pattern list. The fix does
not apply retroactively to the two runs above — their invariant was verified from the dashboard.
