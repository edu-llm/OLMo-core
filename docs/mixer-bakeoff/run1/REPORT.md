# Mixer bake-off -- run 1 analysis

**18 admissible cells** out of 18 found; 18 expected. Control arm: `KDA_BASE`. alpha = 0.05, target power = 0.8.

> **Budget changed after pre-registration.** `seeds.json` froze 1,907 steps; these cells ran 1,144 (599,785,472 tokens/cell, TPP 1.5). Admissibility is held to the realised budget (the modal `steps` across the 18 cell(s) present), not to the plan -- pinning to the plan would have excluded every cell as a short run and reported a confident empty study. **A shorter budget makes the CE magnitudes MORE inflated, not less.**

## 0. Where each arm stands

| arm | n admissible | of expected | excluded | df it contributes | status |
|---|---:|---:|---:|---:|---|
| `KDA_BASE` | **3** | 3 | 0 | 2 | complete |
| `KDA_NOACT` | **3** | 3 | 0 | 2 | complete |
| `KDA_GCONV` | **3** | 3 | 0 | 2 | complete |
| `GDN2` | **3** | 3 | 0 | 2 | complete |
| `KDA_R1` | **3** | 3 | 0 | 2 | complete |
| `KDA_R2` | **3** | 3 | 0 | 2 | complete |

## 1. Admissibility (step 0, before any pooling)

**0 cell(s) excluded.** No cell was excluded.

Saturation-excluded from the sigma pool: 0. Pre-registration s4.2, fail-open: the CE endpoint has no ceiling in this design, so no cell is dropped from the sigma pool on saturation grounds.

## 2. Primary endpoint -- held-out `val_ce` (nats)

| arm | n | mean | sd | per-cell values |
|---|---:|---:|---:|---|
| `KDA_BASE` | 3 | 3.0514 | 0.01075 | 3.0421, 3.0632, 3.0488 |
| `KDA_NOACT` | 3 | 3.0400 | 0.00258 | 3.0392, 3.0380, 3.0429 |
| `KDA_GCONV` | 3 | 3.0451 | 0.01678 | 3.0321, 3.0392, 3.0641 |
| `GDN2` | 3 | 3.0884 | 0.00404 | 3.0838, 3.0910, 3.0905 |
| `KDA_R1` | 3 | 3.0749 | 0.04516 | 3.0542, 3.1267, 3.0439 |
| `KDA_R2` | 3 | 3.0581 | 0.00642 | 3.0567, 3.0651, 3.0524 |

### 2.1 Pooled within-arm sigma -- a pre-registered deliverable in its own right

**sigma_hat = 0.02042 nats** at df = 12 (6 arms contributing).

chi-squared interval at df = 12: sigma_hat x [0.717, 1.651] = [0.01464, 0.03370] nats -- a **factor-2.3 bracket at df = 12**. This narrows the 5.5x guess, it does not close it. **Run 2 must be sized from this number, not from either literature estimate.**

For reference the two prior estimates were 0.0019 (optimistic) and 0.0105 (pessimistic) nats.

### 2.2 Variance homogeneity

- **Levene (median-centred), THE decision test**: W = 0.9506, df = (5, 12), p = 0.4840
- Bartlett (reported, not deciding): chi2 = 16.3667, df = 5, p = 0.0059

Levene does not reject; the pooled analysis stands. At n = 3 per arm Levene and Bartlett have almost no power. Failing to reject is weak evidence of homogeneity, not a demonstration of it.

### 2.3 Pooled-variance one-way ANOVA

F(5, 12) = 2.4893, p = 0.0908. Error df = sum over arms of (n_i - 1) = 12.

### 2.4 Dunnett contrasts vs `KDA_BASE` (two-sided)

Critical value **2.9013** at k = 5, df = 12, rho = 0.500. Method: Gaussian quadrature (Gauss-Legendre) over the equicorrelated max-|t| integral.

Sanity-checked two ways: (1) quadrature refinement 48 vs 96 nodes -> 1.10e-11 (PASS); (2) the k = 1 reduction, where max-|t| is just |t| and the critical value must equal the Student t quantile computed by a different route (inverse regularized incomplete beta) -> 6.84e-14 (PASS).

| arm | n | estimate (nats) | Dunnett 95% CI | half-width | clears MDE? | adj. p |
|---|---:|---:|---|---:|:--:|---:|
| `KDA_NOACT` | 3 | -0.01135 | [-0.05971, +0.03701] | 0.04836 | no | 0.9340 |
| `KDA_GCONV` | 3 | -0.00624 | [-0.05460, +0.04212] | 0.04836 | no | 0.9944 |
| `GDN2` | 3 | +0.03705 | [-0.01131, +0.08542] | 0.04836 | no | 0.1585 |
| `KDA_R1` | 3 | +0.02358 | [-0.02479, +0.07194] | 0.04836 | no | 0.5097 |
| `KDA_R2` | 3 | +0.00669 | [-0.04167, +0.05505] | 0.04836 | no | 0.9924 |

Negative estimate = better than control (lower CE). A contrast is only a detected difference if its CI excludes zero AND it clears the MDE. **Where the CI includes zero, the honest statement is the BOUND: the difference is smaller than the half-width in that row.**

### 2.5 MDE at 80% power -- exact non-central t, dominant tail only

| sigma basis | sigma (nats) | SE(difference) | MDE (nats) |
|---|---:|---:|---:|
| MEASURED (pooled, this run) | 0.02042 | 0.016669 | 0.06360 |
| literature optimistic | 0.00190 | 0.001551 | 0.00592 |
| literature pessimistic | 0.01050 | 0.008573 | 0.03271 |

The normal approximation is 2.2x too optimistic at n = 3 and is not used anywhere here.

### 2.6 The gate contrast -- `KDA_GCONV - KDA_NOACT` (EXPLORATORY)

With `gate_structure="depthwise"`, `gated_conv_activation=None` does NOT mean activation-free -- the depthwise pre-gate is exactly a SiLU with a learnable per-channel slope. So the contrast that isolates the gate is `KDA_GCONV - KDA_NOACT`, never `KDA_GCONV - KDA_BASE`. It is not against the control, so per the pre-registration it is EXPLORATORY and uncorrected.

Estimate +0.00511 nats, uncorrected 95% CI [-0.03121, +0.04143], n = 3 vs 3, uncorrected p = 0.7645.

## 3. Co-primary endpoints -- throughput and peak memory

Headline throughput source: **steady-state, total across devices (from `throughput_tok_s_steady` x18)**.

| arm | n | mean tok/s | sd | ratio to control |
|---|---:|---:|---:|---:|
| `KDA_BASE` | 3 | 419,288 | 4,590 | 1.000x |
| `GDN2` | 3 | 416,894 | 5,691 | 0.994x |
| `KDA_R1` | 3 | 326,513 | 2,363 | 0.779x |
| `KDA_R2` | 3 | 295,638 | 2,538 | 0.705x |
| `KDA_NOACT` | 3 | 418,364 | 5,995 | 0.998x |
| `KDA_GCONV` | 3 | 410,690 | 1,972 | 0.979x |

Per-device steady-state (`tps_device_avg`):

| arm | n | mean tok/s/device | sd | ratio to control |
|---|---:|---:|---:|---:|
| `KDA_BASE` | 3 | 52,410 | 576 | 1.000x |
| `GDN2` | 3 | 52,117 | 716 | 0.994x |
| `KDA_R1` | 3 | 40,818 | 298 | 0.779x |
| `KDA_R2` | 3 | 36,957 | 319 | 0.705x |
| `KDA_NOACT` | 3 | 52,295 | 750 | 0.998x |
| `KDA_GCONV` | 3 | 51,338 | 250 | 0.980x |

### 3.1 Peak memory (GiB, rank 0)

Source: `per_step_running_max` x18.

| arm | n | mean GiB | sd | ratio to control |
|---|---:|---:|---:|---:|
| `KDA_BASE` | 3 | 9.15 | 0.000 | 1.000x |
| `GDN2` | 3 | 9.33 | 0.000 | 1.020x |
| `KDA_R1` | 3 | 9.22 | 0.000 | 1.007x |
| `KDA_R2` | 3 | 9.36 | 0.000 | 1.023x |
| `KDA_NOACT` | 3 | 9.15 | 0.000 | 1.000x |
| `KDA_GCONV` | 3 | 9.43 | 0.000 | 1.031x |

Throughput cross-check: `tps_device_avg` sits below `throughput_tok_s_steady_per_device` on all 18 cell(s) with both, as expected. No compilation leak into upstream's window.

### Median step time (s)

| arm | n | mean | sd | ratio to control |
|---|---:|---:|---:|---:|
| `KDA_BASE` | 3 | 1.2351 | 0.0106 | 1.000x |
| `GDN2` | 3 | 1.2403 | 0.0150 | 1.004x |
| `KDA_R1` | 3 | 1.5855 | 0.0114 | 1.284x |
| `KDA_R2` | 3 | 1.7692 | 0.0173 | 1.432x |
| `KDA_NOACT` | 3 | 1.2358 | 0.0144 | 1.001x |
| `KDA_GCONV` | 3 | 1.2571 | 0.0063 | 1.018x |

### p90 step time (s)

| arm | n | mean | sd | ratio to control |
|---|---:|---:|---:|---:|
| `KDA_BASE` | 3 | 1.2626 | 0.0233 | 1.000x |
| `GDN2` | 3 | 1.2750 | 0.0180 | 1.010x |
| `KDA_R1` | 3 | 1.6243 | 0.0121 | 1.287x |
| `KDA_R2` | 3 | 1.8057 | 0.0178 | 1.430x |
| `KDA_NOACT` | 3 | 1.2713 | 0.0185 | 1.007x |
| `KDA_GCONV` | 3 | 1.2947 | 0.0061 | 1.025x |

## 4. RECOMMENDATION

### Use `KDA_BASE`

**Basis: throughput and memory (CE unresolved): the fastest arm not shown to be worse on CE.**

CE resolved at this n? **NO** (MDE = 0.06360 nats at the measured sigma).

- NO arm's CE difference from the control clears the MDE with a Dunnett-adjusted CI that excludes zero. THE CE RANKING IS NOT RESOLVED AT n = 3. The arms are not ranked on CE below, and the recommendation falls back to throughput and memory. That fallback is a choice forced by the design's power, and it is stated rather than hidden behind an ordering of point estimates.
- KDA_BASE: throughput 419,288 tok/s (1.000x control), peak memory 9.15 GiB (1.000x control).
- KDA_NOACT: throughput 418,364 tok/s (0.998x control), peak memory 9.15 GiB (1.000x control).
- GDN2: throughput 416,894 tok/s (0.994x control), peak memory 9.33 GiB (1.020x control).
- KDA_GCONV: throughput 410,690 tok/s (0.979x control), peak memory 9.43 GiB (1.031x control).
- KDA_R1: throughput 326,513 tok/s (0.779x control), peak memory 9.22 GiB (1.007x control).
- KDA_R2: throughput 295,638 tok/s (0.705x control), peak memory 9.36 GiB (1.023x control).

### Caveats that travel with this recommendation

- TPP IS 1.5 -- 599,785,472 tokens per cell for a 390,125,472-parameter model, computed from what the cells actually report, not from the plan. That is BELOW the academic literature cluster of 3-20, and further below this project's own 1B flagship at TPP 27-44. NOTE the budget moved after pre-registration: seeds.json planned 1,907 steps, the cells ran 1,144 (planned TPP was 2.6). The cut makes every CE magnitude below MORE inflated, not less.
- ARCHITECTURE EFFECTS MEASURED AT LOW TOKEN BUDGETS SYSTEMATICALLY OVERSTATE. Measured in-tree: GDN's edge over baseline shrank 0.0103 nats @1B -> 0.0059 @15B, roughly halved for a 15x budget increase. So the CE result here is DIRECTIONAL WITH INFLATED MAGNITUDES: treat every CE number below as an UPPER BOUND on the production effect, never as the effect itself.
- THROUGHPUT AND PEAK MEMORY ARE BUDGET-INDEPENDENT AND FULLY VALID. They do not inflate at a short budget the way CE does. At this TPP they are very likely to be what carries the recommendation, and they carry it soundly.
- A NULL IS A BOUND, NOT EQUIVALENCE. A non-significant CE contrast licenses exactly one claim: the difference is smaller than the Dunnett-adjusted CI half-width. Equivalence needs a pre-declared margin and a TOST; no margin was declared, so no equivalence claim may be made from this design.
- Say it that way when this is presented: the speed and memory conclusions are STRONG, the CE conclusions are INDICATIVE.
- A 1.0B-token run structurally CANNOT see any effect that only emerges late in training, or any long-horizon recall advantage. That is a limitation, not a null.
- The literature CE gap between these mixer families is ~0.01-0.03 nats, which may sit below this design's MDE. The pre-registration committed to that as an expected outcome before the run.

## 5. Deviations from the pre-registration

- The unequal-variance fallback uses Dunnett's T3 (studentized maximum modulus) rather than Games-Howell (studentized range). Games-Howell is the ALL-PAIRS procedure; these comparisons are k arms against ONE control, whose unequal-variance analogue is T3. The two differ only in the critical-value distribution, and T3 is the correct family here. This is the only substantive departure from the pre-registered plan.
- The pre-registration's saturation rule (s4.2 -- never pool sigma over cells where the endpoint cannot move) is implemented as FAIL-OPEN: the CE endpoint has no ceiling in this design, no cell is dropped from the sigma pool on saturation grounds, and the count of such drops is reported as zero rather than the rule being silently skipped.
- Sliced-eval / W&B trajectory (the pre-registered SECONDARY endpoint) is not analysed here: sliced_eval is null unless --slice-mask-uri was set, and the W&B series is not in the per-cell JSON. Secondary means secondary; it moves no gate.
- The cross-run drift check (s7) and run 2 are out of scope for this script, which analyses run 1 only.

## 7. Provenance

- Seed schedule: `/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-bakeoff--mixer-bakeoff/docs/mixer-bakeoff/seeds.json` (available: True)
- Parameter ledger (`ARM_L0_DELTA`): `/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-bakeoff--mixer-bakeoff/src/olmo_core/nn/transformer/core6_arms.py` (available: True)

