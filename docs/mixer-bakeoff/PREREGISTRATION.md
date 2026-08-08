# Pre-registration — two-run mixer bake-off

**Written 2026-08-08, before run 1 dispatches.** This file and
[`seeds.json`](seeds.json) are the **contract** between run 1 (submitted tonight) and run 2
(submitted days later). If they are wrong or ambiguous the two runs cannot be paired and run 2's
money is wasted. `seeds.json` is the machine-readable authority for every integer; this file
carries the reasoning and the decision rules.

> **UNVERIFIED — confirm against `ARMS` before submitting.** As of the freeze,
> `src/olmo_core/nn/transformer/core6_arms.py` on `edullm/mixer-bakeoff` still holds only the
> CORE-6 *layer-schedule* arms (`L0`, `K2`, `G4R0`, `G4R2`, `G2R0`, `S14`, `G0R0`). The six mixer
> arms below are **not in `ARMS` yet** — another agent is adding them. Arm *names* here are
> placeholders; each is bound to a real arm by a **`mixer_config` fingerprint** in `seeds.json`
> that **is** verified against `src/olmo_core/nn/attention/recurrent.py` on this branch.
> Reconciling renames rows — **it does not renumber them.** The seed integers are frozen now.

---

## 1. Arms

Run 1 — six arms, three replicates each, 18 cells. Run 2 — two arms, three replicates, 6 cells.

| arm (UNVERIFIED name) | run | mixer config (verified against `recurrent.py`) |
|---|---|---|
| `kda-paper` | **1 and 2** | `KimiDeltaAttentionConfig(allow_neg_eigval=False, conv_activation="silu", gated_conv=False)` — the Kimi Linear paper / `fla` variant, **not** K3. Branch defaults. |
| `kda-noact` | 1 | `KimiDeltaAttentionConfig(gated_conv=False, conv_activation=None)` |
| `kda-gated-conv` | 1 | `KimiDeltaAttentionConfig(gated_conv=True, gated_conv_activation=None, gate_structure="depthwise")` |
| `gdn2` | 1 | `GatedDeltaNet2Config(expand_v=1.0, allow_neg_eigval=False)` |
| `kda-hh-r1` | 1 | `KimiDeltaHouseholderConfig(num_householder=1, allow_neg_eigval=True)` |
| `kda-hh-r2` | 1 | `KimiDeltaHouseholderConfig(num_householder=2, allow_neg_eigval=True)` |
| `kda-k3` | **2 only** | scaled-sigmoid forget gate, `gate_lower_bound=-5.0`, `A_log` zero-init, full-rank output gate |

**`kda-paper` is the shared control** for every arms-vs-baseline comparison in both runs.

Three traps, recorded now:

- `allow_neg_eigval` defaults to **`False`** on `KimiDeltaHouseholderConfig` (`recurrent.py:1850`).
  The R=1 and R=2 arms need it set to `True` **explicitly**.
- With `gate_structure="depthwise"`, `gated_conv_activation=None` **does not** mean
  activation-free — the depthwise pre-gate is exactly a SiLU with a learnable per-channel slope
  (`recurrent.py:1119-1130`). So the contrast that isolates the *gate* is
  **`kda-gated-conv` − `kda-noact`**, never `kda-gated-conv` − `kda-paper`.
- **The K3 gate does not exist in this tree.** `gate_lower_bound`, a scaled-sigmoid gate and a
  full-rank output gate all return zero hits in `recurrent.py`. Run 2 needs an implementation
  before it can be submitted. **UNVERIFIED — confirm against `ARMS`.**

## 2. The seed schedule

`seeds.json` is authoritative. Reproduced here in full.

**Data seeds, shared by every arm** (this is what pairing rests on):

| replicate | `--data-seed` |
|---:|---:|
| 1 | `210007` |
| 2 | `220014` |
| 3 | `230021` |

**Init seeds, per (arm, replicate)** — never shared across arms:

| arm | r1 | r2 | r3 |
|---|---:|---:|---:|
| `kda-paper` | `110007` | `120014` | `130021` |
| `kda-noact` | `113008` | `123015` | `133022` |
| `kda-gated-conv` | `116009` | `126016` | `136023` |
| `gdn2` | `119010` | `129017` | `139024` |
| `kda-hh-r1` | `122011` | `132018` | `142025` |
| `kda-hh-r2` | `125012` | `135019` | `145026` |
| `kda-k3` *(run 2)* | `128013` | `138020` | `148027` |

### 2.1 Pair on DATA ORDER — and the precondition that makes it true

Data order is a function of `(data_seed + epoch, len(dataset))` **only**, and is independent of
the model. Verified on this branch: `_build_global_indices` does
`get_rng(self.seed + self.epoch).shuffle(arange(len(dataset)))` (`data_loader.py` ~674), and
`get_rng` is `np.random.Generator(PCG64(seed))` (`data/utils.py:585`).

So the same `--data-seed` gives every arm a **byte-identical token stream — if and only if**
sequence length, global batch size, DP world size, dataset release and step count are also
identical. Sequence length and global batch enter through
`instances_per_batch = global_batch_size // sequence_length`; world size enters through
`indices[:, dp_rank::dp_world_size]` (`data_loader.py:707`). **Any one of them differing breaks
the pairing while every run still trains and prints a plausible number.** A cell that changes any
of them is not paired and must be analysed as an independent sample.

### 2.2 Init seeds cannot be paired across arms

Arms have different tensor inventories — Householder R=2 widens `w_k`/`w_v`/`w_b` and the k/v
convolutions by R, GDN2 has a `w_w` that KDA has not, the gated arm adds gate tensors — so a
single RNG stream diverges at the first differing tensor and every draw after it is unrelated.
**Reusing one integer across arms buys no variance reduction and this document does not claim
any.** Init seed is recorded, and **reused identically for the same arm in run 2**, because that
is what makes run 2's repeated `kda-paper` a true replicate rather than a fresh sample.

### 2.3 `--init-seed` plumbing — verified on this branch

- **`.edullm/train_core6_arm.py` — PLUMBED.** `build_config` passes `init_seed=opts.init_seed`
  into `build_arm` → `TransformerConfig.llama_like` → `cls(...)`, reaching the generator that
  actually draws weights (see the comment block at `train_core6_arm.py:623-640`). Default `12536`.
- **`.edullm/train_on_corpus.py` — NOT PLUMBED.** It exposes `--data-seed` only; there is no
  `--init-seed`. Its config carries `init_seed=12536` (line 341) and calls
  `seed_all(config.init_seed)` (line 803), but weights come from `TransformerConfig.init_seed`,
  default **0** (`nn/transformer/config.py:326`). Every "n seeds" on that entrypoint is n data
  orderings of **one** initialisation.

**Run 1 uses `.edullm/train_core6_arm.py`. Required** — the other entrypoint cannot vary the init
component at all.

**The guard at `train_core6_arm.py:1582`** refuses the run when
`config.init_seed != config.model.init_seed`. Both are set together from `--init-seed`, so they
can only diverge via a dotlist override (`init_seed=99`, `model.init_seed=42`) that moves one.
**Never override either field — pass `--init-seed`.**

### 2.4 Why these integers

- **Not `0`, not `12536`** — the two defaults on this entrypoint. Using either makes "the flag
  never reached the weight draw" indistinguishable from "it did".
- **Not small sequential integers.** Data order keys on `seed + epoch`, so `0,1,2` makes a cell at
  epoch 1 alias a different cell at epoch 0. Spacing the data seeds `10007` apart makes aliasing
  unreachable at any epoch count this study can hit.
- Construction: `data_seed(r) = 200000 + 10007r`, `init_seed(i, r) = 100000 + 3001i + 10007r`,
  both primes, **0 collisions over all 24 integers** (checked). The formula is documentation —
  **the literal integers above are the source of truth.**

## 3. Endpoints

Exact keys in the rank-0 JSON that `summarise()` prints (`train_core6_arm.py:1471-1543`) — the
only channel the platform reads results back through.

- **Primary: `val_ce`** — held-out CE in nats from `evaluate_val_aggregate`, over the corpus's own
  `val` partition, on every rank. A difference is only a paired difference if the denominators
  match, so `val_tokens`, `val_tokens_present`, `val_tokens_declared` and `val_shards` are
  compared before any CE is differenced.
- **Co-primary: steady-state throughput** — `tps_device_avg` (per device) and `tps_total_avg`
  (total). These come from `SpeedMonitorCallback`'s `throughput/device/TPS (actual avg)`, whose
  clock resets **after step 1**, so they exclude process start, dataset open, FSDP wrap and the
  first-step compile. **`tps_naive_wall_clock` is not an endpoint** — it charges every fixed cost
  against the hardware and penalises bigger shapes hardest.
- **Co-primary: `peak_memory_gib`** — `torch.cuda.max_memory_allocated()` on rank 0.
- **Secondary: the ladder trajectory** — `sliced_eval` (`null` unless `--slice-mask-uri` is set)
  and the W&B `train/CE loss` series. Secondary means secondary: it moves no gate.

**Final TRAIN loss (`last_loss`) must NOT be used as an endpoint.** A decay-to-zero LR schedule
ends at a mechanically lower train loss at equal held-out quality and can **invert** the
comparison. `first_loss` is kept only as an init sanity check — expect ≈ `ln(100352)` = **11.52**.

## 4. Analysis plan, pre-committed

1. **Admissibility is step 0, and it runs BEFORE pooling.** A cell that diverged (non-finite
   loss), failed to converge, OOM'd, or ran a different step count is **excluded before any
   variance is pooled**, and the exclusion is **declared with a count** in the results table.
   Never exclude-then-cite. Absolute-magnitude gate, not an existence check: `val_ce` must be in
   a plausible band and `first_loss ≈ 11.52`.
2. **Never pool σ over cells where the endpoint cannot move.** A ceiling- or floor-saturated cell
   contributes ~0 variance and would halve apparent σ. Such cells are excluded from the σ pool and
   counted separately; **fail open** — if it is unclear whether a cell can move, keep it in.
3. **Pooled-variance one-way ANOVA across arms**, `df = n_arms × (3 − 1) = 12` in run 1 — **not**
   independent pairwise t-tests, which throw away 4/5 of the df.
4. **Dunnett** correction for the five arms-vs-`kda-paper` comparisons. Dunnett is the exact case
   for *k* arms against **one shared control**, and is uniformly less conservative than Bonferroni.
   Two-sided α = 0.05, **k = 5, df = 12 → critical value 2.902** (computed two ways — Gaussian
   quadrature over the ρ = ½ equicorrelated max-|t| and a 6M-draw Monte Carlo — agreeing to 3 d.p.).
   Arm-vs-arm comparisons that are not against the control are **exploratory**, declared as such.
5. **Variance homogeneity: Levene (median-centred, robust) as the decision test, Bartlett
   reported alongside.** If Levene rejects at α = 0.05, **fall back to Welch's ANOVA with
   Games-Howell**, do not pool, and report σ per arm instead of one pooled σ. Committing to this
   now is what stops the fallback being chosen after seeing which answer it gives.
6. **Reporting**: effect, Dunnett-adjusted CI and n for every contrast — never a bare p-value,
   and never "n.s." as though it meant "no effect".

## 5. Power — honest, and pre-committed as an expected outcome

**Estimator: exact non-central t** (`scipy.stats.nct`), dominant-tail survival only
(`nct.sf(crit, df, ncp)`) — the naive two-tail form suffers catastrophic cancellation. Validated
by reproducing `moe/audit/findings/power.md` (0.03917 at n=3, 0.02018 at n=5, σ=0.0120) to 5 s.f.
The **normal approximation is 2.2× too optimistic at n=3** and is not used anywhere here.

σ has **never been measured at this scale in this project**, and the two available estimates
differ **5.5×**: 0.0019 nats (161 modded-nanogpt runs at 124M/0.45B) and 0.0105 nats (13 in-repo
KDA runs). MDE at n=3, 80% power, α=0.05 two-sided, unpaired difference of two arm means
(SE = σ√(2/3)):

| σ (nats) | SE(diff) | **MDE, Dunnett k=5, df=12** | MDE uncorrected |
|---:|---:|---:|---:|
| 0.0019 (optimistic) | 0.001551 | **0.0059** | 0.0047 |
| 0.0105 (pessimistic) | 0.008573 | **0.0327** | 0.0262 |

Run 2 (2 arms, df = 4, k = 1 so Dunnett = plain t, t = 2.776): MDE **0.0058** / **0.0322**.

**Say it plainly: the measured softmax-vs-gated-delta CE gap in the literature is ≈ 0.010 nats,
which is BELOW the MDE at the pessimistic σ. This design may not resolve CE differences between
these arms. That is pre-committed as an expected outcome, not something to be discovered during
analysis.** Reaching 0.010 nats at σ = 0.0105 needs **n ≈ 24 per arm** (MDE 0.0103) — 144 cells,
which is not this study.

**What a null licenses, and what it does not.** A non-significant CE contrast licenses exactly
one claim: **a bound** — "the difference is smaller than the Dunnett-adjusted CI half-width".
It does **not** license equivalence. Equivalence needs a pre-declared margin and a TOST, and no
margin is declared here, so **no equivalence claim may be made from this design.**

### 5.1 Locked run parameters, and the TPP 2.6 caveat

**18 cells = 6 arms × 3 seeds, 1.0B tokens per cell.** Locked 2026-08-08.

| parameter | value |
|---|---|
| global batch | **524,288** tokens |
| sequence length | **4096** (→ 128 instances/batch) |
| steps/cell | **1,907** (1.0e9 / 524,288; realised 999,817,216 tokens) |
| parameters | 390,135,552 → **TPP = 2.6** |
| compute profile | `gpu-8xa100` |
| attempts | **1, PINNED** — `olmo-core-train` defaults to 2, which doubles cost and flips the approval gate |
| declared hours | 2.6/cell → worst-case ceiling **~$1,014** for all 18 |
| throughput basis | **303,072 tok/s** (conservative 20-step figure; a 200-step run read 455,789 and that higher number is a run-length artifact) |
| makespan | **≈3.2h** at 6-way concurrency (Batch array jobs refill slots as they free, so makespan is sum-of-cells ÷ concurrency, not serial waves) |

The **R=2 Householder arm is expected ~2× slower** — it runs a non-chunked sequential Triton
kernel — so **3 of the 18 cells are the long pole** and dominate the makespan.

**TPP 2.6 is the single most important caveat in this document.**

- **TPP 2.6 is at the *bottom edge* of the academic literature cluster (3–20)** — slightly
  *below* even that cluster, and that is stated rather than rounded away. Published effect sizes
  are nonetheless **more** transferable here than to this project's 1B flagship at TPP 27–44,
  which sits in the gap between the academic and production clusters.
- **Architecture effects measured at low token budgets systematically OVERSTATE.** Measured
  in-tree: GDN's edge over baseline shrank **0.0103 nats @1B → 0.0059 @15B** — roughly **halved
  for a 15× budget increase**. So a CE ranking from this run is **directionally informative,
  magnitudes inflated**. Report CE magnitudes as upper bounds on the production effect.
- **Throughput and peak-memory rankings are budget-independent and fully valid.** The goal is a
  production mixer choice, so state it plainly: **the speed/memory conclusions are strong and the
  CE conclusions are indicative.**
- **What a 1.0B-token run structurally CANNOT see:** any effect that only emerges late in
  training, and any long-horizon recall advantage. That is a **limitation, not a null**.
- **The relevant literature CE gap between these mixer families is ~0.010–0.03 nats. At the
  pessimistic σ = 0.0105 the Dunnett-corrected MDE at n=3 is ~0.034 nats — larger than the effect
  we are hunting.** Pre-committed: **a null is a BOUND, not equivalence. A flat CE table must not
  be read as "no difference".**

## 6. What run 1 measures for free that run 2 needs: σ

Arguably this design's most valuable output. Run 1 yields a **pooled within-arm σ at
`df = n_arms × (3 − 1) = 12`** — the first measurement of σ at this scale in this project, ending
the 5.5× guess. **Pre-registered as a deliverable of run 1** regardless of what the CE contrasts
do, and **run 2 must be sized from it**, not from either literature estimate.

Report it with its own uncertainty: a χ² interval at df = 12 is only `σ̂ × [0.717, 1.651]`, so
even this estimate is a factor-2.3 bracket. It narrows the guess; it does not close it.

## 7. Reproduction contract — run 2 re-runs `kda-paper` bit-identically

Every flag below must match run 1 **exactly**. This is the whole point of the seed contract.

```bash
python -m torch.distributed.run --nproc-per-node=<N> --standalone \
  .edullm/train_core6_arm.py "$EDULLM_RUN_ID" \
    --save-folder     "$EDULLM_CHECKPOINT_DIR" \
    --arm             kda-paper      `# UNVERIFIED -- confirm against ARMS` \
    --data-seed       210007         `# replicate 1; 220014 / 230021 for r2 / r3` \
    --init-seed       110007         `# replicate 1; 120014 / 130021 for r2 / r3` \
    --sequence-length 4096 \
    --global-batch-size 524288 \
    --rank-microbatch-size <MICRO> \
    --steps           1907 \
    --warmup-steps    <WARMUP> \
    --learning-rate   <LR> \
    --save-interval   <SAVE_INTERVAL>
```

Submitted on compute profile `gpu-8xa100` with **`--attempts 1` pinned** (the `olmo-core-train`
profile defaults to 2). `<MICRO>`, `<WARMUP>`, `<LR>` and `<SAVE_INTERVAL>` are whatever run 1
used — **UNVERIFIED here; copy them from run 1's `intent/<run_id>.json` in the S3 lineage bucket,
which is the durable record, not from `edullm status` (last 30 dispatches only).**

**Must match bit-for-bit between run 1 and run 2** — arm name; `--data-seed`; `--init-seed`;
`--sequence-length`; `--global-batch-size`; `--rank-microbatch-size`; `--steps`;
`--warmup-steps`; `--learning-rate`; the LR schedule (`CosWithWarmup`, set in code — a change is
a code change, not a flag change); the LM loss implementation
(`LMHeadConfig.loss_implementation`, default `LMLossImplementation.default`, set in code);
`dataset_release` (submit-time flag → `EDULLM_DATASET_ID` / `_VERSION` / `_TOKENIZER`); the
compute profile; and the **DP world size** (`--nproc-per-node` × nodes).

Also fixed in code and therefore fixed by the commit: `compile_model=True`, FSDP with
`param_dtype=bfloat16` / `reduce_dtype=float32`, `max_grad_norm=1.0`, `TRITON_F32_DEFAULT=ieee`
(a **correctness** setting for these Triton mixers, not a speed one — 166× accuracy; it must be
`export`ed in the same shell that execs python so torchrun forwards it to the workers).

**`--steps` and `--global-batch-size` are load-bearing for the pairing, not just for cost.** A run
2 that trims steps to fit an approval band consumes a *prefix* of the stream, not the same stream,
and the repeated arm stops being a replicate.

**Record the image digest on both runs, and compare them.** Run 2 must state whether its image
digest differs from run 1's. **That is the drift the repeated `kda-paper` arm exists to detect** —
a per-replicate `run1(kda-paper, r) − run2(kda-paper, r)` on identical seeds. If those three
differences are not centred on zero relative to run 1's pooled σ, the platform changed underneath
the study and the cross-run K3 comparison is void (the *within*-run-2 K3-vs-paper contrast
survives, which is exactly why run 2 re-runs the baseline instead of borrowing run 1's).
Record the ECR digest, `torch`, `cuda` and `gpu` from the summary JSON on every cell.
