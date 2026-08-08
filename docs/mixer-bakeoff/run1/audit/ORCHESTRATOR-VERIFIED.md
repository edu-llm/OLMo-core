# Orchestrator's own verifications — mixer bake-off audit, 2026-08-08

Everything here I checked myself, independently of the council, from source / the repo's own code /
read-only AWS. Council findings I could NOT reproduce are marked as such.

## Confirmed defects (retract these)

### 1. GDN2 "significant" CI is not reproducible — CONFIRMED, most serious
Shipped claim `[+0.0262, +0.0479]` "excluding zero".
- half-width 0.010850 / Welch SE 0.006631 = **implied crit 1.6362**, BELOW uncorrected z 1.96.
- correct SMM(k=5, df=2.5548) = **5.7087** -> `[-0.00080, +0.07491]`, includes zero.
- Ran the repo's OWN `welch_t3_contrasts()` on all 18 cells: GDN2 `ci_low -0.0008014996193461896`,
  `ci_high 0.0749110520182639`, **`excludes_zero: false`** — and false on ALL FIVE contrasts.
- Ran the repo's OWN `welch_anova()`: **F = 40.905279, df_within 5.258204, p = 0.000344**.
  Shipped claim of "Welch F = 53.29, p = 0.0005" is not what the code emits.
- Centre matched the true estimate to 5 d.p. (0.037050 vs 0.037055) => critical-value error, NOT
  fabrication.
- **Fallback was never licensed:** `PREREGISTRATION.md:164-167` makes Levene THE decision test and
  the fallback conditional on Levene REJECTING. Levene p = 0.4840, did not reject. The generated
  `REPORT.md:45-48` gets this exactly right ("Levene does not reject; the pooled analysis stands").

### 2. MFU is not comparable across arms — CONFIRMED (my own finding)
MFU ratio decomposes EXACTLY into (throughput ratio) x (FLOP-count ratio):
```
arm         tok/s     x_base   mfu%    mfu_x   flop_x   tok_x*flop_x
KDA_BASE   419288     1.0000   41.55   1.0000  1.0000   1.0000
KDA_NOACT  418364     0.9978   41.46   0.9980  1.0000   0.9978
KDA_GCONV  410690     0.9795   40.70   0.9795  1.0000   0.9795
GDN2       416894     0.9943   41.97   1.0102  1.0158   1.0100
KDA_R1     326513     0.7787   32.35   0.7786  1.0000   0.7787
KDA_R2     295638     0.7051   29.11   0.7006  0.9934   0.7004
```
Root cause: `feed_forward.py:206-209` counts `6 * numel` (fwd+bwd); mixers
(`recurrent.py:998-1054`, `:1759-1811`) count `2 *` (fwd only). `solve_widths`
(`core6_arms.py:515-576`) param-matches by SHRINKING FFN width on the 14 non-KDA layers, so the
counted total falls at 6/param while rising at 2/param.
Reconciled exactly for KDA_R2: FFN removed 6*3*1024*1376 = 25,362,432; mixer gained 2x4,456,448 =
8,912,896; predicted net **-16,449,536** vs measured **-16,351,232**; residual exactly **-98,304**
= 3*D*32 = one width-step (`solve_widths`'s unexpressible remainder).
=> **GDN2's "highest MFU in the run (42.0%)" is an artifact** — it is 0.9943x, i.e. SLOWER than
the reference. KDA_R2's MFU is understated for the same reason in reverse.
=> Throughput (tok/s) is wall-clock and does NOT depend on the FLOP count. **The recommendation
survives.**

### 3. "28.6% pure kernel overhead / same function" — CONFIRMED invalid on both halves
Arithmetic at n=3: `1 - R1/BASE = 22.13%`; `BASE/R1 - 1 = **28.41%**`; p50 step time **28.37%**.
Arity at n=3: `1 - R2/R1 = 9.46%`; `R1/R2 - 1 = **10.44%**`; step time **11.58%**.
**SOLVED (F1, verified by me): 28.6/10.9 are a SINGLE-CELL reading** — data seed 210007,
`step_time_s_p50`: cell 0 BASE 1.229610791, cell 12 R1 1.581, cell 15 R2 1.754 give
**28.58%** and **10.94%**. So they are real numbers from one replicate, quoted in the same
sentence as the **n=3 means** 0.779x/0.705x. **Mixed estimators in one claim** — not bad
arithmetic. Composition proves it: 1.286 x 1.109 = 1.42617, but the true n=3 total is
BASE/R2 = **1.41825**, which the n=3 components reproduce exactly
(1.2841 x 1.1044 = 1.41816). Fix by quoting n=3 throughout: **28.4% and 10.4%**.
Function: `KimiDeltaAttentionConfig.allow_neg_eigval` defaults **False** (`recurrent.py:1088`, class
default `:687`); `_kda_householder()` passes **True** explicitly. Plus eager-fp32 gate + explicit
L2-norm vs fused-in-kernel, and a DIFFERENT kernel. Params/FLOPs identical; **function is not**.
There is **no `allow_neg_eigval=False, R=1` arm**, so arity was never isolated from the reflection
regime.
PDF strings to fix: `RESULTS-TABLE.html:76` ("same function ... identical params, FLOPs and
state_dict"), `:151` ("28.6% is pure kernel overhead for zero mathematical difference"), `:159`
(GDN2 "highest MFU"), `:113,:121,:136,:139,:262-264`.

### 4. `results.json` is a stale 8-cell partial — CONFIRMED
`coverage.cells_found = 8`, `cells_expected = 18`; GDN2/KDA_R1/KDA_R2 absent; KDA_GCONV n=2;
committed in the same commit whose message says 18/18. `hard_errors: []`.

### 5. 16 of 18 cell files were hand-transcribed — CONFIRMED
`throughput_tok_s_steady_per_device * 8 == throughput_tok_s_steady` must hold BIT-EXACTLY (exact
binary division; zero `round()` in the entrypoint). Holds in only **5 of 18**. Residuals up to
**111 tok/s** (cell 10); also cell 1 (+37), 7 (-49), 12 (-103), 15 (-46).
`first_loss` = exactly `11.7124` in 16 cells — **not float32-representable**.
Only cells 0 and 2 are self-consistent and full precision.
**Bounded:** `val_ce`, `peak_memory_gib`, `peak_memory_reserved_gib` are FULL precision in all 18.
Throughput rounding <= **0.026%** against arm gaps of **2.1-29.5%**. Rankings safe; CE and memory
analyses untouched.

### 6. Test gaps — CONFIRMED
- **No test anywhere checks kernel accuracy at beta > 1**, the regime both R arms ran in.
  `kda_householder_test.py:126,162` draws `beta = rnd(...).sigmoid()` = (0,1).
  `recurrent_test.py:1137` recomputes `logits.sigmoid() * 2.0` **in the test body** and never calls
  the module or the kernel — a contract check on the test's own arithmetic
  (cf. [[test-must-call-not-recompute]]).
- `core6_bakeoff_guards_test.py` — the ONLY test exercising the fused Triton kernel's causality
  (`use_fla=True`; every other causality test uses `use_fla=False`) — is `@requires_gpu` +
  `@requires_fla` and `git grep bakeoff_guards 9425fea` returns **zero hits outside the file
  itself**. On a laptop with neither, all of it **skips silently**
  (cf. [[a-skip-counts-as-a-pass]]).

## Confirmed good (do not "fix")

- pooled sigma-hat **0.020415** df 12, chi2 [0.01464, 0.03370], F(5,12) **2.4893** p **0.0908**,
  Dunnett crit **2.9013**, MDE **0.063603** — reproduced by three independent agents + scipy.
- The SiLU identity: `max |2*sigmoid(a*u)*u - (2/a)*silu(a*u)| = 8.882e-16` over a grid, in stdlib.
  Exact algebra. **The `KDA_NOACT` control was genuinely necessary.**
- Parameter matching exact for all six arms; `solve_widths` mixer-size fix present at sha `9425fea`.
- `peak_memory_gib` sd exactly 0.0 within every arm (all three KDA_BASE cells report
  `9.153131484985352` verbatim).
- Attention schedule (2,5,8,10,12,14) identical across all six arms; RoPE gated by
  `isinstance(block.attention, (Attention, FusedAttention))` at `model.py:253-268`.
- The generated `REPORT.md` is clean on every point audited. **All defects are in the hand-written
  layer** (`HANDOFF.md`, `RESULTS-TABLE.html`/`.pdf`).

## Adjudicated disagreement: A2 vs E1 — A2 is right
E1 claimed "the seed-230021 block effect is visibly ~0.16 nats" and that a blocked reanalysis may
move the headline. I computed the two-way decomposition from the 18 `val_ce`:
block means 210007 **3.05246**, 220014 **3.06931**, 230021 **3.05719**; max spread **0.01686 nats**
(an order of magnitude smaller than claimed). SS_block 9.0707e-04 df 2, SS_resid 4.0943e-03 df 10,
**F_block(2,10) = 1.1077**, variance ratio **0.0176**. Two-way sigma 0.020234 (df 10) vs one-way
0.020415 (df 12) — a 0.9% gain costing 2 df, so Dunnett crit rises 2.901 -> 2.990 and the
**MDE gets WORSE: 0.06481 vs 0.06360**. Blocking does NOT rescue the headline.

## Negative eigenvalues are FREE on the fast kernel — CONFIRMED
`fla-core` 0.5.1 (the pinned version), `fla/ops/kda/chunk.py`:
- `:190` `allow_neg_eigval: bool = False` in the **public signature**
- `:62` the entire forward impl: `fused_beta_sigmoid(beta_raw, scale=2.0 if allow_neg_eigval else 1.0)`
- `:170` backward twin
- `:400-401` only guard is a *coupling* check (needs `use_beta_sigmoid_in_kernel=True`);
  `:412` only beta assert is a **shape** assert. No range assert, no clamp.
- Referenced by **no Triton chunk kernel, no WY/UT file, no backward file**.
Invariant: the inverted matrix is unit-diagonal, strictly lower triangular
(`solve_tril.py:58,69,83`) => **det = 1 for any beta**.
Literature: arXiv:2411.12537 App. E.4 — *"For DeltaNet ... there is no need to modify the Triton
kernel"*; Fig. 13 is a two-line diff. DeltaProduct arXiv:2502.10297 Prop. 1 keeps |lambda| <= 1 at
beta in [0,2].
In-tree: `GatedDeltaNet` **defaults `allow_neg_eigval=True`** (`recurrent.py:147`, `:513`) and has
trained on the chunked kernel with beta in (0,2) all project. `KimiDeltaAttention` has the same two
lines (`:874-876`) behind a flag it exposes (`:1088`).
`kda_householder.py:64-70` self-describes as *"a simple fused-recurrent (sequential-over-time)
Triton kernel, not a chunked WY-representation kernel ... expected to be materially slower ...
mechanism validation, not throughput."*
=> **The 22% and +2.54 GiB measured the stand-in kernel, not the mechanism. Fix = config flag.**
**GATE IT** on one cheap test: `chunk_kda` vs `naive_recurrent_kda`, fp32, beta > 1 (see gap above).

Also: `dispatch_chunk_kda` (`flash_linear_attn_api.py:45-59`) exposes **neither `safe_gate` nor
`lower_bound`** — a free upstream throughput knob we cannot currently reach. `HEAD_DIM = 64`
(`core6_arms.py:99`), so FlashKDA (needs 128, inference-only) does not apply.

## Money, from measured wall clock
Per-cell mean **1,916 s = 0.532 h**; `gpu-8xa100` at **$21.9576**/node-h, 1 node/cell =
**$11.68/cell**.
- run 1 actual compute **~$210** (declared ceiling was $790.47 — a ceiling, not a spend)
- 3-cell arm **~$35** | 6-cell **~$70** | 12-cell 2-arm confirmation **~$527**
- resolving 0.030 nats at sigma-hat 0.02042: n=11 -> 66 cells **~$2,900**
- resolving 0.010 nats: n~95 -> 570 cells **~$25,000**
- production flagship sits at TPP 27-44 vs this run's **1.5** = **18-29x higher**, and the in-tree
  anchor says architecture effects SHRINK with budget (0.0103@1B -> 0.0059@15B). So the CE numbers
  are upper bounds and the gap may be smaller still at production scale.

## The recall eval is viable — VERIFIED IN S3
Bucket is `sbsandbox-intern-edullm-outputs` (NOT `edullm-scratch`).
`teams/scratch/runs/run_019fe0f9-1bbd-702c-b141-6d58e128bda6/cell-{0..17}/checkpoints/` each contain
`step0/`, `step572/`, **`step1144/`** — the final trained checkpoint, **present for all 18 cells**
(verified via `list-objects-v2` filtered on `step1144/config.json`: cells 0-17 all returned).
`--slice-mask-uri` IS declared in argparse (`train_core6_arm.py:2184`) and wired (`:2096`), so the
pre-registered secondary endpoint is a **config change, not code work** — unlike
`--lm-loss-implementation`, which was never declared and killed the first array.

## Data-seed semantics — CONFIRMED (E1's real finding)
`_build_global_indices` (`data_loader.py:668-680`) builds a FULL permutation; the loader consumes
from the front; the run stops mid-epoch. 1144 steps x 128 seq/step = **146,432 of 61,094,464
sequences = 0.2397%** of the corpus (tokens reconcile exactly: 146,432 x 4096 = 599,785,472).
So a "data seed" selects an almost entirely **different sample** (~99.76% disjoint), not a
reordering. All six arms at a seed share ONE index file, so the draw is common and **cancels in
every arm-minus-arm contrast** — estimates unbiased — but sigma-hat mixes optimisation noise with
corpus-sampling noise, so sizing a future run from it inherits that.

## Pairing precondition for any future arm
`seeds.json` locks `steps` as a **pairing precondition** ("a shorter arm consumes a prefix, not the
same stream"). The frozen plan said **1,907**; all 18 cells ran **1,144**. So they are paired with
each other, but **any future seed-paired arm must run 1,144, not 1,907**, or the pairing and the
run1<->run2 drift check are void. Seed formula (recorded as documentation, literals are the source
of truth): `data_seed(r) = 200000 + 10007*r`; `init_seed(arm_i, r) = 100000 + 3001*i + 10007*r`.
Reserved for arm_index 6: data 210007/220014/230021, init 128013/138020/148027.
Also: `seeds.json` declares `last_loss` **must not be used as an endpoint** (a decay-to-zero LR
schedule ends at a mechanically lower train loss at equal held-out quality) — so the quantised
3.000/3.050/3.060 values feed no conclusion.
