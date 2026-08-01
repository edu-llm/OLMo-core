# 07 — Risk audit: internal contradictions, fail-open gates, and the power problem

**Author:** reassessment team member 07. **Date:** 2026-08-01.
**Posture:** hostile review of the **PLAN**, not the science. Every claim labelled
MEASURED / INFERRED / ASSUMED. Arithmetic shown.
**Constraint honoured:** no code executed anywhere. All arithmetic below is done by hand and is
reproducible by hand.

**Sources read:** `/Users/ericwu/Developer/Capstone_LLM/HANDOFF.md` (full),
`/Users/ericwu/Developer/Capstone_LLM/docs/liv-brainlift-experiment-design.md` (§2, §3, §6, §7, §8,
§9, §9a in full; rest skimmed),
`/Users/ericwu/Developer/Capstone_LLM/KDA/HANDOFF.md` (power/variance sections).

---

## TL;DR — the three findings that matter

1. **The 12-arm × 2-seed screen (Phase 3a) cannot distinguish anything.** At n=2 the paired
   t-test has **1 degree of freedom**; it rejects at two-sided α=0.05 **iff** the two per-seed
   differences satisfy `|d₁+d₂| > 12.706·|d₁−d₂|` — i.e. the two seeds must agree to within a
   **1.171 : 1 ratio**. Against the plan's **own measured** MQAR seed spread of **σ ≈ 39.3 pp** at
   its **own chosen operating point** (`N512_D64`: 0.05 / 0.09 / 0.20 / 0.56 / 0.98), a true effect
   of one full seed-SD gets **~9% power** — barely above the 5% false-positive rate. See §2.
2. **The CE gate is already known to be unreachable, and nobody has drawn the conclusion.**
   The plan says "measure `s_δ` in the pilot." The sibling KDA track **already measured it** at
   comparable scale: MDE ≈ 0.014 nats at n=5 paired ⇒ `s_δ ≈ 0.0113–0.0126 nats`. The gate needs
   `s_δ ≲ 0.0114`. The measured value **straddles or exceeds** the requirement. The pilot will
   confirm what is already on disk. See §2.3.
3. **`6.27%` is a 1.2B number quoted inside a 350M design.** At the frozen d=1024 geometry the P1
   model-level saving is **4.437%** and the mixer-level cut is **37.47%**, not 6.27% / 44%. HANDOFF
   states both the right number and the wrong ones, in adjacent paragraphs. Arithmetic in §1.1.

---

## 1. Consistency audit

### 1.0 First: the frozen geometry does NOT have untied embeddings

The brief I was given says "vocab 65536, **untied** embeddings, total 354,483,968." That is
**wrong**, and the error is inherited from an ambiguous line in HANDOFF. Arithmetic (MEASURED — it
reproduces the frozen ledger to the parameter):

```
d = 1024, ff = 4608, 16 layers (10 LIV + 6 GQA), vocab 65536, hq=16 hkv=8 hd=64

LIV mixer   4d² + kd      = 4,194,304 +   3,072 =  4,197,376   × 10 = 41,973,760
GQA mixer   3.0d²         = 1,048,576 + 524,288 + 524,288 + 1,048,576
                          = 3,145,728              × 6 = 18,874,368
per-head QK-norm          = 6 × 2 × 64                  =        768
SwiGLU MLP  3·d·ff        = 3 × 1024 × 4608 = 14,155,776 × 16 = 226,492,416
RMSNorms    16 × 2 × 1024 + 1024                        =     33,792
embeddings  65536 × 1024                                = 67,108,864   ← TIED (×1)
                                                          -----------
                                                          354,483,968  ✓ EXACT
```

With **untied** embeddings the total would be **421,592,832**, and solving the ledger backwards for
`ff` gives `159,383,552 / (16·3·1024) = 3242.67` — **not an integer**, so untied is arithmetically
impossible for this ledger. Cross-checks that confirm tied: HANDOFF's own "+67,108,864 (~19% of the
model)" ⇒ `67,108,864 / 354,483,968 = 18.93%` (an overshoot that was *removed*), and §3.3's
"embeddings are 67.1M of 354.4M = 18.9%".

**Flag (LOW severity, HIGH contagion):** HANDOFF's geometry-omissions table reads
"`llama_like` defaults to **untied** embeddings | **+67,108,864**" under a column headed "cost".
A reader — including me, via the brief — takes that as "the ledger includes untied embeddings."
It means the opposite: the default was untied, that overshot, and the fix was to tie.
**Fix: reword to "`llama_like` defaults to untied; the frozen ledger is TIED — the default
overshot by +67,108,864."**

### 1.1 The P1 saving: 6.27% is WRONG at the frozen geometry; 4.44% is RIGHT

HANDOFF simultaneously asserts (a) "gate reduction is d/(2r) — 4× at d=1024, so the decode ceiling
is **4.44%**, not 6.27%", and (b) elsewhere "rank-128 → **6.27%**" / "a **44%** mixer cut is a
**6.27%** model cut". Resolution, by arithmetic from the frozen geometry:

Stock LIV mixer `4d² + kd`; factorized `2d² + 4dr + kd`. Saving per LIV layer = `2d² − 4dr`.

| | d=1024 (**frozen, 350M**) | d=2048 (1.2B) |
|---|---:|---:|
| gate reduction factor `d/(2r)`, r=128 | **4×** | 8× |
| stock mixer | 4,197,376 | 16,783,360 |
| factorized r=128 | 2,624,512 | 9,443,328 |
| saving / LIV layer `2d²−4dr` | **1,572,864** | 7,340,032 |
| **mixer cut** | **37.47%** | **43.73% ≈ 44%** |
| × 10 LIV layers | 15,728,640 | 73,400,320 |
| model total | 354,483,968 | 1,170,340,608 |
| **model cut** | **4.437%** | **6.272% ≈ 6.27%** |

Both derived figures reproduce independently: `354,483,968 − 15,728,640 = 338,755,328`, which is
**exactly** the `F-r128` entry in HANDOFF's own cost table. And `73,400,320 / 1,170,340,608 =
6.2717%`.

**VERDICT: `6.27%` and `44%` are d=2048 / 1.2B numbers.** They are correct *there* and wrong *here*.
The frozen design is 350M. **Every P1 headline must read 4.44% model / 37.5% mixer.** HANDOFF's
"4.44%, not 6.27%" line is the right one and the other two occurrences are stale copy from the
pre-freeze draft. This is the plan's single most-repeated wrong number and it inflates P1's
parameter story by **41%** relative (6.27/4.44).

Note what survives: **"0.25× params" for the gate projections is CORRECT at d=1024**
(`2dr/d² = 2r/d = 256/1024 = 0.25`). The 92.6%-energy-at-0.25×-params hook is intact. It is only
the *whole-model* percentage that is inflated.

### 1.2 Arm-ledger checks — all PASS

| claim | check | verdict |
|---|---|---|
| `A16-P` 354,388,992 vs `L0` 354,483,968 "within 0.03%" | diff 94,976; `94,976/354,483,968 = 0.02679%` | **OK** (0.027% < 0.03%) |
| `F-r128` 338,755,328 → "0.956×" | `338,755,328/354,483,968 = 0.955625` | **OK** |
| `N-narrow` 338,804,528 vs `F-r128` → "0.0145%" | diff 49,200; `49,200/338,755,328 = 0.014524%` | **OK** |
| `A-fewer3` 357,638,528 → "1.009×" | 3×(4,197,376 − 3,145,728 − 128) = 3,154,560; `354,483,968 + 3,154,560 = 357,638,528` **exact**; ratio 1.008899 | **OK** |
| `Q-mqa` 348,978,944 → "0.984×" | 6 × 2 × (524,288 − 65,536) = 5,505,024; `354,483,968 − 5,505,024 = 348,978,944` **exact**; ratio 0.984469 | **OK** |
| `A16-P` derived width | needs `Δff = 212` ⇒ ff = 4820; `354,483,968 − 10,515,200 + 10,420,224 = 354,388,992` **exact** | **OK** |

The arm builder's ledger is **clean**. This is genuinely good work and I could not break it.

### 1.3 The topology traffic table — all PASS

| claim | check | verdict |
|---|---|---|
| KV 12 KiB/token (L0) | `6 × 2 × 512 × 2 B = 12,288` | **OK** |
| KV 32 KiB/token (A16-P) | `16 × 2 × 512 × 2 B = 32,768` | **OK** |
| saving 20 KiB/token | 20,480 B | **OK** |
| weight read 708.9 MB | `354,483,968 × 2 = 708.97 MB` | **OK** |
| A16-P traffic @4K = 843.1 MB | `708.97e6 + 4096×32768 = 843.15e6` | **OK** |
| 9.9% saving @4K | `4096×20480 / 843.15e6 = 9.95%` | **OK** |
| 10% win at **T ≈ 4,121** | `0.10 = 20480T/(708.97e6+32768T)` ⇒ `17,203.2T = 70.897e6` ⇒ **T = 4,121.2** | **OK** |
| KV == weight at T = 57,690 | `708.97e6/12,288 = 57,696` | **OK** (rounding) |
| KV share @4K 6.6% / @32K 36.2% | `5.033e7/7.593e8 = 6.63%`; `4.027e8/1.1116e9 = 36.2%` | **OK** |

Every topology **byte** number is exact. §5 explains why that is not the reassurance it looks like.

### 1.4 FAILING check — the FLOPs column is internally inconsistent

HANDOFF's cost table: `A16-P` **flops@4K = 1.297×**, **flops@32K = 1.959×**.
§4 of the design doc: attention-score FLOPs as a share of 6ND are `L0 18.9%` / `A16-P 50.5%` at 32K,
difference "**31.6%**".

Reverse-engineering §4's convention (MEASURED against its own numbers): with `6N = 2.127 GFLOP/token`
(N = 354.4M, embeddings **included**), score FLOPs per attention layer per token = `2·T·d`:
`2 × 32768 × 1024 = 6.711e7`; `6 × 6.711e7 / 2.127e9 = 18.93%` ✓; `16 × 6.711e7 / 2.127e9 = 50.48%` ✓.
Under **that** convention the total-FLOP ratio is `(2.127+0.4027)/(2.127+1.0737) = 3.201/2.530 =
**1.265×**`, not 1.959×.

Under the convention that *does* roughly reproduce 1.959 (6N over **non-embedding** params +
`12·T·d` per attention layer): `B = 6 × 287,375,104 = 1.72425e9`;
@32K `(1.72425 + 6.4425)/(1.72425 + 2.41592) = 8.1661/4.1402 = **1.9724×**` (claim 1.959, 0.7% off);
but @4K the **same** convention gives `(1.72368+0.80531)/(1.72425+0.30199) = 2.52899/2.02624 =
**1.2484×**`, not 1.297× (3.9% off).

**These two cells cannot both be right.** Solve for the implied base-to-attention ratio `B/u`
(u = per-layer score FLOPs at 4K), assuming N is matched (it is, to 0.027%):
- from 1.297 @4K: `B + 16u = 1.297(B+6u)` ⇒ `8.218u = 0.297B` ⇒ **B = 27.67u**
- from 1.959 @32K: `B + 128u = 1.959(B+48u)` ⇒ `33.97u = 0.959B` ⇒ **B = 35.42u**

**No single linear FLOP model produces both.** One of the two cells is computed under a different
convention from the other. **Severity: MEDIUM-HIGH** — this column is the *entire* justification for
the "param-matched ≠ compute-matched" discipline, there is a **test asserting the gap**, and the
test will therefore lock in whichever number is wrong.

**Also flag:** §4's "31.6%" is a difference of **percentage points of 6ND**, and HANDOFF converts it
to "~32% apart in actual compute". As a *relative* compute difference it is
`0.316×2.127e9 / 2.530e9 = **26.6%**`, not 32%. Percentage points ≠ percent.

### 1.5 Other numeric disagreements

| # | claim | where | correct value | severity |
|---|---|---|---|---|
| a | "MLP is **69%** of the model" | design §1.1 (line ~106), HANDOFF trap #3 | **63.9%** at d=1024 (`226,492,416/354,483,968`). 69% is the **d=2048** figure (`805,306,368/1,170,340,608 = 68.8%`). Same class of error as §1.1's 6.27%. | LOW |
| b | "GQA mixer = **2.5d²**" | design §1.2 | true at d=2048 only. §1.2 *does* say "becomes **3.0d²** at d=1024" — internally consistent, but the 2.5d² figure appears unqualified in HANDOFF's "Verified sound" block. | LOW |
| c | "vocab 65536 **tied**" (§3.1) vs "untied embeddings" (brief / HANDOFF table) | §3.1 vs §2 of HANDOFF | **tied** — see §1.0 | LOW / high contagion |
| d | "conv state grows **7×**" retracted to **5×** | HANDOFF | `L_cache = k`, so k=15 vs k=3 = **5×** ✓ correct as retracted | resolved |
| e | `A16-P` "parameter-matched **within 0.03%**" vs "**1.000×**" in the same table | HANDOFF cost table | both fine (0.027% rounds to 1.000×) | none |
| f | "r=512 saves **zero** bytes at d=1024" | HANDOFF | `2dr ≥ d² ⟺ r ≥ d/2 = 512` ✓ — and note this means the **rank sweep {128, 256, 512} has a degenerate top rung**: r=512 is *exactly* break-even (`4dr = 2d²`), so `F-r512` **is** `L0` in parameter count. One of three P1 arms is a null arm by construction. | **MEDIUM — see §5.4** |

---

## 2. Statistical power — the plan is underpowered by one to two ORDERS OF MAGNITUDE

This is the most important section of this audit. Read the arithmetic, not the summary.

### 2.0 Method, and a validation of the method against the repo's own published table

Paired design, seed is the unit. Effect size `d_z = μ_δ / s_δ`. Required n at 80% power,
two-sided α=0.05, normal approximation:

```
n = ((z_0.975 + z_0.80) · s_δ / m)²  =  (2.8016 · s_δ / m)²
```

**Validation (this is not my formula, it is theirs):** `KDA/HANDOFF.md` publishes
σ_within = 48.4 pp and "seeds per arm, paired, 80% power": MDE 10 pp at ρ=0.5 ⇒ **n = 184**.
My formula: `(2.8016 × 48.4 / 10)² = 13.56² = 183.9`. **Reproduces to 0.05%.** So the arithmetic
below is the same arithmetic the sibling track already accepted.

For small n the normal approximation is optimistic, so I also give the exact paired-t result.

### 2.1 What n=2 actually means — the algebra of a two-seed t-test

At n=2 the paired t-statistic collapses to a closed form. With per-seed differences d₁, d₂:

```
d̄ = (d₁+d₂)/2      s_d = |d₁−d₂|/√2      t = d̄/(s_d/√2) = (d₁+d₂)/|d₁−d₂|
df = 1  ⇒  t_crit(0.975, 1) = 12.706
```

**Reject iff `|d₁+d₂| > 12.706 · |d₁−d₂|`.** Writing r = d₁/d₂ (same sign, r ≥ 1):

```
(r+1)/(r−1) > 12.706  ⇒  13.706 > 11.706 r  ⇒  r < 1.1709
```

> **At n=2, significance requires the two seeds to agree to within 17.1% of each other.**

With **Bonferroni** across the 11 arm-vs-`L0` comparisons in a 12-arm screen (α = 0.05/11 =
0.00455, df=1 ⇒ t_crit = 141.5), the requirement becomes **r < 1.0142 — the two seeds must agree to
within 1.4%.** Against a metric whose measured seed spread is 39 pp, that is not a test, it is a
lottery.

And without Bonferroni, 11 comparisons at α=0.05 gives family-wise error
`1 − 0.95¹¹ = 43.1%` — **the 12-arm × 2-seed screen is more likely than not to produce at least one
spurious "winner."**

### 2.2 Exact power at n=2 (I computed the non-central distribution by hand)

For n=2, write U = (d₁+d₂)/(σ√2) ~ N(√2·d_z, 1) and V = (d₁−d₂)/(σ√2) ~ N(0,1), independent.
Power = P(|U| > 12.706·|V|), evaluated as `2[Φ(a/√(1+b²)) − Φ₂(0, a/√(1+b²); ρ=b/√(1+b²))]`
with a = √2·d_z, b = 12.706 (ρ = 0.99692):

| true effect | power at n=2, α=0.05 |
|---|---:|
| d_z = 0 (null) | 5.0% (by construction) |
| **d_z = 1 (a full seed-SD)** | **9.0%** (Simpson cross-check: 9.28%) |
| d_z = 11 | 78% |
| **d_z ≈ 11.2 → 80% power** | **the MDE at n=2 is ~11 standard deviations** |

> **A real effect the size of one full seed-to-seed standard deviation is detected 9 times in 100 at
> n=2. The false-positive rate is 5 in 100. The screen is barely distinguishable from a coin that
> lands heads 9% of the time.**

MDE in SD units across the plan's stated seed counts (exact-t where it matters):

| n | df | MDE (× s_δ) |
|---:|---:|---:|
| **2** (budget table, Phase 3a) | 1 | **≈ 11.2** |
| 3 | 2 | 3.10 |
| **5** (§8 Phase 3a "5 paired seeds") | 4 | 1.66 |
| **8** (Phase 4 "≥8 fresh") | 7 | 1.15 |
| 43 (what KDA said the CE effect needs) | 42 | 0.44 |

### 2.3 Plugging in the plan's OWN measured variance

**MEASURED — and this is the plan's own number, at the plan's own chosen operating point.**
HANDOFF records the `N512_D64` MQAR seeds: **0.05 / 0.09 / 0.20 / 0.56 / 0.98**.

```
mean = 1.88/5 = 0.376
Σ(x−x̄)² = 0.106276+0.081796+0.030976+0.033856+0.364816 = 0.617720
s² = 0.617720/4 = 0.154430   ⇒   s = 0.39298  =  39.3 pp
```

The sibling KDA track independently measured **σ_within = 48.4 pp** on the same family of task.
Two independent measurements, same order. Paired s_δ = σ√(2(1−ρ)); I tabulate ρ ∈ {0, 0.5, 0.8}
(ρ=0.5 is what KDA assumed as its central case).

**Seeds required to detect an effect on MQAR accuracy, at σ = 39.3 pp:**

| MDE | ρ=0 (s_δ=55.6) | ρ=0.5 (s_δ=39.3) | ρ=0.8 (s_δ=24.9) |
|---|---:|---:|---:|
| 5 pp | 970 | **485** | 195 |
| 10 pp | 243 | **121** | 49 |
| 20 pp | 61 | **30** | 13 |
| 30 pp | 27 | **14** | 6 |

**And, run the other way — what the planned seed counts can actually see:**

| n | MDE at ρ=0.5 (s_δ=39.3 pp) | MDE at ρ=0.8 (s_δ=24.9 pp) |
|---:|---:|---:|
| **2** | **440 pp** | **279 pp** |
| 5 | 65 pp | 41 pp |
| 8 | 45 pp | 29 pp |
| 43 | 17 pp | 11 pp |

> ### 🔴 THE FINDING
> **At n=2 the minimum detectable effect on MQAR accuracy is ~440 percentage points. The metric's
> entire range is 100 percentage points. The MDE exceeds the range of the measurement by 4.4×.**
>
> **The 12-arm × 2-seed screen in the budget table cannot distinguish anything. Not "is weak." Cannot.
> There is no true effect expressible on this metric that it would detect at 80% power.**
>
> Even at **n=8** — the plan's *confirmation* seed count, three phases later — the MDE is **45 pp**.
> The largest recall gap in the entire hybrid literature that the design cites is Hymba's
> **20.75 pp**. **The plan's confirmation stage is underpowered for the largest effect it cites as
> motivation, by a factor of 2.2.**

### 2.4 The CE gate is already falsified by measurement — nobody has drawn this conclusion

§6.1: "The existing protocol's gate is CE non-inferiority at **+0.010 nats** … that margin is
reachable only if `s_δ ≲ 0.011` at n≥8. **Action: measure `s_δ` in the pilot.**"

Verify their threshold: one-sided non-inferiority, `n = ceil(((1.645+0.842)·s_δ/m)²)`.
At n=8: `m = 2.487·s_δ/√8 = 0.8793·s_δ`. Set m = 0.010 ⇒ **s_δ = 0.01137**. ✓ their 0.011.

**Now invert the two numbers the sibling track already published.** `KDA/HANDOFF.md` states
(a) "+0.0053 nats … needs n≈43 seeds" and (b) "At n=5 paired, detectable val-loss difference is
**~0.014 nats**". Both are statements about the *same* s_δ:

```
from (a):  s_δ = 0.0053 · √43 / 2.8016 = 0.0053 · 6.5574 / 2.8016 = 0.01241 nats
from (b):  s_δ = 0.0140 · √5  / 2.8016 = 0.0140 · 2.2361 / 2.8016 = 0.01117 nats
```

> **MEASURED (sibling track, LM val loss): s_δ ≈ 0.0112 – 0.0124 nats.
> REQUIRED for the gate at n=8: s_δ ≤ 0.01137.
> The measured value STRADDLES the requirement and its upper end EXCEEDS it by 9%.**

**The conclusion nobody has drawn:** the plan's central statistical remedy — "measure `s_δ` in the
pilot and then decide" — is asking a question that has already been answered, on this repo's own
hardware, at comparable scale, and the answer is *the gate is at best marginal and probably out of
reach.* The plan should **pre-register the CE gate as expected-inconclusive and demote it now**,
not spend a pilot re-discovering it. (Caveat, honestly: the KDA measurement was at ~100M scale and
~1B tokens; s_δ generally *shrinks* with scale and token budget, so 350M/2B could land lower. But
"could be better" is not a plan — and the plan currently has no fallback if it isn't.)

### 2.5 The remedy is itself underpowered — you cannot estimate s_δ from a small pilot

Phase 2's whole job is "measure `s_δ` and publish required n." **The plan never states how many
seeds the pilot runs.** This matters enormously, because required n scales as s_δ², so the
sampling error of the *variance estimate* is squared into the sample-size decision.

χ² confidence interval for σ from n seeds (95%):

| pilot n | df | 95% CI for σ | ⇒ 95% CI for required n (∝ σ²) |
|---:|---:|---|---|
| 3 | 2 | [0.52·s, 6.29·s] | **[0.27×, 39.6×]** the point estimate |
| 5 | 4 | [0.60·s, 2.88·s] | **[0.36×, 8.3×]** |
| 8 | 7 | [0.66·s, 2.04·s] | **[0.44×, 4.1×]** |
| 20 | 19 | [0.76·s, 1.46·s] | [0.58×, 2.1×] |

> **A 5-seed pilot returns a required-n estimate uncertain by a factor of 23.** If the pilot says
> "n=8 suffices," the true requirement could be n=66. The plan's gate ("publish required n per
> endpoint") is therefore **not decidable at the pilot's own seed count**, and the plan does not
> say what the pilot's seed count is. **Fix: pilot at n≥20 on ONE arm pair, or accept that the
> required-n number is decorative.**

### 2.6 Three seed counts for the same phase, in the same document

| location | Phase 3a spec |
|---|---|
| `docs/…design.md` §8 phase table | "5 paired seeds, ~2B tok" |
| `docs/…design.md` compute-budget table | "Rank \| **12 arms × 2 seeds** × **150M**/10B" |
| `docs/…design.md` §6.4 | "Screening at **5** paired seeds … confirmation uses **≥8** fresh" |
| `HANDOFF.md` §4 (Next steps) | "3a (P1): rank sweep … `N-narrow`, `S-shared`, `G-grouped`, `1G`" — **no seed count at all** |

**And the scales disagree too:** §3.1 froze **350M** as "a *scientific* choice" and §9 recommends
dropping the 750M stage — but the budget table still costs the screen at **150M** and the headline
at **750M**. **The compute budget table is stale with respect to the frozen decisions and should
not be used to plan anything.** Anyone reading it will provision a 150M screen for a 350M study.

Cost of fixing the seed count, at the budget table's own throughput model
(`6ND/(8×312 TFLOP/s×0.40 MFU)`, which reproduces its "350M/20B ≈ 12 h" row to 2%):

```
350M, 2B tokens: 6 × 3.545e8 × 2e9 / 9.98e14 = 4,262 s = 1.18 h per run on 8×A100
12 arms × 2 seeds = 24 runs =  28 h = 1.2 days
12 arms × 5 seeds = 60 runs =  71 h = 3.0 days
12 arms × 8 seeds = 96 runs = 113 h = 4.7 days
```

> **Going from n=2 to n=8 in Phase 3a costs +3.5 days out of a ~15-day program.** It is affordable.
> **It also still does not make MQAR a usable screening endpoint (MDE 45 pp).** So more seeds is
> necessary and not sufficient — see §5.6 for what the endpoint has to become instead.

### 2.7 Success-rate scoring at n=2 is mathematically incapable of significance

§6.2 mandates "success rate over seeds" as the MQAR endpoint. At n=2 per arm, a 2×2 Fisher exact
test with **perfect separation** (2/2 vs 0/2) has one-sided p = 1/C(4,2) = **1/6 = 0.167**.

> **No 2-vs-2 outcome, however extreme, can reach p < 0.05. The endpoint the plan designates as
> primary is provably unable to reject at the seed count the budget provisions.**

Minimum achievable p by n (perfect separation, one-sided Fisher): n=2 → 0.167; n=3 → 0.050
(exactly at the boundary); n=4 → 0.014; n=5 → 0.004.
**n=3 is the arithmetic floor for this endpoint and n=5 the practical one.**

### 2.8 An unnoticed ambiguity that changes all of the above

**The plan never states whether MQAR is (a) a synthetic task the arms are TRAINED on from scratch,
or (b) an in-context eval applied to the PRETRAINED arms.** The calibration harness is (a) — 4
layers, d=128, 8000 steps, trained on MQAR. Phase 1's gate is (a). But Phase 3's "retrieval
endpoints primary" reads like (b).

This is not pedantry. **The 39.3 pp variance is a property of (a)** — it is the bimodal
"did this tiny model find the recall circuit" lottery, driven by basin selection during training.
Under (b), the arm is a fixed pretrained checkpoint and the only randomness is task-instance
sampling, which averages down as `1/√(n_instances)` and can be driven arbitrarily low for free.

> **If MQAR is an eval on pretrained arms, the seed variance that dominates this entire audit may
> not apply — and the plan would be far better off than §2.3 says. If it is a from-scratch
> synthetic training task, §2.3 stands and the endpoint is dead.** The plan does not say which.
> **This single ambiguity is worth more than any other clarification in the document**, and it is
> resolvable in a sentence. HANDOFF's "Do NOT transfer the operating point to real `L0` … re-run
> the sweep on real `L0`" implies (a), which is the bad case.

---

## 1.4b Addendum — §4's FLOP percentage table uses a coefficient 6× too small

Following on from §1.4. §4's convention is provably `2·T·d` per attention layer per token
(reproduces its own cells: `6 × 2 × 4096 × 1024 / 2.127e9 = 2.37%` ✓ "2.4%";
`16 × 2 × 32768 × 1024 / 2.127e9 = 50.5%` ✓).

The correct **training** score-FLOP coefficient is **`12·T·d`**: QK^T is `T·d` MACs = `2Td` FLOPs per
query token, AV is another `2Td` ⇒ `4Td` forward, ×3 for fwd+bwd = **`12Td`**. `6N` on the same line
is already a fwd+bwd figure, so the two terms must use the same convention. §4 mixes them.

Corrected table (score FLOPs as a share of 6N, N = 354.4M):

| context T | `L0` (6 attn) — §4 says | corrected | `A16-P` (16 attn) — §4 says | corrected |
|---:|---:|---:|---:|---:|
| 4,096 | 2.4% | **14.2%** | 6.3% | **37.9%** |
| 32,768 | 18.9% | **113.6%** | 50.5% | **302.9%** |

**Relative compute gap at 32K:** `(2.1263+6.4425)/(2.1269+2.4159) = 8.5688/4.5428 = **1.886×**`, i.e.
**+89%**, against the doc's "~32% apart in actual compute".

> **The plan's own flagship methodological warning — "param-matched ≠ compute-matched" — understates
> the confound by ~6× in the percentage table and ~2.8× in the relative gap.** At 32K, attention
> score FLOPs *exceed* the entire dense-matmul budget for `L0` and are 3× it for `A16-P`. A
> parameter-matched `L0` vs `A16-P` at 32K is not a 32%-apart comparison; it is a **~1.9× compute**
> comparison. Any "same params, better loss at 32K" statement is dominated by the compute term.
>
> Silver lining: the cost table's `flops@32K = 1.959×` is close to my `1.886×` (the residual is the
> 1.297-vs-1.248 base-convention discrepancy from §1.4), so **the arm builder is roughly right and
> the prose table is badly wrong.** Fix the prose, re-derive the 4K cell, and re-check the committed
> test that "asserts this gap."

---

## 3. Gate audit — 9 gates, 2 real, 5 fail-open, 2 pass-only

A gate is REAL only if (a) it has a numeric threshold, (b) the threshold is decidable at the seed
count that phase provisions, and (c) something specific *stops* when it is not met. Anything else is
theatre.

| # | Phase | Gate as written | Verdict | Why |
|---|---|---|---|---|
| G1 | 0 | "Full forward ≡ chunked prefill ≡ one-token decode; parity vs `Lfm2ShortConv`; state constant in context length; counts match" | **REAL** — but **currently unpassable** | Binary, float-exact, already tested for prefill/parity/counts. **However `ShortConv` has NO incremental decode path**, so "≡ one-token decode" and "state constant in context length" cannot be evaluated at all. The live risk is not failure, it is **quiet redefinition of the gate to the 3 clauses that can be checked.** HANDOFF already lists 153/427 test passes and calls Phase 0 "✅ BOTH HARD BLOCKERS CLEARED" while this clause is unevaluated. |
| G2 | 0b | "If no rank wins, drop the latency claim from the headline now" | **REAL — and it FIRED** ✅ | The one unambiguous success in this plan. Numeric, pre-committed, cheap, and it produced a decision that cost the project its headline. **This is the template every other gate should copy.** |
| G3 | 1 | "Endpoint discriminates controls; bimodality characterized" | **FAIL-OPEN** | "Characterized" is satisfied by writing a sentence. "Discriminates controls" names no controls and no threshold. HANDOFF's own calibration write-up *corrects* the bimodality claim mid-flight ("holds at low load and BREAKS at high load") — i.e. the gate's premise changed and the gate still passed. |
| G4 | 2 | "Publish required n per endpoint; freeze one recipe per arm" | **PASS-ONLY** | Publishing a number is not a decision rule. **Nothing in the plan says what happens if the answer is n=485.** No branch, no descope, no abort. §2.5 also shows the pilot cannot estimate n to better than a factor of ~23 at its (unstated) seed count. |
| G5 | 3a | "Init-scale parity asserted at step 0; rank curve survives controls" | **half REAL / half FAIL-OPEN** | First clause is excellent — binary, step-0, already implemented, catches a documented 24–48× error. Second clause: "survives" is undefined; and §8's own kill rule says a flat curve + `N-narrow` matching is *"publish that"* — a **reframe, not a stop.** So no outcome of 3a halts anything. |
| G6 | 3b | "Resident KV halved **and** retrieval preserved; latency null reported as predicted; **must beat or tie `A-fewer3`**" | **FAIL-OPEN — the worst one** | See §3.1. "Preserved" and "tie" are *non-inferiority* criteria. Under §2's power, non-inferiority is **automatically satisfied**. Meanwhile "resident KV halved" is true by construction (arithmetic, not a measurement) and "latency null reported as predicted" **passes when the prediction is confirmed AND cannot fail if it isn't** — a null result that isn't null just becomes an interesting finding. Three clauses, none of which can stop the phase. |
| G7 | 3c | "Only proceed to routing if a wider span actually gains" | **REAL (conservative)** | The one gate where underpowering helps: low power ⇒ no gain detected ⇒ don't proceed. Literature prior says flat. It will fire correctly for the wrong reason. Keep it. Tighten by naming the metric and the margin. |
| G8 | 4 | "Pre-registered margins met, or report inconclusive" | **PASS-ONLY** | **The margins are not pre-registered anywhere in the document.** The only numeric margin that exists is the inherited CE +0.010 nats, which §2.4 shows is unreachable and §5.3 shows is vacuous. And "report inconclusive" is an outcome, not a gate — nothing downstream is conditioned on it. |
| G9 | 5 | "No 32K quality claim without a matched 32K training stage" | **REAL in form, FAIL-OPEN in practice** | Correct rule. But §5.5 shows the 32K stage as specified trains ~3% useful tokens, so it will be *run* (satisfying the letter) while providing almost no long-context supervision. **A gate satisfied by a stage that does nothing is fail-open.** |

**Score: 2 REAL (G2, G7), 1 REAL-but-unevaluable (G1), 5 FAIL-OPEN, 2 PASS-ONLY.**
Note the pattern: **the only gate that ever killed anything (G2) was a microbenchmark with a
pre-committed numeric threshold and a 2-day cost.** Every gate attached to a *training* result is
soft. That is exactly backwards from a budget-protection standpoint.

### 3.1 The structural defect: every "tie is acceptable" gate is unfalsifiable under low power

Six of the plan's decision criteria are non-inferiority statements — "preserved", "match", "tie",
"non-inferior", "no regression", "latency null":

- G6 "retrieval **preserved**"; G6 "must beat or **tie** `A-fewer3`"; G6 "latency **null** as predicted"
- CE **non-inferiority** at +0.010 nats (inherited protocol)
- §8 kill rule "if P1's rank curve is **flat** *and* `N-narrow` **matches** it"
- §5.2's prediction that P2's latency effect is **≈0 by construction**

> **A non-inferiority claim is confirmed by FAILING to reject. An underpowered study fails to reject
> almost always. Therefore, at n=2 (or n=5, or n=8 on a 39-pp-σ metric), EVERY ONE of these gates
> passes with probability ≈ 1 — including when the treatment is catastrophically worse.**
>
> Concretely: with s_δ = 39.3 pp at n=5, a treatment that destroys **60 pp** of recall accuracy has
> only a `1 − Φ(2.776 − 60/(39.3/√5)) ≈ 1 − Φ(2.776 − 3.414) = 74%` chance of being caught — so
> **1 in 4 catastrophic regressions is scored as "retrieval preserved."** At n=2 it is ~9%: **91% of
> catastrophic regressions pass the gate.**

**Required fix, and it is standard practice, not an invention:** every non-inferiority gate must
state an explicit **non-inferiority margin Δ** and be decided by whether the **upper bound of the
two-sided 95% CI** on the regression lies inside Δ — not by a p-value. Under that rule an
underpowered study yields a wide CI, the CI exceeds Δ, and the gate **fails**. That single change
converts all six gates from fail-open to fail-closed at zero compute cost.

---

## 4. Dependencies that are not done

Hour estimates are ASSUMED (my engineering judgement against the code as described in HANDOFF);
the blocked-claims column is INFERRED from the docs.

| # | Item | Est. hours | Downstream claims BLOCKED | Failure mode |
|---|---|---:|---|---|
| **D1** | **Incremental decode / conv-state path for `ShortConv`** (no `conv_state` cache exists) | **12–24 h** (cache class + `roll`/ring update + wire into `generation_module` + the 3-way equivalence test + `no_grad` on `copy_`) | Gate G1's decode clause; **the entire topology latency claim** (T≈4,121); P2's "latency ≈ 0" null; P1's 4.44% decode ceiling; every §5.2 memory-bucket measurement; Phase 5 systems | **LOUD** for measurement (nothing runs) but **SILENT for the write-up**: the topology claim gets published as a *bytes* claim with the word "decode" attached and no one notices there was never a timer. See §5.1. |
| **D2** | **Document isolation threaded through ATTENTION** (`ShortConv` already honors `cu_doc_lens` and is tested; attention is not) | **8–16 h** (thread `cu_seqlens` into the attention path + varlen kernel selection + a cross-arm assertion test) | **`L0` vs `A16-P` — the study's headline comparison**; every P2 arm; every mixed-mixer comparison | **SILENT AND TOTAL.** §3.3 states it: *"If document masking is on for the attention arms while the conv silently bleeds (or vice versa), every comparison is broken."* The model trains, the loss curve looks normal, the numbers are wrong. **Highest-severity item on this list.** Note the asymmetry is currently in the *worst* direction: the conv IS isolated, attention is NOT ⇒ attention-heavy arms (`A16-P`, `A-fewer*`) get free cross-document context the LIV arms are denied. **The topology gate is currently biased in favour of the control.** |
| **D3** | **Dolma2 document-length audit** | **2–4 h** (metadata scan; no GPU) | All 16K/32K claims; the extension-stage design; interacts fatally with D2 — see §5.5 | **SILENT.** The stage runs, tokens are consumed, and 97% of them teach nothing (§5.5). Cheapest item here by an order of magnitude and it gates the most expensive stage. **Do this first.** |
| **D4** | **μP not coordinate-checked** | **0 h to accept / 16–40 h to port+check** | Fair comparison for `N-narrow` (different `d_model`) and `A16-P` (different `ff`) | **SILENT** in principle — a mis-transferred LR is indistinguishable from an architecture effect. **But I think this is over-weighted.** `N-narrow` must recover 4.437% of parameters; since params scale between linearly and quadratically in `d`, the width change is bounded to **d ∈ [979, 1002]**, i.e. **≤4.4%**. `fan_in` init + the ladder's empirical LR formula is entirely adequate over a 4% width extrapolation. **Recommend: accept, 0 hours, and say so in the paper.** |
| **D5** | **The mixer branch is UNMERGED** (`agent/claude-01/liv-short-conv-mixer`, 2 commits, worktree) | **1–3 h** (rebase + CI + merge) | Nothing scientific; everything operationally | **LOUD** but it is a *bus-factor* risk: 1,472 lines, 55 tests, four documented traps, and the only copy is a local worktree on a machine HANDOFF says **"has died mid-run before."** Merge or push it today. |
| **D6** | **NOT ON THE LIST — 16 of 27 arms have no implementation** | **60–120 h** | **All of Phase 3b (P2)**; most of Phase 3c beyond widths | **SILENT scope loss.** See §5.4. |
| **D7** | **NOT ON THE LIST — per-rank LR retuning for P1** | **8 h per rank** (a small LR sweep) or a confound | The entire P1 rank curve | **SILENT.** §5.1 states published low-rank LR corrections *"disagree even in direction — 1.5-2×, 0.05-0.1×, and 0.5× all appear"* and prescribes `η_B/η_A = d/r` **plus** "re-tune base LR per rank." Re-tuning base LR per rank is **not in any phase, budget line, or gate.** This is a far larger LR risk than D4 and it is invisible in the plan. **A rank curve run at one LR is an LR curve, exactly as the init-scale trap §5.1 already warns about — and the plan caught the init version and missed the LR version.** |

**Ranked by (silence × blast radius):** D2 ≫ D7 > D3 > D1 > D6 > D5 > D4.

---

## 5. Conclusions already licensed by measurement that nobody has propagated

### 5.1 bytes ≠ time — and the topology claim is the SAME unmeasured roofline argument P1 was

**MEASURED, on L40S, jobs 1670883/1670884:** a 4× byte reduction produced an **8.2% slowdown**.
Dense hit 695 GB/s; the factorized path hit 161 GB/s. The roofline model **predicted a win and the
measurement returned the opposite sign.**

Now inventory the topology claim's evidentiary status:

| | P1 latency claim (pre-2026-07-31) | Topology claim (today) |
|---|---|---|
| argument type | bytes saved ⇒ time saved | bytes saved ⇒ time saved |
| byte arithmetic | exact ✓ | exact ✓ (§1.3) |
| latency measurement | **none**, then **run** | **none** |
| roofline outcome | predicted win | predicts 9.9% win @4K |
| measured outcome | **8.2% loss — sign flip** | **unmeasurable: blocked by D1** |

> **The topology claim is in precisely the state P1's latency claim was in the day before it died.**
> HANDOFF calls it *"the one efficiency claim that IS testable at trainable contexts"* and the design
> doc calls `L0 vs A16-P` *"a real, measurable systems comparison at 4K"* — but **nothing has measured
> it, and nothing CAN until `ShortConv` has a decode path (D1).** The project has one documented
> instance of its own roofline model returning the wrong sign, on the same hardware, in the same
> month, and has not applied that skepticism to its remaining bytes-only claim.

**In fairness — the mechanism probably does not transfer**, and I will say so rather than
overclaim. P1 died from **GEMV inefficiency at M=1, N=128**: a skinny matmul cannot saturate memory.
A paged KV read is a large contiguous streaming load that flash-decoding kernels *do* saturate. So
I expect the topology bytes→time conversion to be far better than P1's. **But "I expect" is what the
project said about P1 in the block immediately above the block that retracted it.**

**Two further unpropagated facts, pulling opposite ways:**
- **Against:** the project's only per-op decode profile (q4 ONNX, 128-token past) measures
  `GroupQueryAttention` at **1.5%** and `Conv` at **1.0%** of decode. Naively scaling attention 6→16
  layers and deleting 10 convs gives `+1.5%×(10/6) − 1.0% = **+1.5%** net` for `A16-P` at short
  context — an order of magnitude below the roofline's 9.9%, because at 128 tokens KV is negligible.
  Consistent (the roofline is a *4K* claim), but it shows how much of the 9.9% is extrapolation.
- **For:** the same profile is **q4**, and every crossover in the plan assumes **bf16**. Quantized
  weights shrink the denominator ~3.3× and move the 10% crossover to **T ≈ 1,236** — well inside 4K.
  (Credited: this is the orchestrator's A.4 finding; it is the strongest un-exploited fact in the
  dossier and it is *not* in the design doc.)

**Action (cheap, high value):** D1 is 12–24 h. Do it, then measure `L0` vs `A16-P` decode latency at
T ∈ {2K, 4K, 8K, 16K, 32K} on the L40S using the *identical* CUDA-graph methodology that killed P1.
**Do not publish the traffic table as a systems result until a timer has confirmed it.**

### 5.2 The spectra result licenses a MUCH bigger experiment than the one being run

**MEASURED (`probes/spectra_v2.py`, 32,768 tokens, `rank(Σ_x)=1024`):** activation-aware rank drops
~36% for the gates (493.3) **and identically for the value stream (507.8)**, while `out_proj` and a
random Gaussian collapse *less* (0.784 ratio). The stated explanation is that **all three `in_proj`
row-blocks read the same `x`**, so the collapse is a property of `Σ_x` — the input distribution —
not of gates.

The plan draws one conclusion ("the premise is falsified; say *gates tolerate* low rank") and stops.
**Two further conclusions follow immediately and neither is in the plan:**

**(a) It applies to every projection that reads the residual stream — including 63.9% of the model.**
If low-rank tolerance is a property of `Σ_x`, then the SwiGLU **`gate_proj` and `up_proj`** — which
read the same `x` — inherit it. The plan factorizes:

| what P1 factorizes | share of model | max saving at r=128 |
|---|---:|---:|
| 10 LIV gate projections (`2d²` each) | **5.92%** | 4.44% |
| 16 SwiGLU up+gate projections (`2·d·ff` each) | **42.6%** | up to ~34% |

> **The measurement licenses attacking 42.6% of the model and the plan attacks 5.92%, because the
> brainlift said "gates."** This is the single largest missed opportunity I found. It is also
> *cheaper* to test than P1 (no gate-variance-parity trap, no multiplicative path). Yes, low-rank
> MLPs are well-trodden — but the plan's *own novelty argument* is that STAR's genome has no rank
> field, and that argument covers both.

**(b) It predicts P1's quality result is a NULL, and a null at 4.44% params is not a paper.**
If any `d→d` map reading this `x` tolerates rank 128, then `F-r128` will match `L0`, `N-narrow` will
also match `L0`, and the outcome is §8's own kill rule: *"spend the parameters wherever you like."*
That is a true and honest finding — **but the plan budgets ~12 arms × multiple seeds × 2B tokens to
produce it**, and it is already the modal prediction of the project's own measurement.

**(c) It makes the whole P1 arm a study of the input distribution.** The correct control is therefore
not `N-narrow` alone but **a `Σ_x`-matched decoy**: factorize a projection that reads a *different*
input (e.g. `out_proj`, which the probe measured as collapsing *less*) at the same rank. If `out_proj`
at r=128 hurts and gate at r=128 doesn't, the result is about `Σ_x` and is *interesting*. If both are
fine, the result is "350M models at 2B tokens are over-parameterized," which is not about LIV at all.
**That decoy arm costs one run and is not in the plan.**

### 5.3 The perplexity gates are not merely underpowered — they are VACUOUS BY MARGIN

This is independent of seeds and cannot be fixed by buying more of them.

**MEASURED (Mamba-2 Table 2, cited in §6.0):** the entire published attention-ratio basin — 2 to 11
attention blocks, 4% to 23% attention — spans **0.06 ppl**, at ppl ≈ 8.3.

Convert to nats, which is what the gate is denominated in:

```
Δ(nats) = ln(ppl₂) − ln(ppl₁) = ln(8.36/8.30) = 0.007202 nats
  (at ppl 15: 0.00399 nats;  at ppl 20: 0.00300 nats — smaller at every realistic ppl)
```

> ### 🔴 The CE non-inferiority margin is +0.010 nats. The **entire span of the phenomenon** is
> **0.0030 – 0.0072 nats**. **The margin is 1.4× to 3.3× WIDER than the full range of architectural
> variation it is supposed to police.**
>
> **A gate whose acceptance region strictly contains every possible outcome accepts everything.**
> It would declare an *all-conv, zero-attention* model non-inferior to a *transformer*. This is not
> a power problem — at infinite seeds it still passes. **It is vacuous by construction, and no one
> has said so.**

Answering the assignment's question directly: **yes — every perplexity-based gate in the plan is
useless.** G4, G8, and the inherited CE gate are all denominated in a currency whose total dynamic
range across the architectures under test is smaller than the gate's own tolerance. **Delete them or
re-denominate at ≤0.002 nats** — and if you re-denominate, §2.4 says you need n ≈ ((2.487×0.0124)/
0.002)² ≈ **238 seeds**, which settles it: **drop CE as a gate entirely.**

**Constructive counterpoint — AR-Hits could actually rescue a likelihood endpoint, and the plan
undersells it.** Zoology: 82% of a 2.1-ppl gap concentrates in 6.4% of tokens. Effect amplification
on the slice = `0.82/0.064 = 12.8×`; standard error inflation from the smaller slice =
`1/√0.064 = 3.95×`. **Net SNR gain ≈ 3.2×**, which divides required n by ~10 (n=43 → n≈4.3).
**This is the plan's one credible path to a powered likelihood endpoint and §6.1 files it under
"highest value-per-GPU-hour" without ever computing the number.** Compute it, state the margin **in
AR-slice nats**, and make *that* the gate. (Caveat, honest: the 3.2× assumes seed variance scales
with slice size; if the between-seed component is common-mode it will not shrink and the gain is
smaller. Measure it in the pilot — this is what the pilot should actually be for.)

### 5.4 The "controls" are a set of alternatives that are ALL expected to win

Assembled from the plan's own text, not my speculation:

| proposal | its own mandatory control | what the plan says about it |
|---|---|---|
| **P1** | `N-narrow` | HANDOFF §4: *"just build a narrower model"*; §8 kill rule pre-writes the outcome where it ties |
| **P1** | `G-grouped` | **MEASURED +15.3% faster** at identical cost; *"in the searched space"* while P1 is not |
| **P1** | `S-shared` | *"STAR's own evolutionarily-selected incumbent"* |
| **P2** | `A-fewer3` | *"**mandatory and it is the strongest competitor** … matches CLA2's capacity **and halves read bandwidth**, which CLA structurally cannot"*; **MEASURED 0.717× FLOPs at 32K** |
| **P2** | `SWA` | *"Hymba shows SWA and CLA are largely substitutes … if SWA gets the same win more simply, that is the answer"* |
| **P3** | `D-dyn` | *"**expected to beat** the constrained router"* |
| **P3** | width sweep | *"published width sweeps flat past k=3"*; prior on the router **20-25%** |

> **Every one of the three proposals has at least one control that the plan itself predicts will
> beat it. The plan is not a treatment study with controls; it is a comparison of seven alternatives
> in which the three brainlift proposals are the underdogs.**
>
> This is *scientifically fine* and *rhetorically fatal* if the arm structure keeps calling them
> "controls." Decision #8 already reframed the **prose** ("we measured what Liquid never ablated")
> but **the arm taxonomy never followed.** Retitle: *"How should a fixed parameter / cache / span
> budget be spent in a short-conv hybrid?"* — with the three proposals as **entrants**, not the
> thesis. Then a clean sweep by `N-narrow` + `A-fewer3` + `k3` is the **headline result**, not a
> triple null.

**And a hard arithmetic constraint nobody has stated: at the frozen geometry there is no
systems-viable rank.** §5.1 advises *"{256, 512} are the systems-viable ranks; treat 128 as an
aggressive probe."* At d=1024:

| r | saving/LIV layer `2d²−4dr` | × 10 | % of 354,483,968 |
|---:|---:|---:|---:|
| 64 | 1,835,008 | 18,350,080 | **5.176%** |
| 128 | 1,572,864 | 15,728,640 | **4.437%** |
| 256 | 1,048,576 | 10,485,760 | **2.958%** |
| **512** | **0** | **0** | **0.000%** |

> **§5.1's recommended "systems-viable" ranks save 2.96% and 0.00% of the model. That advice was
> written at d=2048 and does not survive the freeze to d=1024.** `F-r512` is a parameter-identical
> copy of `L0` with a rank constraint — a legitimate quality datapoint, but it must never appear in
> a parameter-efficiency table, and **12-arm × 2-seed budgets should not be spent on a rung that is
> arithmetically guaranteed to save nothing.**

### 5.5 The 32K headline: compute is NOT the blocker — data and document isolation are

HANDOFF: *"no 32K quality claim without a matched 32K training stage."* Is that stage affordable?

**Compute — yes, easily.** 3% of a 20B-token budget = 600M tokens at 32K, on 8×A100 @ 40% MFU:
```
dense   6ND       = 6 × 3.545e8 × 6e8                     = 1.276e18 FLOP
attn    12·T·d·L  = 6 layers × 12 × 32768 × 1024 = 2.416e9 /token × 6e8 = 1.450e18 FLOP
total   2.726e18 / (8 × 312e12 × 0.40 = 9.98e14 FLOP/s)   = 2,731 s ≈ 0.76 h
```
**Under one hour per arm.** The budget table's "~2 days" for long-context is generous by ~50×.
**The 32K stage is not budget-constrained. Whoever framed it as a budget question was wrong.**

**Data — no, and this is the real blocker, and it is UNMEASURED (D3).** The only doc-length data the
project has is FineWeb-Edu, not the actual corpus: **30.7% of tokens in docs >4K, 8.4% >16K,
**3.1% >32K**.

**And now the interaction nobody has drawn — D2 and the 32K stage are in direct conflict:**

> **Document isolation and long-context training are mutually destructive under packing.** §3.3
> mandates document-isolated packing (`cu_seqlens` through convs *and* attention) because bleed is
> *"a confound with teeth."* But a document-isolated 32K sequence packed from ~50 median-622-token
> documents contains **no dependency longer than the longest document in it.** The attention mask
> forbids it. **The model sees 32,768 positions and learns nothing beyond ~622 tokens of range.**
>
> Upper bound on useful long-range supervision in the extension stage: **only the ~3.1% of tokens
> living in docs ≥32K** can teach 32K-range dependency. So a 600M-token 32K stage delivers at most
> **~18.6M tokens** of genuine long-range signal — **0.09% of the 20B budget.**

**Consequence:** either (i) accept that the extension stage teaches position-embedding range
adaptation but not long-range dependency, and say so; or (ii) deliberately **oversample long
documents** in the extension mix (standard practice, e.g. length-stratified sampling) and report the
mix. **Option (ii) is not in the plan.** As written, G9 is satisfied by a stage that does almost
nothing — a fail-open gate (§3, G9).

**Separate the two topology claims, because they have completely different blockers:**

| claim | needs | blocked by | verdict |
|---|---|---|---|
| **topology SYSTEMS** (20 KiB/token, 10% @ T≈4,121) | a decode timer at T ∈ {2K…32K}. **Training context is irrelevant — you can decode to 32K from a 4K-trained model; bytes and µs don't care that the quality is bad.** | **D1 only (12–24 h)** | ✅ **demonstrable, cheaply, and it is the project's best remaining systems result** |
| **topology QUALITY at 32K** | matched 32K stage with real long documents | D2 + D3 + a long-doc-oversampled mix | ⚠️ reachable only with mitigation (ii) |

> **Answering the assignment: the 32K headline is reachable on compute and NOT reachable on data as
> currently specified. But the topology claim does not need it** — its systems half is demonstrable
> at 4K training for the cost of D1. **Publish the systems claim from a decode measurement and stop
> tying it to the 32K quality stage.**

### 5.6 Two smaller unpropagated conclusions

- **The MQAR calibration measured the very variance that kills the plan and was read as a
  calibration success.** `N512_D64` was chosen *because* it is graded and off-ceiling — the right
  instinct. But its five scores (0.05/0.09/0.20/0.56/0.98) **are** s = 39.3 pp (§2.3). **The
  operating-point selection and the power calculation used the same five numbers, and only the first
  was performed.** HANDOFF even says *"at high-load rungs report success rate AND median accuracy"* —
  a reporting fix for a variance the plan never priced.
- **`grouped` retains 0.130 energy — identical to a random mask — and the plan still schedules it as
  a full training arm.** That is defensible (the caveat about Eckart-Young favouring low-rank is
  correct and honest). But if `grouped` at 0.130 trains to parity with `L0`, **the retained-energy
  metric is falsified as a predictor** and every conclusion derived from it — including P1's
  surviving 92.6% hook — loses its evidentiary basis. **The plan should pre-register that
  implication** rather than discovering it. It is a genuine two-way bet and currently only the
  "grouped loses" branch is written down.

---

## 5.7 Arm inventory — Phase 3a cannot run as specified today

Counting arms declared in design-doc §4 against the 11 built in `liv_arms.py`:

| Tier | declared in §4 | built |
|---|---:|---|
| A | 2 (`L0`, `A16-P`) | **both** ✓ |
| B (P1) | 8 (`F-r64/128/256/512`, `N-narrow`, `S-shared`, `G-grouped`, `1G`) | 4 (`F-r128`, `F-r256`, `G-grouped`, `N-narrow`) |
| C (P2) | 8 (`C-near`, `C-far`, `C-all3`, `Q-mqa`, `A-fewer3`, `A-fewer`, `SWA`, `MLA`) | **2** (`Q-mqa`, `A-fewer3`) |
| D (P3) | 9 (`k5/k9/k15`, `L1b-schedule`, `M2-fixed`, `M-fixed`, `M-router`, `D-dyn`, `RepVGG`) | 3 (`W-k5/k9/k15`) |
| **total** | **27** | **11** |

> **Phase 3a as written is `{128,256,512}` + `N-narrow`, `S-shared`, `G-grouped`, `1G` = 7 arms.
> Three of them — `F-r512`, `S-shared`, `1G` — do not exist.** And of the six **cross-layer-sharing**
> arms that constitute P2's entire mechanism (`C-near`, `C-far`, `C-all3`), **zero are built** — the
> two P2 arms that exist (`Q-mqa`, `A-fewer3`) are both *competitors*, not the proposal.
>
> **P2 currently has a competitor set and no treatment.** The `kv_reuse_group` plumbing described in
> §4 (config field, index validation, `k_proj`/`v_proj` guards, `UserDict` for FSDP2, producer-norm,
> post-rotary K sharing, last-consumer eviction) is a **specification, not code.** Estimate 24–40 h
> for the P2 mechanism alone; 60–120 h for the full 16. **Neither figure appears in the plan's
> budget, which prices GPU-hours only and assumes every arm exists.**

---

## 6. Ranked risks

Probability = my calibrated estimate that it bites if the plan runs as written. Cost = damage to the
deliverable. Score = P × C on a 1–5 scale each.

| # | Risk | P | C | score | Mitigation (and cost) |
|---:|---|:-:|:-:|:-:|---|
| **R1** | **Phase 3a at n=2 selects noise.** MDE ≈ 11 s_δ; family-wise error 43% over 11 comparisons; the "winner" is a coin flip and Phase 4 then spends its budget confirming it (§2.1–2.3) | **5** | **5** | **25** | Delete the 2-seed row. Screen at **n≥5** (+3.5 days, §2.6) **and** switch the screening endpoint to a low-variance one (AR-slice likelihood, §5.3) — seeds alone are insufficient. **Or** cut to ≤5 arms at n=5 for the same compute. |
| **R2** | **Attention lacks document isolation while the conv has it (D2).** Silently biases the headline `L0` vs `A16-P` **in favour of the control**, and every P2 arm | **5** | **5** | **25** | 8–16 h. Thread `cu_seqlens` through attention; add a cross-arm test asserting *both* mixer types match independent per-document forwards. **Blocker — do before any training.** |
| **R3** | **Every non-inferiority gate passes automatically** (§3.1). ~91% of catastrophic regressions pass at n=2; 26% at n=5 | **5** | 4 | **20** | Zero compute. Rewrite all six as **CI-upper-bound vs explicit margin Δ**. Converts fail-open → fail-closed. |
| **R4** | **CE / perplexity gates are vacuous by margin** — +0.010 nats vs a 0.003–0.007 nat total phenomenon (§5.3) | **5** | 4 | **20** | Delete the CE gate. Re-denominate in **AR-slice** nats (~3.2× SNR, §5.3) and measure whether that gain is real in the pilot. |
| **R5** | **P2 has no implementation** — 0 of 3 sharing arms built, ~24–40 h unbudgeted (§5.7) | 4 | **5** | **20** | Decide now: build it, or cut P2 to a literature+arithmetic section. **Do not discover this in week 3.** |
| **R6** | **All three proposals lose to their own controls** (§5.4) and the write-up still frames them as the thesis | 4 | 4 | 16 | Retitle to a budget-allocation study **before** results land. Decision #8 did this in prose; do it in the arm taxonomy. |
| **R7** | **32K headline unreachable on data** — doc-isolated packing + a short-doc corpus ⇒ ≤0.09% of the budget carries 32K-range signal (§5.5) | 4 | 4 | 16 | 2–4 h: run D3. Then either oversample long docs in the extension mix, or **decouple** — publish the topology *systems* claim (needs only D1) and drop the 32K *quality* claim. |
| **R8** | **The topology claim is an unmeasured roofline** — the same argument form that already returned the wrong sign on this hardware (§5.1) | 3 | **5** | 15 | 12–24 h (D1) + one L40S job. Measure decode latency `L0` vs `A16-P` with the CUDA-graph methodology that killed P1. **Highest value-per-hour item in this audit.** |
| **R9** | **Per-rank LR is unbudgeted (D7)** — the rank curve is an LR curve, the exact failure the plan caught for init and missed for LR | 4 | 3 | 12 | 8 h/rank, or fix `η_B/η_A = d/r` **and** pre-register that base LR is held constant, naming it as a limitation. |
| **R10** | **`6.27%` / `44%` / `69%` / `2.5d²` — d=2048 numbers in a d=1024 design (§1.1, §1.5)** | **5** | 2 | 10 | 1 h. Grep the dossier for every d=2048-era figure. The team caught this once (4.72 µs) and never swept. |
| **R11** | **The FLOP tables disagree with each other and with the correct `12·T·d` coefficient (§1.4, §1.4b)** — and a committed test locks one in | 4 | 3 | 12 | 2 h. Fix the coefficient, re-derive both cells, re-check the test. |
| **R12** | **The pilot cannot estimate `s_δ`** to better than ~23× at an unstated seed count (§2.5) | 4 | 3 | 12 | Pilot **one** arm pair at n≥20, not all arms at n=3. Same or lower cost. |
| **R13** | **`F-r512` is a null arm** — saves exactly 0 params at d=1024 (§5.4) | **5** | 2 | 10 | Replace with `F-r64` (5.18%). Free. |
| **R14** | **Mixer branch unmerged on a worktree** on a machine that has died mid-run (D5) | 3 | 4 | 12 | 1–3 h. Merge or push **today**. |
| **R15** | **Phase 0 declared complete with an unevaluable gate clause** (G1's decode/state clauses, blocked by D1) | 4 | 2 | 8 | Mark G1 **PARTIAL** in HANDOFF. Do not let "✅ BOTH HARD BLOCKERS CLEARED" stand for a 5-clause gate with 2 clauses unchecked. |
| **R16** | **MQAR train-vs-eval ambiguity** (§2.8) — decides whether §2's variance applies at all | 3 | 3 | 9 | One sentence. **Do this first — it is free and it changes the interpretation of this entire audit.** |
| **R17** | **Budget table is stale** (150M screen / 750M headline vs a frozen 350M study) | **5** | 2 | 10 | 1 h. Re-cost at 350M. My §2.6 numbers are a start. |
| **R18** | **μP not coordinate-checked (D4)** | 2 | 2 | 4 | **Accept.** `N-narrow`'s width moves ≤4.4%; `fan_in` + the ladder formula is adequate. State the limitation. |

### The three things to do before anything else (total ≈ 30 engineering hours, no GPU-days)

1. **D2 — document isolation through attention (8–16 h).** Without it the headline comparison is
   silently biased toward the control. Everything else is wasted until this is true.
2. **D1 + one L40S decode job (12–24 h + 1 h).** Converts the topology claim from an unmeasured
   roofline into the project's strongest measured systems result — or kills it, cheaply, the way
   G2 killed P1's latency claim. **This project's one unambiguous success was a cheap microbenchmark
   with a pre-committed threshold. Do that again.**
3. **Rewrite the gates (0 h compute).** Every non-inferiority gate as CI-upper-bound-vs-Δ; delete the
   CE gate; delete the 2-seed row; replace `F-r512` with `F-r64`; state whether MQAR is train or eval.

### What I could NOT break

Stated so the audit is calibrated and not merely negative:

- **The arm-builder parameter ledger is exact.** Six independent checks (§1.2), including two I
  reconstructed from geometry rather than from the doc. The two-stage `N-narrow` solve to 0.0145% is
  genuinely careful engineering.
- **Every KV-traffic byte number is exact** (§1.3), including the T ≈ 4,121 crossover to 1 part in
  4,000. The *bytes* are not in question; only the bytes→time step is (§5.1).
- **The P1 latency retraction is exemplary.** Pre-committed threshold, two jobs, ≤0.3% spread,
  profiler-measured kernel counts matching prediction exactly, an iso-byte control that identified
  the *mechanism* (GEMV inefficiency, not launch overhead) and corrected the project's own prior
  explanation. **This is the standard the rest of the plan should be held to, and it is why the
  criticisms above are worth making rather than a reason to distrust the team.**
- **The init-scale trap (G5 clause 1)** is the best gate in the document: it identifies a 24–48×
  monotone-in-r error that would have produced a smooth, plausible, entirely spurious rank curve,
  and it is asserted at step 0 for ~zero cost. The only complaint is that its LR analogue (D7/R9)
  was not caught by the same reasoning.
