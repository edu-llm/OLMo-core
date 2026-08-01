# 06 — Claim 4 verification: CE gate vacuity + fail-open gates

**Verifier:** verification agent 06. **Date:** 2026-08-01. **Posture:** independent re-derivation.
**Constraint honoured:** no code executed on the local Mac. All arithmetic run on Stanford FarmShare
(`login.farmshare.stanford.edu`, `/scratch/users/ericrcwu/kda/venv/bin/python`, numpy 2.5.1).
**scipy is NOT installed there** — I hand-rolled the t-CDF via a continued-fraction incomplete beta
(`betacf`/`betainc`) and validated it against published t-tables to 5 sf
(df=1→12.7062, df=4→2.7764, df=7→2.3646, df=42→2.0181, all exact). Non-central t power was computed
by numerical integration over the chi distribution of `s` (20k–40k-point trapezoid), not approximated.

Scripts left on FarmShare at `/tmp/ericrcwu_claim4.py`, `/tmp/ericrcwu_c4.py`, `/tmp/ericrcwu_c4c.py`.

---

## BOTTOM LINE (full reasoning below)

| Claim | Verdict |
|---|---|
| **(a) "The CE gate is VACUOUS BY MARGIN"** | **CONFIRMED on the arithmetic** — margin 0.010 nats vs basin 0.00724 nats (1.38×). But the *rhetoric* is overstated two ways: (i) "it would declare an all-conv model non-inferior to a transformer" is TRUE ONLY against Mamba-2's Transformer++ row (0.00926 nats < 0.010) and is FALSE against the more relevant pure-SSM-vs-best-hybrid contrast (0.0403 nats = 4× the margin); (ii) "every perplexity gate in the plan is useless" is true on the merits but **low-consequence**, because §6.1 has *already* demoted CE from primary and line 1136 pre-commits to "inconclusive." See §2.4. |
| **(b) "5 of 9 gates are FAIL-OPEN"** | **SUBSTANCE CONFIRMED, DENOMINATOR WRONG.** There are **9 gate *cells*** in §8's table but they contain **~20 separately-decidable clauses**; "5 of 9" counts cells, and the audit's own §3.1 says **six** criteria are non-inferiority. Both "5" and "9" are defensible-but-soft counts. The *quantitative* sub-claims: n=2 ratio 1.171 ✅ EXACT (1.170850); Fisher 2v2 p=1/6 ✅ EXACT; FWER 43% ✅ EXACT (43.12%); **n=5 "1 in 4 pass" ≈ reproduces (I get 27.5%, they get 26%)**; **n=2 "91% pass" DOES NOT REPRODUCE — I get 86.5%, and their own stated formula gives 100%, not 91%.** See §4. |

---

## 1. Gate inventory — verbatim, with line numbers

**Source:** `/Users/ericwu/Developer/Capstone_LLM/docs/liv-brainlift-experiment-design.md` §8 phase
table, lines 1373–1383. This is the *only* place in the document that has a column literally headed
"Gate to pass". There are **9 rows** → the risk audit's denominator of 9 is the count of *table
rows*, which is a fair reading of "gates" but is not the count of decidable propositions.

| # | Line | Phase | Gate to pass (VERBATIM) | Type |
|---|---:|---|---|---|
| G1 | 1375 | 0 | "Full forward ≡ chunked prefill ≡ one-token decode; parity vs `Lfm2ShortConv`; state constant in context length; counts match" | **superiority/equality — fail-CLOSED** (4 clauses, all binary/float-exact) |
| G2 | 1376 | 1 | "Endpoint discriminates controls; bimodality characterized" | **non-inferiority-ish / undefined — FAIL-OPEN** (no threshold, no named controls) |
| G3 | 1377 | 2 | "Publish required n per endpoint; freeze one recipe per arm" | **PASS-ONLY** (an action, not a criterion) |
| G4 | 1378 | 0b | "If no rank wins, drop the latency claim from the headline now rather than defending it later. Cheap, and it decides how P1 is framed" | **superiority — fail-CLOSED** ✅ (and it FIRED) |
| G5 | 1379 | 3a | "Init-scale parity asserted at step 0; rank curve survives controls" | **split**: clause 1 fail-CLOSED (binary, step-0); clause 2 **FAIL-OPEN** ("survives" undefined) |
| G6 | 1380 | 3b | "Resident KV halved **and** retrieval preserved; latency null reported as predicted; must beat or tie `A-fewer3` to be worth anything" | **FAIL-OPEN ×3** — "preserved", "null … as predicted", "tie" are all non-inferiority; "KV halved" is true by construction (arithmetic, not measurement) |
| G7 | 1381 | 3c | "Only proceed to routing if a wider span actually gains" | **superiority — fail-CLOSED** ✅ (requires a *gain*; low power ⇒ correctly does not proceed) |
| G8 | 1382 | 4 | "Pre-registered margins met, or report inconclusive" | **PASS-ONLY** — the "or" makes it unfailable; and the margins are not pre-registered in this document |
| G9 | 1383 | 5 | "No 32K quality claim without matched 32K training" | **fail-CLOSED in form**, fail-open in practice (a stage that runs but teaches nothing satisfies it) |

**NOTE — the risk audit's own G-numbering is OFF BY ONE relative to the document order.** The audit
labels the Phase-0b microbenchmark gate "G2" and the Phase-1 calibration gate "G3", i.e. it inserts
0b in table order rather than document order (the doc lists 0b *after* Phase 2, at line 1378). My
numbering above follows the document's literal line order. **The substance is identical; only the
labels differ.** Anyone cross-referencing "G2 is the one that fired" against my table should read
"G4."

**Also in scope but NOT in §8's table** — additional decision criteria the audit correctly counts:
- **Line 1133** (§6.1): "The existing protocol's gate is **CE non-inferiority at +0.010 nats**."
- **Lines 1409–1412** (kill rules, stated in advance): "if P1's rank curve is **flat** *and*
  `N-narrow` **matches** it … publish that"; "If P2 **preserves** retrieval, report a capacity win
  and an **explicit latency null**"; "If `k15` **doesn't beat** `k3`, drop P3."
- **Line 1257–1258** (§6.4): "Screening at 5 paired seeds selects one configuration; confirmation
  uses ≥8 *fresh* paired seeds never used in selection."

> **DENOMINATOR FINDING:** "9 gates" = 9 *table rows*. Counting decidable clauses gives ~20.
> Counting *non-inferiority* criteria gives **6** (which is what the audit's own §3.1 says:
> G6×3 + CE + kill-rule-flat/matches + latency-null). So the headline "**5** of **9**" and the body
> "**six** criteria" are two different counts of two different things, in the same document.
> **Neither is wrong; the pairing is sloppy.** The substance — that the majority of the plan's
> training-attached gates cannot fail — reproduces.

---

## 2. The CE margin: is it real, and is it vacuous?

### 2.1 The margin exists, verbatim

`docs/liv-brainlift-experiment-design.md:1133`:

> "The existing protocol's gate is CE non-inferiority at **+0.010 nats**. With paired seeds and
> `n = ceil(((1.645+0.842)·s_δ/m)²)`, that margin is reachable only if `s_δ ≲ 0.011` at n≥8."

**Provenance traced.** "The existing protocol" is
`/Users/ericwu/Developer/Capstone_LLM/docs/liv-kda-gqa-sub500m-experiment.md`, which states it
**twice**, and — importantly — in the *correct* CI-upper-bound form the risk audit demands as its fix:

- line 386: `U95(CE_L0 - CE_A16-P) <= +0.010 nats` (Gate 1, topology survival)
- line 396: `U95(CE_K2 - CE_L0-P) <= +0.010 nats` (Gate 2, KDA incremental value)

> **FINDING the risk audit missed (favourable to the plan):** the *inherited* protocol already
> specifies the gate as a **one-sided 95% upper confidence bound vs an explicit margin** — which is
> exactly the "required fix" the risk audit proposes in §3.1 ("state an explicit Δ and decide by
> whether the CI upper bound lies inside Δ … converts fail-open → fail-closed"). **The fix is not
> an invention; it is already written down in the parent document and was lost in transcription
> into the brainlift design doc**, which states the margin as a bare number without the U95 wrapper.
> That is a *transcription* defect, not a design defect, and it is cheaper to fix than the audit
> implies. Same for the recall gate: the parent doc uses `LCB(recall composite) >= -2.0 points`
> (line 387) and `>= +2.0 points` (line 397) — again the correct CI form.

Direction and metric confirmed: **cross-entropy, in nats, one-sided, "treatment minus control ≤
+0.010"** — i.e. the treatment is allowed to be up to 0.010 nats *worse*.

### 2.2 ppl ↔ nats: my own re-derivation

`ppl = exp(CE)` ⇒ `ΔCE = ln(ppl₂/ppl₁)`. **Primary source located and read directly:**
`/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/06_baselines_infra.md:202-215`
quotes Mamba-2 (arXiv 2405.21060) **Table 2** verbatim — 350M, 48 layers, 7B tokens, Pile, GPT-2
tokenizer, "same number of parameters, same hyperparameters, same training and validation set":

| Num. attn blocks | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 9 | 11 | 15 | 24 | Tf++ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Perplexity** | 8.60 | 8.38 | 8.32 | 8.29 | 8.29 | 8.28 | **8.26** | 8.27 | 8.28 | 8.30 | 8.34 | 8.50 | 8.68 |

**I verified the 0.06 ppl figure myself from these numbers:** blocks 2–11 = {8.32, 8.29, 8.29, 8.28,
8.26, 8.27, 8.28, 8.30}; max − min = 8.32 − 8.26 = **0.06 ppl exactly**. ✅ The figure is sourced from
a primary paper (not a secondary claim) and **the baseline ppl is ≈ 8.3**, at the low end of the
range the assignment flagged.

My conversions (FarmShare):

```
ln(8.32/8.26) = 0.007238 nats          ← the actual basin, at the actual baseline
risk audit used ln(8.36/8.30) = 0.007203   (0.5% low — they shifted both endpoints
                                             by +0.04; harmless, same answer)
0.06 ppl at baseline 10 → 0.005982 nats
0.06 ppl at baseline 15 → 0.003992 nats
0.06 ppl at baseline 20 → 0.002996 nats
```

**Implied-baseline check requested:** 0.06/0.0072 = **8.33** and 0.06/0.0030 = **20.0**. So the
audit's stated "0.0030–0.0072 nats" range is *not* a range of measurement uncertainty — it is the
same 0.06 ppl evaluated at hypothetical baselines from 8.3 to 20. **The audit does not label it that
way, which makes it look like an interval estimate when it is a sensitivity sweep.** Minor
presentational defect. The *relevant* number for the Mamba-2 sweep is the single value **0.00724
nats**, and the low end (0.0030) corresponds to a ppl-20 model that is not the cited one.

**Is baseline ppl ~8.3 plausible for the arms under test?** Yes, and the plan's own arms may be
*higher*. Cross-check from the sibling audit (`KDA-LIV/docs/claude-audit/04-prior-art.md`, quoting
arXiv 2607.07953 Table 2 at 350M/15B): val CE 2.273–2.452 nats ⇒ **ppl 9.7–11.6**. At ppl 10 the
0.06-ppl basin is **0.00598 nats**, still below 0.010. **So the claim holds across the whole
plausible baseline range (8.3 → 20 gives 0.0072 → 0.0030 nats), and holds MORE strongly at higher
ppl.** Direction of the sensitivity is favourable to the audit.

### 2.3 Margin vs phenomenon — every relevant contrast

Computed on FarmShare from the Table-2 row above:

| Contrast | ΔCE (nats) | × the 0.010 margin |
|---|---:|---:|
| **Basin width, 2–11 blocks (THE CLAIM)** | **0.007238** | **0.72×** ← inside the margin |
| 15 blocks (31% attn) vs best | 0.009639 | 0.96× ← inside |
| **Transformer++ vs pure-SSM (all-conv, 0 attn)** | **0.009259** | **0.93× ← inside** |
| 1 attn block vs best | 0.014423 | 1.44× — outside |
| Transformer++ vs 50%-attn | 0.020955 | 2.10× — outside |
| 50%-attn (24/48) vs best hybrid | 0.028642 | 2.86× — outside |
| **Pure SSM (0 attn) vs best hybrid (6/48)** | **0.040338** | **4.03× — well outside** |
| Transformer++ vs best hybrid | 0.049597 | 4.96× — well outside |
| **Full table dynamic range (8.26 → 8.68)** | **0.049597** | **4.96×** |

**Also useful:** 0.010 nats expressed back in ppl at baseline 8.26 = **0.083 ppl** — i.e. the margin
tolerates a perplexity regression **1.4× larger than the entire published basin**.

### 2.4 The LOGIC — what this does and does not imply

**What is TRUE:**
1. ✅ **The margin (0.010) exceeds the basin (0.00724) by 1.38×.** Confirmed exactly. A
   non-inferiority test with Δ > the full dynamic range of the contrast it polices cannot
   discriminate *within* that range, at **any** sample size. This is a *design* defect, not a power
   defect, and the audit is right to separate the two. At infinite n the gate still accepts every
   ratio in 4–23%.
2. ✅ The audit's headline sentence "**it would declare an all-conv model non-inferior to a
   transformer**" is **literally true** — but only just, and only against one specific pair:
   Transformer++ (8.68) vs 0-attention Mamba-2 (8.60) = **0.009259 nats < 0.010**. It clears by 7%.

**What is OVERSTATED:**
3. ⚠️ **That "all-conv vs transformer" example is the weakest possible framing of the audit's own
   case, and it is arguably the wrong comparison.** Mamba-2's Transformer++ row is *worse* than its
   pure-SSM row, so the pair the audit chose is one where the true difference genuinely is tiny. The
   comparison the plan actually runs — `L0` (mostly-conv hybrid) vs `A16-P` (attention-heavy) — maps
   onto **pure-SSM vs best-hybrid = 0.0403 nats = 4× the margin**, which the gate WOULD catch. **So
   the gate is vacuous for the *ratio-within-the-basin* question and NOT vacuous for the
   *some-attention-vs-none* question.** The audit's blanket "the acceptance region strictly contains
   the entire phenomenon" is false as stated: it contains the *basin*, not the *table*.
4. ⚠️ **"Every perplexity gate in the plan is useless" — true on the merits, materially overstated
   in consequence.** The CE gate is **not a primary endpoint** in this plan. Evidence, all from the
   design doc itself:
   - §6.1's own *title*, line 1126: "**Held-out CE is almost certainly underpowered — do not make it
     the primary endpoint**."
   - Line 1136: "If it is out of reach, say '**inconclusive**' rather than 'non-inferior' — the
     protocol already mandates this."
   - Line 1106 (§6.0): "**rank arms on recall, not perplexity** … and pre-register that."
   - Lines 1138–1145 designate four primary endpoints: recall composite, length extrapolation,
     AR-Hits sliced perplexity, component-level state accounting.
   - `HANDOFF.md:143`: "**Primary endpoints: recall + length extrapolation + AR-Hits sliced
     perplexity. NOT held-out CE.**"

   **So the plan diagnosed this and demoted the endpoint before the audit did.** The audit's own §5.3
   concedes the constructive path (AR-Hits) is already in the plan and only faults it for not
   computing the SNR number. The residual defect is real but small: the +0.010 margin is still
   *quoted* in §6.1 and is still the only numeric margin G8 could refer to, so **G8 ("pre-registered
   margins met") currently points at a vacuous margin.** That is the load-bearing residue — and it
   is a one-line fix (delete the CE gate, or re-denominate in AR-slice nats).
5. ⚠️ One thing that would make the gate *worse* than the audit says: **AR-Hits sliced CE is itself a
   perplexity metric.** If "every perplexity gate is useless" were taken literally it would kill the
   plan's own best remedy. The correct statement is narrower: *aggregate held-out CE at a 0.010-nat
   margin is useless; a re-denominated slice metric is not.*

**Independent corroboration I found that the audit did not cite** — the sibling
`KDA-LIV/docs/claude-audit/04-prior-art.md:54` reaches the same verdict by a completely different
route: arXiv 2607.07953's authors state that val-loss differences "**below roughly 1e-3 to 1e-2**"
are not treated as conclusive, and **0.010 nats = 1e-2 sits exactly at the upper edge of the band
the closest prior work refuses to interpret.** Two independent arguments, same conclusion. That
strengthens (a).

---

## 3. Fail-open power arithmetic — my re-derivation vs the audit's

### 3.1 The seed SD

`HANDOFF.md:405` records the `N512_D64` MQAR seeds: **0.05 / 0.09 / 0.20 / 0.56 / 0.98**.

```
mean = 0.376
sample SD (ddof=1) = 0.392980 = 39.30 pp   ← this is the audit's 39.3 pp  ✅ EXACT
population SD (ddof=0) = 0.351489 = 35.15 pp
```

✅ **The audit's σ ≈ 39.3 pp is the *sample* SD (ddof=1), correctly computed.** (Population SD would
be 35.1 pp; using it would make the plan look ~12% better. Sample SD is the right choice.)

**Caveat on transferability that the audit itself raises and I endorse (its §2.8):** this 39.3 pp is
the SD of a *from-scratch-trained ~1M-param MQAR model's* accuracy — a bimodal "did it find the
recall circuit" lottery. If MQAR is instead an *eval on pretrained 350M arms*, the only randomness
is task-instance sampling, which shrinks as 1/√n_instances and is essentially free to drive down.
**The entire fail-open power argument is conditional on the from-scratch reading.** The plan does not
say which. This is a genuine and unresolved ambiguity, and it caps the confidence of claim (b).

### 3.2 t critical values (my own, validated)

| df | α | sided | my t_crit | published |
|---:|---:|---:|---:|---:|
| 1 | 0.05 | 2 | **12.70620** | 12.7062 ✅ |
| 1 | 0.05 | 1 | 6.31375 | 6.3138 ✅ |
| 1 | 0.05/11 | 2 | **140.054** | — (audit says 141.5 — see below) |
| 4 | 0.05 | 2 | 2.776445 | 2.7764 ✅ |
| 4 | 0.05 | 1 | 2.131847 | 2.1318 ✅ |
| 7 | 0.05 | 2 | 2.364624 | 2.3646 ✅ |

### 3.3 The n=2 agreement-ratio condition — ✅ REPRODUCES EXACTLY

Derivation (mine, independent): with two paired differences d₁, d₂,
`d̄ = (d₁+d₂)/2`, `s_d = |d₁−d₂|/√2`, `SEM = s_d/√2 = |d₁−d₂|/2`, so

```
t = d̄/SEM = (d₁+d₂)/|d₁−d₂|
```

Reject at two-sided α=0.05, df=1 iff `|d₁+d₂| > 12.70620·|d₁−d₂|`. With r = d₁/d₂ ≥ 1, same sign:

```
(r+1)/(r−1) > 12.70620  ⇒  r + 1 > 12.70620r − 12.70620  ⇒  13.70620 > 11.70620·r
⇒  r < 1.170850
```

> ✅ **The audit's "1.171 : 1 ratio" is EXACT.** My value: **1.170850**. Their derivation, their
> t_crit, and their algebra all reproduce.

Extensions I computed that they did not: **one-sided** gives r < **1.376** (a 37.6% agreement window,
noticeably less brutal); **Bonferroni over 11 comparisons, two-sided** gives t_crit = **140.054** and
r < **1.014383**. The audit reports t_crit = 141.5 → r < 1.0142. **Their t_crit is 1.0% high
(140.054 vs 141.5) but the resulting ratio is right to 4 sf** because the map is so insensitive
there. Immaterial.

### 3.4 The 60-pp-regression power claims — ONE REPRODUCES, ONE DOES NOT

**Assumptions I state explicitly** (the answer is very sensitive to all four):
- **Paired** t-test, seed as unit (this is the plan's stated design, §6.4 line 1253).
- σ = 39.298 pp is a **per-arm** SD. Converting to a **difference** SD requires
  `s_δ = σ√(2(1−ρ))`. **The audit's central case sets s_δ = σ = 39.3 pp, which silently assumes
  ρ = 0.5.** I report all three.
- Effect = 60 pp regression. α = 0.05.
- **Exact non-central t**, not the normal approximation.

**My numbers (exact non-central t):**

| s_δ assumption | n | 1-sided power / MISS | 2-sided power / MISS |
|---|---:|---|---|
| **ρ=0.5, s_δ=39.30** | **2** | 26.5% / **73.5%** | 13.5% / **86.5%** |
| **ρ=0.5, s_δ=39.30** | **5** | 87.2% / **12.8%** | 72.5% / **27.5%** |
| ρ=0.5, s_δ=39.30 | 8 | 98.7% / 1.3% | 95.7% / 4.3% |
| ρ=0, s_δ=55.58 | 2 | 19.2% / 80.8% | 9.7% / 90.3% |
| ρ=0, s_δ=55.58 | 5 | 63.5% / 36.5% | 45.2% / 54.8% |
| ρ=0.8, s_δ=24.85 | 2 | 40.7% / 59.3% | 21.1% / 78.9% |
| ρ=0.8, s_δ=24.85 | 5 | 99.6% / 0.4% | 97.5% / 2.5% |

**Reconciliation against the audit's claims:**

**(i) "at n=5, 1 in 4 catastrophic 60-pp regressions scores as 'retrieval preserved'" — ✅ REPRODUCES
(within rounding).**
The audit computes `1 − Φ(2.776 − 60/(39.3/√5)) = 1 − Φ(2.776 − 3.4138) = 1 − Φ(−0.638) = 0.7382`
⇒ 26.2% miss. I reproduce that arithmetic exactly (0.7382). My **exact non-central t** two-sided
value is **72.50% power / 27.50% miss** — so their normal approximation is 0.7 pp optimistic on
power. **"1 in 4" (25%) is a fair rounding of 27.5%.** ✅
*But note:* the audit plugged `t_crit(df=4)=2.776` into a **Φ** (normal) — a hybrid of the two
methods. The result happens to land close to the exact answer, but the method is not defensible as
written; the exact calculation is what supports the claim.

**(ii) "At n=2 it is ~9%: 91% of catastrophic regressions pass the gate" — ❌ DOES NOT REPRODUCE.**
- **Their own stated formula gives 0%, not 9%.** `1 − Φ(12.706 − 60/(39.3/√2)) = 1 − Φ(12.706 −
  2.1591) = 1 − Φ(10.547) ≈ 0.0000`. So the miss rate by their own method is **100%**, not 91%.
- **My exact non-central t at n=2, ρ=0.5, two-sided: power 13.49% ⇒ miss 86.5%**, not 91%.
- **Where their 9% actually comes from:** §2.2 of the audit computes exact power at n=2 for
  **d_z = 1** (an effect of *one full seed-SD*) and gets 9.0%. I reproduce that independently:
  **9.06%** at n=2, two-sided, ncp=√2. ✅ That number is correct — **but it is for a 39.3-pp effect,
  not a 60-pp effect.** §3.1 imports the 9% from §2.2 and re-attaches it to the 60-pp scenario. That
  is a **transposition error**: two different effect sizes, one number.

> 🔴 **FLAG: the "91% pass at n=2" figure is wrong.** The correct exact value for a 60-pp regression
> at n=2 (ρ=0.5, two-sided) is **86.5%**. The 9%/91% pair belongs to a *one-seed-SD* (39.3 pp)
> effect, where it is right. **The direction and severity of the conclusion are unaffected** — 86.5%
> is still catastrophic — but the specific number should not be quoted.

**(iii) The audit's exact-power table in §2.2 — I spot-checked it and it holds.** d_z=1 at n=2
two-sided: they report 9.0% (with a Simpson cross-check of 9.28%); I get **9.06%** by non-central-t
integration. ✅ Their bivariate-normal method and my chi-integration method agree to 0.06 pp.

### 3.5 Fisher exact at 2-vs-2 — ✅ REPRODUCES EXACTLY

Perfect separation (2 successes / 0 failures vs 0 / 2), one-sided:
`p = 1/C(4,2) = 1/6 = 0.166667`. ✅ **Confirmed.** Two-sided = 2/6 = **0.3333**.

| n per arm | one-sided min p | two-sided min p |
|---:|---:|---:|
| 2 | **0.166667** | 0.333333 |
| 3 | **0.050000** (exactly at the boundary) | 0.100000 |
| 4 | 0.014286 | 0.028571 |
| 5 | 0.003968 | 0.007937 |

> ✅ **"Provably unable to reject at any outcome" is CORRECT** at n=2 for both one- and two-sided
> (0.167 and 0.333 both exceed 0.05). It is the **most extreme possible table**, so no other outcome
> can do better. The audit's "n=3 is the arithmetic floor" is also right, with the caveat it states:
> at n=3 one-sided p = **exactly 0.050**, which fails a strict `p < 0.05` and passes `p ≤ 0.05` —
> and it is unusable two-sided (0.100). **n=4 is the first seed count that rejects two-sided.**
> The audit says "n=3 arithmetic floor, n=5 practical" — I would say **n=3 one-sided-only knife
> edge, n=4 two-sided floor**. Minor.

### 3.6 Family-wise error over 11 comparisons — ✅ REPRODUCES EXACTLY

```
1 − 0.95^11 = 0.431200  =  43.12%
```
✅ **43% is right.** (k=9 → 36.98%; k=12 → 45.96%.) The "11" is the count of arm-vs-`L0` comparisons
in a 12-arm screen, which matches §8 line 1379's arm list. Sound.

---

## 4. s_δ / KDA cross-reference — the "already measured" claim

### 4.1 Do the KDA numbers exist?

✅ **Both located and read directly in `/Users/ericwu/Developer/Capstone_LLM/KDA/HANDOFF.md`:**
- **line 158–159:** "The paper's one strictly parameter-matched LM pair is **+0.0053 nats** (needs
  **n≈43 seeds**)."
- **line 575–576:** "At **n=5 paired, detectable val-loss difference is ~0.014 nats**. The paper's
  own parameter-matched gap is **0.0053 nats** — so val loss is *underpowered by design* and a null
  there is expected, not informative."
- **line 417:** `sigma_within (pooled seed-to-seed) = 48.4pp` — the MQAR-family figure.
- **lines 428–431:** the seeds-per-arm table (ρ=0.5, MDE 10pp → **184**).

**Validation of the audit's method against KDA's own published table:** `(2.8016 × 48.4/10)² =
183.9` vs their 184. ✅ Reproduces to 0.05% — so the audit is using the same formula the sibling
track already accepted. Good.

### 4.2 My inversions

```
z(.975)+z(.80) = 2.80159 ;  z(.95)+z(.80) = 2.48647

from n=43 @ m=0.0053, two-sided:  s_δ = 0.0053·√43/2.80159 = 0.012405
from n=5  @ MDE=0.014, two-sided: s_δ = 0.0140·√5 /2.80159 = 0.011174
```
✅ **Both reproduce the audit's 0.01241 and 0.01117 exactly.** (The synthesis quotes the range as
"0.0113–0.0126" and the audit's TL;DR as "0.0113–0.0126"; the body says "0.01117–0.01241". **The
0.0126 upper end does not follow from either inversion** — it looks like a rounding drift. Use
**0.0112–0.0124**.)

Gate requirement, my derivation:
```
one-sided, n=8, m=0.010:  s_δ ≤ 0.010·√8/2.48647 = 0.011375   ← ✅ matches the doc's "≲0.011"
two-sided, n=8, m=0.010:  s_δ ≤ 0.010·√8/2.80159 = 0.010096
```

**Straddle verdict:** measured 0.01117–0.01240 vs requirement 0.011375. The low end is **1.8%
below**; the high end is **9.1% above**. ✅ **"Straddles the requirement and its upper end exceeds it
by 9%" is EXACT.** Note this is against the *one-sided* requirement, which is the generous one; under
a two-sided requirement (0.010096) **both** measured values fail.

### 4.3 Transferability — my own assessment (the audit under-weights this)

**Is the KDA s_δ in the same units and at a comparable scale?** Partially.

| dimension | KDA measurement | LIV plan |
|---|---|---|
| metric | LM **val loss in nats** — same unit ✅ | held-out CE in nats |
| pairing | paired by seed ✅ | paired by seed |
| scale | **~100M params** (audit's own caveat; the KDA LM runs are the `d_model=616` refit family) | **350M** |
| token budget | ~1B (the KDA LM stage) | **2B screen / 20B confirm** |
| provenance of 0.0053 | **the DeltaProduct paper's** number, not measured locally — KDA *inferred* n≈43 from it | — |

> ⚠️ **The 0.0053-nat effect is a LITERATURE number, not a local measurement.** KDA/HANDOFF:158
> says "**The paper's** one strictly parameter-matched LM pair is +0.0053 nats (needs n≈43 seeds)."
> The **n≈43** is KDA's *derived* seed requirement, and to derive it KDA must already have had an
> s_δ estimate — which line 575's "~0.014 nats at n=5" is. So **the two numbers the audit inverts
> are not independent**; they are two expressions of one underlying s_δ estimate, which is why they
> agree to 10%. **That agreement is not corroboration.** The audit presents them as if inverting two
> published facts triangulates the value; it triangulates one value stated twice.
>
> ⚠️ **s_δ generally shrinks with scale and token budget.** Going 100M→350M and 1B→2-20B tokens
> should reduce seed variance in val loss. The audit states this caveat honestly (§2.4 parenthetical)
> and then proceeds as if the number transfers. **My assessment: the KDA s_δ is a reasonable ORDER-OF-
> MAGNITUDE prior and NOT a substitute for the Phase-2 measurement.** The plan's "measure s_δ in the
> pilot" is therefore **not redundant** — the audit's strongest phrasing ("the pilot will confirm
> what is already on disk", "asking a question that has already been answered") **overreaches**.
>
> **What survives:** even if s_δ halves to 0.006 at 350M/2B, the achievable margin at n=8 one-sided
> would be `2.48647×0.006/√8 = 0.00528 nats` — still *below* 0.010, meaning the gate becomes
> *reachable* but remains **vacuous by margin** (§2.3), because the phenomenon is 0.0072 nats. **So
> the vacuity finding (a) is robust to the transferability objection; the "already unreachable"
> finding is not.** These are genuinely two different criticisms and only one of them is scale-proof.

**Re-denomination cost, my calculation:** if the margin is tightened to 0.002 nats (below the basin),
at s_δ = 0.0124: **n = 238** (one-sided) or **n = 302** (two-sided). ✅ The audit's "≈238 seeds"
reproduces exactly. At the optimistic s_δ = 0.006 it would be n = 56 (one-sided) — still far past the
plan's ≥8. **Conclusion stands: CE cannot be a gate at this program's seed budget under any
re-denomination that would make it non-vacuous.**

---

## 5. HANDOFF's own claims — verified

| HANDOFF claim | Location | Verdict |
|---|---|---|
| "This repo's own KDA study measured that a **+0.0053-nat effect needs ~n=43 seeds**" (`HANDOFF.md:145`) | `KDA/HANDOFF.md:158-159` | ⚠️ **PARTIALLY SUPPORTED.** The n≈43 is real and appears in KDA. But "**this repo's own KDA study measured**" is inaccurate: KDA:158 attributes +0.0053 to "**the paper's** one strictly parameter-matched LM pair" (DeltaProduct). The repo *derived* the seed requirement; it did not *measure* the effect. **Attribution error, not a numeric error.** My check: n = (2.80159×0.0124/0.0053)² = 42.9 → 43 ✅ self-consistent. |
| "+8.92pp at n=3 collapsed to +2.01pp (ns) at n=8" (`HANDOFF.md:182`) | `KDA/HANDOFF.md:167` ("A '+8.92pp KDA>GDN' result at n=3 collapsed to **+2.01pp ns** at n=8. Sign-consistency across 3 seeds is p=1/8, not evidence") **and** `KDA/HANDOFF.md:131` ("KDA's per-channel gate is NOT better than GDN's per-head gate at this scale: **S5 +2.01pp ns**") | ✅ **FULLY SUPPORTED**, two independent locations in KDA/HANDOFF, one of which is the headline results block. My check of the corollary: sign-consistency across 3 seeds under H₀ is (1/2)³ = **1/8 = 0.125** ✅ correct, and correctly noted as not significant. |
| σ_within = 48.4 pp on the same task family | `KDA/HANDOFF.md:417`, ANOVA block at 405–420 | ✅ **VERIFIED, with the full supporting ANOVA** (n=20, 4 levels, F(3,16)=0.337 vs F_crit 3.24, η²=5.9%). This is a genuinely independent second measurement of MQAR seed variance (48.4 pp vs the plan's 39.3 pp) — **same order, and the audit is right to cite it as corroboration.** |

---

## 6. VERDICTS

### (a) "The CE gate is VACUOUS BY MARGIN" — **CONFIRMED** (arithmetic), with two qualifications

**CONFIRMED:**
- The +0.010-nat margin exists verbatim at `design.md:1133` and traces to
  `liv-kda-gqa-sub500m-experiment.md:386,396`. ✅
- The 0.06-ppl basin is sourced from a **primary** paper (Mamba-2 Table 2, 350M/48L/7B tokens) and I
  re-verified it from the raw row: 8.32 − 8.26 = 0.06 exactly. ✅
- **ΔCE = ln(8.32/8.26) = 0.007238 nats.** The margin is **1.38× wider** than the entire basin. ✅
- The claim is **robust to baseline ppl**: at ppl 10 (the realistic figure for 350M/15B runs, from
  arXiv 2607.07953: CE 2.27–2.45 nats ⇒ ppl 9.7–11.6) it is 0.00598 nats; at ppl 20, 0.0030. **Every
  plausible baseline puts the phenomenon inside the margin.** ✅
- Independent second route: prior-art audit finds 0.010 nats = 1e-2 sits at the top of the band the
  closest prior work "refuses to interpret." ✅

**QUALIFICATION 1 — the "all-conv vs transformer" line is true but cherry-picked.** It holds only for
Mamba-2's Transformer++ (8.68) vs 0-attention (8.60) = 0.00926 nats, clearing by 7%. Against the
contrast the plan actually runs (mostly-conv vs attention-heavy ≈ pure-SSM vs best hybrid) the gap is
**0.0403 nats = 4× the margin** and the gate WOULD fire. **"The acceptance region strictly contains
the entire phenomenon" is false as stated** — it contains the *basin* (0.0072), not the *table*
(0.0496). The vacuity is specific to the **ratio-within-basin** question.

**QUALIFICATION 2 — "every perplexity gate in the plan is useless" is true on the merits but
LOW-CONSEQUENCE.** §6.1's own title already says "do not make it the primary endpoint"; line 1136
already pre-commits to "inconclusive"; §6.0 line 1106 already says "rank arms on recall, not
perplexity"; HANDOFF key decision #2 already names recall / length-extrap / AR-Hits as primary and
"NOT held-out CE". **The plan diagnosed this before the audit did.** The residue that genuinely
binds: **G8 ("pre-registered margins met") has no other numeric margin to point at**, so the only
pre-registered margin in the program is a vacuous one. That is real, and it is a one-line fix.
Also: taken literally the claim would kill AR-Hits sliced CE, which is the plan's own remedy — the
correct scope is "aggregate held-out CE at Δ=0.010", not "perplexity".

### (b) "5 of 9 gates are FAIL-OPEN" — **CONFIRMED IN SUBSTANCE, DENOMINATOR AND ONE NUMBER WRONG**

**CONFIRMED:**
- §8's table has **exactly 9 rows** with a "Gate to pass" column (lines 1375–1383), so **9 is a
  defensible denominator**. ✅
- The **structural** argument is correct and is the most important finding here: a non-inferiority
  claim is confirmed by *failing to reject*, so an underpowered study passes it. ✅
- **n=2 agreement ratio 1.171 : 1** — my exact value **1.170850**. ✅ EXACT.
- **Fisher 2-vs-2 one-sided p = 1/C(4,2) = 1/6 = 0.1667**, two-sided 0.3333; **no 2-vs-2 outcome can
  reach α=0.05**, so "provably unable to reject at any outcome" is ✅ CORRECT.
- **FWER = 1 − 0.95¹¹ = 43.12%.** ✅ EXACT.
- **σ = 39.30 pp** is the correct *sample* SD of 0.05/0.09/0.20/0.56/0.98. ✅ EXACT.
- **"1 in 4 catastrophic 60-pp regressions passes at n=5"** — my exact non-central-t value is
  **27.50%** miss (two-sided, ρ=0.5). ✅ REPRODUCES (they say 26%).
- The proposed fix (CI upper bound vs explicit Δ) is correct **and is already the form used in the
  parent protocol document** — a fact the audit missed, which makes the fix cheaper than presented. ✅

**DENOMINATOR IS SOFT:**
🔴 **"5 of 9" and the body's "six criteria are non-inferiority statements" are two different counts.**
9 = table *rows*; the rows contain ~20 decidable clauses; the non-inferiority count is 6 (G6×3 + CE +
kill-rule flat/matches + latency null), and two of those six (the CE gate and the P2 latency null)
are **not in the 9-row table at all**. So the ratio "5/9" mixes numerator and denominator from
different populations. **The honest statement is: "of the 9 gate rows, 5 cannot fail; and separately,
6 non-inferiority criteria across §6 and §8 are all confirmed by failing to reject."** Also note the
audit's G-numbering is off-by-one vs document order (it lists Phase 0b as G2; the doc puts it 4th).

**ONE NUMBER DOES NOT REPRODUCE:**
🔴 **"At n=2, 91% of catastrophic [60-pp] regressions pass the gate" is WRONG.**
- My exact non-central t (n=2, 60 pp, s_δ=39.3, ρ=0.5, two-sided): power **13.49%** ⇒ miss **86.5%**.
- The audit's *own stated formula* `1 − Φ(12.706 − 2.1591)` gives ≈ **0**, i.e. miss **100%**.
- The **9%** they quote is the correct exact power for a **one-seed-SD (39.3 pp)** effect at n=2
  (I independently get **9.06%**) — it was transposed from §2.2 onto §3.1's 60-pp scenario.
- **Severity: LOW.** 86.5% is still catastrophic and the conclusion is unchanged. But the specific
  "91%" should be corrected to **~87%**, or re-labelled as applying to a 39-pp effect.

**ONE CONDITIONALITY THE AUDIT FLAGS AND THE SYNTHESIS DROPS:**
⚠️ The whole 39.3-pp variance argument holds **only if MQAR is a from-scratch training task**. If it
is an eval on pretrained arms, seed variance is replaced by instance-sampling variance which shrinks
for free. The audit says this plainly in its §2.8 ("worth more than any other clarification"); the
synthesis's C.6 bullet and error-table row 16 do **not** carry the caveat. **Row 16 should.**

---

## 7. Summary table — audit's number vs mine

| Quantity | Risk audit | My re-derivation | Status |
|---|---:|---:|---|
| Mamba-2 basin in nats | 0.007203 (via 8.36/8.30) | **0.007238** (via 8.32/8.26) | ✅ 0.5% |
| Margin ÷ basin | "1.4×–3.3×" | **1.38×** at ppl 8.26 | ✅ |
| Basin at ppl 15 / 20 | 0.00399 / 0.00300 | **0.003992 / 0.002996** | ✅ EXACT |
| Implied baselines for the 0.0030–0.0072 range | (not stated) | **ppl 8.33 to 20.0** — a sensitivity sweep, not an interval | ⚠️ mislabelled |
| Tf++ vs all-conv | (asserted "would pass") | **0.009259 nats < 0.010** — passes by 7% | ✅ true but marginal |
| Pure-SSM vs best hybrid | (not computed) | **0.040338 nats = 4.03× margin** — would FAIL the gate | 🔴 undercuts "contains the entire phenomenon" |
| Sample SD of the 5 MQAR seeds | 39.3 pp | **39.298 pp** (ddof=1); pop. SD 35.15 | ✅ EXACT |
| n=2 agreement ratio | 1.171 | **1.170850** | ✅ EXACT |
| Bonferroni-11 t_crit (df=1) | 141.5 | **140.054** | ⚠️ 1.0% high (ratio unaffected) |
| Power, d_z=1, n=2, 2-sided | 9.0% | **9.06%** | ✅ EXACT |
| Miss rate, 60 pp, n=5 | 26% | **27.50%** (exact t, 2-sided) / 12.80% (1-sided) | ✅ ≈ |
| **Miss rate, 60 pp, n=2** | **91%** | **86.51%** (exact t, 2-sided); their own formula gives 100% | 🔴 **DOES NOT REPRODUCE** |
| Fisher 2v2 one-sided | 1/6 = 0.167 | **0.166667** | ✅ EXACT |
| Fisher n=3 one-sided | 0.050 | **0.050000** (knife edge; 2-sided 0.100) | ✅ |
| FWER, 11 comparisons | 43.1% | **43.1200%** | ✅ EXACT |
| s_δ from n=43 @ 0.0053 | 0.01241 | **0.012405** | ✅ EXACT |
| s_δ from n=5 @ 0.014 | 0.01117 | **0.011174** | ✅ EXACT |
| s_δ range as quoted in TL;DR/synthesis | "0.0113–0.0126" | **0.0112–0.0124** | ⚠️ 0.0126 is drift |
| Gate requirement, n=8, 1-sided | 0.01137 | **0.011375** | ✅ EXACT |
| Straddle: high end exceeds by | 9% | **9.1%** | ✅ EXACT |
| n for a 0.002-nat margin at s_δ=0.0124 | ≈238 | **238** (1-sided) / **302** (2-sided) | ✅ EXACT |
