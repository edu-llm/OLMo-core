# HANDOFF — DP2-KDA / KDA-Householder

**Last updated:** 2026-08-04 (third update this day; supersedes `8972066` and `9e36632`).

**Status in one paragraph.** The research question was re-pointed at **Kimi K3's** KDA node,
answered by a 48-cell paired sweep on AWS, written up as a **9-page conference paper**
(`paper/kda-regime-arity.{tex,pdf}`), and then **stress-tested by two follow-up runs (72 more
cells) that close both of the paper's open confounds**. Capacity matching leaves the effect
intact and slightly larger; a 33× learning-rate sweep rejects the leading optimization
explanation. Both cut against an earlier claim as well: matching costs the single band that
survived Holm. Writing the paper had already forced a re-analysis that overturned several
numbers — the metric is prefix-averaged, so the five "lengths" are nested. Totals across all
three runs: **120 cells, 120/120 succeeded, 8.67 GPU-hours, ~$122 worst case** (actual far
lower; cells run 228–305 s against a 1 h bound).

> **BRANCH:** `edullm/a5-solvability` at `b26bbd1`, pushed, clean.
> The image build only fires on `edullm/**` or `main`; an `agent/**` branch pushes green while
> **publishing no image**.

---

## Goal

Does the **Kimi K3** KDA node benefit from R>1 Householder factors per token (DeltaProduct arity)?

K3 §2.1.1 Eq. 2 fixes `β = Sigmoid(W_β x) ∈ (0,1)` — **strict**. The question is whether arity
pays under that constraint. Answer: essentially no, and the reason is that **β range, not arity,
is the dominant lever.**

---

## Current Progress

### The paper — the primary deliverable

`paper/kda-regime-arity.tex` → `paper/kda-regime-arity.pdf`, **9 pages, US Letter**,
two-column. Author: Eric Wu, Stanford University / Alpha AI Engineering.

**Build: `cd paper && tectonic -X compile kda-regime-arity.tex`.** Exit 0. **Only `tectonic`
0.16.9 is installed on this machine — there is NO MacTeX, no TeX Live, no `pdflatex`, no
`tlmgr`** (all four standard install paths verified empty). Tectonic fetches packages on demand
and caches them. Do not add a conference `.cls`; MacTeX is a 6.39 GB download against ~11 GB
free and would not fit.

Title: *Write Strength, Not Write Count: The β Range Dominates Householder Arity in Gated
Delta-Rule Attention.* Sections: Intro · Background · The K3 KDA node · Setup · Results ·
Two failed explanations · Limitations (6 subsections) · Related work · Conclusion. Seven
booktabs tables (Table 6 = capacity-matched square, Table 7 = LR sweep).

### The experiment

Run `run_019fce2f-9b5b-70ea-8a77-6703e1b76605`, 48/48 SUCCEEDED. Records synced to
`/tmp/sweep48/cell-*/` (**re-sync from S3 if gone**:
`s3://sbsandbox-intern-edullm-outputs/teams/scratch/runs/run_019fce2f-.../`). Aggregated at
`/tmp/sweep48-result.json`. Aggregator: `probes/analyze_regime_arity.py`.

4 arms × 2 tasks (`a5_words`, `s5_words`) × 6 bundles (1101–1106), fully paired, 4000 steps,
one A10G. All arms `kda_hh`/triton.

| arm | R | β regime | non-embed params | β_mean reached |
|---|---|---|---|---|
| `R1` | 1 | strict (0,1) | 998,092 | 0.899 |
| `DP2-strict` | 2 | strict (0,1) | 1,400,524 | 0.804 |
| `R1-refl` | 1 | reflection (0,2) | 998,092 | 1.705 |
| `Reflection` | 2 | reflection (0,2) | 1,400,524 | 1.625 |

### ⚠️ THE RE-ANALYSIS — read this before quoting any number from the old handoff

**The metric is prefix-averaged token accuracy over positions 1..L** (`probes/train_probe.py:427`;
group tasks mask nothing, so the denominator is all 64·L positions). The five evaluation lengths
are therefore **nested, not independent**. Four consequences, all verified independently by me:

1. **A zero-parameter dilution model — "correct below 40, chance above" — reproduces both strict
   arms' entire length curves to within ~1.3pp.** The strict arms do not extrapolate at all; the
   smooth decay is arithmetic. (I checked: R1/S5 predicted 7.81% vs observed 6.63% at L=512.)
2. **De-nesting into disjoint position bands reverses the two long-length S5 cells:**
   +8.57 → **−0.77** (129–256) and +4.44 → **+0.31** (257–512). It *strengthens* A5:
   +13.47 → **+30.14** (65–128), +35.28 → **+57.09** (129–256).
3. **The "18.6×" headline in the previous handoff sat on one of the reversed cells and is
   withdrawn.** Replaced by band-level ratios where both arms are measurable: 25.7× (A5 41–64),
   124.7× (A5 65–128), 45.5× (S5 41–64).
4. **Zero of ten prefix cells survive Holm correction; exactly one of eight bands does.** With
   df=5, p<0.00625 requires |t|>4.03 — no prefix cell reaches it, so none *could* have.
   **Superseded:** that one survivor does not survive capacity matching either (see below).

**Also withdrawn:** the "+3.50pp upper bound on the capacity confound" argument I constructed. It
is circular — both strict arms sit within 2× chance in every extrapolation band (chance 1.67% A5,
0.83% S5), so it bounds one floored model against another.

### ⚠️ THE CONFOUND IS INSTABILITY, NOT NON-CONVERGENCE — RE-DIAGNOSED 2026-08-04

The previous version of this file, and the paper's §"unconverged loss", call this "10 of 24
strict runs end with unconverged loss". **The word "unconverged" is wrong; the confound is
real.** Both halves matter, so do not carry either one alone into the next revision.

**Wrong: "those are not converged models."** Every one of the ten reaches **1e-4 or below**
after warmup (verified, all ten). `loss_trace` samples one minibatch every 500 steps and its
final entry is a single batch at step 3999, where OneCycle has driven the LR to **4e-9** and
the weights are frozen. `R1`/S5/b1106 reads 1e-4 at step 3000 and 0.824 at step 3999. They fit
the training set. Fixed at `70dc155`: records now carry `loss_summary` (tail mean/median/max
over the last 5% of steps, plus `trace_min_after_warmup`). **Never read `loss_trace[-1]` as
convergence.**

**Right: the instability is real and asymmetric.** Log final loss correlates with accuracy at
L=40 at **r = −0.468** across the 24 strict runs (the paper says −0.43; I get −0.468 — same
finding). Two runs (`R1`/S5/b1102 and b1103) are high at **5 of 7** sampled steps, which is
sustained oscillation, not one unlucky batch. Spiked runs average 74.46% at L=40 against 80.73%
for clean ones. Zero of 24 reflection runs spike at all.

Both were fixed in the paper at `b26bbd1`: §7.1 now states the instability finding and
explicitly retracts the "not converged models" sentence. And the LR sweep went further than a
restatement — over a 33× range the two accounts *are* separable, and the evidence favours
expressivity.

**The sharper effect this uncovered.** The strict arms fit to ~0.0000 training loss and still
score only **68–89% at L=40** — a length *inside* the training range 3–40 — where `R1-refl`
scores 100% (A5) / 92.6–99.0% (S5) and `Reflection` 100% / 89.8–100%. That is a generalization
gap on in-distribution data: not length extrapolation, and untouched by the nesting artifact.
This is what the LR sweep tested, and no rate closed it.

### What survives

- The **A5 mid-range interaction**, now capacity-matched: **+58.88pp** (se 14.85, t 3.96,
  L95 +28.95) at positions 129–256. Unmatched it was +57.09 (se 16.08, t 3.55).
- The **cost asymmetry**: switching β regime is **free** (998,092 params both sides) while
  doubling R costs **+40.3%**.
- **K3-relevant conclusion**: K3 fixes strict β, and under strict β arity buys almost nothing
  here. Whether K3-scale models would benefit from R>1 is a separate question this scale cannot
  settle — the paper says so.

### Bugs fixed earlier this session (all pushed, all blocking)

| Commit | Fix | Why |
|---|---|---|
| `621eaba` | Call mixer `init_weights` in `probes/train_probe.py` | `A_log`/`dt_bias` are `torch.empty` (`recurrent.py:1162-1163`); `build()` does not initialize. **All 155 archived records and the LM study trained the decay gate from uninitialized memory.** |
| `e96dd89` | `.[wandb,fla]` in `.edullm/Dockerfile` | Bare `assert has_fla()` kills a run 4 s in with no message. |
| `b146c45` | `EDULLM_COMMIT_SHA` provenance fallback + `--run-id` | Image excludes `.git` on purpose (token in `.git/config`), so provenance wrote `"unknown"`, which `analyze_sigma.py:23-25` **rejects**. |
| `4b5b9cf` | Add `R1-refl` arm id | Fourth cell of the 2×2 had no canonical id. |
| `76502c6` | `probes/analyze_regime_arity.py` | `analyze_sigma.py` keys on filenames with no task axis; a two-task sweep silently overwrites. |
| `70dc155` | `R1-P` carries its own `match_arm`; adds `R1-refl-P`, `--match-arm`, `lr` and `loss_summary` in the record; de-nesting in the aggregator | `R1-P` resolved to settings **byte-identical to `R1`**, so it trained, succeeded, and recorded `arm: R1-P` while controlling for nothing. A missed parameter match printed `within_tolerance: false` and trained anyway. |
| `5989dbb` | `probes/analyze_lr_gap.py` | Nothing could read an LR sweep: records carried no `lr`, so its cells collided as duplicate `(arm, task, bundle)` keys. |

---

## What Worked

- **Rendering the PDF and looking at it.** Caught that `\affil{A \\ B}` collapses to
  "Stanford University , Alpha AI Engineering" — one line, stray comma. Correct form is two
  `\affil` entries with `\author[1,2]`.
- **Verifying subagent claims against the data before relaying them.** Also the reverse: the
  paper team's claim that my framing was wrong was *correct*, and checking it myself is what
  established that rather than a guess about who to believe.
- **Reading `statusReason`, not `status`.** A Batch job at `RUNNABLE` looks identical whether
  cold-starting or permanently unplaceable.
- **Preferring a queue with SUCCEEDED history.** `gpu-1xl4` had 0 ever; `gpu` (A10G) had 27 with
  the identical container shape.
- **Testing the fanout command through the platform's exact `shlex.join`-in-`bash -c` wrapping**
  before submitting. 48/48 indices distinct.

## What Didn't Work

- **My K3 novelty hypothesis was false.** I claimed the new decay floor forecloses reflections.
  `α = exp(g) > 0` always, so `det(Diag(α)) > 0` under *both* parameterizations. Sign is governed
  entirely by β. K3 forecloses odd permutations via **strict β**, which Kimi Linear already had.
- **The parity prediction failed.** Predicted a large interaction on S5 (odd generator) and small
  on A5. Opposite at long lengths. The paper reports this as a finding.
- **Three citation claims in my brief were wrong**, caught against sources: Grazzi Prop 1
  **item 3 is spectral** (item 2 concerns permutations, and item 3 carries a published
  correction); **arXiv:2606.26560 is "Erase-then-Delta Attention"** and treats the per-channel
  gate as a KDA-credited *baseline*; **DeltaProduct App. B.4 Eq. 6 is a product of RWKV-7
  matrices**, not gated Householders.
- **My first pre-validation was worthless.** `images.json` asserting `critical: 0` compiled clean
  at $1.86; the registry had 4 CRITICAL / 8 HIGH. **The pre-validator only checks what you
  assert.**
- **Pre-validation cannot see IAM.** `gpu-1xl40s` compiled clean, passed admission, then 403'd.
- **`HANDOFF.md`'s old "Measured: 6.9–9.5× slower vs `chunk_kda`" was never measured** — those
  `perf.tsv` rows read `PENDING`.

## Key Decisions

1. **Tectonic, not MacTeX.** Self-contained, reproducible, already working. MacTeX does not fit
   on this disk.
2. **Report the de-nested (band-level) analysis as primary**, with prefix numbers shown alongside
   so a reader can see the artifact rather than take it on trust.
3. **Publish the failed predictions.** Two mechanistic hypotheses were pre-registered and
   contradicted; the paper says so in its own section.
4. **`gpu-1xa10g` is the default profile.** Both newly-promoted L-series shapes are broken.
5. **Route future LM runs through `TransformerConfig`** — `transformer/init.py:134` is the *only*
   caller of mixer `init_weights`, so that path fixes the uninitialized-gate bug structurally.

---

## Next Steps

### 1. Decide the venue, and whether to close the two open confounds first

The paper is submittable as an empirical/negative result. Two cheap experiments would materially
strengthen it, both ROUTINE on the platform:

**BOTH RAN. 72/72 cells succeeded, 5.16 GPU-hours, and the paper is updated (`b26bbd1`).**

| Run | id | Cells | Result |
|---|---|---|---|
| Parameter-matched | `run_019fcf14-0f7c-70f0-bbb3-7f5d6b45482a` | 24 | Interaction survives, strengthens |
| In-distribution LR | `run_019fcf14-6197-70b6-867c-29b766298103` | 48 | No rate closes the gap |

Records synced to `/tmp/matched24/` and `/tmp/lrsweep48/`; re-sync from
`s3://sbsandbox-intern-edullm-outputs/teams/scratch/runs/<run-id>/` if gone. Combine the
matched run with the original sweep into one directory and analyse with `--square matched`.

- **Capacity confound: CLOSED.** `R1-P`/`R1-refl-P` solved to `ffn_dim=174` → 1,399,756 params
  against 1,400,524, **−0.055%**, verified in the live logs. Every new cell's eval bank hashes
  identically to the original sweep's, so the squares are paired across runs. A5 129–256 goes
  **+57.09 → +58.88pp** (t 3.96, L95 +28.95).
- **Optimization confound: REJECTED over a 33× range.** Best strict cell is 91.33% at L=40
  against a 100% ceiling; at the top two rates the strict arms fit *stably* (tail-median loss
  0.0006–0.0084) and still fall 9–21pp short. Caveat recorded in the paper: `DP2-strict` on A5
  peaks at the grid's top rate, so its optimum is unbracketed.

### ⚠️ The cost of doing this properly: the last Holm survivor is gone

Capacity matching **weakens** the family-wise evidence. S5 41–64 drops from t=4.26 to t=3.64,
and **no band clears Holm in the matched square** — strongest is A5 129–256 at p=0.00535
against a 0.00500 threshold. Higher point estimates, weaker corrected significance. Both facts
are now in the abstract, results, Table 4's caption and the conclusion. Do not quote the "one
band survives" line from the old draft.

### 2. Everything prior to `621eaba` is suspect

The 155-record archive and the LM study trained with an uninitialized decay gate, and **no repeat
runs exist**, so it cannot be determined whether the bytes were zeros (harmless) or recycled
garbage (arm-dependent). The 48-cell sweep is the only clean data in the project.

### 3. Open engineering items

| Item | Note |
|---|---|
| **Open the `l2_normalize` PR** | Required by `guides/olmo-core.md:13`. CHANGELOG entry already in `320495a`. Still unopened. |
| **Report two platform bugs** | `gpu-1xl40s` IAM gap (CFN template at `infra/iam/admission-service-roles.yaml:133-135` grants the queue; deployed role does not). `gpu-1xl4` unplaceable shape, 0 successes ever. Both pass pre-validation and fail on AWS. |
| **`assert has_fla()` is bare** at `recurrent.py:87,617,1111` | No message. One-line upstream fix. |
| **`perf.tsv` `PENDING` rows** | The `chunk_kda` comparison was never run. Run it or delete the claim. |
| `docs/dp2-kda/` not formally closed | Three audits recommended terminating it; its premise is strict β, which this sweep shows gains little from arity. |

---

## Environment

**AWS (eduLLM platform)** — mandated venue. `gpu-1xa10g` $1.006/hr. Routine ceiling **$500**
(`config/policy.yaml:27`); auto-approve under **$5 AND 1 h** (`:103-104`). Both the $1.01 cell and
the $48.29 sweep self-authorized as `routine_self_authorized` (submitter is approver). Always
pre-validate with `tools/compile_submission.py`; inputs kept at `/tmp/pv/`.

**LaTeX** — `tectonic` only, at `/opt/homebrew/bin/tectonic`. See the build note above.

**FarmShare** (free, L40S sm_89) — socket was dead this session; nothing measured there.

**`probes/` exists twice** — vendored here (canonical, tracked) and at `Capstone_LLM/probes` (own
repo, `main`). Byte-identical as of `76502c6`; both received every fix.

**The LM harness `/Users/ericwu/Developer/Capstone_LLM/KDA/lm/` is in NO git repository** (737
lines). It produced the +0.0357 nats result. Vendoring it is a precondition for reproducibility.

---

## Governing documents

| Path | What |
|---|---|
| `paper/kda-regime-arity.tex` | **The paper.** Build with tectonic; see above. |
| `probes/analyze_regime_arity.py` | Sweep aggregator. Point it at a directory of records. |
| `docs/kda-householder/kda-householder.md` | Earlier write-up, 1262 lines. §7 LM result, §7.2 the vacuous-endpoint lesson. **Its probe sections use the prefix metric and are subject to the same nesting artifact.** |
| `docs/dp2-kda/phase-0-1-runbook.md` | Old plan, recommended for termination. Its gate demands power ≥0.80 at a +5pp floor, which caps at exactly 0.50 by construction. |
| `/tmp/k3.txt` | Extracted K3 paper text (`pdftotext -layout`; WebFetch returns compressed binary). Regenerate if gone. |
| `.claude/skills/edullm-platform-runs/` | Platform skill; `references/prevalidate.md` is the offline compiler. |
