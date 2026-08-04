# HANDOFF — DP2-KDA / KDA-Householder

**Last updated:** 2026-08-04.

**Status in one paragraph.** The research question was re-pointed at **Kimi K3's** KDA node
(arXiv 2607.24653, published 2026-07-28) and then **answered by measurement**: a 48-cell paired
sweep on AWS shows that under **strict β ∈ (0,1) — which is what K3 specifies — R>1 buys almost
nothing** (+0.98pp at L=256, +0.35pp at L=512), while under reflection β ∈ (0,2) it buys up to
+36pp. Four blocking bugs were found and fixed along the way, including one that invalidates every
prior result in the project. Total AWS spend to date: **~$50 worst case, 48/48 cells succeeded.**

> **BRANCH:** `edullm/a5-solvability` at `76502c6`, pushed, clean. The image build only fires on
> `edullm/**` or `main`; an `agent/**` branch pushes green while **publishing no image**.

---

## Goal

Originally: does DeltaProduct (R Householder factors per token) improve LLMs when combined with
Kimi Delta Attention?

**Sharpened, and this is the question the sweep answers:** *does the **K3** KDA node benefit from
R>1?* K3 differs from the implemented Kimi-Linear-style KDA in two ways (§2.1.1 of the paper), and
one of them — **strict β = Sigmoid(W_β x) ∈ (0,1), Eq. 2** — is decisive.

---

## Current Progress

### The result (new, and it settles the K3 question)

Run `run_019fce2f-9b5b-70ea-8a77-6703e1b76605`, 48/48 SUCCEEDED, records in
`s3://sbsandbox-intern-edullm-outputs/teams/scratch/runs/run_019fce2f-.../cell-*/`.
Design: 4 arms (`R1`, `DP2-strict`, `R1-refl`, `Reflection`) × 2 tasks (`a5_words`, `s5_words`)
× 6 bundles (1101–1106), paired. Both tasks report `PAIRED`, 0 rejected, provenance `4b5b9cf`.

**R effect of going R=1 → R=2, within each β regime (pp, n=6 paired):**

| | L=40 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| A5 strict | +3.31 | +3.50 | +2.01 | **+0.98** | **+0.35** |
| A5 reflection | +0.01 | +0.30 | +15.48 | **+36.26** | **+31.36** |
| A5 interaction | −3.30 | −3.20 | +13.47 | **+35.28** (L95 +16.37) | **+31.01** (L95 +9.45) |
| S5 strict | +2.67 | +1.92 | +0.82 | **+0.38** | **+0.21** |
| S5 reflection | +1.02 | +15.60 | +18.73 | +8.95 | +4.65 |
| S5 interaction | −1.65 | +13.68 (L95 +5.15) | +17.91 (L95 +5.15) | +8.57 (L95 +2.15) | +4.44 (L95 +1.05) |

**Read this as:** the entire benefit of extra Householder factors lives in the **reflection**
regime. K3 uses strict β, so **the K3 KDA node should not be expected to benefit from R>1.** That
is now measured, not merely argued from determinants.

### Two findings that contradict things previously believed

1. **The parity hypothesis is WRONG.** I predicted a large interaction on S5 (its transposition
   generator is odd) and a small one on A5 (both generators even). The opposite holds at long
   lengths: **A5 +35.28 vs S5 +8.57 at L=256.** Reachability-of-reflections is *not* the mechanism.
2. **The dominant effect is β itself, not R — and it is enormous.** Mean accuracy at L=256,
   `s5_words`: `R1` 12.51%, `DP2-strict` 12.89%, `R1-refl` 19.58%, `Reflection` 28.53%. Switching
   β regime at fixed R=1 (12.51 → 19.58) beats doubling R at fixed strict β (12.51 → 12.89) by
   roughly **18×**. The project spent its history studying R; the lever was β.

### Bugs fixed (all pushed; each was blocking)

| Commit | Fix | Why it mattered |
|---|---|---|
| `621eaba` | Call the mixer's `init_weights` in `probes/train_probe.py` | `A_log`/`dt_bias` are allocated with `torch.empty` (`recurrent.py:1162-1163`) and `build()` does not initialize them. **All 155 archived probe records AND the LM study trained the decay gate from uninitialized memory.** |
| `e96dd89` | `.[wandb,fla]` in `.edullm/Dockerfile` | Every recurrent mixer opens with a bare `assert has_fla()`. Without it a run dies 4 s in with an `AssertionError` and no message. |
| `b146c45` | `EDULLM_COMMIT_SHA` provenance fallback + `--run-id` | `probe_source_revision()` shells to git, but the image excludes `.git` on purpose (the token lives in `.git/config`). It wrote `"unknown"`, which `analyze_sigma.py:23-25` **rejects** — a paid run would be silently dropped at aggregation. |
| `4b5b9cf` | Add the `R1-refl` arm id | R=1 + reflection was the only cell of the 2×2 with no canonical id. Without it the interaction is unidentified: `Reflection − DP2-strict` differs in R *and* β. |
| `76502c6` | `probes/analyze_regime_arity.py` | `analyze_sigma.py` parses arm/bundle from *filenames* and has no task axis, so a two-task sweep collides in `recs[(arm, bundle)]` and S5 silently overwrites A5. |

### The K3 architecture delta (verified against the paper, text at `/tmp/k3.txt`)

K3 is 2.8T MoE, 104B active, **69 KDA + 24 MLA layers at 3:1**, 1M context, NoPE everywhere.

| Aspect | K3 | This repo | Delta? |
|---|---|---|---|
| Decay | `g = g_min·σ(e^A z)`, `g_min = −5`, `α ∈ (e⁻⁵,1)` (Eq. 5) | `g = −e^A·softplus(z)`, unbounded (`recurrent.py:1265-1269`) | **YES**, ~25 LOC, no kernel edits |
| β | strict `σ(W_β x) ∈ (0,1)` (Eq. 2) | default `False` = strict (`recurrent.py:1102`) | no — but the LM study overrode to `True` (`train_lm.py:165`) |
| Output gate | full-rank `W_g` (Eq. 6) | low-rank `nn.Sequential` (`recurrent.py:1192`) | **YES**, ~20 LOC |
| ShortConv, q/k L2Norm order, RMSNorm-then-gate, low-rank `z` + per-head bias | — | already faithful | no |

**The decay floor does NOT foreclose reflections** — an earlier claim of mine that was false.
`α = exp(g) > 0` for any finite `g`, so `det(Diag(α)) > 0` under *both* parameterizations. Sign is
governed entirely by β. K3 forecloses odd permutations via **strict β**, which Kimi Linear already
had, not via the new floor. fla shipped K3's exact `lower_bound = -5.0` formula in Dec 2025.

---

## What Worked

- **Verifying every subagent claim before relaying it.** An orchestrator's own agent claimed the
  headline LM contrast was "0.52σ, not significant" — it used an *unpaired* SE and discarded the
  design. Correct paired t is 7.12. Recomputing by hand caught it.
- **Reading `statusReason`, not `status`.** A Batch job stuck at `RUNNABLE` looks identical whether
  it is cold-starting or permanently unplaceable.
- **Preferring a queue with SUCCEEDED history over one that reports "Healthy."** `gpu-1xl4` had
  0 successes ever; `gpu` (A10G) had 27 with the *identical* container shape.
- **Testing the fanout command through the platform's exact `shlex.join`-inside-`bash -c` wrapping**
  before submitting. All 48 indices verified distinct, correct arm/task/bundle, own S3 prefix.
- **Validating the aggregator against the existing archive.** Staging the 48 `s4000` records as a
  fake sweep reproduced the hand-computed +11.32pp interaction — and exposed a bug in my own
  integrity check (counting eval banks per *task* rather than across arms *within* a bundle).

## What Didn't Work

- **My K3 novelty hypothesis was false.** I claimed the decay floor kills R>1's justification. It
  does not — `exp` is positive, so the determinant sign never changes. Caught by a subagent, then
  verified numerically. The repo's own `HANDOFF.md:145-151` already said the β result was
  "dominated by Grazzi Prop 1 item 3 — not novel," and I built a hypothesis it refutes.
- **The parity prediction failed** (see above). A5 shows the larger interaction, not the smaller.
- **My first pre-validation was worthless.** I wrote `images.json` asserting `critical: 0`; it
  compiled clean at $1.86. Real registry: 4 CRITICAL / 8 HIGH. **The pre-validator only checks what
  you assert.** `blocking_findings` must *enumerate* findings so each can be matched to a review.
- **Pre-validation cannot see IAM.** `gpu-1xl40s` compiled clean, passed admission, then 403'd:
  the deployed role lacks `batch:SubmitJob` on that queue even though
  `infra/iam/admission-service-roles.yaml:133-135` grants it. **The CFN template is ahead of the
  deployed stack.**
- **`HANDOFF.md:158`'s "Measured: 6.9–9.5× slower vs `chunk_kda`" was never measured.** Those rows
  in `perf.tsv` read `PENDING`; the job was blocked on Slurm QOS. Key Decision #3 hung on it.
- **I nearly shipped a `NameError`** — used `n_layers` inside a factory where only `d_model` and
  `layer_idx` are in scope. Caught before running.

## Key Decisions

1. **Route through `TransformerConfig`, not the old harness, for any future LM run.**
   `transformer/init.py:134` is the *only* caller of mixer `init_weights`, so that path fixes the
   uninitialized-gate bug structurally rather than by remembering to call something.
2. **`gpu-1xa10g` is the default profile.** 27 prior successes; the two newly-promoted L-series
   shapes are both broken in different ways.
3. **A new aggregator rather than extending `analyze_sigma.py`.** Keying on record fields instead
   of filenames makes a task axis possible and raises on duplicate cells.
4. **Fanout over 48 separate submissions.** One approval, one manifest, per-cell S3 prefixes from
   the platform's prologue.
5. **The `fla` extra goes in the shared image.** Additive and pinned; the build-time check
   *constructs a mixer* rather than importing the package, following the reasoning already at
   `.edullm/Dockerfile:169-175`.

---

## Next Steps

### 1. Decide what to write up — the science is now sufficient for a paper

Three results are in hand and mutually reinforcing:
- **β regime dominates R** by ~18× at matched cost (new, 48-cell sweep).
- **K3's strict β means R>1 should not help there** (new, and directly relevant to a frontier
  model published 2026-07-28).
- **The LM study's methodological lesson**: an endpoint can be *vacuous* rather than underpowered
  (write-up §7.2; MDE 0.010–0.018 nats vs arm differences ≤0.003).

TMLR does not require novelty. ~4 weeks.

### 2. If more measurement is wanted, the cheap high-value one is n

n=6 gives se 3–11pp on the interaction. `--fanout_size 64` at 8 bundles × 4 arms × 2 tasks is
**$64.38, still ROUTINE**. Would tighten L=256 where the effect is largest.

### 3. Everything prior to `621eaba` is suspect

The 155-record archive and the LM study both trained with an uninitialized decay gate, and **no
repeat runs exist** so it cannot be determined after the fact whether the bytes were zeros
(harmless, arm-independent) or recycled garbage (arm-dependent). Any figure quoted from them needs
this caveat. The 48-cell sweep is the only clean data in the project.

### 4. Open engineering items

| Item | Note |
|---|---|
| **Open the `l2_normalize` PR** | Required by `guides/olmo-core.md:13`. CHANGELOG entry already in `320495a`. Still unopened. |
| **Report two platform bugs** | `gpu-1xl40s` IAM gap; `gpu-1xl4` unplaceable shape with 0 successes. Both pass pre-validation and fail on AWS. |
| **`assert has_fla()` is bare** at `recurrent.py:87,617,1111` | No message. Worth a one-line fix upstream. |
| **`perf.tsv` `PENDING` rows** | The `chunk_kda` comparison was never run. Either run it or delete the claim from `HANDOFF`/Key Decision #3. |
| TBPTT gradients wrong 29–39%; double-backward wrong HVP | Judged structurally unreachable for plain AdamW (module never passes `initial_state`, no `create_graph` in `train/`), but unfixed. |
| `docs/dp2-kda/` not formally closed | Three audits recommended terminating it. Its premise is strict β, which this sweep now shows gains nothing from R. |

---

## Environment

**AWS (eduLLM platform)** — the mandated venue. `gpu-1xa10g` $1.006/hr. Routine ceiling **$500**
(`config/policy.yaml:27`); auto-approve under **$5 AND 1 h** (`:103-104`) — which is why the single
$1.01 cell self-authorized but the $48.29 sweep did too (submitter is approver, `routine_self_authorized`).
Pre-validate with `tools/compile_submission.py` before every submission; inputs at `/tmp/pv/`.

**FarmShare** (free, L40S sm_89) — socket was dead this session; nothing measured there.
`ssh -S /tmp/farmshare-ericrcwu.sock ... ` and `--exclude=wheat-01` are mandatory.

**`probes/` exists twice** — vendored in this repo (canonical, tracked) and at
`Capstone_LLM/probes` (own repo, `main`). Kept byte-identical this session; both got all fixes.

**The LM harness `/Users/ericwu/Developer/Capstone_LLM/KDA/lm/` is in NO git repository** (737
lines). It produced the headline +0.0357 nats result. Vendoring it is a precondition for anyone
else reproducing that work.

---

## Governing documents

| Path | What |
|---|---|
| `docs/kda-householder/kda-householder.md` | The write-up, 1262 lines. §7 is the LM result, §7.2 the methodological lesson. |
| `docs/dp2-kda/phase-0-1-runbook.md` | The old plan. **Recommended for termination**; its gate demands power ≥0.80 at a +5pp floor, which caps at exactly 0.50 by construction. |
| `docs/dp2-kda/evidence/` | 158 preserved records + a stratification warning (pooling across geometries inflates a contrast to +16.4pp). |
| `probes/analyze_regime_arity.py` | The sweep aggregator. Run it on a directory of records. |
| `/tmp/k3.txt` | Extracted K3 paper text (`pdftotext -layout`; WebFetch returns compressed binary). Regenerate if gone. |
| `.claude/skills/edullm-platform-runs/` | Platform skill; `references/prevalidate.md` is the offline compiler. |
