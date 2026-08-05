# HANDOFF — DP2-KDA / KDA-Householder

**Last updated:** 2026-08-05. Supersedes `8972066`, `9e36632`, `2630fcd`, `0ebc9f9`.

**Status in one paragraph.** The research question was re-pointed at **Kimi K3's** KDA node and
answered by a **120-cell paired factorial experiment on AWS** (120/120 succeeded, 8.67
GPU-hours), written up as a **10-page conference paper that compiles today**
(`paper/kda-regime-arity.{tex,pdf}`). The two confounds a reviewer would raise first — capacity
and optimization — are **factors in the design**, not caveats: parameter matching leaves the
effect intact and slightly larger, and no learning rate over a 33× range closes the strict
arms' deficit. The work is **done and submittable**. Everything below is either context for
defending it or optional strengthening.

> **BRANCH:** `edullm/a5-solvability` at `c9c57eb`, pushed, clean.
> The image build only fires on `edullm/**` or `main`; an `agent/**` branch pushes green while
> **publishing no image**.

---

## Goal

Does the **Kimi K3** KDA node benefit from R>1 Householder factors per token (DeltaProduct
arity)?

K3 §2.1.1 Eq. 2 fixes `β = Sigmoid(W_β x) ∈ (0,1)` — **strict**. The question is whether arity
pays under that constraint. **Answer: essentially no. β range, not arity, is the dominant
lever** — and β range is free while arity costs +40.3%.

---

## Current Progress — the work is complete

### The paper (primary deliverable)

`paper/kda-regime-arity.tex` → `.pdf`, **10 pages, US Letter**, two-column, **7 booktabs
tables**, zero overfull boxes, zero undefined references. Author: Eric Wu, Stanford University /
Alpha AI Engineering.

**Build: `cd paper && tectonic -X compile kda-regime-arity.tex`.** Exit 0. **Only `tectonic`
0.16.9 exists on this machine — no MacTeX, no TeX Live, no `pdflatex`, no `tlmgr`** (all four
install paths verified empty). Tectonic fetches packages on demand. Do not add a conference
`.cls`: MacTeX is a 6.39 GB download against ~11 GB free and will not fit.

Title: *Write Strength, Not Write Count: The β Range Dominates Householder Arity in Gated
Delta-Rule Attention.*

Structure — note §5.4–5.6 are **Results**, not Limitations:

| § | Content |
|---|---|
| 5.1–5.3 | Prefix-metric artifact · the interaction · β dominates arity and is free |
| **5.4** | **Capacity factor** — arity separated from parameter count (`tab:matched`) |
| **5.5** | **Learning-rate factor** — a generalization gap, not an optimization failure (`tab:lrgap`) |
| **5.6** | **Replication check** the design gives for free |
| 6 | Two failed explanations (both pre-registered, both contradicted) |
| 7 | Limitations — only what the design does *not* address |

### The experiment — ONE design, three factors, 120 cells

| Block | Arms | Cells | Run id |
|---|---|---|---|
| 2×2 core | `R1`, `DP2-strict`, `R1-refl`, `Reflection` | 48 | `run_019fce2f-9b5b-70ea-8a77-6703e1b76605` |
| Capacity | `R1-P`, `R1-refl-P` | 24 | `run_019fcf14-0f7c-70f0-bbb3-7f5d6b45482a` |
| Learning rate | `R1`, `DP2-strict` × 4 rates | 48 | `run_019fcf14-6197-70b6-867c-29b766298103` |

Records at `/tmp/sweep48/`, `/tmp/matched24/`, `/tmp/lrsweep48/`. **Re-sync from
`s3://sbsandbox-intern-edullm-outputs/teams/scratch/runs/<run-id>/`** if gone. 2 tasks
(`a5_words`, `s5_words`) × 6 bundles (1101–1106), 4000 steps, one A10G, all `kda_hh`/triton.

| arm | R | β regime | non-embed params | β̄ |
|---|---|---|---|---|
| `R1` | 1 | strict (0,1) | 998,092 | 0.899 |
| `R1-P` | 1 | strict | 1,399,756 | 0.878 |
| `DP2-strict` | 2 | strict | 1,400,524 | 0.779 |
| `R1-refl` | 1 | reflection (0,2) | 998,092 | 1.800 |
| `R1-refl-P` | 1 | reflection | 1,399,756 | 1.767 |
| `Reflection` | 2 | reflection | 1,400,524 | 1.658 |

**How to reproduce the analysis.** Combine `/tmp/sweep48/` + `/tmp/matched24/` into one
directory, then:

```bash
python probes/analyze_regime_arity.py <dir>                 # unmatched square
python probes/analyze_regime_arity.py <dir> --square matched  # capacity-controlled
python probes/analyze_lr_gap.py /tmp/lrsweep48 --reference /tmp/sweep48
```

### What the experiment establishes

- **A5 mid-range interaction, capacity-matched: +58.88pp** (se 14.85, t 3.96, L95 +28.95) at
  positions 129–256. Unmatched it was +57.09 — **matching moves it up, not down**.
- **Cost asymmetry** (the most robust claim, independent of effect size): β regime is **free**
  (byte-identical ledgers at equal R) while doubling R costs **+40.3%**.
- **The strict deficit is a generalization gap, not a training failure.** Strict arms reach
  stable near-zero training loss and still score 68–89% at **L=40, inside the training range
  3–40**, where reflection arms are at 97–100%. No rate over 33× closes it.
- **K3-relevant conclusion**: K3 fixes strict β, and under strict β arity buys almost nothing at
  this scale. Whether K3-scale models benefit from R>1 is a separate question the paper
  explicitly declines to settle.

---

## ⚠️ Three corrections that must not be re-broken

These were all errors *in earlier versions of this file or the paper*. Each is fixed; each is
easy to reintroduce by quoting an old draft.

### 1. The metric is prefix-averaged, so lengths are NESTED

`probes/train_probe.py:evaluate` averages over **all** positions 1..L, and group tasks mask
nothing. The five "lengths" are not independent measurements.

- A **zero-parameter** model ("correct below 40, chance above") reproduces both strict arms'
  entire length curves to ~1.3pp. Their smooth decay is *arithmetic*, not extrapolation.
- De-nesting **reverses** two S5 cells (+8.57 → −0.77 at 129–256; +4.44 → +0.31 at 257–512) and
  **strengthens** A5 (+35.28 → +57.09).
- An old **"18.6×" headline sat on a reversed cell — withdrawn.** Use band-level ratios: 25.7×
  (A5 41–64), 124.7× (A5 65–128), 45.5× (S5 41–64).

`analyze_regime_arity.py:denest()` implements the correction and prints bands as primary. It
reproduces the paper's hand-computed values exactly.

### 2. "Unconverged" is the WRONG word — the confound is instability

Do not say the ten flagged runs failed to converge. **All ten reach 1e-4 or below after
warmup.** `loss_trace` samples one minibatch per 500 steps, and its last entry is a single batch
at step 3999 where OneCycle has driven the LR to **4e-9** — the weights are frozen. `R1`/S5/b1106
reads 1e-4 at step 3000 and 0.824 at step 3999.

**But the confound is real, just misnamed.** r = −0.47 between log final loss and accuracy;
two runs high at 5 of 7 sampled steps (sustained oscillation); 0 of 24 reflection runs spike.
Say **instability**.

Fixed at `70dc155`: records carry `loss_summary` (tail mean/median/max + `trace_min_after_warmup`).
**Never read `loss_trace[-1]` as convergence.**

### 3. Capacity matching COSTS the one Holm survivor — say so

Under Holm at α=0.05 over ten bands, the unmatched cells yield exactly one survivor (S5 41–64,
t=4.26, p=0.00401). **Matched, that band drops to t=3.64 and no band survives** — strongest is
A5 129–256 at p=0.00535 against a 0.00500 threshold.

Higher point estimates, weaker family-wise evidence. The paper reports the better-controlled
analysis and states the cost in the abstract, §5.2, `tab:paired`'s caption, §5.4 and the
conclusion. **Do not quote "one band survives" without the matching caveat.**

Also withdrawn: the "+3.50pp upper bound on the capacity confound" — circular, because both
strict arms sit within 2× chance in every extrapolation band.

---

## What Worked

- **Removing confounds by experiment rather than by argument.** The capacity bound I first tried
  to *reason* my way to was circular; running 24 cells settled it in 20 minutes for ~$3.
- **Rendering the PDF and looking at every page.** Caught `\affil{A \\ B}` collapsing to one
  line with a stray comma, and a 28pt overfull table that no exit code flagged.
- **Verifying agent claims against the data before relaying them** — in both directions. A
  subagent computed an unpaired SE on a paired design and declared the LM headline
  insignificant; that was wrong (correct paired t = 7.12). But the paper team's claim that *my*
  framing was wrong was right, and checking rather than adjudicating is what established it.
- **Programmatic number-checking after every paper edit.** A 26-number sweep against the source
  records, re-run after the reframe. Caught nothing the second time, which is the point.
- **Reading `statusReason`, not `status`.** A Batch job at `RUNNABLE` looks identical whether
  cold-starting or permanently unplaceable.
- **Preferring a queue with SUCCEEDED history.** `gpu-1xl4` had 0 ever; `gpu` (A10G) had 27 with
  the identical container shape.
- **Testing the fanout command through the platform's exact `shlex.join`-in-`bash -c` wrapping**
  before submitting. 24/24 and 48/48 indices distinct on the first try.
- **Watching the first container's live logs** before trusting 24 cells. `MATCH_ARM`,
  `FFN_SOLVE`, and the eval-bank hash all confirmed the new code path in the first 90 seconds.

## What Didn't Work

- **My K3 novelty hypothesis was false.** I claimed the new decay floor forecloses reflections.
  `α = exp(g) > 0` always, so `det(Diag(α)) > 0` under *both* parameterizations. Sign is governed
  entirely by β. K3 forecloses odd permutations via **strict β**, which Kimi Linear already had.
- **The parity prediction failed.** Predicted a large interaction on S5 (odd generator) and small
  on A5. Opposite at long lengths. Published as a finding.
- **I over-retracted once.** On finding the "unconverged" label wrong I first concluded the whole
  confound was an artifact. It is not — the instability is real (see correction 2). Getting this
  right required checking both halves rather than flipping the conclusion.
- **Three citation claims in my brief were wrong**, caught against sources: Grazzi Prop 1 **item
  3 is spectral** (item 2 concerns permutations, and item 3 carries a published correction);
  **arXiv:2606.26560 is "Erase-then-Delta Attention"** and treats the per-channel gate as a
  KDA-credited *baseline*; **DeltaProduct App. B.4 Eq. 6 is a product of RWKV-7 matrices**.
- **My first pre-validation was worthless.** `images.json` asserting `critical: 0` compiled clean
  at $1.86; the registry had 4 CRITICAL / 8 HIGH. **The pre-validator only checks what you
  assert.** `blocking_findings` must enumerate `(vulnerability_id, package_name)` pairs.
- **Pre-validation cannot see IAM.** `gpu-1xl40s` compiled clean, passed admission, then 403'd.
- **A shell poller for Batch status.** The bare `aws` CLI has no region or credentials here; only
  the `mcp__sb-aws__*` broker does, so `Monitor`/`until` loops over `aws batch` spin uselessly.
  Poll through the MCP tool.
- **Batch's `SUCCEEDED` counter lags badly.** It read 1 while S3 held 13 finished records. **Count
  objects in the S3 prefix, not array status.**

## Key Decisions

1. **One design, three factors — not a sweep plus follow-ups.** The capacity and LR blocks are
   presented as factors that exist to exclude specific alternatives. This is why §5.4–5.6 are
   Results and Limitations opens by pointing at them.
2. **Match capacity in BOTH regimes, not just strict.** The estimand is an interaction;
   controlling one contrast and not the other makes the halves differently constructed, so their
   difference mixes the correction with the effect. Worse than controlling neither.
3. **Cross each extra factor only where it is identified.** The LR factor skips the reflection
   arms — they are already at 97–100%, so no rate can close a gap that is not there. Flagged as
   the design's one deliberate asymmetry rather than hidden in the cell counts.
4. **Derive the parameter target, never hardcode it.** `--match-arm` builds the target arm and
   reads its ledger. 1,400,524 is only correct at d_model=256/3 layers/4 heads/head_dim=64; a
   hardcoded number stops matching when geometry moves, and stops in the direction that still
   runs and still reports success.
5. **Report the de-nested analysis as primary**, with prefix numbers alongside so a reader sees
   the artifact rather than taking it on trust.
6. **Publish the failed predictions and the lost significance.** Both pre-registered mechanisms
   were contradicted, and matching cost the Holm survivor. All in the paper.
7. **Tectonic, not MacTeX.** Self-contained and already working; MacTeX does not fit on disk.
8. **`gpu-1xa10g` is the default profile.** Both newly-promoted L-series shapes are broken.
9. **Route future LM runs through `TransformerConfig`** — `transformer/init.py:134` is the *only*
   caller of mixer `init_weights`, so that path fixes the uninitialized-gate bug structurally.

---

## Next Steps

Nothing is blocking. In priority order:

### 1. Decide the venue and submit

The paper is complete and defensible as an empirical/partly-negative result. No experiment is
outstanding for it. The most attackable points, all disclosed in-paper: n=6; no band survives
Holm under capacity matching; no mechanism (both candidates falsified); `DP2-strict` on A5 peaks
at the LR grid's top rate so its optimum is unbracketed; and single cells are not resolvable
below ~4pp at short lengths (§5.6).

### 2. Optional strengthening, in descending value per dollar

| Experiment | Cells | ~Cost | Buys |
|---|---|---|---|
| More bundles (n=6 → 12) on the matched square | 24 | ~$25 | The only thing that can recover a Holm survivor; se scales 1/√n |
| `DP2-strict` A5 at LR 3e-2 and 1e-1 | 6 | ~$6 | Brackets the one unbracketed optimum |
| An `R=3` arm | 12 | ~$12 | Where Eq. 3 actually predicts something (Grazzi's k≤2 fails) |
| Solvable-group control | 12 | ~$12 | Would begin to address the missing mechanism |

Pre-validate with `tools/compile_submission.py` first; `/tmp/pv/*-inputs.json` are working
templates. Submission is `gh workflow run submit-run.yml` — and **releasing the run is a human
click**; the approval API call is blocked for agents by design.

### 3. Everything prior to `621eaba` is suspect

The 155-record archive and the LM study trained with an **uninitialized decay gate**, and no
repeat runs exist, so it cannot be determined whether the bytes were zeros (harmless) or
recycled garbage (arm-dependent). The 120 cells above are the only clean data in the project.

### 4. Open engineering items

| Item | Note |
|---|---|
| **Open the `l2_normalize` PR** | Required by `guides/olmo-core.md:13`. CHANGELOG entry already in `320495a`. Still unopened. |
| **Report two platform bugs** | `gpu-1xl40s` IAM gap (CFN at `infra/iam/admission-service-roles.yaml:133-135` grants the queue; deployed role does not). `gpu-1xl4` unplaceable, 0 successes ever. Both pass pre-validation and fail on AWS. |
| **`assert has_fla()` is bare** at `recurrent.py:87,617,1111` | No message. One-line upstream fix. |
| **`perf.tsv` `PENDING` rows** | The `chunk_kda` comparison was never run. Run it or delete the claim — an earlier handoff asserted "6.9–9.5× slower" that was never measured. |
| **`probes/` is black-dirty** | Pre-existing, ~36 diff lines, none from this session's edits. `black .` covers it, so `make style-check` fails on it. |
| `docs/dp2-kda/` not formally closed | Three audits recommended terminating it; its premise is strict β, which this experiment shows gains little from arity. |

---

## Environment

**AWS (eduLLM platform)** — mandated venue, and the only one. `gpu-1xa10g` $1.006/hr. Routine
ceiling **$500** (`config/policy.yaml:27`); auto-approve under **$5 AND 1 h** (`:103-104`);
fanout ≤64. Roster handle is **`ericrcwu001`**, not `ericrcwu` — the latter is refused with
`submitter_not_in_roster`. Pre-validation needs Python ≥3.12, so use
`uv run --python 3.12 python tools/compile_submission.py`. Image tags in ECR are **12-char**
short SHAs. Read-only inspection goes through `mcp__sb-aws__aws`; the bare CLI has no
credentials.

**LaTeX** — `tectonic` only, at `/opt/homebrew/bin/tectonic`. See the build note above.

**FarmShare** (free, L40S sm_89) — socket was dead this session; nothing measured there.

**`probes/` exists twice** — vendored here (canonical, tracked) and at `Capstone_LLM/probes` (own
repo, `main`). Byte-identical as of `5989dbb`; both received every fix.

**The LM harness `/Users/ericwu/Developer/Capstone_LLM/KDA/lm/` is in NO git repository** (737
lines). It produced the +0.0357 nats result. Vendoring it is a precondition for reproducibility.

---

## Governing documents

| Path | What |
|---|---|
| `paper/kda-regime-arity.tex` | **The paper.** Build with tectonic; see above. |
| `probes/analyze_regime_arity.py` | Aggregator. `--square matched` for the capacity-controlled analysis; `--lr` to slice an LR sweep; `denest()` implements the band correction. |
| `probes/analyze_lr_gap.py` | LR-sweep analyzer. Reports accuracy at L=40 against the reflection ceiling, with fit loss beside it so the two cannot be conflated. |
| `docs/kda-householder/kda-householder.md` | Earlier write-up, 1262 lines. §7 LM result, §7.2 the vacuous-endpoint lesson. **Its probe sections use the prefix metric and are subject to the same nesting artifact.** |
| `docs/dp2-kda/phase-0-1-runbook.md` | Old plan, recommended for termination. Its gate demands power ≥0.80 at a +5pp floor, which caps at exactly 0.50 by construction. |
| `/tmp/k3.txt` | Extracted K3 paper text (`pdftotext -layout`; WebFetch returns compressed binary). Regenerate if gone. |
| `.claude/skills/edullm-platform-runs/` | Platform skill; `references/prevalidate.md` is the offline compiler. |

---

## Session commits

`621eaba` · `e96dd89` · `b146c45` · `4b5b9cf` · `76502c6` (harness + aggregator fixes) ·
`320495a` · `01713e4` · `bd2ede9` (earlier) · `0a9afa4` · `ef85c82` (paper v1) ·
`70dc155` (matched arms, `--match-arm`, `lr`, `loss_summary`, de-nesting) ·
`5989dbb` (`analyze_lr_gap.py`) · `b26bbd1` (paper: both factors) ·
`c9c57eb` (paper: reframed as one design) · plus handoff commits.
