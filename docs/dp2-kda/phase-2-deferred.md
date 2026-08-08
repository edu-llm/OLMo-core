# Phase 2 deferred plan — small language-model screen

**Status:** deferred. This plan is not a launch authorization, instance choice, or spending commitment.

**Entry gate:** The program owner may activate this document only after the signed Phase-1 decision package satisfies every P1.4 criterion in [the Phase 0–1 runbook](phase-0-1-runbook.md).

## 1. Purpose

Phase 1 asks whether strict DP2 has a large, controlled synthetic-memory signal. Phase 2 asks whether the selected construction survives on natural-language training at small scale.

It can support this narrow claim:

> In a frozen all-KDA micro-LM, does the Phase-1-selected strict DP2 construction improve a preregistered long-context endpoint while remaining non-inferior to R1-P on held-out cross-entropy?

It cannot support a K3, target-model, target-hybrid, target-cache, or target-scale efficiency claim.

## 2. Activation checklist

Do not create a Phase-2 run until all items are true:

1. Phase-0 and Phase-1 decision packages are complete and immutable.
2. The P1.4 gate says “Phase 2 eligible.”
3. A single strict-beta DP2 arm is named, including factor mode, beta initialization, decay mode, and all parameter counts.
4. The exact Phase-2 arm list is frozen:
   - R1;
   - R1-P;
   - selected DP2;
   - tied-K only if the rank-two mechanism claim remains in scope.
5. The Phase-2 runner passes a source/image/manifest smoke test.
6. The program owner chooses a Phase-2 GPU shape only after reviewing the Phase-1 measured runtime/memory and a new EC2 value assessment.
7. A fresh data split and fresh Phase-2 seed sets are reserved. No Phase-1 seed, task instance, or selection result becomes a confirmation seed.

If any condition fails, Phase 2 remains deferred.

## 3. Mandatory harness changes

KDA/lm/train_lm.py is scaffolding, not a valid Phase-2 runner. Before P2A it must:

| Requirement | Definition of done |
|---|---|
| Strict-beta arm dispatch | R1, R1-P, tied-K, DP2-budgeted where applicable, and selected DP2 are named manifests; reflection is separate. |
| Same-geometry R1-P | Same \(d_{\rm model}\), heads, state geometry, and tokenization. Match the DP2 non-embedding delta only through FFN width. |
| Correct token accounting | Record loss tokens as steps × microbatch × accumulation × sequence length. The current record omits accumulation. |
| Document semantics | Use document-isolated packed data for the primary result, or explicitly label a run continuous-stream and exclude it from the primary comparison. |
| Evaluation contract | Freeze one held-out document bank, train-length CE, long-context CE lengths, and any long-memory diagnostic before training. |
| Run manifest | Store source/data/image/seed/geometry/optimizer/runtime/output information for every run. |
| Failure contract | NaN, OOM, malformed manifest, missing checkpoint, or mismatched loss-token count is a failed run, not a silent retry. |

## 4. Frozen experimental constants

The starting small-LM geometry is the current all-KDA harness geometry:

| Field | Provisional value |
|---|---:|
| \(d_{\rm model}\) | 512 |
| layers | 12 |
| heads | 8 |
| head dimension | 64 |
| training context | 2,048 |
| microbatch | 4 |
| accumulation | 4 |
| effective sequences/step | 16 |
| tokens/optimizer step | 32,768 |

These values are provisional only until P2 activation. The activated manifest must record the actual parameter counts and prove R1-P matching; changing any field creates a new Phase-2 protocol revision.

## 5. P2A — broad pilot / triage

**Purpose:** eliminate unstable or plainly poor variants and estimate paired variance. It is not a confirmation study.

| Field | Value |
|---|---|
| Arms | R1, R1-P, tied-K, DP2-budgeted, selected DP2 |
| Seed bundles | 3101, 3102, 3103 |
| Horizon | 7,630 optimizer steps |
| Loss tokens/run | 250,019,840 |
| Training jobs | 15 |
| Allowed decision | eliminate instability, clear CE inferiority, local-control failure, or malformed setup |

Use the same deterministic seed mapping convention as Phase 1, with a non-overlapping seed namespace. P2A cannot declare a winner and its seeds cannot be pooled into P2B/P2C.

## 6. P2B — fresh survivor screen

**Purpose:** select one DP2 construction for the small-LM confirmation.

| Field | Value |
|---|---|
| Arms | R1-P plus selected DP2; tied-K may remain only if rank-two wording is intended |
| Seed bundles | five fresh bundles, 3201 through 3205 |
| Horizon | same as P2A unless the approved protocol revision changes it before P2B starts |
| Allowed decision | nominate one final DP2 construction and freeze its final contrast |

P2B is still a selection screen. A non-significant result is “uncertain,” not a negative architecture conclusion. If the endpoint is saturated, floor-limited, or mixture-like, stop and issue a new protocol rather than forcing a mean-based analysis.

## 7. P2C — fresh small-LM confirmation

**Purpose:** decide whether the selected DP2 construction has earned a target-model study.

| Intended claim | Arms | Fresh paired seed bundles | Jobs | Horizon |
|---|---|---:|---:|---:|
| Practical small-LM value | R1-P, selected DP2 | 8, IDs 3301–3308 | 16 | 31,800 steps / 1,042,022,400 loss tokens per run |
| Rank-two mechanism | R1-P, selected DP2, tied-K | 8 shared IDs | 24 | same |

Do not spend eight confirmation seeds on EDA or reflection unless the program owner opens an independent pre-registered contrast.

## 8. Endpoints and inference

### Primary endpoints

1. Held-out cross-entropy at the training context, in nats/token; lower is better.
2. A frozen long-memory composite from the selected diagnostic suite; higher is better.

### Decision bounds

For paired difference \(d_i\), use:

\[
U_{95}=\bar d+t_{0.95,n-1}\frac{s_d}{\sqrt n},\qquad
L_{95}=\bar d-t_{0.95,n-1}\frac{s_d}{\sqrt n}.
\]

The proposed practical gate against R1-P is:

\[
U_{95}(CE_{\mathrm{DP2}}-CE_{\mathrm{R1\text{-}P}})\le0.010
\]

and

\[
L_{95}(M_{\mathrm{DP2}}-M_{\mathrm{R1\text{-}P}})\ge0.020.
\]

The seed bundle is the sample. Context lengths, document bootstrap resamples, and evaluation examples quantify conditional measurement uncertainty but never increase training-sample \(n\).

### Power rule

Use the fresh P2B paired variance and a noncentral-\(t\) calculation to finalize P2C \(n\). As a planning approximation:

\[
n\approx\left\lceil
\left(
\frac{(1.645+0.842)s_d}
{|\mu_{\mathrm{alt}}-\theta_0|}
\right)^2
\right\rceil,\qquad n_{\mathrm{final}}=\max(8,n).
\]

The denominator is the anticipated effect’s distance from the decision boundary, not automatically the acceptance margin. If the required \(n\) exceeds the budget, report the study as inconclusive rather than changing the criterion after seeing results.

## 9. Outputs and decision handoff

The Phase-2 reviewer must produce:

1. p2-manifest-index.json;
2. raw per-seed result table including all failures/retries;
3. paired endpoint analysis with one predeclared scalar per seed;
4. throughput/memory ledger separated from quality analysis;
5. p2-decision.md that names the result: positive, negative, or inconclusive;
6. an explicit recommendation either to activate [target-model-confirmation-deferred.md](target-model-confirmation-deferred.md) or to retain R1.

Phase 2 does not authorize target-model training by itself. The target plan has its own source, cache, checkpoint, systems, and fresh-seed gates.
