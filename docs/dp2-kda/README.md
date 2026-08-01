# DP2-KDA experimentation program

**Canonical status:** Phase 0 and Phase 1 are planned for execution. Phase 2 and target-model confirmation are intentionally deferred behind explicit decision gates.

**Program question:** Does strict-beta DP2-KDA, which applies two ordered delta updates per real token, earn its extra cost through better long-memory behavior than ordinary KDA and fair controls?

## Read this first

This folder is the authoritative plan. It supersedes the earlier combined Phase 0–2 document.

| Document | Use it for | Status |
|---|---|---|
| [Phase 0–1 execution runbook](phase-0-1-runbook.md) | Exact work packages, artifacts, test gates, task matrix, seed schedule, and Phase-1 decision rule | execute after owner review |
| [AWS operations and cost guardrails](aws-operations.md) | Approved instance shapes, read-only preflight facts, cost equations, launch gates, and on-node scheduling | no AWS launch authorized |
| [Phase 2 deferred plan](phase-2-deferred.md) | The small-LM program that becomes eligible only after Phase 1 | deferred |
| [Target-model confirmation deferred](target-model-confirmation-deferred.md) | K3/target-model source confirmation, checkpoint expansion, cache/decode work, and target-scale confirmation | deferred |

## Program map

~~~mermaid
flowchart TD
  A["Phase 0: semantic and numerical gate<br/>g6e.xlarge, 1× L40S"] -->|all required checks pass| B["Phase 1: fresh synthetic triage<br/>p5.48xlarge, 8× H100"]
  A -->|any semantic failure| AX["Fix implementation; do not train"]
  B -->|pre-registered resource gate passes| C["Phase 2: small-LM plan becomes eligible"]
  B -->|gate fails or is inconclusive| BX["Stop or revise only with a new protocol"]
  C -->|fresh small-LM confirmation passes| D["Target-model confirmation becomes eligible"]
  C -->|fails or is inconclusive| CX["Retain R1; do not launch target study"]
~~~

No arrow may be skipped. A later phase must not be started merely because infrastructure is available.

## What DP2 means

Normal KDA performs one delta-rule memory update for each token. DP2 performs two ordered updates after a single channel-wise decay:

\[
S_{t,0}=D_tS_{t-1},
\]

\[
S_{t,j}=S_{t,j-1}+
\beta_{t,j}k_{t,j}
\left(v_{t,j}^{\mathsf T}-k_{t,j}^{\mathsf T}S_{t,j-1}\right),
\qquad j=1,2.
\]

The second factor reads the state after the first. This can create a rank-two state transition, but it does not automatically make training faster or quality better. The runbook is designed to distinguish a real mechanism from extra parameters, extra write strength, or a numerical artifact.

## Arm names

| Name | Meaning | Claim it controls |
|---|---|---|
| R1 | Ordinary strict-beta KDA | baseline |
| R1-P | R1 with the DP2 parameter delta spent only in FFN width | generic capacity |
| R1-2step-tiedK | Two sequential writes with the same key direction | sequential work without independent key directions |
| DP2-budgeted | Two independent factors sharing a total beta budget | extra write mass |
| DP2-strict | Two independent strict-sigmoid factors | practical candidate |
| Reflection | Separate exploratory \(2\sigma\) beta regime | does not belong in the main strict-DP2 claim |

## Working rules

1. **Strict beta means \(\beta\in(0,1)\).** Any \(2\sigma\) reflection result is labeled separately.
2. **The unit of inference is a seed bundle.** Different context lengths or examples from one trained model are repeated measurements, not new samples.
3. **No phase reuses selection seeds as confirmation seeds.**
4. **No K3 or target-model claim is allowed from Phase 0 or 1.**
5. **No AWS mutation is authorized by these documents.** All launch steps require a separate, concrete approval after the read-only preflight is current.
6. **The existing OLMo-core tree is dirty.** Do not create a clean worktree from HEAD until the DP2 implementation has been preserved in a reviewed commit or explicit patch bundle.

## Roles and decision rights

| Role | Responsibilities | May approve |
|---|---|---|
| Program owner | Approves scope, phase gates, spend, and claim wording | phase advancement and AWS launch |
| Implementation owner | Changes code, adds tests, pins environment, records source revision | code-ready declaration only |
| Experiment operator | Builds image, runs declared manifest, preserves all outputs/failures | operational stop for safety or corruption |
| Statistical reviewer | Checks manifest completeness, paired analysis, and decision memo | analysis-ready declaration only |

One person may hold multiple roles, but the Phase-1 decision memo must be reviewed by someone other than the person who selected the task difficulty.

## Immediate next action

Start with P0.0 in the [Phase 0–1 execution runbook](phase-0-1-runbook.md): preserve the current DP2 source state, freeze a manifest, and make the semantic test suite reproducible on the designated g6e.xlarge environment.
