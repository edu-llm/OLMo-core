# Deferred DP2-KDA target-model confirmation

**Status:** deferred. This file is a future gate specification, not authorization to alter a target model, launch an AWS job, or claim K3 fidelity.

**Dependencies:** [Phase 0–1 execution runbook](phase-0-1-runbook.md) and [Phase 2 deferred plan](phase-2-deferred.md).

## 1. Future question

Only after Phase 2 passes, ask:

> In a frozen, source-confirmed Kimi/K3-like hybrid backbone, does replacing only the predeclared R1 KDA layers with strict DP2 improve preregistered quality/memory outcomes at acceptable measured cost?

This is a materially different question from a probe or all-KDA micro-LM result.

## 2. Entry requirements

All conditions must pass:

1. Phase 0 semantics and numerical gates pass.
2. Phase 1 has a signed positive eligibility decision.
3. Phase 2 has a fresh, paired, preregistered positive confirmation or a documented adequate-power result.
4. Official target architecture sources are pinned and independently checked.
5. Full inference state is implemented and passes prefill/decode equivalence.
6. The target runtime/kernel path has a measured target-shape memory profile.
7. The program owner explicitly authorizes a new target-model budget and AWS plan.

## 3. Primary-source architecture packet

Before using the word “K3” or “K3-faithful,” archive primary evidence for:

| Fact | Required evidence | Why |
|---|---|---|
| exact release/config revision | official report, model card, configuration, code revision | prevent variant mixing |
| KDA/MLA layer count and placement | released config and source | freeze intervention surface |
| decay equation/range | source implementation and report | identify whether bounded decay is target-faithful |
| output-gate construction | source implementation | determine whether a dense/direct gate is required |
| beta, normalization, and preconditioning | source implementation | separate DP2 from another recurrence change |
| ShortConv and cache semantics | source implementation | preserve virtual-token and decode correctness |
| head/state dimensions and dtype | config/kernel | feasibility and memory accounting |
| recipe/checkpoint provenance | documented training recipe | calibrate the scope of any replication claim |

If an item cannot be confirmed, use the narrower label “DP2 on the local KDA substrate,” not “K3 DP2.”

## 4. Frozen target intervention

When eligible:

- preserve model width, depth, tokenizer, data, context curriculum, optimizer, packing, and global-attention/MLA locations;
- replace only predeclared R1 KDA layers with R2 DP2;
- do not alter MLA/GQA cache layout in the DP2 contrast;
- keep beta range, Q/K normalization, state precision, and output-gate mode constant across R1/R2;
- use R1-P by changing FFN width only;
- include tied-K only for a rank-two mechanism claim.

Reflection and EDA remain separate research branches.

## 5. K3 gate replication sequence

Do not combine a new DP2 recurrence with a target gate change in one comparison.

1. Establish R1 parity under the current local gate.
2. Establish R2 semantics under the current local gate.
3. Port the source-confirmed target decay/output-gate path and validate R1.
4. Port that exact same path to R2.
5. Compare R1 and R2 under the one frozen target mode.

An R1 checkpoint cannot be called exactly preserved if the decay/output-gate family changes at the same time.

## 6. Checkpoint expansion requirement

Any R1-to-R2 continuation study requires a post-gate factor mask:

\[
\beta^{\mathrm{eff}}_{t,j}=m_j\beta_{t,j},\qquad(m_1,m_2)=(1,0).
\]

The expander must copy shared paths and factor-1 K/V/beta/ShortConv channels, initialize factor 2 independently, then prove dormant R2 equals R1 exactly before activation. The continuation controls are R1 continued, dormant R2 continued, and activated R2.

## 7. Full inference and systems gate

Before any serving claim, implement typed state for:

- recurrent matrix state;
- Q, K, and V ShortConv histories;
- document reset, beam/batch reorder, dtype, and placement;
- prefill-to-decode and arbitrary prefix/suffix equivalence.

After a quality survivor exists, benchmark equal-runtime R1/R2 on:

1. forward, backward, and full training step;
2. accelerator-hours to a fixed quality target;
3. allocated/reserved memory by recurrence, activations, and workspace;
4. prefill throughput/latency;
5. decode latency/cache bytes;
6. target parallelism support.

A pair-fused kernel must beat the best sequential/fused DP2 baseline, not merely a naive virtual-token implementation.

## 8. Target confirmation statistics

Use fresh paired seed bundles not used at any earlier stage:

| Claim | Minimum fresh design |
|---|---|
| practical target-model DP2 | 8 paired R1-P versus DP2 bundles, subject to power escalation |
| rank-two mechanism | the same 8 IDs across R1-P, DP2, and tied-K, for 24 jobs |

Eight is a floor, not a guarantee. Use a target-horizon pilot and noncentral-\(t\) power calculation; final inference is seed-level paired-\(t\), never seed-times-context.

## 9. Stop conditions

- Phase 2 is negative/inconclusive: do not launch target study.
- Target facts cannot be verified: narrow or stop the claim.
- Dormant R2 parity fails: do not run continuation.
- Cache/decode parity fails: prohibit serving claims.
- Memory/time exceeds the declared budget: report the systems failure rather than weakening controls.
- Final bound misses: retain R1 and report a negative/inconclusive result.
