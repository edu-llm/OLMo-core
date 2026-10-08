# Proposed confirmation audit and launch route

*2026-10-08. Proposal only. This document does not authorize or certify confirmation. Prepared from protocol, implementation and pinned source metadata, without confirmation model outcomes.*

The smallest valid route keeps the five declarative review arms and adopts the successful development acquisition recipe only after its sensitivity and cost checks pass. Confirmation needs its own source catalog, independently reviewed evaluation variants, feasible schedules, GPU restart verification and frozen preregistration. The existing development-only validators should remain closed until those artifacts exist.

## What changes in the protocol

Replace the statement-only Stage-1 requirement with a shared mixed acquisition stage. Each cycle exposes every old declaration once and every supported source-QA answer target once. Two independently authored source question forms rotate across cycles. Count declaration exposures and QA target/variant exposures separately. The source QA pool ends at the shared checkpoint; Stage 2 and its buffer contain no acquisition QA.

The estimand becomes the timing of declarative review after mixed statement-and-QA acquisition. Evaluation questions are different wordings of deliberately trained facts. Report lexical overlap and independently reviewed paraphrase performance; describe this as near transfer. It cannot establish acquisition from declarations alone or generalization to novel facts.

Amend the existing human-audit requirement to documented independent agent source review before confirmation. This is a change to the previously proposed method, not evidence that human review occurred. Keep `human_review_complete: false`. Require named author and independent reviewer records, exact pinned evidence, complete coverage and explicit unresolved conflicts. The amendment must update PRD §1.3, the experiment README and validation schemas together. Disclose model-assisted auditing and its limitations in the paper. Future human checking can supplement this record.

Retain the 40% to 70% development acquisition gate, every evaluation probe, fixed confirmation E, exact paired review invariants and the prohibition on outcome-based selection. Fixed-dose development must not bypass its behavioral gate. Confirmation uses fixed E irrespective of its observed acquisition score and retains poorly acquired valid replicates.

## Verified source coverage and the schedule constraint

Read-only reconstruction used `/private/tmp/p4-spacing-schema/fict_qa.parquet`, whose SHA-256 matches the pinned source:

`c178467bfd7db74b17c2d57333bb8908f881c0e1abfbe216c7db75f0934765b7`.

The immutable outer partition, seed `20261007`, has SHA-256:

`f49376659a758f5169c6ad7f28735a8fa5a3fe880dbab9385ac758a1af1f8d1a`.

| Source partition | Events | Canonical probes | Exact source-statement units | Distinct answer targets within units |
|---|---:|---:|---:|---:|
| Development | 20 | 348 | 301 | 314 |
| Confirmation | 80 | 1,449 | 1,258 | 1,327 |
| Total | 100 | 1,797 | 1,559 | 1,641 |

These are source counts across all roles, before semantic grounding/grouping. The source has 7,500 QA rows and no cross-event links detected by the existing duplicate-root/exact-text checks. That result does not settle entity aliases or semantic overlap. Each confirmation replicate still uses 20 old, 30 new, 20 control and 10 teaching events. Ten teaching events contribute 750 original QA rows if the existing all-source-question policy is retained, subject to actual membership validation.

The development span of 84 updates does not fit typical confirmation splits under cohort size 2, review cap 8 and maximum stagger/span ratio 0.75. For N old units, stagger is `floor((N - 1) / 2)`. The feasible common span is a multiple of 21 satisfying `L >= stagger / 0.75` for every frozen replicate.

Metadata-only examples with split seeds `2026100801` through `2026100816` have 268 to 347 old source units and require up to L=231 before semantic grouping. These seeds illustrate scale, not a frozen seed selection. Their first eight splits collectively cover 77 old events, 1,399 canonical old probes and 1,279 source answer targets; sixteen cover all 80 events. Authoring nearly the entire held-out catalog once is therefore more efficient than independent per-replicate authoring.

Determine the final count from audited groups, compile every candidate manifest, and choose a common feasible L. Test acquisition and interference on development at that L before freezing confirmation. If a new L requires more development work than the declared limit, timestamp a justified development amendment before those jobs. Do not change L after observing confirmation outcomes or relax the capacity invariant.

## Required audit artifacts

Build one role-independent catalog for the 80 confirmation events, then derive each replicate's old/new/teaching subsets from its frozen assignment. The following logical artifacts can be separate files or sections of one hash-bound catalog.

1. **Pinned source inventory.** Preserve every source row/root, original assertion, canonical answer label, event and style. Bind QA, MCQ, blind-answer and factsheet byte hashes, the full dataset/model revisions, tokenizer and outer partition. Report complete probe-to-unit coverage before and after grouping.
2. **Grounded declaration and grouping ledger.** Source authors receive only original assertions, source answer labels and pinned event factsheets. They receive neither evaluation questions nor outcomes. Review every identifying-context addition against exact evidence. Preserve original assertions for explicit conflicts and retain unsupported/nonliteral-label flags. Merge equivalent assertions only with a reviewed membership rationale; sharing an answer is insufficient. Independently review every declaration and merge, retaining all original units and probes. Grounding should not create context-length truncation.
3. **Cross-event entity and fact-link ledger.** Extract entities, aliases and event-qualified subject/relation/object assertions from all 100 events. Review candidate links between development and confirmation and across confirmation roles. Shared names or answers alone do not prove leakage. Classify each candidate with evidence and an explicit disposition. Genuine cross-role fact links require an audited grouping/exclusion and partition amendment before training. Do not mark unresolved links complete. A partition correction that invalidates prior separation also requires fresh relevant development checks.
4. **Source-QA catalog.** Author two closed-book question forms per supported unit/answer target using the same source-only packet. Include every target, stable IDs, evidence and conflict handling. Independent semantic review must establish that each question requests its paired answer and identifies the intended event/relation. Preparation rejects exact normalized canonical/paraphrase evaluation forms, embedded declarations, wrong-role membership and unsupported targets. The overlap report remains diagnostic; it never selects or excludes probes.
5. **Answer and evaluation-paraphrase ledger.** A separate author/reviewer channel may see canonical evaluation questions, source answers and evidence. It produces one meaning-preserving evaluation paraphrase per canonical probe that can be old in any frozen replicate. It does not supply wording to the source-QA author. Freeze accepted aliases and normalization before any confirmation model output. Source QA membership in an answer-similarity cluster does not establish an alias. Prefer canonical-only acceptance where additional strings cannot be justified. Retain all probes with flags when source ambiguity remains.
6. **Independent review record.** Identify authors and reviewers, model/provider versions, their input packet hashes, per-record decisions and unresolved objections. Reviewers must not approve their own authored records. A second model family can provide an additional semantic review, but exact evidence checks and complete membership checks are still required. Keep human review status separate from agent review status.

Review all records, not an output-selected sample. Automatic checks verify hashes, evidence quotations, roles and coverage; agent review addresses meaning. A new schema should require both checks and the actual review record. Boolean claims alone do not establish completion.

## Minimal implementation support

Preserve the present development path and add a separately reviewed confirmation path. Do not simply delete development restrictions.

- `grounding.py`: validate a frozen, role-independent source catalog and derived per-replicate group membership. Bind its revision, outer partition, semantic audit and grouping policy. Continue validating complete source/probe retention and conflict preservation.
- `acquisition.py`: support confirmation only with a frozen reviewed catalog and preregistration binding. Derive the old subset, preserve two forms and exact per-target doses, and prohibit acquisition QA in reviews or buffer.
- `teaching.py` and `prepare.py`: validate the complete teaching-event source pool, all held-out wording exclusions, alias/paraphrase coverage and reviewed cross-event dispositions. Emit truthful human/agent review fields. Keep the existing pinned byte/hash checks.
- `training.py`: bind acquisition policy, grounding catalog, target/variant rotation, fixed E and complete shared state in the confirmation validator. Record observed acquisition separately from permission to follow the fixed protocol. Preserve exact resume and exposure audits.
- `analysis.py`: allow only the explicitly preregistered grounded aggregation/acquisition combination. Check all n manifests, margins and delays; refuse mixed recipes. Retain event/unit/probe sensitivity results and inference at the independent paired-replicate level.

Reuse `grounded_assertion_v1` and its metric version only if semantics and aggregation remain identical. Introduce a distinct audit/protocol schema for confirmation eligibility. If either unit construction or aggregation changes, version that policy/metric too and rerun the relevant development checks. Add focused tests for rejected incomplete audits, catalog/role mismatch, exact evaluation leakage, wrong acquisition doses, unsupported confirmation recipes and mixed analysis. Run the real tiny `hf_olmo` integration before deployment.

## Development evidence, GPU A/A and cost measurement

Before confirmation, complete at least one development five-arm bundle with the final recipe and enough independent UNI/EXP pairs to estimate paired variability, within the amended limit if necessary. Use whole-set canonical behavior to establish acquisition. Prespecify an assay-sensitivity criterion before inspecting development arm outcomes, using measurable NONE interference and a review-content comparison such as UNI versus NONE/GEN. Never choose the recipe because EXP beats UNI.

Add a GPU A/A restart check on the actual 371.5M model, intended L40S backend and frozen numerical settings. Starting from one shared development checkpoint, run the same short declared UNI prefix twice. One copy is uninterrupted; the other saves and restores model, Adam state, RNG, all clocks and its cursor midway. Neither enters confirmation. Bind the exact prepared manifest/code commit, software, GPU and environment. Compare realized row/content hashes, all doses/clocks, optimizer counters, model and optimizer tensors, RNG and deterministic predictions. Require bitwise-equal tensors/state on the same deterministic backend and evaluation loss agreement within a prespecified `1e-6` absolute tolerance. Investigate a failure before confirmation; do not widen tolerances after seeing it. File serialization bytes need not be identical.

CPU smoke tests establish logic, not GPU numerical reproducibility. The GPU check is a short bounded allocation, not two full spacing runs. Record its actual Slurm elapsed time and maximum memory. Baseline development showed 7.52 GB peak allocation and approximately six-minute acquisition screens, but those figures do not price the new source-QA recipe or confirmation geometry.

Measure initialization, Stage 1, continuation and buffer updates, loss/generation/MCQ evaluation, checkpoint writes/restores and allocated idle time. The new 73-declaration/78-target development cycle has 13 updates. A source-count confirmation example with 347 old units and 361 targets has 59 updates per cycle at 12 content positions per update, before reviewed semantic grouping. This is a geometry forecast, not measured runtime.

For N old groups and T source-QA targets, forecast Stage-1 updates as `E * ceil((N + T) / 12)` for batch 16 and four teaching positions. Forecast each continuation as `L + stagger + 1 + buffer_steps` updates. All five arms share one Stage-1 allocation/state. Budget full evaluation over all 1,449 canonical probes plus the frozen old paraphrases. Supervised QA diagnostics can be development-only or a frozen confirmation subset if justified before launch; primary evaluations remain complete.

Total forecast allocation is `sum_i(Stage1_i + sum_a(load_ia + train_ia + eval_ia + I/O_ia)) + setup + GPU_AA + retries`. Add Slurm elapsed and setup/storage transfers to process timing. Record the academic venue and accounting unit without inventing a dollar rate. A provisional 20% contingency is a forecast line, not a cap. Keep per-job runtime below 24 hours, task-owned retry limits and the existing node exclusion.

## Sample size and frozen launch packet

Use independent paired replicates as the uncertainty unit. Freeze the primary contrast EXP minus UNI at D*, the proposed meaningful loss margin 0.02 nats per answer token, 90% TOST and 95% directional intervals, and Holm's secondary family H1/H3/HG. A 2-point behavioral equivalence claim requires separate precision/power planning; otherwise report behavior descriptively.

Plan n from the final development paired SD and conservative larger-SD scenarios. Four development pairs give an uncertain variance estimate; show its uncertainty rather than treating it as known. Run the existing finite-sample t simulation under true differences zero, plus/minus half the margin, and plus/minus twice the margin. State the scenario required to meet 80% power. Candidate n values 8, 12 and 16 are planning points, not a maximum or an assurance.

A read-only calculation with the current planner, 20,000 draws and seed 20261007 illustrates the distinction. At paired SD 0.02 and n=16, simulated TOST power is about 96.9% for a true zero contrast but only 60.5% at a true difference of 0.01. At SD 0.04 and n=16, zero-contrast TOST power is about 23.8%. At SD 0.01 and n=8, it is about 81.7% at a true difference of 0.01. These are hypothetical planning scenarios, not measurements or an n recommendation. Increase n or explicitly choose an estimation claim if meaningful precision is unaffordable. Never increase the equivalence margin to obtain a convenient result.

Freeze one timestamped packet containing all audit/catalog hashes, n unique prepared-manifest hashes, role/order/train/evaluation seeds, acquisition policy and E, model/tokenizer revisions, full code commit, L/cohort/cap, review count, buffer/delays, loss and behavioral margins, inference/multiplicity rules, missing-pair policy, maximum attempts, actual hardware/numerical settings, power scenarios, measured cost/contingency and assay-sensitivity evidence. Register before the first confirmation training update. Operational monitoring cannot change the sample size based on main-effect significance. Infrastructure retries retain the same seed/config; valid poor performance is retained.

## Next tasks

1. Finish the isolated source-QA review and bounded adaptive acquisition screen. Fix the development fixed-dose gate loophole before any fixed-dose continuation.
2. Build the held-out role-independent source packet and audit inventory while development runs. Delegate declaration/source-QA authors and independent reviewers separately from the evaluation-paraphrase channel.
3. Derive role manifests from predetermined metadata seeds, compute audited unit counts and compile the common feasible span. Complete development sensitivity and variance checks at that span.
4. Implement the separately guarded confirmation schemas and preregistration/analysis support against real completed audit artifacts. Run focused CPU checks and the actual-model GPU A/A prefix.
5. Populate measured costs and power scenarios, choose n, finish every audit disposition, hash the n prepared manifests and timestamp/register the full launch packet.
6. Submit fixed-E shared acquisition and five matched continuations per frozen replicate only after all prerequisites pass. Keep confirmation acquisition and arm outcomes out of further tuning.
