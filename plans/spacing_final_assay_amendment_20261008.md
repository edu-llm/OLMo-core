# Final assay development amendment, October 8, 2026

This amendment is frozen before the scale acquisition trajectories and L=252 development continuations described here. The user authorized repeated bounded development and independent review until a valid main run can launch. It extends the earlier development limit and explicitly replaces the earlier proposed human-only review requirement with completed, hash-bound independent agent audits. Human review remains false. Confirmation training remains blocked until the actual audits, sensitivity, costs, precision plan and preregistration are complete.

## Pilot status and geometry

The completed L=84 five-arm array 1806140 is an initial pilot. Its partial and final NONE/UNI/GEN results have already been inspected; no claim of retrospective blinding is made. It established interference and a declaration-review benefit in one replicate, but it cannot establish final-span eligibility or precision. The original confirmation role counts remain 20 old, 30 new, 20 control and 10 teaching events per replicate. Geometry is unchanged by pilot outcomes.

Freeze the common span at L=252, cohort size 2 and review cap 8. The source-only first 32 seed inventory, split seeds 2026100801 through 2026100832, has at most 355 ungrouped old units and stagger177. L=252 is the smallest multiple of 21 satisfying stagger/L<=0.75 for every candidate. Grouping can only decrease the count. This metadata rule and its resulting span do not depend on arm outcomes. The source catalog SHA-256 is 69a4e36f7d2e465df991131851d99f0aaa722d01cea4f408b0412947d58c7cd6. Keep primary delay 84, diagnostics21/84/168 and buffer 168. Report all per-replicate stagger ratios and exposure-to-test clocks.

## Acquisition stability and scale

The first source-QA recipe passed at E4 with 42.15% old group/event-macro exact match. E4 is not frozen for confirmation yet. More factual content per cycle may change acquisition.

Run complete fresh development acquisition trajectories on the prespecified grid E=2,3,4,6,8 without early stopping. Preserve complete checkpoints at every milestone and disclose every result. Model, LR 1e-4, warmup 10, batch 16, four separate-event teaching positions, context96, loss budget36, source-only authoring, original gold labels and 40–70% gate remain fixed. No arm comparison chooses E.

Use six standard development trajectories, with the original split999 and independent order/train/evaluation seeds2701–2706,2801–2806,2901–2906. They retain the existing 73 grounded old units and 78 source-QA targets.

Use three additional acquisition-only scale probes with order/train/evaluation seeds3701–3703,3801–3803,3901–3903. All content remains in the 20 development events. Preserve the original teaching events event_010,event_017,event_070. Of the remaining 17 events, the two smallest ungrouped counts supply one unused new placeholder, event_004 with6 units, and one never-trained control, event_078 with7 units. The other15 events supply232 ungrouped old units and241 answer targets, before reviewed grouping. This gives40 content updates per cycle, closer to the confirmation range than the standard13. The exact scale counts and limits are reported after source review. These scale probes cannot launch review arms or enter spacing-effect analysis.

Choose a common E only from acquisition. Eligible grid doses must put every standard and scaled trajectory within the unchanged 40–70% gate. Among eligible doses, choose the scaled mean closest to 55%; ties choose the smaller E. If no grid dose qualifies, retain the failed candidate and declare a new bounded amendment before another trial. Never remove a failed trajectory, alter aliases, filter learned facts or increase the gate ceiling. A scale probe remains an approximation to confirmation size and does not establish performance on held-out events.

## Final-span sensitivity and uncertainty

Raise the development bundle limit to 8. The planned decision uses six complete fresh paired five-arm bundles at L252 and the acquisition-selected common E. All five arms within each replicate fork the same full model/optimizer/RNG/acquisition state. Existing count, content, token, endpoint and ordered-stream assertions remain mandatory. Source acquisition QA ends at the shared checkpoint. Review and buffer streams contain only the existing declarations, separate teaching and generic/interference content.

The prespecified sensitivity criteria at the primary delayed endpoint are:

- Mean NONE old answer-loss increase from acquisition is at least 0.10 nats/token, with a one-sided 95% paired t lower bound above 0, and mean old exact-match decline is at least 5 percentage points.
- Mean NONE minus UNI answer loss is at least 0.02 nats/token, with a one-sided 95% paired t lower bound above 0.
- Mean GEN minus UNI answer loss is at least 0.02 nats/token, with a one-sided 95% paired t lower bound above 0.

All criteria must pass. EXP minus UNI does not enter assay eligibility or recipe selection. Source-QA diagnostic and lower-overlap sensitivity remain separate from the gate. These criteria do not guarantee detection of every timing effect. A failed review-benefit check requires a disclosed new endpoint/review-content/dose amendment; it does not establish schedule equivalence. A failed forgetting check requires a new interference/delay amendment before more trials.

Estimate final-span EXP-minus-UNI paired SD only after the six bundles. Use its one-sided 80% upper confidence bound as the primary planning SD, with unshrunk and larger-SD sensitivity scenarios. Keep the meaningful loss margin0.02. Choose n from finite-sample simulation at true differences0, plus/minus0.01 and plus/minus0.04, with80% power as the stated planning target. The first 32 metadata seeds bound this candidate planning design. If no n<=32 achieves the stated conservative power scenario, freeze an estimation claim and expected95% CI half-width rather than promise equivalence or widen the margin. The chosen n and inference claim are frozen before confirmation outcomes.

## Confirmation audit and launch

Source authors see only pinned assertions, canonical source answer labels and event factsheets. Evaluation paraphrases are authored in a separate channel. Independent review records bind exact packet/content hashes, actual identities, coverage, evidence and objections. Cross-event audit distinguishes shared entities or answer labels from shared factual propositions. Unresolved same-proposition role/partition links remain blocking. Unknown identity and source-label ambiguity remain disclosed even if no same-proposition leakage is evidenced. Canonical-only answer acceptance remains the default; no outcome-dependent alias addition is allowed.

The actual L40S midpoint save/restore A/A check1806148 passed with source artifacts unchanged. Repeat it only if relevant numerical/training code changes. The final preregistration binds source/audit/policy/model/code hashes, all n manifests, role/order/train/eval seeds, common E and L, delays, margins, inference/multiplicity rules, hardware, measured allocation cost and missing-pair/attempt rules. Confirmation uses fixed E and retains valid poorly acquired replicates. A main acquisition-distribution check, if adopted, must be a whole-experiment arm-blind rule frozen in the preregistration, never a replicate filter or a quiet restart.

All current FarmShare jobs are terminal. The next allocations are separately bounded acquisition stability/scale checks, followed by final-span sensitivity after E is chosen. No confirmation training has launched.
