**TL;DR:** The user authorized further assay development and repeated independent subagent review on October 7, 2026. The next bounded candidate adds source-authored QA to shared initial acquisition. All spacing-stage reviews remain declarations, and the generated-answer acquisition gate remains 40–70%. Confirmation requires a separately frozen, fully audited protocol after development succeeds.

# Acquisition amendment

The original declaration-only acquisition recipe failed its behavioral gate across the completed development candidates. The grounded candidate peaked at 15.87% old group/event-macro generated exact match at E32 and finished at 13.32% at E64. In-sample teaching QA reached 92.66%, while old-fact MCQ accuracy reached 65.64%. Saved predictions include wrong entities from the teaching pool. These observations support inadequate question-answer transfer, without identifying a unique causal mechanism.

The next recipe tests direct factual QA supervision during the shared initial learning stage. It retains the same model, original LR, event assignment, grounded assertion groups, evaluation probes, answer normalization, aliases and 40–70% gate. No confirmation outcomes inform this development choice.

## Source construction and review

- An isolated author receives only original assertions, source answer labels, reviewed declarations and pinned event factsheets. The author receives no evaluation question strings or model outputs.
- Every old assertion group retains its declaration and every distinct source answer target. The current development packet contains 73 groups and 78 targets. Each target receives two independently phrased, closed-book questions with source evidence.
- An independent reviewer checks all questions and targets against the source assertions and factsheets. Contradictory source claims remain visible; reviewers do not silently replace gold labels.
- Exact normalized overlap with canonical or paraphrase evaluation questions is rejected by the loader. Evaluation questions are available to the leakage checker, not the content author. All original evaluation probes remain unchanged.
- The packet, source inputs, review record and tokenized examples are hashed. This candidate uses a new acquisition policy and cannot be pooled with declaration-only acquisition.

The QA targets deliberately concern the same factual relations later tested. Evaluation wording is excluded verbatim, but this is near transfer of trained QA knowledge, not knowledge held out from training or evidence that declarations alone produce QA generalization. Before preparation, compute a word-token Jaccard similarity report between acquisition questions and the corresponding canonical/paraphrase probes. Freeze 0.7 as the diagnostic high-similarity threshold and report performance on lower-similarity probes alongside the full gate. This diagnostic never changes the primary probe set, training targets or acceptance threshold.

Two isolated semantic reviewers, including a different model family, judge the source packet without seeing author rationales or each other's judgments. Actual reviewer identities, agreement, source conflicts and adjudications are retained. The independent-review flag reflects these completed reviews; it is not a substitute label for human review.

## Dose and scheduling

One acquisition cycle includes one declaration per old group and one QA example per source answer target. QA wording alternates deterministically across cycles. Declaration doses and QA target doses are recorded separately; a combined cycle is not described as one declaration exposure alone.

The mixed content stream occupies the same 12 available content slots per update, with the existing four separate-event QA teaching slots. The 151 content examples require 13 updates per cycle. The recipe retains 36 loss-bearing tokens per example, the 96-token training context and complete, untruncated content. Generic filler remains an explicit part of the objective.

The next fresh screen fixes milestones at E = 1, 2, 3, 4, 6, 8, 12, 16, 32 and 64 before submission. It uses LR 1e-4 and the existing model initialization. It stops at the first usable milestone after warmup. Above 70% is a ceiling failure; below 40% through E64 is an acquisition failure. A valid failed screen is retained.

Evaluate both source-authored question variants per target as an explicitly in-sample acquisition diagnostic, separate from the gate. Report source-QA target and group doses and wording-specific exposure counts, including the one-exposure imbalance at odd E. Groups with multiple answer targets receive more QA content; report their performance separately rather than imply uniform QA dose per group.

If the first usable milestone overshoots the ceiling, retain that failure. The next candidate may reduce source-QA frequency or cap its cycles at a prespecified choice from 1, 2 and 4, while declaration acquisition continues. Such a candidate is separately frozen before running. If all milestones remain below the floor, the next candidate tests source-grounded declaration variants or a same-family scale change, with a written amendment and fresh initialization. No alias relaxation or learned-fact filtering is permitted.

Stage 2 uses the existing compiled declaration reviews only. Source acquisition QA never enters an arm or its no-review buffer. All five arms fork the same full model, optimizer, RNG and acquisition state within a replicate. Existing count, content, token-budget, endpoint and ordered new/teaching-stream assertions remain mandatory.

## Decisions after the screen

If acquisition passes, complete a five-arm development bundle and enough independent paired continuations to assess interference sensitivity, paired uncertainty and measured end-to-end cost. Select the assay from acquisition and sensitivity, without selecting a spacing treatment for its apparent advantage. Freeze the final model, content, acquisition dose, endpoint, margins, sample size and failure rules before confirmation.

For the first sensitivity bundle, the existing primary delay remains 84 updates and the full diagnostic delays remain 21, 84 and 168. Initial usability targets are a NONE old-QA loss increase of at least 0.10 nats per answer token and an exact-match decline of at least 5 percentage points from acquisition to the primary delayed endpoint. Declaration-review benefit is measured as NONE minus UNI and GEN minus UNI loss at the paired primary endpoint, with a provisional minimum of 0.02 nats. Estimate uncertainty over at least three independent shared-state continuations before confirmation; require the paired mean benefits to exceed that minimum and their intervals to establish positive review benefit. These criteria evaluate the ability to detect factual interference and review, not whether EXP beats UNI. A failed sensitivity check triggers a new assay amendment, not a claim of schedule equivalence.

New-fact QA remains a declaration-to-question transfer and interference-load diagnostic. Its acquisition differs from old-fact acquisition, so raw old/new QA accuracy is not presented as a like-for-like learning comparison. If declaration reviews cannot influence QA retention, mixed-format reviews would be a different study requiring explicit content, dose and schedule redesign.

If acquisition fails, send the unchanged results to new independent brainstorming rounds and choose another explicitly bounded development candidate. Candidate selection and every amendment are recorded before its run. The user's renewed instruction authorizes further development beyond the earlier initial screen limit; it does not authorize threshold relaxation or outcome-dependent confirmation selection.

The resulting main claim concerns declaration-review timing after mixed declaration-and-QA acquisition. It does not establish declaration-only factual QA learning or human retrieval-practice mechanisms.

## Confirmation audit

The existing human audit remains incomplete and its flag remains false. No development success alone opens confirmation. Any replacement of the earlier human-review requirement must be an explicit protocol amendment with actual independent semantic/source verification, entity and overlap checks, separately verified evaluation paraphrases, reviewer identities and honest disclosure of the review method. Until those checks and the final preregistration exist, the confirmation loader must keep rejecting the run.

## Operational preflight

Reuse the pinned model and dataset caches on FarmShare. Prior same-model screens consumed approximately 6–7 minutes including evaluation and checkpointing. The maximum acquisition trajectory here has 832 updates versus 448 in the last candidate; the provisional end-to-end estimate is 8–12 minutes, with the actual stopping milestone potentially earlier. Queue wait is unknown and reported separately. Request one GPU, four CPUs, 32 GB host memory and a 30-minute cap; exclude wheat-01. Source preparation runs separately on a CPU Slurm node. All artifacts remain on FarmShare, with remote inspection sufficient for this stage.
