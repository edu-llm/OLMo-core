# PRD: Reruns for the spacing study (Priority 1) and the interleaving study (Priority 2)

*Owner: P4 · Drafted 2026-10-04 · Spacing reviewed 2026-10-07 · Status: staged proposal; calibration and preregistration required before confirmation*
*Builds on:*
- *[methods_verification.md](../research/methods_verification.md): what the existing code did;*
- *[research.md](../research/research.md) Parts I–II: the literature;*
- *[research/notes/05_methods_stats_review.md](../research/notes/05_methods_stats_review.md) and [research/notes/10_final_report_methods_review.md](../research/notes/10_final_report_methods_review.md): the mock reviews;*
- *[PRD_mastery_gated_curriculum.md](PRD_mastery_gated_curriculum.md): Priority 3, a separate document.*

## TL;DR

- **Why rerun.**
  - **Spacing:** the pilot's review schedule was applied to *document-style batches*, not to individual facts. Only about half of the evaluated facts were ever reviewed, so the expanding-vs-uniform comparison wasn't really tested. The metric also mixes fact memory with format adaptation, and three seeds over one fixed set of 80 items is too thin.
  - **Interleaving:** there is one run with one order, the blocked arm's result is mostly recency, and the interleaved arm hadn't converged.
- **What we run.**
  - **Spacing (S-series):** one 300M-class model with full fine-tuning, five arms, exact per-fact schedules and exposure accounting, independent paired replicates, and several retention delays from each trajectory. Choose replication once from meaningful precision and measured cost. LoRA, another span and 1B are targeted extensions, not a compulsory factorial grid.
  - **Interleaving (I-series):** all four counterbalanced orders × seeds, evaluation after every block, training to convergence, a block-length sweep, a replay arm, a skill-similarity manipulation and a size sweep.
- **Cost.** The original GPU-hour promises were not benchmarked and must not be used as a budget. Measure complete-run training, evaluation and checkpoint costs before confirmation. Spacing uses 5n continuations plus n shared Stage-1 runs, rather than 100 main continuations and an automatic 20-run scale check. CPU validation comes first; every later stage has a measured cost forecast and a scientific reason to proceed. The user prefers a dynamic budget with no hard total cap; avoid unbenchmarked estimates and unjustified grids.
- **Code location.** The team implements this on `edu-llm/OLMo-core` (branches `anshulm/fictionalqa-review` and `p4/blocked-vs-interleaved`). The publication checkout keeps the paper workspace separate from experiment implementation; experiment code can also be inspected in local remote-tracking git objects.
- **Pipeline.** The AWS platform is retired, and the P4 code never depended on it. Everything except a thin launcher adapter can be built now; see **Part 3**.
- **Paper text.** The [RERUN] sections in [P4_spacing_paper_sections.md](../paper/drafts/P4_spacing_paper_sections.md) were drafted for the October 4 plan. Reconcile them with this reviewed protocol before assembling a manuscript.

---

# Part 1: Spacing rerun (Priority 1)

*Reviewed 2026-10-07 for scientific validity and cost. This section supersedes the October 4 spacing design and the matching [RERUN] manuscript placeholders. No experiments have been launched.*

## 1.1 Question, contribution and scope

**Question:** at a fixed replay budget, how much does the timing of repeated exposure change delayed factual retention, and what does it cost in new learning?

The main experiment tests **fixed schedule shape**, with exposure count, reviewed content and per-fact endpoint timing controlled. It also tests spaced versus massed review. It does not establish that all replay schedulers are equivalent, that adaptive replay is unnecessary, or that a human retrieval mechanism explains gradient-based restudy.

This distinction matters for prior work. FOREVER uses optimizer-update magnitude as its clock and includes replay regularization; its schedule ablation also compares increasing, uniform and decreasing intervals. SRT adapts per-example review and selection using perplexity. Neither is the same estimand as a fixed-count, endpoint-matched comparison. A null here would bound the benefit of these particular fixed shapes; it would not contradict all gains from adaptive replay. See [FOREVER, §2 and §3.4](https://arxiv.org/html/2601.03938v2) and [SRT, §3–4](https://arxiv.org/html/2608.17530v1).

The smallest defensible main study uses **one 300M-class model, full fine-tuning, five arms, one prespecified span, and several delays measured along each run**. Spend replication on this comparison before adding adaptation methods, models or schedule grids. Generality extensions have separate decisions in §1.8.

| Pilot or previous-plan weakness | Required correction |
|---|---|
| Style-level review left many evaluated facts unreviewed | Every old fact receives the declared exposures; compile and verify schedules before training |
| Review counts and last exposure were not controlled per fact | Match counts, content and first/last review for `UNI`/`EXP`; match count, content and last review for `MASS` |
| Three seeds on one event split | Independent replicate bundles with paired arms and fresh event-role assignments; choose sample size from precision and cost |
| Old-fact loss mixes factual retention and general adaptation | Raw paired retention is primary; show never-trained controls and a generic-insertion comparator; qualify drift correction |
| Weak initial learning and incomplete behavioral measurement | Development calibration, free generation with a stopping rule, and multiple-choice scoring |
| The proposed 100-cell adaptation factorial and 20-cell scale check had no measured cost basis | One adaptation method in the core; measured complete-run costs; optional contrasts only after a decision gate |
| A nonsignificant result could become an equivalence claim | Freeze a meaningful margin and fixed sample size; distinguish equivalence, useful differences and unresolved uncertainty |

## 1.2 Estimands and decision rules

Let `L(a, i, d)` be replicate `i`'s event-macro old-fact answer loss for arm `a` at the common checkpoint `d` updates after the final scheduled review anywhere in that replicate. Loss excludes EOS. Average tokens within each answer, eligible question probes within each supporting-statement rehearsal unit, units within each event, and events within the replicate. Lower is better. Retain unit-micro and all-probe micro averages as sensitivity analyses. Multiple source question IDs for the same statement do not create additional rehearsal doses or independent facts.

| ID | Contrast at the prespecified primary delay `D*` | Interpretation |
|---|---|---|
| **S-H2, primary** | `EXP − UNI` | Fixed schedule shape at matched per-fact first/last exposure and dose |
| S-H3 | `UNI − MASS` | Distributed versus consecutive review, with last review and dose matched |
| S-H1 | `UNI − NONE` | Practical effect of spending part of a fixed training budget on review, including displaced new data |
| S-HG | `UNI − GEN` | Old-fact review versus a specified generic-text insertion at the same positions and new-data budget |

`UNI − GEN` is evidence about the value of the reviewed content relative to that control. Generic web text is not a perfect match for old facts' domain, difficulty or gradient magnitude, so do not describe this as complete identification of a memory mechanism. Report old- and new-fact outcomes together for every contrast.

For S-H2, propose **ΔL = 0.02 nats per answer token** as the smallest loss difference worth acting on. This corresponds to about a 2% ratio in geometric-mean gold-token probability; it is not a 2% change in answer accuracy. Explain this choice in the preregistration and freeze it before confirmation. Do not define ΔL as a fraction of the review gain observed in the main data, or widen it after seeing uncertainty. A proposed behavioral equivalence margin is **2 percentage points of generated exact match**; justify and freeze that separately if a behavioral-equivalence claim is planned. These are design choices, not validated universal thresholds.

- Report the paired estimate, its 95% CI and its 90% CI for S-H2. A 90% CI strictly inside `[-ΔL, +ΔL]` establishes loss equivalence by TOST at α = .05.
- A 95% CI wholly below `−ΔL` or above `+ΔL` supports a practically meaningful directional difference. A point estimate beyond ΔL with a CI spanning it does not.
- Otherwise, report the precision actually achieved. A small statistically detectable effect can still be practically equivalent. Failure to reject zero does not establish equivalence.
- A loss-equivalence conclusion remains a loss-equivalence conclusion unless the prespecified behavioral equivalence test also passes. Report discordant loss, exact-match and MCQ results.
- Test S-H1, S-H3 and S-HG as one prespecified secondary family with Holm correction. Report raw estimates and 95% CIs as well as adjusted p-values. All other contrasts and delays are descriptive unless assigned a separate correction in the preregistration.

The former S-H4, the during-review advantage, and S-H5, the forgetting-rate claim, become secondary trajectory analyses. Do not make "same forgetting rate" a required success condition. Rate depends on outcome scale and initial performance; an insignificant arm-by-time interaction proves little.

## 1.3 Data, development and evaluation separation

- Pin FictionalQA and every used configuration to revision `131cb74fdc3e601b5e896ed768ad9852ea35a8f9`; validate the joins among `fict_qa`, MCQ and blind-answer metadata. The audited source contains 100 events and 7,500 QA rows before canonicalization and filtering.
- **Reserve 20 events for development once**, before any new calibration. Use the remaining 80 only for confirmation. Hash and publish this outer partition. Development outcomes may set training lengths, formatting, LR, spans and test budgets; confirmation outcomes may not.
- In each confirmatory replicate, independently assign the 80 confirmation events to **20 old, 30 new, 20 never-trained controls and 10 QA-format teaching events**, balanced by the available style metadata. Every arm in that replicate uses the same assignment. Separate RNG seeds cover split assignment, ordering, adapter initialization if applicable, and evaluation. This is replication conditional on the available corpus and pretrained checkpoint, not independent pretraining or evidence about arbitrary factual domains.
- Development runs use a smaller disjoint event pool. Keep their data geometry and differences from confirmation explicit. Their variance is a planning estimate, not proof of the precision of a larger confirmatory split. Supplement it with a conservative variance sensitivity analysis rather than treating the old three-seed pilot as a reliable power estimate.
- Deduplicate at the complete duplicate-cluster level before splitting. Keep every version of an event, fact, answer and supporting statement in one role. Audit cross-event entity aliases, duplicated facts and supporting text. Exclude or group genuinely linked events by a rule fixed before inspecting confirmation outcomes. If grouping changes the available counts, publish a corrected manifest before training.
- Default to one canonical fact statement per duplicate cluster and **no blind-guessability filter**. Also group identical normalized supporting statements within an event into one stable rehearsal unit, retaining every eligible question probe. One complete statement exposure counts once per unit. For units with distinct answer labels, preserve the complete source statement; do not choose a rewrite that omits another probe's target. Audit near-duplicate meanings and cross-event links separately; exact-text grouping does not settle semantic equivalence. Pretraining and QA-format effects are measured with baselines and never-trained controls. If a filter is needed, choose it on development data and apply the frozen metadata rule to all roles; keep an unfiltered sensitivity set. Never filter confirmation facts because the trained model learned or forgot them poorly.
- Stage 1 and review use declarative statements. A development-selected relation-first representation must preserve the original assertion, retain source text and audit hashes, and keep evaluation questions out of training. QA-format teaching uses distinct events in every arm at the same prespecified rate and positions. A broader teaching recipe can include source QA variants from those events, each paired with its own source answer rather than a duplicate-root answer. Freeze the teaching pool and rate after acquisition calibration. It may improve format transfer; do not assume that it removes all format effects.
- Evaluate all eligible facts in each old/new/control role. One canonical QA form is the primary loss probe. Add one verified, meaning-preserving question paraphrase per old fact for behavioral transfer, created and checked before training. Training sees neither evaluation question form for those facts. Hold variants within their fact; they are not independent samples.
- Audit every training statement and question/answer span against the tokenizer and context limit. No truncated gold answers or supporting facts. Freeze answer aliases, normalization, MCQ distractors and exclusions before confirmation; retain an audit sample of raw predictions.

"Held out" means held-out question wording or never-trained events, as applicable. The answers and supporting statements for old facts are deliberately trained. Do not describe these as unseen facts.

## 1.4 Arms, schedule construction and exposure accounting

Use **k = 4** reviews per old fact in the core. Use one `L` selected for assay sensitivity during development, with **expanding gaps in the ratio 1:2:4**. At k = 4, r = 1.35 is a weak shape manipulation. Retain it only as a later pilot-recipe bridge, not another compulsory grid axis.

Choose `L` as a multiple of 21 updates so both `UNI` and `EXP` have exact integer offsets. For fact `f`, let its first spaced review be `s_f` and its last be `e_f = s_f + L`.

| Arm | Review offsets from `s_f` | What occupies review positions |
|---|---|---|
| `NONE` | No old-fact review | Additional new-fact training at the matched total budget |
| `UNI` | `0, L/3, 2L/3, L` | The same canonical statement for that fact at every exposure |
| `EXP` | `0, L/7, 3L/7, L` | Identical old statements and per-fact counts to `UNI` |
| `MASS` | `L−3, L−2, L−1, L` | Identical old statements and counts; four consecutive optimizer updates |
| `GEN` | The complete `UNI` insertion manifest | A pinned generic-text sample, length matched to the displaced review material |

For illustration only, `L = 84` gives `UNI = [0,28,56,84]`, `EXP = [0,12,36,84]`, and `MASS = [81,82,83,84]`. This example is not a selected training duration. Massing cannot match both endpoints of a spaced schedule; match the last exposure and state the remaining difference.

**Compile all schedules on CPU before reserving a GPU.** Starts are assigned independently of fact difficulty, stratified by event and statement length where feasible, and paired across arms. Reject an infeasible manifest. Never defer, drop, duplicate or shift reviews at runtime to satisfy a batch cap.

A simple capacity-safe starting construction is to assign small cohorts to successive start steps. With k distinct offsets, at most k cohorts can be due at one step. If each cohort has at most q facts and the review cap is Bᵣ sequences, `kq ≤ Bᵣ` bounds occupancy in every arm. For batch size 16 and a cap of 8 review sequences, q = 2 is feasible before token-length checks. Use the same cohorts across arms and log their membership. A more efficient solver is acceptable if it satisfies the same invariants.

The compiler must also show that `number_of_old_facts × k` fits the available review slots. Record the width of staggered start times relative to L, and avoid letting that width dominate the intended spacing manipulation. A short 180-update phase may not fit all facts. Choose the duration from the actual manifest and the desired old/new token allocation; do not inherit the pilot's duration or replay percentage.

Required invariants:

1. **`UNI`, `EXP` and `MASS` have the same complete training-example multiset**, including old, new and QA-format material, with the same tokenization and loss masks. Only the permitted schedule/order changes. Preserve the within-stream order of new facts and QA teaching; document any changed placement imposed by review timing.
2. Match global optimizer updates, batch shapes and total non-padding loss-bearing tokens among arms. Padding tokens are a separate compute count. Use deterministic length-aware packing and a fixed generic token source to make the `GEN` replacement budget exact without truncating facts. If equalization needs fillers, declare and include their counts; do not silently add extra factual exposure.
3. `UNI` and `GEN` have the same new-fact examples, counts, placement and token budget. `NONE` intentionally has more new-fact exposure under the fixed total budget. Its contrast cannot identify review independently of displacement.
4. First/last old review steps match per fact in `UNI`/`EXP`; last steps match in `MASS`; every reviewed old fact has exactly k exposures. Common Stage-1 exposure histories are identical across all five arms.
5. The no-review buffer contains identical new/QA material in the same order in every arm, with no old facts or their duplicates. Log updates and cumulative new/QA tokens since each fact's last review. A step is not a biological time unit, and constant LR does not make optimizer update norms equal.
6. Log the entire planned and realized exposure manifests, including per-fact times, content hashes, source role, tokens, loss weights and per-step occupancy. Abort on a mismatch. Assert optimizer-state and sampler-state restoration when forking or resuming.

Do not randomize schedules among facts inside a single model as a substitute for independent arm runs. Facts interact through shared parameters; that cheaper design estimates a different treatment and can hide interference between arms.

## 1.5 Model, training and calibration

- **Core model:** `allenai/DataDecide-dolma1_7-300M`, audited checkpoint `step45787-seed-default`, revision prefix `4b1b42ff`, with 371.5M base parameters. Resolve and record the full immutable revision and tokenizer hash before execution.
- **Core adaptation:** full fine-tuning. This avoids making the main retention finding conditional on one LoRA rank. It is not automatically cheaper or more expensive in elapsed time; profile it. If available hardware makes full FT impractical, choose LoRA before confirmation and narrow the claim, rather than changing method after seeing schedule results.
- AdamW β = 0.9/0.95, weight decay 0.1 and clip 1.0 are starting settings. Select the LR from at most three development candidates using stable acquisition and new-fact learning, not the `EXP − UNI` contrast. Freeze one LR and one constant-after-warmup schedule for every arm. Record global loss normalization and effective batch size.
- Warm up once before schedule comparisons begin. Carry optimizer moments, global step, LR state and RNG/data state across stages. A shared Stage-1 checkpoint includes all this state, not just weights. Each replicate has its own Stage-1 run; branch it into the five arms only when their manifests first diverge.
- Stage 1 gives every old fact exactly E complete exposures with shuffled, balanced ordering. Use a bounded, increasing development schedule to find the smallest E with usable behavioral acquisition. A suggested calibration target is 40–70% old-fact generated exact match, away from floor and ceiling. If this is unattainable within the calibration cap, diagnose format transfer or data construction before spending on confirmation.
- Freeze E after calibration. Do not stop each confirmatory replicate or arm on its own test score, and do not discard poorly learned confirmation facts. Show the whole Stage-1 acquisition distribution.
- Choose the shortest `L`, Stage-2 duration and primary buffer delay `D*` that produce measurable interference without collapsing all arms to chance on development data. Verify that `NONE` changes over the buffer and that review can affect the measurements. Require both acquisition and assay sensitivity; selecting settings because expanding wins is prohibited.
- Use bf16 if supported and preserve the same numerical settings across arms. Verify that microbatch accumulation, padding and reduction implement the declared loss. An A/A restart check should reproduce the same exposure sequence and agree within a recorded numerical tolerance.

Budget calibration itself. Start with one bounded LR/acquisition screen, then at most four independent development replicate bundles for the final recipe, completing the five-arm diagnostic on at least one bundle and enough paired `UNI`/`EXP` continuations to estimate variance. Timestamp that limit before running. If it cannot establish a usable assay or a credible cost forecast, revise the assay rather than launching a wide grid. Development results are reported separately and are not pooled into confirmation.

## 1.6 Measurement with bounded evaluation cost

Use one training trajectory per arm through the longest prespecified buffer. Multiple delays require extra evaluation, not fresh retraining. Suggested checkpoints are pre-Stage-1, end of Stage 1, a small fixed grid during Stage 2, global buffer onset, `D*/4`, `D*`, and `2D*`, rounded once in the manifest. Confirm the longest delay is affordable before freezing it; never pick the primary delay after reading confirmation curves.

Because facts have staggered final reviews, "buffer onset" is not an immediate post-review measurement for every fact. Report each checkpoint's range and distribution of per-fact review-to-test delays. `UNI`, `EXP` and `MASS` have identical delay distributions within each replicate. A delay curve is indexed by the global buffer clock unless explicitly computed per fact.

- **Primary:** raw answer-token cross-entropy excluding EOS, with the event-macro aggregation in §1.2. Include EOS loss separately for comparison with the pilot. Report exact masks and denominators.
- **Behavioral checks:** actual greedy generation, maximum answer length, stop sequence/EOS rule, normalization and accepted aliases fixed on development. Save unnormalized outputs. A teacher-forced all-token argmax match is at most an answer-prefix check and does not test free-generation termination.
- **MCQ:** score full candidate answers by the sum of conditional answer-token log probabilities, excluding EOS, with a frozen prompt, fixed option text and paired, balanced option orders. Report length-normalized scoring as a sensitivity if candidate lengths differ. Report accuracy and correct-versus-distractor log-likelihood margin. MCQ recognition and free recall support different claims.
- **Transfer:** report canonical and held-out paraphrase questions separately, averaging probes within rehearsal units before event aggregation. Do not multiply the effective sample size by the number of variants. In-sample QA-teaching performance is a separate debugging outcome and never substitutes for old-fact acquisition.
- **Plasticity and general performance:** report new-fact loss and accuracy beside retention. Run a small frozen generic-language loss set at Stage 1 end and the final checkpoint as a diagnostic of broad degradation.
- **Drift:** show old and never-trained control trajectories separately. As a secondary sensitivity, report the difference of their changes from the shared Stage-1 baseline. That correction assumes control and old items share the same nuisance drift. Differences in difficulty and cross-fact transfer can violate the assumption; disagreement with raw outcomes must be visible.
- **Curves:** compute a time-weighted trapezoidal AUC over Stage 2 only. Report the no-review buffer separately. Use discrete interval changes or a prespecified log-time curve for forgetting; do not assume one linear slope or infer equal rates from nonsignificance.

Batch and cache teacher-forced scoring. Run full old-fact loss at the prespecified checkpoints; run expensive full generation/MCQ at Stage-1 end, buffer onset, `D*` and the longest delay. During-training generation may use one frozen stratified panel for debugging, never as a replacement for the full primary evaluation. Score new and never-trained facts on a sparser fixed grid if profiling shows a worthwhile saving. Pin every evaluation subset before confirmation and share it across arms.

## 1.7 Replication, power and inference

**The training replicate is the primary uncertainty unit.** Facts within a run share parameters and event content. Hundreds of questions cannot substitute for independently repeated arm comparisons.

- For each contrast, first reduce each replicate to its paired difference. Primary intervals and tests use those paired values. Publish every replicate, its split seed and initialization/order seeds. The n count is the number of complete independently assigned replicate bundles.
- Analyze item heterogeneity secondarily with event/fact and replicate structure. If events recur across replicate assignments, keep their stable IDs and account for recurrence. Use event-level resampling or mixed models as sensitivity analyses; do not bootstrap individual tokens or question variants as independent observations. With a small n, complex mixed-model fits should not replace the paired result.
- Fix n before confirmation. Use development paired standard deviations, their uncertainty and plausible larger-variance scenarios to calculate power for the prespecified directional claim and TOST at ΔL. For TOST, show scenarios with true contrasts of zero and ±ΔL/2; for a CI wholly beyond ΔL, plan against a true contrast larger than ΔL, not exactly on its boundary. A rough precision forecast is `t_(.975,n−1) × s_difference / sqrt(n)`; use finite-sample power calculations or simulation for the final design. Target at least 80% power under the prespecified effect scenario and report the scenario. For behavioral equivalence, plan its precision too.
- Planning candidates are n = 8, 12 and 16, rather than an automatic ten. They correspond to 40, 60 and 80 continuations on one adaptation method, plus n shared Stage-1 runs. Select the smallest candidate that meets the precision goal at a reasonable measured cost. These counts are planning choices, not assurances that 16 is enough or a maximum permitted sample size.
- If the required n is substantially larger or costlier than these candidates, compare the scientific value of that precision with a narrower estimation claim before confirmation. More replication is justified when it answers the main question; do not inflate the margin to make an underpowered design look conclusive. There is no preset total spending ceiling.
- Do not inspect the main effect after each few runs and stop when a p-value or CI becomes attractive. If unexpectedly high costs force an operational stop, record the reason and report the resulting uncertainty and incomplete design. Any valid sequential alternative needs its stopping boundaries and error control specified before data collection.
- Infrastructure failures may be rerun under the original seed/config with a logged reason and capped attempts. A divergent or poorly performing valid run is a result, not an infrastructure failure. Report missing arm pairs and use a preregistered rule for the primary analysis; do not quietly replace them with new seeds.

The primary inference is conditional on this benchmark, pretrained model, adaptation method, k, span and interference regime. Even well-powered equivalence at one setting cannot establish that scheduling generally does not matter.

## 1.8 Extensions chosen for a stated question

Do not launch a factorial expansion. Freeze each extension's arms, sample size, endpoint and budget before its new runs. A setting chosen after seeing the core is a follow-up; use new replicate bundles for confirmatory evidence and do not pool it with the core as if selected in advance.

| Priority | Extension and minimum useful arms | Launch condition and claim it supports |
|---|---|---|
| 1 | A second, prespecified spacing span with `UNI`, `EXP`, `MASS`; keep dose and per-fact final review matched within that span | The core is a usable assay and a claim about span sensitivity is worth the measured cost, whether the core shows a difference, equivalence or a boundary condition. This is preferable to testing many ratios at one span. Cross-span comparisons must state which endpoints moved. |
| 2 | Same-model LoRA bridge, initially `UNI`/`EXP`; include `MASS` or `NONE` only for the corresponding claim | Needed to connect to the old pilot or test adaptation sensitivity. Use r = 16, α = 32, dropout 0.05 and all linear layers for the pilot bridge; recalibrate acquisition fairly on development data. Do not repeat every control by default. |
| 3 | A 1B model in the same family and **the same adaptation method**, with the contrasts supporting the core finding | Broader scale is needed, memory/throughput are measured, and the comparison can reach useful precision. A 300M full-FT versus 1B LoRA comparison cannot identify a scale effect. Pin the checkpoint first. |
| 4 | A second factual task or model family | Needed for claims beyond FictionalQA or this family. Choose one orthogonal generality check rather than a large model ladder. |
| Claim-dependent | One adaptive scheduler plus uniform-random replay and, if needed, a difficulty-prioritized control | Required before recommending fixed uniform review over adaptive policies. Equalize total replay/new tokens, document per-item dose differences and charge scoring/scheduling overhead. This changes the estimand from fixed-count timing to allocation policy. |

If the intended paper is specifically about adaptive replay, move the last row ahead of scale rather than adding it on top. A practical scheduler comparison needs measured cost versus retention/new-learning quality, and sensitivity to tuning effort. A timing-only ablation cannot stand in for the full FOREVER method; teacher-forced QA review is not a model retrieval attempt.

`CON`, added-budget review, more expansion ratios, an LR-decay interaction and a large LoRA-rank sweep are deferred. Added-budget review can address displacement, but it answers a different compute question and must show its extra tokens and cost. Do it only if that question becomes central.

## 1.9 Stages, measured cost and stop decisions

| Stage | Deliverable | Decision before spending more |
|---|---|---|
| **S0, CPU work** | Source-data audit; immutable event partitions; schedule compiler and invariants; metric checks; tiny-model A/A/resume smoke; draft preregistration and cost ledger | Every proposed arm is feasible. Recover the pilot logs in parallel; missing pilot logs do not block a well-specified new experiment. Old retrospective TOST remains exploratory. |
| **S1, bounded development** | Acquisition and interference calibration; complete-run benchmark on intended hardware; paired variance scenarios; chosen endpoint, margin, n and manifest | Freeze all choices and timestamp the final preregistration **after calibration and before confirmation**. The assay is usable and the measured core cost forecast is proportionate to the scientific question. |
| **S2, confirmation** | One model, full FT, five arms × fixed n; shared Stage-1 state per replicate; frozen delayed evaluations | Complete the declared experiment and analyze it once. No silent extensions to obtain a preferred result. |
| **S3, targeted follow-up** | At most the extension that answers the next unresolved question | A written scientific reason, sample-size rationale and measured incremental cost forecast exist. No preset total cap is required. Scale is optional. |

**The old ≤20 GPU-hour spacing estimate is withdrawn.** The pilot's token count does not price stronger initial acquisition, all-fact evaluation, free generation, full FT or the new schedules. The interleaving estimate elsewhere in this document is also unbenchmarked and is not a combined-project budget.

Measure separately on the intended hardware: initialization/download, Stage 1, Stage-2 plus buffer training, each evaluation mode, checkpoint writes/restores, and teardown. Include billed idle allocation. Record peak memory and effective tokens per second with the actual sequence lengths and evaluation set. Prefer one GPU if it fits; choose hardware by complete-run cost, not its hourly rate alone. Free academic compute still has a GPU-hour and elapsed-time cost.

For n replicates and five arms, the total allocated time is approximately:

`H_core = sum_i [H_stage1_i + sum_a(H_load_ia + H_train_ia + H_eval_ia + H_io_ia)] + H_shared_setup`

`C_total = C_development + sum_jobs(rate_per_allocated_job × billed_hours × attempts) + C_storage_and_transfer + C_extensions`

Use the rate for the whole allocated job, not a per-GPU rate accidentally applied to a multi-GPU node. GPU-hours are the sum of GPU count × allocated wall-hours. Account for Stage 1 once per replicate, never once per arm. A LoRA bridge needs its own Stage 1; adapters cannot be forked from the full-FT Stage-1 state and called the same experiment.

Cost controls:

- Cache tokenized data, prompts, candidate answers and model weights; reuse the shared Stage-1 state and any later prefix that is byte-identical before the first differing update.
- Evaluate frozen delays in memory where possible. Keep restart checkpoints and the few checkpoints needed for audit or later rescoring, not full optimizer snapshots at every metric tick.
- Use a cheap frozen panel for development diagnostics and full declared sets for confirmation. Never claim savings by reducing independent replication while counting more correlated questions.
- Set maximum attempts and a hard allocated-time cap per job; recover at checkpoints. Failed jobs, calibration, evaluation and storage all count toward cost.
- Put the measured quote, selected n and a separately stated contingency in the run plan. A provisional 20% contingency may be used for forecasting, then updated from observed overhead; it is not a spending limit or a measured failure rate. Track forecast versus actual cost after each operational batch without using main-effect significance to decide continuation. Diagnose substantial runtime overruns or repeated failures before launching more jobs. Choose one useful extension at a time and reforecast it; avoid multiplying all models, adaptation methods, schedules and seeds into an automatic grid.

**Fields that must be filled before S2:** hardware/venue and billing unit; development spent/remaining; actual full-FT peak memory; measured Stage-1/continuation/evaluation/checkpoint durations; frozen E/L/Stage-2 length/`D*`/maximum delay; n and predicted precision; core cost and contingency; maximum attempts; next stage decision and its scientific justification. Until then, report counts and formulas rather than invented dollar or GPU-hour promises.

## 1.10 Acceptance criteria

- [ ] The outer development/confirmation partition, per-replicate event roles, fact clusters, tokenizer and content manifests are immutable and audited.
- [ ] Every schedule passes count, endpoint, capacity, content-multiset, token-budget and no-leakage assertions before launch; realized logs agree.
- [ ] Acquisition and forgetting are measurable on development data; confirmation settings were not selected for an arm's advantage.
- [ ] The final preregistration fixes the primary endpoint/delay, meaningful margins, n, multiplicity and failure policy before S2; the measured cost forecast and dynamic budgeting policy are recorded.
- [ ] All five paired arms use shared Stage-1 optimizer/RNG/data state within each replicate; A/A and resume checks pass.
- [ ] Results include raw loss, generated exact match, MCQ, new-fact learning, never-trained drift, per-replicate contrasts and actual costs.
- [ ] Claims distinguish a loss bound from behavioral equivalence, finite fixed schedules from adaptive policies, and model-time interference from human memory.
- [ ] Every number in the paper traces to a versioned result file and analysis script; manuscript [RERUN] sections are reconciled with this protocol before use.

---

# Part 2: Interleaving rerun (Priority 2)

## 2.1 Problems this rerun fixes

| Pilot problem (methods_verification.md Part II; research/notes/10 §B1) | Fix |
|---|---|
| One run, one order (A→B→C→D) | 4 counterbalanced orders × 5 seeds |
| Blocked accuracy is mostly recency (D 94.5%, others 0–15%) | Evaluation after every block; per-skill learning and forgetting curves |
| Interleaved arm not converged (training loss 0.21 vs. 0.02) | Train to convergence; report curves |
| Interleaving confounded with spacing and re-exposure | Block-length sweep (cyclic) and a blocked + replay arm |
| Four skills share inputs and differ only in their bracket cue | A second, low-similarity skill set |
| Random-init 162M only | Pretrained small models and a size sweep |
| Adam momentum may blur skill-pure batches | Momentum diagnostic |

## 2.2 Hypotheses

| ID | Hypothesis |
|---|---|
| **I-H1** (primary) | Averaged over counterbalanced orders, interleaved training gives higher final macro accuracy than blocked training (after the interference phase). |
| I-H2 | Accuracy increases as block length decreases (dose–response across *L*_b ∈ {128, 32, 8, 2, 1}). |
| I-H3 | Blocked + 10% replay closes most of the gap, i.e. the interleaving advantage is mainly re-exposure. |
| I-H4 | The interleaving advantage is larger for the high-similarity skill set than for the low-similarity set. |
| I-H5 | The advantage changes with model size (direction not pre-specified). |

## 2.3 Design

- **Code:** the existing `generate_data.py` and `train_compare.py` on `p4/blocked-vs-interleaved`, extended by the team.
- **Skill sets:**
  - **High similarity:** the current four transformations of 6-digit strings, each cued by its delimiter (rotate left, rotate right, reverse, swap pairs).
  - **Low similarity:** [TODO: define; candidates are sort ascending, digit sum mod 10, duplicate each digit, count of a target digit]. Use the same generator rules: disjoint train/test digit strings, and outputs that differ across skills.
- **Arms:**
  - **Block-length sweep:** blocks of *L*_b updates per skill, cycled until the budget ends, with *L*_b ∈ {128, 32, 8, 2, 1}. *L*_b = 128 with a single cycle is the pilot's blocked arm; *L*_b = 1 is the interleaved arm.
  - **`BLOCK+R`:** blocked with *L*_b = 128, plus 10% of each batch drawn from previously seen skills.
- **Orders:** 4 sequences forming a Williams square, so that each skill appears once in each serial position and each ordered pair of adjacent skills appears once.
- **Budget:** [TODO: number of epochs, e.g. 4 epochs = 2,048 updates], chosen in calibration so that the interleaved arm converges on all skills. All arms use identical examples, tokens and constant LR.
- **Interference phase:** 160 updates of generic text after the curriculum (as in the report's proposal), with an evaluation before and after.
- **Models:**
  - the random-init 162M OLMo-architecture model (as in the pilot);
  - pretrained Pythia-160M and Pythia-410M (continued training);
  - Pythia-1B for the size sweep [TODO: confirm checkpoints].
- **Seeds:** 5 per order → 20 paired runs per arm.
- **Momentum diagnostic:** repeat `BLOCK` vs. `INTERLEAVED` with β₁ = 0 on one model, to check whether single-skill batches act like mixed batches through Adam's moving average.
- **Measurement:**
  - per-skill exact match after every block and every 64 updates;
  - final macro accuracy, before and after interference;
  - forgetting per skill (peak minus final);
  - the area under the macro-accuracy curve.
- **Analysis:** paired differences averaged over orders × seeds; a mixed model with skill, order and seed random effects; Holm correction across I-H2–I-H5.

## 2.4 Phases and compute

| Phase | Work | Compute |
|---|---|---|
| **I0** (now) | Build the low-similarity generator, Williams-square orders, per-block evaluation and the replay arm; write the preregistration. | none |
| **I1** | Calibrate the budget for convergence (1 order × 2 seeds) | <1 GPU-h |
| **I2** main | (5 block lengths + `BLOCK+R`) × 4 orders × 5 seeds × 2 skill sets = 240 runs on the 162M model | ~5–8 GPU-h (the pilot pair took about 71 s) |
| **I3** scale | Pythia-160M/410M/1B, `BLOCK` vs. `INTERLEAVED` vs. `BLOCK+R`, 4 orders × 3 seeds | ~5–8 GPU-h |

## 2.5 Acceptance criteria

- [ ] All four orders and all seeds have been run, with per-block curves committed.
- [ ] The interleaved arm reaches its training-loss plateau within the budget.
- [ ] The report describes the actual skills (not "subjects") and reports no uncommitted numbers (the toy-pilot figures are removed unless their code is found).


---

# Part 3: Implementation that does not depend on the compute pipeline

*Added 2026-10-04 after surveying every edu-llm repository ([research/notes/11_compute_pipeline_survey.md](../research/notes/11_compute_pipeline_survey.md)).*

## 3.1 What we found
- **The AWS platform is retired.** The `edu-llm/platform` README has said "PLATFORM FROZEN (2026-08-18)" since that date, and edullm-p1 commits from 2026-09-22 to 2026-09-27 remove its AWS, S3 and RunPod paths. No replacement pipeline is named in any repository we can read.
- **The P4 code never depended on the platform.**
  - The spacing code (`anshulm/fictionalqa-review`) has no `.edullm/` folder, reads no `EDULLM_*` variables, and runs "on a CUDA machine" through its own `olmo-review` CLI, writing to a local `runs/` folder.
  - The interleaving code (`p4/blocked-vs-interleaved`) is plain Python plus a FarmShare Slurm wrapper.
- **The org has used five compute paths:** the AWS platform, MIT ORCD Engaging (Slurm), Stanford FarmShare (Slurm), RunPod, and hand-launched EC2. **Weights & Biases (entity `eduLLM`) is the one constant across all of them.**

## 3.2 Design rule: separate the science from the launcher

Everything scientific lives in a **core** that runs the same way on a laptop GPU, a Slurm node or a container. A thin **launcher adapter**, written last, maps one pipeline's conventions onto the core.

| Layer | Contents | Depends on pipeline? |
|---|---|---|
| Data | FictionalQA download pinned to revision `131cb74`, the event splits, any development-selected metadata filter, cached locally; the interleaving generators | No |
| Schedules | the per-fact review schedule generator; the block-length and Williams-square order generator; unit tests that check the realised exposure logs | No |
| Training | the training loop with the same model, optimizer and LoRA/full-fine-tuning options, a constant LR, and one optimizer state across stages | No |
| Evaluation | loss, exact match, MCQ, likelihood margin; per-item logs | No |
| Analysis | paired-replicate estimates, TOST, Holm, heterogeneity sensitivities and figures; reads only logged results files | No |
| Run interface | `python -m <pkg> run --config <arm>.yaml --run-id <id> --out <dir> [--resume auto] [--dry-run]` | No (defines the contract) |
| **Launcher adapter** | a Slurm `sbatch` template (ORCD or FarmShare), a container recipe (`.edullm/Dockerfile` style), or a platform `run.yaml` | **Yes, and it's the only part that is** |

## 3.3 The contract the core must meet

This is drawn from what every org pipeline required (research/notes/11, "What a new experiment must provide").
1. **One non-interactive command**, with all scientific settings in a committed config file per arm. Hardware settings (GPU count, microbatch) are overridable and excluded from the experiment fingerprint.
2. **Runs from a pushed commit**, with pinned dependencies (torch, transformers, peft, OLMo-core revision) and an optional Dockerfile that starts `ARG BASE_IMAGE` / `FROM ${BASE_IMAGE}`.
3. **Run id and output locations from arguments or environment variables**, accepting both local paths and `s3://` URIs, so `EDULLM_RUN_ID` / `EDULLM_CHECKPOINT_DIR` / `EDULLM_OUTPUT_PREFIX` can be mapped in if a platform returns.
4. **Resumable**, via `--resume auto` from the latest checkpoint, and safe under Slurm requeue or preemption.
5. **W&B logging** to entity `eduLLM` with a P4 project, the study as the group, and a deterministic run name. Results files are also written locally and optionally uploaded as W&B artifacts, the fallback P1 used once S3 was gone.
6. **A `--dry-run` / smoke config** that resolves data and prints the plan without training.
7. **Caches on scratch storage** (HF, torch), explicit precision, and no secrets in git.

## 3.4 What can start now vs. what waits
- **Now, with no GPU:**
  - the data and schedule generators with unit tests;
  - the exposure-log checker;
  - the config files for every arm;
  - the analysis scripts (develop them on the pilot logs once Anshul shares `runs/`);
  - CPU smoke tests on a tiny model.
- **Needs a GPU, but not the final pipeline:** the S1 and I1 calibration runs. Profile peak memory and complete-run time on the intended card before reserving the main runs; FarmShare, ORCD or a rented GPU remain options. A 24 GB fit is a target to verify, not an assumption.
- **Waits for the pipeline decision:** the launcher adapter only, roughly a day's work for a Slurm template or a container recipe.

## 3.5 Open questions for the team
1. Which compute will P4 use: ORCD Engaging (Amy is listed as an operator in the archived roster), FarmShare, or a new AWS path? And who owns the decision (the platform README names Max McCorkel for resumption)?
2. Is there a P4 W&B project under `eduLLM` to log into?
3. Should the new code live on the existing branches, or on a new `p4/...` branch? Note that the retired platform only built images for `edullm/**` branches.

---

## References

- Allen-Zhu, Z., & Li, Y. (2024). Physics of language models: Part 3.1, knowledge storage and extraction. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235, pp. 1067–1077). https://proceedings.mlr.press/v235/allen-zhu24a.html
- Baayen, R. H., Davidson, D. J., & Bates, D. M. (2008). Mixed-effects modeling with crossed random effects for subjects and items. *Journal of Memory and Language, 59*(4), 390–412. https://doi.org/10.1016/j.jml.2007.12.005
- Barr, D. J., Levy, R., Scheepers, C., & Tily, H. J. (2013). Random effects structure for confirmatory hypothesis testing: Keep it maximal. *Journal of Memory and Language, 68*(3), 255–278. https://doi.org/10.1016/j.jml.2012.11.001
- Biderman, D., Portes, J., Gonzalez Ortiz, J. J., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., & Cunningham, J. P. (2024). LoRA learns less and forgets less. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=aloEru2qCG
- Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). Pythia: A suite for analyzing large language models across training and scaling. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 2397–2430). https://proceedings.mlr.press/v202/biderman23a.html
- Kang, S. H. K., Lindsey, R. V., Mozer, M. C., & Pashler, H. (2014). Retrieval practice over the long term: Should spacing be expanding or equal-interval? *Psychonomic Bulletin & Review, 21*(6), 1544–1550. https://doi.org/10.3758/s13423-014-0636-z
- Kirchenbauer, J., Mongkolsupawan, N., Wen, Y., Goldstein, T., & Ippolito, D. (2026). FictionalQA: A dataset for studying memorization and knowledge acquisition. In *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=SUNC1eJGMr
- Lakens, D. (2017). Equivalence tests: A practical primer for *t* tests, correlations, and meta-analyses. *Social Psychological and Personality Science, 8*(4), 355–362. https://doi.org/10.1177/1948550617697177
- Loftus, G. R. (1985). Evaluating forgetting curves. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 11*(2), 397–406. https://doi.org/10.1037/0278-7393.11.2.397
- Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=T5wkZJqzkz
- Magnusson, I., Tai, N., Bogin, B., Heineman, D., Hwang, J. D., Soldaini, L., Bhagia, A., Liu, J., Groeneveld, D., Tafjord, O., Smith, N. A., Koh, P. W., & Dodge, J. (2025). DataDecide: How to predict best pretraining data with small experiments. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR Vol. 267, pp. 42487–42502). https://proceedings.mlr.press/v267/magnusson25a.html
- O'Neill, C. (2026). *Can a language model learn facts continually in its weights?* [Preprint]. arXiv. https://arxiv.org/abs/2607.11020
- Roediger, H. L., III, & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science, 17*(3), 249–255. https://doi.org/10.1111/j.1467-9280.2006.01693.x
- Yang, Y., Jones, M., Mozer, M. C., & Ren, M. (2024). Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training. In *Advances in Neural Information Processing Systems 37*. https://openreview.net/forum?id=YSs1z5udBY
