# Opus scientific rescue review, round 6

*2026-10-08. Independent review sub-agent. Read-only assessment of `plans/spacing_acquisition_rescue_20261007.md`, `plans/spacing_confirmation_audit_proposal_20261008.md`, `plans/PRD_spacing_interleaving_reruns.md` and `experiments/spacing_rerun/` (`schedule.py`, `data.py`, `training.py`, `analysis.py`, `configs/calibration-sourceqa.json`). No data, authoring, job, paper or source edits. No confirmation outcomes exist or were seen. Array 1806140 outcomes were not seen.*

## Brief, prior findings and parent position

**Original brief.** Rescue the failed declaration-to-QA acquisition without manipulating the gate, the evaluation set or the aliases.

**Prior rounds.**
- Round 1 diagnosed that the model learned the declarations but QA retrieval transfer was inadequate.
- Round 2 accepted shared Stage-1 declarations plus source QA. Its narrow estimand is declaration-review timing after mixed acquisition. It also required truly isolated source review, an in-sample diagnostic, an overlap sensitivity analysis and inference planning.
- Rounds 3–5 corrected the source packet. Both reviewer families now approve 78 targets and 156 forms. The 16 source limitations are retained and `human_review_complete` stays false.

**Current evidence.** GPU trial 1806120 passed the fixed 40–70% gate at E4, its first usable milestone:

| Metric | Value |
|---|---|
| Old-fact generated exact match (EM) at E4 | 0.4215 |
| Answer loss | 1.197 |
| MCQ accuracy | 0.707 |
| EM at E1, E2, E3 | 0.039, 0.151, 0.299 |
| Content | 73 declarations, 78 targets, all 95 old canonical probes below word Jaccard 0.7 |
| Stage-2 acquisition QA | zero |
| Commit | `d7e9bb5f` |
| Manifest | `cd22ed06…aefd` |

Five-arm development array 1806140 is running at L=84, primary delay 84 and buffer 168.

**Parent proposal.** Choose a common L from metadata. Seeds 2026100801–16 give at most 347 old source units, so L=231; geometry for the first 32 seeds is pending. Then run three fresh, independent fixed-E4 development bundles with five arms each at that L. Retain gate failures, then assess sensitivity, variance and power. The sensitivity targets are:
- NONE old loss increases by at least 0.10 and EM falls by at least 5 percentage points.
- The NONE−UNI and GEN−UNI mean benefits are at least 0.02, with positive paired CIs.

## Verdict

1. **Choosing L from metadata is valid.** Two conditions apply: the rule must be written down and timestamped before any development outcome at that L, and it must not depend on 1806140. Unit counts come from the pinned source and the frozen role-seed list, not from model behaviour. The capacity invariant `stagger / L <= 0.75` (`schedule.py:53`) is a feasibility constraint, not a tuning knob.
2. **This is not yet the fastest valid route, and it leaves extrapolation risk.** Three issues need fixing first:
   - **(a)** Freeze L from a conservative upper bound now. Do not wait for audited grouping.
   - **(b)** Three bundles are too few for either the sensitivity decision or power planning.
   - **(c)** A development replicate with 73 groups at L=231 still differs from confirmation in N, Stage-1 length and stagger. The acquisition dose E4 was chosen from a single seed just above the floor. It is the main threat to confirmation, and the parent plan does not yet test it.
3. **Decide the geometry before anyone reads 1806140.** Under the PRD's 20/30/20/10 design, L=84 is infeasible for confirmation, so 1806140 can only be a pilot. Under the alternative B below, 1806140 *is* confirmation geometry. Choosing between them after seeing its results would be geometry shopping.

## Findings

### F1. Freeze L now from an upper bound that is independent of grouping (high)

- Grouping can only merge units, so it can only lower N. L computed from **ungrouped source-unit counts** is therefore feasible for any audited grouping.
- **Proposed rule:** replicate i uses split seed `2026100800 + i`. Set `L = 21 * ceil(max_{i<=n_max} floor((U_i - 1)/2) / (0.75 * 21))`, where U_i is the ungrouped old-unit count.
  - Choose n_max = 32 now. Run the pending 32-seed metadata count on FarmShare or a CPU node, not the laptop. It is cheap.
  - With n_max fixed, a later power-driven choice of n ≤ 32 cannot force an L change after development.
  - If 16 seeds give a maximum of 347, L=231. If seeds 17–32 exceed 347, L could be 252.
- **Freeze delays at the same time.** The PRD says inference is conditional on span (§1.7, line 161). Moving the core span from 84 to 231 is therefore a core amendment to PRD §1.3 and the configs, not an extension.
  - Keep primary delay 84 and eval delays 21/84/168, unless a delay-relative-to-L argument is written down now.
  - If added, a diagnostic delay of L=231 needs buffer 252 and must be declared before running.
- Keep cohort 2 and review cap 8. Raising cohort to 3 (cap 12, L≈168) changes review density and fills all 12 content slots at peaks. It is a design change with no scientific benefit.
- A larger-than-necessary L only lowers each replicate's stagger ratio, which is harmless. Report the per-replicate ratios and the per-fact review-to-test delay distributions (PRD line 137).

### F2. Separate 1806140 from the decision, in writing, now (high, time-critical)

Commit a note **before** 1806140 outcomes are inspected:
- Under A, 1806140 is a pilot at geometry that cannot be used for confirmation. Its results are reported but cannot qualify or disqualify the L=231 assay, except for detecting bugs or invariant failures.
- The L=231 development bundles decide the assay whatever 1806140 shows.

Without this note, a bad L=84 result followed by an L=231 rerun is indistinguishable from a forking-paths rescue.

### F3. A single E4 at the floor is the main risk to confirmation (high)

1. **E4 is a first passage.** E3 gave 0.299 and E4 gave 0.4215, 2 points above the floor from one seed. The adaptive "first usable milestone" rule selects the first crossing, so fresh seeds can plausibly fall below 0.40 at E4. That is a fragile operating point for a confirmation stage that cannot gate.
2. **Confirmation acquisition is about 4.5× larger.**
   - Development has 5 old events (`data.py:85`), and its cycle of 73 + 78 = 151 items takes 13 updates.
   - Confirmation has 20 old events. About 265–347 groups and about 280–361 targets need roughly 50–59 updates per cycle.
   - At the same dose E, items wait about four times as many intervening updates between exposures. Total gradient steps rise and so does interference within acquisition.
   - The teaching pool grows from 225 to about 750 rows.
   - None of these changes has a known sign, and development at N=73 cannot reveal it.
3. **Stage-2 stagger differs as well.**
   - At L=231, development with N=73 has stagger 36 (ratio 0.16). Confirmation reaches 0.75, about 173 updates.
   - In confirmation, late-starting facts sit up to about 173 updates of interference between acquisition and their first review. Reviews there act more as relearning than maintenance.
   - The NONE−UNI benefit and the forgetting magnitude can therefore differ systematically between development and confirmation.

**Valid checks that never touch confirmation outcomes, ranked:**

- **C1. Scale-matched development acquisition pilot.** Author and review source QA for the remaining development events. This is the same pipeline the 80-event catalog needs, at about a quarter of the work, so it doubles as a dress rehearsal. Then run Stage-1 only:
  - Treat all non-teaching development events as old. That gives about 250 source units and about 40–45 updates per cycle, close to confirmation scale.
  - Use 3+ fresh seeds and evaluate the fixed grid {2, 3, 4, 6, 8}. Do not stop adaptively.
  - This measures how E transfers to scale. It uses development events only, under a timestamped development amendment.
- **C2. Choose E by an arm-blind rule, frozen before C1 and before the new development bundles.**
  - Run Stage-1 through E8 for every new development seed and save E4/E6 checkpoints. Apply the rule to acquisition data only, then fork Stage-2 from the chosen E.
  - **Rule:** use the smallest grid E at which every scale-matched seed and every N=73 seed lies in [0.40, 0.70] and the scale-matched mean lies in [0.45, 0.65]. If both E4 and E6 qualify, take the one whose scale-matched mean is closest to 0.55.
  - If no E qualifies, write an amendment. Do not relax the gate.
  - Choosing E from acquisition alone keeps arm outcomes out of the dose choice.
- **C3. Arm-blind Stage-1-first confirmation (protocol rule, frozen in preregistration).**
  - Run all n confirmation Stage-1s to fixed E before forking any arm.
  - Prespecify a whole-experiment validity rule on the acquisition distribution only, for example median EM in [0.30, 0.80].
  - Never apply it per replicate. Every replicate is retained (PRD line 127).
  - Acquisition is shared before the arms and carries no information about EXP−UNI, so this is not outcome selection.
  - An abort must be disclosed. Any restart is a new, disclosed confirmation attempt, because confirmation-event acquisition will have been seen. C1 and C2 exist to make an abort unlikely.

### F4. Three development bundles cannot support the sensitivity or power decisions (high)

- **Positive-CI criterion.** With n=3 the paired t has 2 df. A positive two-sided 95% interval needs mean/SE > 4.30. The test will fail often for real effects, and any SD estimate is nearly uninformative.
- **Planning multiplier.** A one-sided 80% upper confidence bound on SD multiplies the observed SD by:

  | Bundles | df | SD multiplier |
  |---:|---:|---:|
  | 3 | 2 | ≈2.12 |
  | 6 | 5 | ≈1.46 |
  | 8 | 7 | ≈1.35 |

- **Cost is negligible.** At L=231, N=73, each arm runs 36 + 231 + 1 + 168 = 436 Stage-2 plus buffer updates, against 289 now. The run measured at 118 s of application time spent most of it on evaluation. Six to eight bundles cost GPU-minutes to low GPU-hours.
- **Recommendation:** timestamp an amendment raising the development bundle limit (`development_design_limit.replicate_bundles`, currently 4; PRD line 131) to 8. The user's authorization covers further development, and the amendment must precede the runs.
  - Run 6–8 fresh, independent five-arm bundles at the frozen L and E.
  - Retain every bundle, including gate failures, with no conditional filtering. Report gate status per bundle.
- **Sensitivity rule, frozen before the bundles run:**
  - **Interference:** mean NONE old-loss increase ≥ 0.10, with a one-sided 95% lower bound above 0, and mean EM decline ≥ 5 points.
  - **Review:** mean NONE−UNI ≥ 0.02 **and** mean GEN−UNI ≥ 0.02 at the primary endpoint, each with a one-sided 95% paired t lower bound above 0.
  - Both review conditions must hold. This is intersection–union, so no multiplicity adjustment is needed.
  - EXP is not part of the rule. Its values are logged but not inspected for the decision.
  - The scale-matched replicate (F3, C1, extended to Stage-2) can be one additional prespecified sensitivity bundle, labelled with its role-count deviation.

### F5. Sample size: precision under a conservative SD, no automatic n=8 (medium-high)

- **Planning SD.** Use the unshrunk development paired SD for EXP−UNI at D*, inflated to its 80% upper confidence bound.
  - Development averages over 5 old events and confirmation over 20, so the event-averaging part of the variance should shrink.
  - Between-replicate training noise need not shrink, so do not take credit for the shrinkage. Report it only as a scenario that would favour the design.
- **Approximate TOST size** (margin Δ=0.02, true difference 0, 80% power): `n ≈ (t_.95 + t_.90)² σ² / Δ² ≈ 8.6 σ²/Δ²`, plus a small-sample correction.
  - σ = Δ gives n ≈ 10. σ = 2Δ gives n ≈ 35–37.
  - This agrees with the existing simulation (SD 0.02, n=16 → 96.9%; SD 0.04, n=16 → 23.8%). Use the simulation for final numbers, including the ±Δ/2 scenarios required by PRD line 155.
- **Precommitment.** If the conservative-SD n for TOST exceeds what can be run with n ≤ n_max = 32, the primary claim becomes an estimate:
  - a 95% CI for EXP−UNI with its expected half-width `t_.975,n-1 · σ_cons / sqrt(n)` stated in advance, and
  - the directional test.
  - No equivalence claim is promised and the margin is never widened.
- Confirmation replicates all draw on the same 80 events (20+30+20+10 = 80), so inference is conditional on this event pool. Keep event-level resampling as the sensitivity analysis (PRD line 154).

### F6. Alternative geometry B: confirmation at development scale (medium; decide now or drop)

- **Design:** each confirmation replicate uses development role counts, 5 old / 7 new / 5 control / 3 teaching events drawn from the 80 confirmation events, at L=84 with the frozen E.
- **Advantages:**
  - Development and confirmation recipes become identical apart from the events. The N, Stage-1 length and stagger extrapolations of F3 disappear.
  - It cuts Stage-1 per replicate by about 4.5× and evaluation by about 4×, so larger n is affordable.
  - 1806140 and further L=84 development bundles become directly relevant.
- **Costs:**
  - It amends PRD line 76 (20/30/20/10) and makes replicate-level contrasts noisier, because the event-macro averages over 5 events.
  - Replicates still overlap in events. Disjoint 20-event sets allow only 4 replicates, so overlap must be modelled as recurrence.
- **Condition:** B is valid only if chosen **before** 1806140 or any L=231 outcome is read. My recommendation is to keep A, which follows the PRD design. Choose B now only if C1 cannot be authored promptly, because without C1 the A-route E choice rests on unmeasured extrapolation.

### F7. If declaration reviews fail to protect QA retention (medium; plan now)

Write the branch now so a failure leads to an amendment, not a claim of equivalence. If the NONE interference condition passes but the NONE−UNI or GEN−UNI condition fails at 6–8 bundles, the assay cannot detect review benefit. In that case, in order of preference:

1. **Declare the assay insensitive** for declaration review on QA. Report it honestly and make no spacing claim. This is always valid.
2. **Test the declarative endpoint.** Promote statement-content loss on old declarations (already computed by `diagnose.py`) to a co-primary or alternative estimand: "timing of declaration review on declarative memory". QA stays a transfer diagnostic. This is a narrower but coherent study and needs a fresh amendment plus development bundles.
3. **Run a mixed-format review study.** Reviews include source-QA forms with matched declaration and QA review doses. The plan already calls this a different study (retrieval-practice-like). It needs new content, dose accounting and its own development sensitivity check.
4. **Change review dose or density** (more than 4 reviews per fact) within capacity. This changes k and therefore amends the core design.

If the interference condition itself fails, the remedy is a longer delay or buffer or a larger interference stream, declared before rerunning. Raising Stage-2 LR is a recipe change and needs fresh acquisition checks.

## Ranked next actions

1. **Now, before reading 1806140:** commit the decision note (F2), choose geometry A or B (F6), and freeze the L rule with n_max = 32 and the delays/buffer (F1). Run the 32-seed metadata count on FarmShare or CPU.
2. Timestamp the development amendment:
   - bundle limit raised to 8;
   - the scale-matched acquisition pilot C1;
   - the arm-blind E rule C2;
   - the F4 sensitivity rule;
   - the F7 failure branch.
3. Author and review source QA for the remaining development events through the existing isolated channels, as a rehearsal of the global catalog. Run C1 (Stage-1 only, 3+ seeds, full grid). Apply C2 and freeze E.
4. Run 6–8 fresh five-arm development bundles at the frozen L and E. Apply the F4 rule without filtering. Compute the conservative SD and power or precision (F5) and record measured costs.
5. In parallel and independent of outcomes, finish the remaining prerequisites:
   - the global 80-event source catalog, cross-event entity/fact-link ledger and evaluation-paraphrase channel;
   - the confirmation schemas and loader;
   - GPU A/A on L40S at the frozen settings;
   - the measured cost forecast.
6. Write the preregistration with fixed E, L, n, the C3 Stage-1-first validity rule, margins, Holm family, missing-pair policy and attempts. Register it, then submit.

## Blockers to a valid main launch (all remain open)

- The L rule is not frozen and the 32-seed geometry is pending.
- The E choice for confirmation scale is untested (F3).
- The development sensitivity check and its variance estimate are missing (F4).
- The global 80-event source catalog, cross-event ledger, evaluation paraphrases and the full agent-review record are incomplete. `human_review_complete` is honestly false; this is disclosed, not faked.
- The confirmation schemas and loader path are missing, and the loader must keep rejecting runs until they exist.
- The GPU A/A restart check has not been run.
- Measured confirmation cost, the n decision and the preregistration packet are missing.

## What not to do

- Choose L, E, geometry or n from 1806140 or any arm contrast.
- Filter gate-failed development bundles or poorly acquired confirmation replicates.
- Widen the 0.02 margin or the 40–70% gate.
- Use EXP-vs-UNI development results for any assay decision.
- Pool development with confirmation.
- Run counts or analyses on the laptop.
