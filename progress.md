# Progress log: P4 papers

Goal: turn P4's work into rigorous, publishable journal/conference paper(s). Three studies are in scope:
- **spaced review**: [P4_whitepaper_writeup.md](paper/source/P4_whitepaper_writeup.md);
- **blocked vs. interleaved training**: [P4_final_report.md](paper/source/P4_final_report.md), Test 02;
- **mastery-gated curricula**: [P4_final_report.md](paper/source/P4_final_report.md), Test 03.

Synthetic-student material is out of scope. **No paper text has been edited yet; we are waiting for the go-ahead.**

## TL;DR

- **Spacing.** The experiment is real and its headline numbers match the code. The paper misdescribes several parts of the method, 3 citations are misused, and the design has gaps reviewers will find.
  - **Reviewed plan, 2026-10-07:** the spacing PRD now defines a five-arm 300M full-fine-tuning core, calibration-based sample size and cost accounting, with optional extensions gated separately. Recover the old logs where possible, then calibrate and freeze the new protocol before confirmation.
  - **Evidence recovery:** the `runs/` outputs and the CI script from Anshul remain missing. They are needed to verify the old paper, but need not prevent a new self-contained experiment.
- **Interleaving.** The report describes a 195M study that was never run. The only run (162M, digit skills, n = 1, one order) is unmentioned, and the "toy pilot" numbers have no code.
  - Cheap fix: counterbalanced orders × seeds plus recency and spacing controls, using the existing code.
- **Mastery gating.** Two preliminary experiments, no code on any branch, no seeds. Reviewers will require real experiments.
  - Plan: [PRD_mastery_gated_curriculum.md](plans/PRD_mastery_gated_curriculum.md) (Phase 1 ≈ 20–50 GPU-hours).
- **Cross-cutting.** Every reported number must trace to committed code. Fix the one-pager, which mislabels the mastery panel and repeats the untraceable numbers.

Key files:
- [research.md](research/research.md): literature synthesis for all three studies, with the top-10 papers to add per study and the bibliographies.
- [methods_verification.md](research/methods_verification.md): the papers checked against the code (Parts I–IV).
- [PRD_mastery_gated_curriculum.md](plans/PRD_mastery_gated_curriculum.md): the experiment plan for mastery gating.
- [PRD_spacing_interleaving_reruns.md](plans/PRD_spacing_interleaving_reruns.md): the experiment plan for the spacing rerun (Priority 1) and interleaving rerun (Priority 2).
- [P4_spacing_paper_sections.md](paper/drafts/P4_spacing_paper_sections.md): the section collage for the spacing paper ([PILOT] / [BOTH] / [RERUN] versions of each section).
- [research/notes/](research/notes/): detailed notes 00–11 (11 covers the compute pipelines).
- [README.md](README.md): map of the project folders.
- Papers, unchanged: [P4_whitepaper_writeup.md](paper/source/P4_whitepaper_writeup.md) and [P4_final_report.md](paper/source/P4_final_report.md) (the synthetic-student section is omitted from the md copy).

---

## Checklist: open tasks and issues

## Part I: Spacing whitepaper

Section numbers like "§2.4" refer to the whitepaper unless marked otherwise.

### A. Citations (details: [research/notes/01_citation_verification.md](research/notes/01_citation_verification.md))
- [ ] **Intro: "expanding schedules … greatly improve memory (Landauer & Bjork 1978; Cepeda et al. 2006)".**
  - Cepeda found **no** reliable expanding-vs-fixed difference (62.0% vs 58.6%, p = .61) and calls L&B's claim one with "little apparent empirical backing".
  - L&B found about +10 points at 30 minutes, for test-type practice only. For restudy, uniform was slightly better.
  - Fix: rewrite the sentence and cite Latimier et al. 2021.
- [ ] **Intro: mechanisms "consolidation, retrieval difficulty, encoding variability (Cepeda)".**
  - Cepeda's candidate theories are consolidation, *study-phase retrieval* and encoding variability. "Retrieval difficulty" comes from Bjork and from Karpicke & Roediger.
  - Fix: rename them and call them "candidate accounts".
- [ ] **Conclusion: our null "reproduces" Karpicke & Roediger 2007.**
  - Their equal > expanding result was caused by delaying the first retrieval. With the first retrieval matched (Exp. 3, which is our design), there was no difference.
  - Their task was retrieval practice; ours is restudy.
  - Fix: say "consistent with" and cite Exp. 3, Latimier 2021 and Kang 2014.
- [ ] Use archival versions: LoRA → ICLR 2022; OLMo → ACL 2024; DataDecide → ICML 2025; FictionalQA → ICLR 2026. Fix names: Hwang, J. D.; Mongkolsupawan, N.; Roediger, H. L., III; Walsh, E. Add the editor and DOI for McCloskey & Cohen.
- [ ] Soften "one of the most robust results in cognitive psychology" (not Cepeda's wording), or cite Dempster 1988.
- [ ] Add a related-work section covering the 10 papers in research.md §0. Address FOREVER head-on, since it contradicts our null.

### B. Paper text that doesn't match the code or is misleading (details: [methods_verification.md](research/methods_verification.md))
- [ ] **What a review is.** In the code, each review event is one update of 16 statements drawn randomly from **one document style**, with styles rotated. About half the 80 evaluated facts are never reviewed. Describe this exactly. It also undercuts "same reviews, only timing differs" at the level of individual facts.
- [ ] **Evaluation set (§2.4).** It is described as "held-out paraphrased questions … rather than memorized training text". In fact it is the question form of *trained* statements, and the answer appears verbatim 93% of the time. There are 80 items, fixed across seeds; EOS is included in the loss; and losses are averaged over the 5 styles.
- [ ] **"Trajectory change" (§2.4, §3.5).** It is described as a Stage-2 average, but it averages 9 checkpoints, 3 of which are in the buffer. Fix the definition and re-check the §3.5 interpretation.
- [ ] **Model.** Base model is 371.5M parameters; "377M" includes LoRA. Give the exact ID and checkpoint: `allenai/DataDecide-dolma1_7-300M`, `step45787-seed-default`.
- [ ] **Training details missing:**
  - AdamW β = (0.9, 0.95), weight decay 0.1, gradient clipping 1.0, fp16, LoRA dropout 0.05;
  - **the optimizer and 10-step warmup restart at Stage 2**, so the first review (step 8) falls in warmup.
- [ ] **Schedules.** List both arms' review steps and the 1.35 ratio. State that first and last reviews are matched only globally, and that per-style last-review times differ by up to 57 updates.
- [ ] **Data.** 20 old and 40 new events (386 and 689 facts); split by event; the seeds don't change the split or the evaluated items.
- [ ] **Statistics.** State the CI method (paired t, df = 2) and add per-seed values.
- [ ] Delete the stray drafting note in §3.2 ("This is the natural place for it…").

### C. Claims to tone down or reframe
- [ ] Use "expanding vs. uniform review", not "the spacing effect" (there is no massed arm), including in the title.
- [ ] §3.5: the expanding "trajectory" advantage is a known human pattern (Kang et al. 2014), not merely an artifact.
- [ ] §3.3: "starting point, not forgetting rate". The CI is about half the size of the effect, and the conclusion flips on a probability scale (Loftus 1985). Add the caveat, or present a horizontal comparison (review is worth more than 180 updates of delay).
- [ ] "Statistically and practically indistinguishable" needs an equivalence test (TOST) with a pre-set margin.
- [ ] Conclusion: drop "how you space it does not [matter]" and the production-default advice. Call it a pilot, as the team's own `REVIEW_LAB.md` does.
- [ ] Report effects in nats or probability ratios rather than "% loss".
- [ ] Old-fact loss falls during new-fact training in every arm, which is format/domain adaptation. Don't claim old facts became "better learned".

### D. Information needed from the team
- [ ] `runs/review_lab/olmo370m-fictionalqa-3seeds-buffer180/` (REPORT.md, per-seed and aggregate CSVs, metrics.jsonl).
- [ ] The script or notebook that computed the paired 95% CIs.
- [ ] Were any settings (Stage-1 length, LR, ratio 1.35, buffer length, 12 events) chosen after seeing results?
- [ ] Code for the Pythia 20-seed spacing pilot (in the one-pager), if we want to cite it.

### E. Reanalyses that need no retraining (once we have `runs/`)
- [ ] Report exact match (already logged).
- [ ] Item-level mixed-effects model: reviewed vs. never-reviewed facts, dose–response on the number of reviews, and facts never seen in Stage 1.
- [ ] TOST for expanding vs. uniform.
- [ ] Condition × time over all buffer checkpoints, on two scales (loss and probability or accuracy).

### F. New spacing runs at 300M (reviewed 2026-10-07)
- [x] Review [PRD Part 1](plans/PRD_spacing_interleaving_reruns.md) with an Astra subagent at xhigh for scientific validity and cost. The revised PRD governs implementation; older checklists and draft methods are not an additional mandatory grid.
- [ ] Per-fact schedules: every old fact gets *k* reviews at its own expanding or uniform positions, with first and last exposure matched per fact.
- [ ] Split calibration and confirmation events; reserve never-trained control events and separate QA-format teaching events. Evaluate the frozen fact roster without post-treatment filtering.
- [ ] Validate the five-arm core: no review, uniform, expanding, massed, generic-data insertion. Verify exposure, endpoint and token invariants before GPU runs.
- [ ] Benchmark complete 300M full-fine-tuning calibration runs, including evaluation and checkpoint overhead. Share Stage-1 checkpoints and optimizer states across paired arms.
- [ ] Freeze meaningful effect/equivalence margins, primary delay, sample size and measured cost forecast before confirmation; no unadjusted significance peeking or margin changes after seeing main results.
- [ ] Report event-macro answer loss excluding EOS, generated-answer exact match, MCQ accuracy, old/new trade-offs and multiple delays from the same buffer. Treat drift correction as a secondary analysis with assumptions.
- [ ] Keep LoRA comparisons, a second review span, adaptive schedulers and larger models as separately justified extensions, including informative null results.

### G. Spacing extensions (after F; PRD Part 1 governs)
- [ ] Choose an extension for a specific unresolved question and budget it using measured costs. Run selected paired contrasts, preserving adaptation method for any size comparison. The former broad size ladder is not a launch requirement.

### H. Decisions and writing
- [x] Draft the spacing paper as a section collage (pilot-accurate, timeless and rerun versions): [P4_spacing_paper_sections.md](paper/drafts/P4_spacing_paper_sections.md). It has 25 open placeholders, listed at the end of the file.
- [ ] When compute arrives, run PRD Part 1 (spacing), then Part 2 (interleaving), then the mastery PRD.
- [ ] **Decide where P4 runs.** The AWS platform is retired; the options are ORCD Engaging, FarmShare or a new AWS path. Also confirm the W&B project under `eduLLM`. See PRD Part 3.5.
- [ ] **Start building the pipeline-independent core now:** data and schedule generators with tests, per-arm configs, the analysis scripts, and a `--dry-run` smoke mode (PRD Part 3.4). The launcher adapter waits for the pipeline decision.
- [ ] Decide: (a) pilot paper now, or (b) reruns first.
- [ ] Pick a target venue. TMLR is the best fit; CogSci/CCN 2027 are options if the human comparison is made rigorous.
- [ ] Draft the paper in research-paper format.


## Part II: Blocked vs. interleaved (Final Report, Test 02)

### II-A. Citations (details: [research/notes/09_final_report_citation_verification.md](research/notes/09_final_report_citation_verification.md))
- [ ] Brunmair & Richter (2019): g = 0.42 is correct, but add the moderators (math g = 0.34; words g = −0.39; similarity decides the direction). Our dissimilar, rule-like skills are the case where humans may *not* benefit.
- [ ] Rolnick et al. (2019): accurate as written, but it studies RL agents. Add LM evidence (Dong et al., 2024; Russin et al., 2025) and update to the NeurIPS 2019 version.
- [ ] Add the missing key work: Flesch et al. (2018), Russin et al. (2025), Lee, Cho & Yoo (2024), Foster et al. (2019), Carvalho & Goldstone (2014a). See research.md §II.2.
- [ ] Add a reference list. The report currently has none.

### II-B. Text vs. code (details: [methods_verification.md](research/methods_verification.md) Part II)
- [ ] The report describes a *proposed* 195M pretrained-checkpoint study (4 school subjects, sessions, interference phase, 12 paired runs). What was run: a **random-init 162M model, 4 digit-manipulation skills, 512 updates, one order (A→B→C→D), one run**. Describe the actual run, or label the 195M design as proposed.
- [ ] Remove or substantiate the "toy pilot" numbers (52.8% → 79.2%). No code exists for them on any branch. They also appear in the one-pager.
- [ ] Report the actual result honestly: 27.4% vs. 61.6%, but blocked = recency (D 94.5%, others 0–15%) and interleaved not converged (loss 0.21 vs. 0.02).
- [ ] Don't call the digit transformations "subjects".

### II-C. Experiments needed (details: research/notes/10 §B1 and research/notes/07 synthesis)
- [ ] All 4 counterbalanced orders (already generated by the code) × ≥5 seeds, with evaluation after each block (recency control).
- [ ] Train to convergence, or report learning curves.
- [ ] A blocked-but-spaced control (separates interleaving from spacing), a block-length sweep, and a skill-similarity manipulation.
- [ ] A size sweep (e.g., 160M / 410M / 1B), since order effects emerge between 160M and 410M.

## Part III: Mastery gating (Final Report, Test 03)

### III-A. Citations (details: [research/notes/09_final_report_citation_verification.md](research/notes/09_final_report_citation_verification.md))
- [ ] **Anthropic "biology" link.** It points at the wrong section (`#dives-tracing`), and the source never claims CE loss produces a cheap heuristic that "collapses" out of distribution. Cite Nikankin et al. (2025) and Lee et al. (2024) instead.
- [ ] **Robins (1995)** did not "introduce" replay. He replicated Ratcliff (1990) and introduced *pseudo*rehearsal. Replace the Stanford proxy link with the DOI.
- [ ] **Reversed-digit paper (arXiv 2403.05845)** is a 13B fine-tune with step-by-step reasoning and no OOD tests. Use Lee et al. (ICLR 2024) for reversed output and McLeish et al. (2024) for place-value tags.
- [ ] **CAMPUS "about 7%"** is against static-curriculum baselines. Over shuffle it is +1.67 points (≈4.6%).
- [ ] **RLAAR:** cite the ACL 2026 version. 62.6→75.1 compares the full method with the base model; the curriculum-only ablation is 63.2→71.9.
- [ ] **Zaremba & Sutskever:** "combined" is best for program evaluation only, and it mixes in *harder* examples. That supports look-ahead mixing, not replay.
- [ ] **LR-decay paper (Luo et al., ICLR 2026):** it concerns data *quality*. "Identical schedule across arms" doesn't fix the confound, and WSD still decays. Use constant LR, moderate decay or checkpoint averaging.
- [ ] Cite the chain task's origins: LEGO (Zhang et al., 2022), variable binding, Yao et al. (2025). Cite Nanda et al. (2023) for the clock algorithm.
- [ ] Add the human counter-evidence: Slavin (1987), about 0 effect when time is equalized. State Kulik et al. (1990) precisely (0.52 SD on end-of-unit exams; 0.29 on standardized tests).
- [ ] Add a reference list. The report currently has none.

### III-B. Text vs. evidence (details: [methods_verification.md](research/methods_verification.md) Part III)
- [ ] **No code on any branch** for Preliminary Experiments I and II. Recover and commit it, or don't report the numbers.
- [ ] Preliminary Experiment II text "(0.59, 0.67)" doesn't match the table (0.67, **0.55**).
- [ ] The one-pager mastery panel is labelled "Prerequisite-chain task" but shows Preliminary Experiment II numbers. The real chain-task OOD values are 0.37 / 0.33 / 0.75, where the gate without replay is *worse* than shuffle.
- [ ] Report model size, seeds, token budget, gate threshold, probe and exam construction, and replay fraction.
- [ ] "Depth 5 not unlocked" → "unlocked late" (0.18).
- [ ] Remove the informal sentence in Limitations. Move cost analysis, competitive programming and attribution graphs to future work.
- [ ] "Preregistered" (mission statement): link a timestamped registration or remove the word.

### III-C. Experiments needed
- [ ] Run Phase 0–1 of [PRD_mastery_gated_curriculum.md](plans/PRD_mastery_gated_curriculum.md):
  - a 2 × 2 of gate × replay, plus yoked clock, tuned clock, mixture-matched shuffle, pacing-only and mixed-difficulty arms;
  - ≥10 paired seeds and matched tokens;
  - disjoint gate-probe and exam families;
  - a preregistered primary contrast.
- [ ] Phase 2: a size ladder and continued training of open 0.4–1.4B models.

## Cross-cutting
- [ ] Decide on paper structure: a combined "learning science in LM training" paper (submit only when all three studies meet the bar), or separate papers. See research/notes/10 Part C for venues.
- [ ] Fix the one-pager ([P4_onepager.md](original/onepager/P4_onepager.md) / .html / figure): mislabelled mastery panel, untraceable interleaving and Pythia numbers.
- [ ] Get from the team: the spacing `runs/` folder and CI script, the toy-interleaving-pilot code, the mastery Preliminary Experiments I/II code, and the Pythia spacing-pilot code.

---

## History (completed work)

### 2026-10-02: Conversion
- Converted `P4 Whitepaper Writeup.docx` to [P4_whitepaper_writeup.md](paper/source/P4_whitepaper_writeup.md), verbatim. The figure is in [figures/](paper/figures/).

### 2026-10-02: Deep research, citation check and code audit
- **Research.** Six areas: human spacing, ML replay/spacing, LLM fact learning and forgetting, statistics and mock review, model scale, and citation verification. The synthesis, with 141 sources each formatted with a URL, is in [research.md](research/research.md). The detailed notes are in [research/](research/notes/).
- **Citations.** All 9 references are real; 3 are misused and 4 need their archival versions (now checklist A).
- **Code audit.** Reconstructed the exact data split, schedules and per-fact exposures, which surfaced the per-fact review problem (now checklist B and F).
- **Coverage.** No related-work section; the 10 papers to add are listed in research.md §0.
- **Scale.** Most comparable work is at ≤3B. Some popular claims come only from 7B+ models or vision networks, so they need caveats. Order effects emerge between 160M and 410M.

### 2026-10-02: Methods and results checked against GitHub
- Checked against `edu-llm/OLMo-core`, branches `anshulm/fictionalqa-review`, `p4/blocked-vs-interleaved` and `andrew/peer-distillation`. Everything was read remotely. Report: [methods_verification.md](research/methods_verification.md).
- **Verdict: real experiment, real headline numbers.** The inaccurate descriptions are now checklist B; the missing data is checklist D.
- The other two branches are not part of the paper. The interleaving branch's result (27.4% → 61.6%, one run) differs from the old Final Report (52.8% → 79.2%).

### 2026-10-02: Readability pass
- Added TL;DR sections to every markdown file, a "10 papers to add" section to research.md, and this checklist.

### 2026-10-04: Final Report (interleaving and mastery gating)
- Converted `P4 Validating Learning Science - Final Report.docx` to [P4_final_report.md](paper/source/P4_final_report.md), with 3 figures and the synthetic-student section omitted.
- Research: interleaving (research/notes/07), mastery and curricula (research/notes/08), citation check of Tests 02–03 (research/notes/09), and statistics, mock reviews and venues (research/notes/10). Synthesised in research.md Parts II–III, each with its top-10 papers to add.
- Code check:
  - Read `p4/blocked-vs-interleaved` remotely and scanned all 152 branches.
  - Found that the report's interleaving design was never run, and that the toy pilot, both mastery experiments and the Pythia spacing pilot have no code. Report: [methods_verification.md](research/methods_verification.md) Parts II–IV.
- Verified the close precedents ourselves: Kohli et al. (2026) gate + replay (full text), Yao et al. (2025), Russin et al. (2025), Lee et al. (2024). Corrected a mislabelled verification tag in research/notes/10.
- Wrote [PRD_mastery_gated_curriculum.md](plans/PRD_mastery_gated_curriculum.md).

### 2026-10-04: Writing skill
- Added the project skill `.claude/skills/scientific-writing/`. It contains:
  - `SKILL.md`, which loads automatically whenever paper prose is written or edited;
  - a banned-phrase list;
  - a section guide;
  - project conventions;
  - `check_prose.py`, a mechanical self-check.
- Added a pointer to it in `CLAUDE.md`.
- Baseline check of the whitepaper: 27 em-dashes, "spacing effect" misuse, and several "Note that" / "Notably" openers. Fix these when editing begins.
- **Update (same day):**
  - **Em-dashes are now banned outright** in paper prose (including `--` and spaced hyphens or en-dashes used as dashes); the checker reports each one as an error.
  - "note that" and sentence-initial "Notably / Interestingly / Importantly" are **soft warnings**: keep them only if they pass the test in banned_phrases.md.

### 2026-10-04: Rerun PRD and spacing draft
- Wrote [PRD_spacing_interleaving_reruns.md](plans/PRD_spacing_interleaving_reruns.md).
  - **Spacing rerun:** per-fact review schedules with first and last review matched per fact; massed and generic-data arms; never-trained control events; QA-format teaching events; balanced Stage-1 exposure; 10 seeds with resampled splits; LoRA plus full fine-tuning; a 1B check; TOST and mixed models. About 20 GPU-hours.
  - **Interleaving rerun:** a Williams-square order design × 5 seeds; a block-length sweep; a replay arm; a low-similarity skill set; a size sweep. About 15 GPU-hours.
- Wrote [P4_spacing_paper_sections.md](paper/drafts/P4_spacing_paper_sections.md) using the scientific-writing skill.
  - **Content:** an abstract, introduction, related work, methods, results, discussion, limitations and conclusion that are accurate to the pilot as run; reusable background sections; and methods written for the reruns, with [TODO] slots.
  - **Marking:** unverified whitepaper numbers are marked †, and reconstructed exposure counts ‡.
  - **Checks:** it passes the prose check apart from intentional exceptions.
- Re-verified against full texts: Tirumala et al. (2022), the spacing quote; Kang et al. (2014), .49 vs .41 average and .49 vs .46 final; FOREVER Table 3, 42.5 vs 40.9; Landauer & Bjork (1978), .62 vs .58 for restudy.

### 2026-10-04: Project reorganised
- Moved files into four folders:
  - `original/`: the team's files, untouched;
  - `paper/`: Markdown paper sources, drafts and figures;
  - `plans/`: the PRDs;
  - `research/`: the synthesis, the code verification and `notes/`.
- `progress.md`, `README.md` and `CLAUDE.md` stay at the top level.
- Rewrote all 103 relative links, and the paths in the writing skill. A link check found none broken.
- A full pre-move backup is at `/tmp/p4/edu-llm-backup-20261004`. It is temporary, so delete it once the new layout is confirmed.

### 2026-10-04: Compute pipeline survey
- Deleted the temporary pre-move backup.
- Surveyed all 24 edu-llm repositories remotely ([research/notes/11_compute_pipeline_survey.md](research/notes/11_compute_pipeline_survey.md)).
  - The AWS `platform` has been frozen since 2026-08-18 and was retired in practice; edullm-p1's 2026-09-22 to 09-27 commits remove its AWS, S3 and RunPod paths.
  - The org has also used ORCD Engaging, FarmShare, RunPod and hand-launched EC2. W&B (entity `eduLLM`) is the one constant.
  - The P4 spacing and interleaving code never used the platform.
- Added PRD Part 3: a pipeline-independent design (science core + launcher adapter) and what can start now.

### 2026-10-06: Pushed to GitHub
- Pushed this workspace to `edu-llm/OLMo-core` as the standalone branch `p4-publication`. It shares no history with `main`.
- Added `.gitignore` (excludes the local writing skill, `CLAUDE.md`, the synthetic-student documents, OS/editor/Python junk and run artifacts).
- Added [HANDOFF.md](HANDOFF.md) for the next agent: where everything is, and to ignore the synthetic-student content left in the unmodified originals.

### 2026-10-07: Spacing PRD reviewed for scientific validity and cost
- Used a GPT-6 Astra subagent at xhigh, as requested, to revise [PRD Part 1](plans/PRD_spacing_interleaving_reruns.md). Interleaving's experiment design remains unchanged.
- Replaced the automatic full-FT/LoRA factorial and 1B check with one five-arm 300M full-FT core and separately justified extensions. Proposed planning sample sizes are 8, 12 or 16 complete paired bundles, selected once from calibration, precision and measured cost.
- Added a development/confirmation event partition, exact capacity-safe review schedules, content/token invariants, stronger expanding gaps, shared Stage-1 optimizer state, actual generated-answer scoring and explicit bounds on the claims.
- Made raw event-macro answer loss excluding EOS primary and drift correction secondary. Prespecified equivalence and practical-difference rules replace nonsignificance-as-equivalence; main-effect peeking cannot determine sample size.
- Withdrew the unsupported spacing GPU-hour estimate. The new cost ledger includes development, initialization, training, evaluation, checkpointing, allocated idle time and retries. The user clarified that the budget is dynamic with no hard total cap; spending decisions should use measured costs and scientific value, with operational runtime/retry limits retained. No price or runtime has been measured and no experiments were launched.
- Marked the October 4 [RERUN] paper passages as requiring reconciliation with the reviewed protocol. Updated the active spacing checklist; historical notes retain their original estimates as history.

### 2026-10-07: Spacing implementation and first FarmShare calibration
- Delegated implementation to GPT-6 Astra at xhigh. Pushed the standalone runner to `p4-publication` at `d879576a001eb1d7c98a9f623a3c4fdc2b702314` before submission.
- The runner pins model/data revisions, validates exact matched schedules, saves model/optimizer/RNG/cursor state, and blocks continuation after failed acquisition. The original 18 CPU tests and actual tiny `hf_olmo` integration checks passed.
- FarmShare setup job `1801924` completed in 2:29. GPU calibration job `1801933` completed with exit code 0 on `oat-04`, using one NVIDIA L40S, 32 GB requested host memory and four requested CPUs. Slurm elapsed time was 11:35; application time was about 677 seconds. Peak allocated CUDA memory was 7.52 GB. Jobs used wall-time caps below 24 hours and excluded `wheat-01`.
- The development acquisition screen reached E=64 and failed its gate. Old-fact generated-answer event-macro accuracy was 4.24%, below the required 40–70%; answer-token loss was 3.015 and MCQ accuracy was 47.83%. No five-arm continuation or confirmation job was launched. This is an assay failure, not a spacing result.
- Checkpoint writes consumed about 384 seconds, training 222 seconds and evaluation 60 seconds. Subsequent code reduces checkpoint frequency while retaining milestone, periodic, final and signal-triggered saves; a dropout-enabled mid-epoch resume test checks exact continuation.
- Immutable source and artifacts remain under `/scratch/users/ericrcwu/agent-runs/p4-spacing-20261007-d879576a/`. The prepared manifest SHA-256 is `58ba9dddfd5b19898b655eecfe05747802e50b5912a6e69743e65083e60404fe`.
- Pushed the diagnostic/cadence code at `4555e8d63866119e65205424d4475ae9f267caf3` before diagnostic job `1801956`. All 24 tests passed, including exact mid-epoch continuation. The diagnostic completed with exit code 0 on `oat-04` in 21 seconds, with 15.1 seconds application time and 6.00 GB peak allocated CUDA memory.
- QA-teaching generated-answer event-macro accuracy was 96.08%, versus 4.24% for old-fact QA. Old statement-content token loss was 0.388, and audited statement-answer-span loss was 0.425 across all 95 old facts, versus 3.015 for old-question answer loss. These different contexts are diagnostic comparisons, not a mechanism test. The result suggests statement-to-question transfer is inadequate despite good training-context fit. Some questions also reverse the statement's relation direction. Revise acquisition on development data before running the spacing comparisons.
- All three task-created jobs are terminal. No confirmation or continuation jobs remain running; the SSH socket remains open. Artifacts were inspected remotely and not downloaded locally.

### 2026-10-07: Transfer-focused development follow-up
- The user requested a transfer fix and another calibration before the main experiment. Pushed `3e360efd863e458a8b8786b6d05b44b3935a00d2` before submitting the second LR screen. This candidate changes only LR from `1e-4` to `3e-5`, with the same facts, questions, exposure cap, seeds and 40–70% acquisition gate.
- Preparation job `1803688` completed in 25 seconds on `barley-01`. Its manifest is `af3081925fe4b2a2a06e4fe32005eee175eb0effc39ae5078e8eb6e953541d64`; split and tokenized-example hashes match the original run. GPU job `1803691` completed with exit code 0 on `oat-04` in 6:34. Acquisition still failed at E=64: old QA event-macro exact match 1.62%, loss 3.710 and MCQ accuracy 36.70%. Application time was 388.3 seconds, comprising 224.2 training, 64.3 evaluation, 91.7 checkpointing and 5.7 initialization. Peak CUDA allocation was 7.52 GB. The reduced checkpoint cadence saved most of the original 384-second write overhead.
- CPU performance jobs `1803727` and `1803732` both completed with exit code 0. The first started before a pending-only cancellation check and was allowed to finish. Actual 300M training updates took about 15–16 seconds on four threads, 11.3 seconds on eight, and 18–36 seconds on sixteen. That implies roughly 85–115 minutes for training alone at E=64, before evaluation and checkpointing. A CPU screen was not launched because the end-to-end advantage was uncertain and would consume considerably more CPU time.
- Pushed the audited relation-first declarative candidate at `71cc7ef9e9ef7beb24c2bad86d5cf17dfaed6994`. All 29 tests passed. Independent preparation confirmed all 348 evaluation records unchanged; old56/new78 training statements changed, with original text retained. One ambiguous new source stays unchanged rather than acquire unsupported detail from its answer label. The audit is by agents, not humans. Preparation job `1803956` completed in 16 seconds on `barley-01`; GPU calibration `1803957` completed with exit code 0 on `oat-04` in 6:25. Old QA event-macro accuracy peaked at 15.21% at E8 and finished at 10.89% at E64, with loss 2.369 and MCQ accuracy 48.39%. The representation helped early acquisition compared with the original 2.02% at E8, but did not meet the unchanged 40–70% gate.
- A metadata audit found old95 canonical root IDs correspond to86 normalized supporting statements, new127 to102, QA64 to56 and control62 to57. Exact supporting-statement repetition therefore complicates per-fact exposure claims. This remains a confirmation blocker; the LR-only screen retains the original data to isolate its single change.
- Implemented stable source-statement rehearsal units and unit-aware scoring for fresh recipes. Every question probe is retained; repeated statements train once per unit. Multiple-answer units retain full source statements, and colliding rewrites fall back to source text. Versioned aggregation averages probes within units, units within events, then events; legacy and unit-aware results cannot be pooled. The code rejects legacy rehearsal units for confirmation. Near-duplicate semantic content still needs its own audit.
- Pushed the transfer-v2 recipe at `297c60f260e6905cf8f558706dc5a1ea86d850d9` before running it. It adds all 225 source QA examples from the same three teaching events, each with its own source answer, at four slots per update. None overlaps an old/new/control evaluation question. All 40 tests and an actual tiny-model Stage-1-to-UNI integration passed. Independent preparation retained all 348 evaluation records unchanged.
- Preparation job `1804401` completed on `barley-04` in 20 seconds. GPU calibration `1804402` completed with exit code 0 on `oat-04` in 7:11, with 425.9 seconds application time and 7.52 GB peak CUDA allocation. At E64, old QA unit/event-macro accuracy was 16.15%, with 12.63% across the same 95 question probes, answer loss 2.334 and MCQ accuracy 50.17%. QA-teaching event-macro accuracy was 92.90%. Acquisition failed the unchanged 40–70% gate; no continuation or main experiment was launched.
- Read-only diagnostic job `1804405` completed with exit code 0 on `oat-04` in 20 seconds. Old statement-content loss was 0.357 and statement-answer-span loss 0.035, compared with old-question answer loss 2.334. Strong fit in the trained context therefore remains insufficient for generated QA transfer.
- Source review identified short supporting statements that omit entity, location or relation context needed to identify the assertion. A new development-only candidate is being authored from the original statements and pinned event factsheets, without evaluation questions or model outcomes as authoring inputs. It will retain all original probes and explicitly group equivalent assertions, with source conflicts recorded. This changes the development representation and rehearsal-unit policy; its results must not be pooled with earlier recipes.
- Completed the agent source review for `calibration-grounded.json`. All 188 old/new source units map exactly once into 157 reviewed assertion groups, 73 old and 84 new. Original source IDs and all 348 evaluation probes remain in the manifest; source conflicts and related assertions left separate are recorded in the ledger. The `grounded_assertion_v1` policy and `spacing-metrics-v3-grounded-unit-macro` schema keep this recipe distinct from earlier calibrations. Human review remains incomplete, and the loader rejects this recipe for confirmation.
- All 45 tests passed. Fresh pinned preparation verified the actual factsheet parquet hash and all evidence excerpts, with manifest `46c2bfb3eb98f62a9da92bf62c18fe044ee1b7de0ee02e111cf4ca7704dbe9fd`. Root independently verified all 348 evaluation records unchanged and the entire 225-row teaching pool identical to transfer-v2. The longest declaration is 28 tokens, so the 36-token loss budget and 96-token context remain unchanged, as do LR, QA rate, E64 cap and the 40–70% gate. Group counts change the schedule and total teaching dose at a fixed E; this combined development correction does not isolate individual causes.
- Pushed the grounded correction at `bc1fdfe3af23040f8043a4973b6aff68dc05e327` before submission. The actual tiny `hf_olmo` integration passed Stage 1, full 121-step UNI continuation, checkpoint reload, four reviews per old group and 95-probe-to-73-group exposure clocks. FarmShare preparation job `1804815` completed with exit code 0 on `rye-01` in 50 seconds, with manifest `89fb2a300ffad7f376e4de1b4b8ae198627416de712320370db6c804adc2060b`. Dependent single-GPU calibration job `1804816` started on `oat-04` at approximately 21:04 Pacific, with a 30-minute cap and `wheat-01` excluded.
- The SSH control socket disappeared after the GPU job started. After reconnecting, the user supplied Slurm accounting and `dev-grounded/stage1/acquisition_decision.json`, establishing completion with a failed acquisition gate. Initial agent inspection was blocked by T3's automatic approval reviewer, which reported unsupported `all_turns` for its configured `gpt-5.3-codex` model. The user then enabled full access; independent remote inspection succeeded and verified the decision's manifest hash against preparation.
- Grounded calibration job `1804816` completed with exit code 0 on `oat-04` in 6:20. It used one NVIDIA L40S and 32 GB allocated host memory; Slurm reports 10 allocated CPUs. Application runtime was 373.3 seconds: 208.7 training, 67.8 evaluation, 90.1 checkpointing and 5.2 initialization. Peak allocated CUDA memory was 7.52 GB. Old-fact generated-answer group/event-macro accuracy at E1/2/4/8/16/32/64 was 0.00%, 2.22%, 13.92%, 11.74%, 14.86%, 15.87% and 13.32%. Answer-token loss reached 1.932 at E32, then rose to 2.115 at E64; E64 MCQ accuracy was 65.64% and in-sample QA-teaching accuracy was 92.66%. No milestone met the unchanged 40–70% gate. The source-context correction therefore did not produce usable generated QA acquisition; these observations do not isolate the cause of failure or establish a spacing effect. Preparation and calibration are both terminal, and no continuation or main experiment was submitted. Remote artifacts remain on FarmShare; no local download was performed. This result uses grounded aggregation and must not be pooled with earlier recipes.
- Read-only inspection of the grounded run's saved predictions and training records supports inadequate statement-to-question transfer plus memorization of teaching answers. Old units train as standalone declarative next-token examples; question/answer supervision uses 225 rows from three separate teaching events. Of 95 old probes at E64, 11 are exact matches and 84 fail; 19 failed predictions exactly equal an answer in the teaching pool, including New Chicago instead of Crestwood and Leopold von Strauss instead of Jane Arthur. This is consistent with dependence on teaching-answer associations, but no ablation establishes their causal contribution. Fifty failed generations select the correct MCQ answer under the separate choice-containing prompt. All old generations terminate before the 32-token limit; five are empty. Strict matching rejects some plausible paraphrases, such as innovative strategies for innovative tactics, but substantial wrong-entity and wrong-date errors remain. Realized training totals are 258,048 loss tokens, of which 156,011 are generic filler; QA content contributes 8,021 tokens including EOS. Batch training loss falls from 5.393 to 0.217 while old-question loss worsens between E32 and E64. These measurements support a context-dependent generalization failure; they do not prove an optimizer fault, a model-capacity limit, or the exact balance of interference, memorization and scoring effects. No new job was submitted.

### 2026-10-07: Acquisition rescue authorized

- The user requested sustained independent subagent brainstorming, implementation and bounded development trials until a valid main spacing run can launch, including Opus 5.5. This renews development authorization beyond the original small calibration sequence; it does not weaken acquisition, leakage or confirmation-integrity gates.
- Independent code review found no label-shift, loss-mask or generation-batching fault. Ten tiny actual `hf_olmo` models produced matching greedy tokens with batched versus singleton decoding and cache enabled versus disabled. The recipe remains the primary concern.
- The [acquisition amendment](plans/spacing_acquisition_rescue_20261007.md) specifies a fresh shared Stage-1 declaration-plus-source-QA candidate. The main review stage still uses only declarations. This changes the acquisition initialization and narrows the scientific claim explicitly.
- An isolated source-only author is creating two acquisition question forms for each of 78 answer targets across all 73 old grounded groups. The sanitized packet contains no evaluation questions or model outputs. Source input hash is `08050cf14f3d3af87476e4931291b3ba855b4bef3e509492731be8dbabf063e2`. An independent source review and exact evaluation-overlap rejection are required before preparation.
- Opus 5.5 brainstorming is running through the required Claude Alpha/TrueFoundry provider. The first non-Alpha delegation was blocked by the project account guard before research began. Further brainstorming rounds must include prior evidence and unresolved objections.
- No rescue candidate has been submitted yet. Confirmation remains blocked pending successful development acquisition, interference sensitivity, audited content and the final preregistration.

**Historical note:** the team's GitHub code must not be stored on this computer. Two clones and three copied code files made during the research were deleted on 2026-10-02. The October 7 implementation and push were explicitly authorized by the user.

- Finished two isolated source-QA reviews, including Opus 5.5 through ClaudeAlpha. Eight targets required wording revisions; final re-review approved all 78 targets and 156 forms, with 16 source limitation flags retained by either reviewer. The final source-QA bundle SHA-256 is `53820aa79beaa1d82dbc104017e0b0dd3d31be81f9f3475e0e90c5a62278c971`, and adjudication SHA-256 is `542916d9e640d5dbd9a3301ebee919f5d0bdc15749f459fb60eac24ac2a2b304`. Human review remains false. No evaluation strings, aliases or source answers changed.
- Independent code review found and corrected the fixed-dose development acceptance bypass. Both adaptive and fixed-dose development acquisition now enforce the original 40–70% gate and warmup requirement. The actual tiny `hf_olmo` integration checks below-floor rejection, above-ceiling rejection, blocked continuation after failure, and a controlled valid metric for continuation. All 57 tests passed. The next submitted command is adaptive `calibrate`, with the milestones frozen in the acquisition amendment.

- Pushed source-QA recipe `d7e9bb5fdf762b1c8c5e69259eca299ff3d53e4e`. FarmShare preparation `1806119` completed in 20 seconds on `barley-01`, manifest `cd22ed06d7815332bd5020b082023ca187f61057559e71acdf84f91020c6aefd`. Independent comparison verified all 348 probe records, 225 teaching rows, existing tokenized examples and five Stage-2 streams unchanged. GPU calibration `1806120` completed with exit 0 on `oat-05` in 2:09, using one NVIDIA L40S and 7.52 GB peak CUDA allocation. Application runtime was 118.1 seconds, including 20.6 training, 50.9 evaluation, 36.6 checkpointing and 9.7 initialization.
- Acquisition passed at the first usable milestone E4, with 42.15% old group/event-macro generated exact match, answer loss 1.197 and MCQ accuracy 70.66%. E1/E2/E3 exact match was 3.89%/15.05%/29.90%. Supervised source-QA forms scored 83.78%; all 156 forms received two exposures, all 78 targets four, all 73 declarations four. All 95 old canonical probes fall below the frozen Jaccard 0.7 threshold. This is near transfer of supervised factual QA, not declaration-only acquisition.
- Submitted the first five-arm development sensitivity array `1806140`, limited to three parallel GPUs with separate immutable outputs under `/scratch/users/ericrcwu/agent-runs/p4-acquisition-rescue-20261008-01/`. Each arm forks the identical accepted full Stage-1 checkpoint, runs 121 review/interference updates plus a 168-update common buffer, and receives no source acquisition QA. Confirmation remains blocked pending actual source/entity/paraphrase audits, feasible-span development, precision/cost planning and preregistration.
- Added a bounded numerical A/A verifier. It compares two uninterrupted UNI updates with an actual save/perturb/restore between updates, including full model/Adam/RNG/progress/clocks, realized exposures, conditional losses and greedy answers. Both actual tiny `hf_olmo` dropout replay and deliberate failure detection pass; the suite now has 59 tests. The live GPU check is a separate prerequisite, not yet certified.

- All five pilot array tasks `1806140_0` through `_4` completed with exit 0. Slurm elapsed times were 4:10/4:57/5:03/4:45/4:12, across `oat-05`, `oat-06` and `oat-02`. At the prespecified primary delay84, NONE old loss was1.597 and EM18.44%, UNI loss1.245 and EM30.42%, MASS loss1.191 and EM26.94%, and GEN loss1.547 and EM21.31%. NONE minus UNI loss benefit was0.352 and GEN minus UNI0.301 in this single development pilot. This supports interference and declaration-review sensitivity descriptively; it is not a final-span eligibility or timing-inference result.
- Actual L40S A/A job `1806148` completed with exit0 in43 seconds on `oat-05`. Application time35.6 seconds; bitwise model/Adam/RNG/progress/clock/exposure replay and prediction/loss comparisons passed, and all bound source artifacts remained unchanged. The check's code commit is `fb27597ffa60e8debc334e706bfb231740ad70f9`.
- Metadata-only source inventories cover all80 held-out events,1258 exact statement units,1449 canonical probes and1327 answer targets. Eight source-only chunks exclude canonical/paraphrase/MCQ/model outputs. First32 candidate confirmation splits require common L252 before semantic grouping. The [final assay development amendment](plans/spacing_final_assay_amendment_20261008.md) fixes that span and retains20/30/20/10 role counts. The existing L84 bundle remains a disclosed pilot whose results have been read.
- Opus scientific round6 identified fragile first-passage E4 acquisition and content-scale extrapolation. The amendment prespecifies six standard and three acquisition-only scaled development trajectories, all complete E2/3/4/6/8 grids; acquisition alone chooses a common eligible E, without filtering failures. Three original teaching events remain separate; the source-only scaled plan has232 original old units/241 targets before grouping, around40 updates per cycle. Two fresh development source packets are being authored and must receive actual independent reviews. Six paired final-span bundles will then assess sensitivity and conservative variance. Confirmation remains blocked.
- The all100-event source/entity audit found omitted repeated names and a possible same-technology origin conflict between held-out events000/071. No clean-data assertion has been made. Reviewers distinguish entity identity uncertainty from same-proposition leakage; conservative, independently reviewed same-role component binding is being considered without dropping units or changing gold. Full cross-event adjudication and separate evaluation paraphrases are still required. No task-created Slurm job currently remains active.

- Froze the acquisition policy core at actual UTC2026-10-08T06:59:52.626772+00:00, SHA256 `56d0d72fcfbff828dafe27e6713041fc9b0d6ed0ad9de2a799c312083d9975cc`. It binds nine config/seed/role/source frames prospectively; actual prepared manifests will be bound later before scaled training. Normal acquisition may start under the core while scale content receives independent review. No timestamp is backdated.
- Implemented complete acquisition grids with five immutable full-state dose checkpoints, crash-window/terminal-decision recovery, a guarded common-dose selector, and scale-only role overrides/arm-analysis exclusions. The full suite passed77 tests plus27 mutation subtests; root independently passed29 relevant tests plus27 subtests. All numerical Session methods remain identical to the actual GPU A/A commit.
- Development scale authoring now preserves all301 original source units and348 probes in265 groups and290 answer records. Both chunks remain unreviewed pending two isolated actual reviews and repairs. The first Opus review approved all144 groups but identified three question records with semantic answer hints. Confirmation authoring chunks01/02 are also complete and unreviewed; source-only authors for03–08 are active. No main training has launched.

- Pushed grid code `1b6f6e99d99b9a2d1c3ced6bb9c39fbffc4f9ec7`, archive SHA256 `5d786c0ea63579d350c4361028d6977ad78928103e48f06af532b396a94db5b8`, before normal preparation/training. New immutable scratch root `/scratch/users/ericrcwu/agent-runs/p4-acquisition-grid-20261008-1b6f6e99`. Preparation1806304 created allsixmanifests, but an overly strict operator check failedfive tasks because evaluationseeds intentionally rotateMCQchoices. Corrected read-only verification1806312 passedallsix in2–3seconds, confirming348originalquestions/gold/candidates and225teachingrows/trainingexamples unchanged, withonlyplanned MCQorder differences. No preparedfiles were rebuilt or overwritten.
- Complete normal acquisition array1806319 finished allsix with exit0 on oat02/05/06, Slurmelapsed4:46/4:04/4:22/3:56/4:41/4:31. Allcomplete E2/3/4/6/8 trajectories were retained and no dose selected. The firstthree showed E4 EM42.09/42.70/52.24%; E6 EM60.86/56.04/55.17%; E8 EM59.24/57.74/65.57%. Selection still requires allnine grids and the frozen rule. Observed fullgrid hostRSS approached32GiB; future scalejobs will request48GiB for measuredheadroom. None of the completed or runningjobs was cancelled.
- Froze exact final-span planningpolicy at actualUTC2026-10-08T07:13:53+00:00, SHA256 `ff7e69ee8351de9e6ae6583960ea743b7e38e26d18c37d63ad619bda0a84da17`, before anyfinal-span arms. Implemented fullstate/stream/endpoint evidence validation, pairedsensitivity gates and conservativefinite-t/MC uncertainty planning. Root independently passed9 focusedtests plus14mutation subtests. Actualevidence/allocatedcost reports remain pending.
- Both independentcross-event reviews are now actual recorded artifacts. Opus adjudication `00c2434c81608c09649152fa528f89db318a883ab5c0440c86be2b258ae2e40a` retains unknownidentity and validates conservative same-roleprotection for000/071 withoutclean-data orhuman-clearance claims. Constrained first32 roleassignments require max372rawoldunits/stagger185; L252stillfits185/252=0.73413. Human review remainsfalse.
- All80confirmationevents now have unreviewed source authoring and all1449probes unreviewed evaluationparaphrases in separatechannels. Actualindependentreviews and correctiveversions continue. Canonical questions, answers andaliases remainfixed. Source03/04/07/08 required purelymechanical canonicalopaque-ID correction; original expandedduplicatecluster files remain preserved. No task-createdSlurmjob remainsrunning at this checkpoint, and no mainrun haslaunched.
