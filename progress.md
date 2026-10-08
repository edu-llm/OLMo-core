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

**Historical note:** the team's GitHub code must not be stored on this computer. Two clones and three copied code files made during the research were deleted on 2026-10-02. The October 7 implementation and push were explicitly authorized by the user.
