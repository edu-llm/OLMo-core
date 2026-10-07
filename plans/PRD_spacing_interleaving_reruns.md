# PRD: Reruns for the spacing study (Priority 1) and the interleaving study (Priority 2)

*Owner: P4 · Drafted 2026-10-04 · Status: proposal for team review; to be preregistered before main runs*
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
  - **Spacing (S-series):** a 300M rerun with **per-fact review schedules** (first and last review matched per fact), plus massed and generic-data control arms, a **never-trained control set**, behavioural metrics, 10 seeds with resampled event splits, LoRA and full fine-tuning, and a 1B check.
  - **Interleaving (I-series):** all four counterbalanced orders × seeds, evaluation after every block, training to convergence, a block-length sweep, a replay arm, a skill-similarity manipulation and a size sweep.
- **Cost.** Both studies use small models and short runs. Estimated **≤20 GPU-hours (spacing) + ≤15 GPU-hours (interleaving)**, dominated by evaluation. Phase 0 needs no GPUs.
- **Code location.** The team implements this on `edu-llm/OLMo-core` (branches `anshulm/fictionalqa-review` and `p4/blocked-vs-interleaved`). No team code is stored on this machine.
- **Pipeline.** The AWS platform is retired, and the P4 code never depended on it. Everything except a thin launcher adapter can be built now; see **Part 3**.
- **Paper text.** The matching methods sections are already drafted in [P4_spacing_paper_sections.md](../paper/drafts/P4_spacing_paper_sections.md) (sections tagged [RERUN]).

---

# Part 1: Spacing rerun (Priority 1)

## 1.1 Problems this rerun fixes

| Pilot problem (evidence in methods_verification.md Part I, research/notes/00, research/notes/05) | Fix |
|---|---|
| A review event was one 16-statement batch from one style; about 31–40 of 80 evaluated facts were ever reviewed; per-style last-review times differed by up to 57 updates | **Per-fact schedules** with first and last review matched per fact (§1.4) |
| Old-fact loss fell during new-fact training in every arm (format/domain adaptation) | **Never-trained control events**, evaluated at every checkpoint; QA-format teaching events shared by all arms (§1.3) |
| Facts were weakly learned (about 3.8 nats per answer token; about 3.5 Stage-1 exposures per fact; uneven across styles) | **Balanced Stage-1 exposure**, calibrated to a target exact-match band (§1.5) |
| No massed arm, so the "spacing effect" was untestable | **Massed arm** (§1.4) |
| Review replaced new-fact updates; no control for "a break from new facts" | **Generic-data arm** at the same insertion positions (§1.4) |
| 3 seeds, 80 fixed items, one event split; no equivalence test; scale-dependent rate claim | **10 seeds with resampled splits**, all old facts evaluated, TOST, item-level mixed models, two outcome scales (§1.7) |
| Cross-entropy only; exact match unreported | **Exact match, 4-choice MCQ accuracy, likelihood margin** (§1.6) |
| Optimizer and warmup reset at Stage 2; LoRA only | **Optimizer state carried across stages**; LoRA *and* full fine-tuning (§1.5) |
| One 300M model | A **1B** confirmation (§1.8) |

## 1.2 Hypotheses (preregister before main runs)

| ID | Hypothesis | Primary test |
|---|---|---|
| **S-H1** | Review improves *fact-specific* delayed retention: uniform review lowers drift-corrected old-fact loss relative to no review. | Paired contrast, 95% CI |
| **S-H2** (primary) | Schedule shape: expanding and uniform review are equivalent in drift-corrected delayed retention within ±Δ. | TOST, Δ = 0.02 nats [TODO: confirm Δ from the Phase 0 reanalysis, e.g. 10% of the S-H1 effect] |
| S-H3 | Spaced vs. massed: uniform review beats massed review at the long delay. | Paired contrast |
| S-H4 | During training, expanding gives lower average old-fact loss than uniform (the pattern Kang et al., 2014, report in humans). | Paired contrast on the Stage-2 average |
| S-H5 | Review changes the *level*, not the *rate*, of forgetting during the buffer. | Condition × time interaction on two scales (loss, accuracy) with an equivalence bound |

**Decision rules:**
- **S-H2:**
  - TOST significant → report equivalence within ±Δ.
  - A one-sided test significant in either direction with |difference| ≥ Δ → report the difference.
  - Otherwise → report "inconclusive".
- **No post hoc search** for secondary contrasts that clear significance.

## 1.3 Data

- **Source:** FictionalQA (Kirchenbauer et al., 2026), configs `fict_qa`, `gend_mcq_w_grades_03-01-26` and `blind_answer_attempts`, pinned revision `131cb74fdc3e601b5e896ed768ad9852ea35a8f9`. There are 7,500 QA rows over 100 events.
- **Split per replicate:** each seed draws a fresh event-level split from the 100 events:

  | Split | Events | Use |
  |---|---|---|
  | Old | 20 | Stage 1, then reviewed |
  | New | 40 | Stage 2 and buffer |
  | Never-trained control | 20 | Evaluated only |
  | QA-format teaching | 10 | Trained in every stage and arm, never evaluated |

  The remaining 10 events are held in reserve.
- **Facts:** one canonical question per duplicate cluster (`question_id == duplicate_root`), as in the pilot.
- **Guessability filter:** drop questions with `blind_grade_avg` ≥ [TODO: threshold, e.g. 0.5]. Log the retained counts per split.
- **Training text:** the fact's declarative statement (`fict`), as in the pilot. Optionally, where a "similar" duplicate exists, also train on its statement as a paraphrase. [TODO: decide; if used, apply in every arm.]
- **QA-format teaching events:** question–answer pairs from these 10 events are mixed into every arm at a fixed rate. The model learns the answer format without seeing any evaluated fact in QA form. This removes most of the train/test format mismatch (Allen-Zhu & Li, 2024).
- **Evaluation items:** **all** retained old facts, plus all new and control facts. Formats:
  - the `fict_qa` question with `natural_answer`;
  - the 4-choice MCQ item from `gend_mcq_w_grades`.

## 1.4 Arms and per-fact schedules

Each old fact *f* is assigned a start step *s_f* in Stage 2 (staggered evenly across facts) and receives exactly **k** review exposures. In every spaced arm, the first exposure is at *s_f* and the last at *s_f* + *L*, so first and last review times are matched **per fact**. All arms train for the same number of updates and process the same number of tokens.

| Arm | Review offsets for fact *f* (relative to *s_f*) | Role |
|---|---|---|
| `NONE` | none; slots filled with new-fact statements | baseline |
| `UNI` | 0, *L*/(k−1), 2*L*/(k−1), …, *L* | uniform |
| `EXP` | 0, …, *L* with gaps growing by ratio *r* (default *r* = 1.35, as in the pilot; sensitivity *r* = 2) | expanding |
| `MASS` | *L*−k+1, …, *L* (k consecutive updates ending at the shared last review) | massed; recency matched |
| `GEN` | `UNI` positions, filled with generic web text (a fixed Dolma sample) instead of old facts | "break from new facts" control |
| `CON` (optional) | time-reversed `EXP` | contracting |
| `ADD-UNI` (optional) | `UNI`, but review statements are *added* to the batch instead of replacing new-fact statements | displacement control |

- **Insertion:** at each Stage-2 update, the review statements scheduled for that step replace an equal number of new-fact statements in the 16-sequence batch (`ADD-UNI` excepted). The schedule generator caps the review load per step and logs the realised per-fact exposure times.
- **Defaults** [TODO: set in Phase 0]: k = 4; *L* = [TODO]; Stage-2 length = [TODO]; buffer length = [TODO, at least as long as the pilot's 180 updates]. Choose these so that the per-step review share stays below 50% of the batch in every arm.
- **Review content:** the fact's training statement (restudy). Optional secondary arm: review in QA format (retrieval-like), to relate to the testing-effect literature (Roediger & Karpicke, 2006).

## 1.5 Model and training

- **Model:** `allenai/DataDecide-dolma1_7-300M`, revision `4b1b42ff` (`step45787-seed-default`); 371.5M parameters (Magnusson et al., 2025).
- **Adaptation:**
  - LoRA r = 16, α = 32, dropout 0.05, all linear layers (as in the pilot);
  - **plus full fine-tuning** at a learning rate chosen in Phase 0 (Biderman et al., 2024).
- **Optimizer:** AdamW (β = 0.9, 0.95), weight decay 0.1, clip 1.0. Constant LR after a single 10-step warmup at the start of Stage 1 (Luo et al., 2026). **One optimizer state across all stages**; the pilot re-initialized it at Stage 2.
- **Stage 1:** each old fact appears exactly *E* times in shuffled order, with style-balanced, fact-level sampling. Calibrate *E* in Phase 0 so that mean old-fact exact match after Stage 1 is in [TODO: target band, e.g. 40–70%].
- **Stage 2 and buffer:** new facts as in the pilot; QA-format teaching events at a fixed share in all stages.
- **Precision:** bf16 if supported (the pilot used fp16 autocast).
- **Seeds:** 10 per arm. A seed fixes the event split, LoRA initialization and data order, and is shared across arms (paired design). All arms in a seed start from that seed's shared Stage-1 checkpoint.

## 1.6 Measurement

- **Checkpoints:**
  - every [TODO] updates in Stage 2;
  - buffer delays of 1, 15, 60, 180 and [TODO: final] updates.
- **Metrics, per fact:**
  - answer-token cross-entropy (nats), reported with and without EOS;
  - greedy exact match;
  - 4-choice MCQ accuracy;
  - likelihood margin: log p(correct) − logsumexp over the distractors.
- **Primary outcome:** **drift-corrected delayed retention**, the mean old-fact answer loss minus the mean never-trained-control answer loss at the final delay (O'Neill, 2026, uses a similar drift control).
- **Secondary outcomes:**
  - exact match and MCQ accuracy at the final delay;
  - the Stage-2 average (S-H4);
  - forgetting during the buffer (S-H5);
  - new-fact loss and accuracy (plasticity).
- **Exposure log:** per fact, the number and times of every Stage-1 and review exposure. This makes the item-level dose–response analysis possible.

## 1.7 Analysis

- **Unit:** item-level linear mixed-effects models with crossed random effects for fact (nested in event) and seed (Baayen et al., 2008; Barr et al., 2013). Arm is a fixed effect, with random slopes where supported. As a check, also report seed-level paired differences with 95% CIs and all per-seed values.
- **Equivalence:** TOST for S-H2 at the preregistered Δ (Lakens, 2017).
- **Multiplicity:** Holm correction across S-H1, S-H3, S-H4 and S-H5.
- **Forgetting:** model loss and accuracy over all buffer checkpoints. Report the condition × time interaction on both scales (Loftus, 1985). Add a horizontal comparison: the delay at which a review arm reaches the no-review arm's buffer-onset level.
- **Power:** use the Phase 0 reanalysis variance to confirm that 10 seeds give ≥80% power for S-H1 and a TOST bound of Δ. Increase seeds if they don't.

## 1.8 Scale and generality (after the 300M grid)

- **1B confirmation:** `NONE`, `UNI`, `EXP` and `MASS` at the DataDecide 1B size on the same Dolma 1.7 recipe [TODO: confirm repository and revision], LoRA only, 5 seeds.
- **Optional:** Pythia-410M/1B for a second model family (Biderman et al., 2023). Yang et al. (2024) report order effects emerging between 160M and 410M.

## 1.9 Phases and compute

| Phase | Work | Compute |
|---|---|---|
| **S0** (now, no GPUs) | Collect the pilot `runs/` folder and CI script. Reanalyse: exact match; reviewed vs. never-reviewed facts; TOST; buffer slopes. Implement the schedule generator and exposure logging. Write and timestamp the preregistration. | none |
| **S1** calibration | Choose *E*, k, *L*, the Stage-2 and buffer lengths, and the full-FT LR. 2 seeds; small grid. | ~2 GPU-h |
| **S2** main grid | 5 core arms × 10 seeds × {LoRA, full FT} = 100 runs (+ optional arms) | ~8–12 GPU-h |
| **S3** scale | 1B: 4 arms × 5 seeds | ~4–6 GPU-h |

A pilot run trains for about 0.4M tokens, so evaluation dominates the cost. Replace these estimates with measured throughput after S1.

## 1.10 Acceptance criteria

- [ ] Per-fact exposure logs show every old fact received exactly k reviews at its scheduled offsets in every spaced arm.
- [ ] Every number in the paper traces to a committed results file and analysis script.
- [ ] The preregistration is timestamped before S2 starts.
- [ ] The results report S-H1–S-H5 with CIs, TOST and per-seed values, on both the loss and accuracy scales.

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
| Data | FictionalQA download pinned to revision `131cb74`, the event splits, the guessability filter, cached locally; the interleaving generators | No |
| Schedules | the per-fact review schedule generator; the block-length and Williams-square order generator; unit tests that check the realised exposure logs | No |
| Training | the training loop with the same model, optimizer and LoRA/full-fine-tuning options, a constant LR, and one optimizer state across stages | No |
| Evaluation | loss, exact match, MCQ, likelihood margin; per-item logs | No |
| Analysis | the mixed models, TOST, Holm, figures; reads only the logged results files | No |
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
- **Needs a GPU, but not the final pipeline:** the S1 and I1 calibration runs. These fit on one 24 GB card, so FarmShare, ORCD or a single rented GPU will do.
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
