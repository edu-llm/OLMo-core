# PRD: Controlled experiments on mastery-gated curriculum training

*Owner: P4 mastery-gating subteam · Drafted 2026-10-04 · Status: proposal for team review*
*Builds on: [P4_final_report.md](../paper/source/P4_final_report.md) (Test 03), the mock review in [research/notes/10_final_report_methods_review.md](../research/notes/10_final_report_methods_review.md) §B2, the literature in [research/notes/08_mastery_curriculum.md](../research/notes/08_mastery_curriculum.md), and the code check in [methods_verification.md](../research/methods_verification.md) Part III.*

## TL;DR

- **Should the subteam run experiments? Yes.**
  - Reviewers at any ML venue, and at CogSci, will not accept a proposal-only section.
  - They will not accept the two current preliminary experiments as evidence either. Those have no code anywhere in the repository, appear to be single runs, have no stated model size or seeds, and contain one number that contradicts its own table.
  - The good news: the experiment that would convince a fair reviewer is small. Phase 1 below is about **≤50 GPU-hours** on tiny models.
- **The question:** does *performance-contingent* progression (unlock the next skill only when held-out probes show mastery) beat *fair non-contingent* schedules when replay and total tokens are held equal?
  - The current preliminary results can't answer this. Experiment I compares gate + replay against baselines without replay. In Experiment II, once the clock arm also gets replay, the gap shrinks to 0.97 vs. 0.91, which is within single-run noise.
- **Design (Phase 1):**
  - **Tasks:** two synthetic prerequisite tasks (variable-chain depth; addition → multi-operand addition).
  - **Arms:** 9 schedules, including a **yoked clock** (same unlock times, not contingent on the model's own performance) and a **mixture-matched shuffle** (same skill proportions, no order).
  - **Controls:** ≥10 paired seeds, matched *tokens*, constant LR, and gate probes that are disjoint from a sealed exam.
  - **Pre-registration:** one primary contrast, registered before the main runs.
- **Then (Phase 2):** scale toward the team's target size (from-scratch 20M → 300M, then continued training of open 0.4–1.4B models). A natural-domain study (math/code skill graphs) only comes after that.
- **Novelty is real but narrow.** Threshold-gated curricula with replay are already used as a *training tool* (Kohli et al., 2026; Yao et al., 2025), and progress-driven curricula exist (Graves et al., 2017; Matiisen et al., 2020). Theory says an easy + hard mixture can match a staged curriculum on chain tasks (Wang et al., 2025). Nobody has run a controlled test of contingent vs. yoked/tuned progression at matched tokens. That is the contribution, and a rigorous null would also be publishable.

---

## 1. Background and motivation

The human mastery-learning literature reports gains when learners advance only after demonstrating competence (Bloom, 1968). Kulik et al. (1990) found about 0.52 SD on end-of-unit exams, but only about 0.29 SD on standardized tests, and Slavin (1987) found roughly zero once *time on task* is equalized. A matched-token design is exactly the time-equalized case, so the human evidence gives no strong prior for a benefit. That makes the experiment informative whichever way it comes out. In machine learning:
- Curricula help mainly under small budgets or noisy labels, and often fail to beat well-tuned random orderings (Wu et al., 2021; BabyLM: Warstadt et al., 2023; Diehl Martinez et al., 2023).
- Easy-to-hard data can be necessary for compositional tasks: training only on deep chains needs exponentially many samples, while a staged curriculum *or* a simultaneous easy + hard mixture makes the task learnable (Wang et al., 2025; Yao et al., 2025).
- Removing a skill from training once it is "mastered" causes catastrophic forgetting unless it is replayed (Robins, 1995; Rolnick et al., 2019). The subteam's own preliminary data show the same (the no-replay arms).

**Gap.** Gated curricula with replay are used in practice: Kohli et al. (2026) unlock k+1-hop data at ≥95% held-out k-hop accuracy and jointly train all earlier stages. But no study isolates the *contingency* of progression from its *timing*, its *data allocation*, and *replay*, at matched tokens and with enough seeds. Without that, "mastery gating helps" cannot be separated from "replay helps", "seeing hard data at the right time helps", or "spending more tokens on hard skills helps".

## 2. Research questions and hypotheses (to be preregistered)

| ID | Question | Hypothesis | Test |
|---|---|---|---|
| **H1 (primary)** | Does contingent progression beat non-contingent progression when replay is equal? | Gate + replay > tuned clock + replay, yoked clock + replay, and mixture-matched shuffle on sealed-exam macro accuracy | Three one-sided paired contrasts, Holm-corrected; SESOI = 3 accuracy points |
| H2 | Is replay necessary for gating to work? | Gate + replay > gate without replay, mainly on early-skill retention; gate × replay interaction > 0 | 2 × 2 (gate/clock × replay/none) |
| H3 | Is any gating benefit about *order* or about *allocation*? | If the gain is from order, gate + replay > mixture-matched shuffle; if from allocation, they tie | Paired contrast + TOST |
| H4 | Is gating more sample-efficient? | Fewer tokens to reach a target sealed-exam accuracy on the deepest skill | Tokens-to-threshold, censored at the budget |
| H5 (Phase 2) | Does the effect change with model size? | The size of the effect changes monotonically with size; no direction pre-specified | Size × arm interaction |

**Decision rules (fixed in advance):**
- **H1 supported** if all three Holm-adjusted contrasts are significant *and* each effect is ≥ 3 points.
- **Equivalence (publishable null)** if TOST shows gate + replay within ±3 points of the best non-contingent arm.
- **Inconclusive** otherwise. Report it as such; don't search for a secondary contrast that clears significance.

## 3. Phase 0: recover and instrument (week 1)

1. **Recover or rewrite the code for both preliminary experiments** and commit it to a new branch (e.g., `p4/mastery-gated`) with configs, generators and logs. *Every number in a paper must trace to a script and a results file.*
2. Re-run both preliminary experiments with **3 seeds** to estimate seed variance for the power analysis. Fix the reported numbers, including the text/table mismatch (0.59 vs. 0.55) and the one-pager's mislabelled panel.
3. Add logging of **tokens per skill per step**, unlock times, probe results with binomial CIs, and probe compute.
4. Write and timestamp the **preregistration** (OSF, or a dated commit) before Phase 1 main runs. If none is made, remove the word "preregistered" from the report.

## 4. Phase 1: controlled synthetic study (weeks 2–4)

### 4.1 Tasks

| Task | Skills (prerequisite order) | Training distribution | Gate-probe family (G) | Sealed-exam family (E), disjoint from G |
|---|---|---|---|---|
| **T1: variable chain** (LEGO-style pointer chasing; Zhang et al., 2022). Primary | Skill *k* = resolve a depth-*k* chain of scrambled definitions, *k* = 1…6 | Depth *k*; variable names from name set A; 0–4 distractor definitions | Depth *k*; **unseen names** (set B); 0–4 distractors | Depth *k*; unseen names (set C) **and** 5–10 distractors. Secondary: depth extrapolation to *k*_max + 1, + 2 |
| **T2: arithmetic** (secondary) | 1) 2-operand addition; 2) multi-operand addition; 3) optional third skill with real prerequisite dependence | Up to 10 digits; 3–5 operands | 11–12 digits; 6–7 operands | 15–20 digits; 8–25 operands |

- **Formatting held fixed across all arms:** digit-level tokenization and reversed-output targets for arithmetic, the standard choices that make carrying learnable (Lee et al., 2024; McLeish et al., 2024).
- **Generators** are seeded, published, and checked for train/probe/exam disjointness.
- **Format-aid ablation (T2, one size):** reversal and place-value aids on vs. off. Gating effects may depend on whether the task is learnable without them.
- **Calibration rule:** tune task difficulty so that shuffle reaches 30–80% on the sealed exam at the budget. This avoids floor and ceiling effects.

### 4.2 Arms

| Arm | Description | Role |
|---|---|---|
| `SHUF` | IID over all skills from step 0 | Standard baseline |
| `SHUF-MM` | IID, but with skill proportions matched to `GATE+R`'s realised per-skill token allocation (averaged over seeds) | Separates **order** from **allocation** (H3) |
| `CLOCK-T+R` | Fixed stage quotas, **tuned** on G-family validation with the same tuning budget as the gate's τ × ρ grid; replay fraction ρ | Fair non-contingent baseline |
| `CLOCK-Y+R` | **Yoked**: unlock times copied from a `GATE+R` run with a *different* seed (and, as a second variant, the same seed); replay ρ | Isolates **contingency** from timing (Church, 1964, explains why to yoke across runs) |
| `GATE+R` | Unlock skill *k*+1 when the lower 95% Wilson bound of probe accuracy on skill *k* is ≥ τ; replay of unlocked skills at fraction ρ | **Main arm** |
| `GATE` | Same gate, no replay | Replay ablation (H2); expected to forget |
| `CLOCK-T` | Tuned clock, no replay | Completes the 2 × 2 (H2) |
| `PACE-RAND` | Same unlock times as `GATE+R`, but each newly unlocked slice is a random skill | Pacing-only control (Wu et al., 2021) |
| `MIX` | Fixed easy + hard mixture throughout, no gating | The alternative theory predicts works (Wang et al., 2025; Zaremba & Sutskever, 2014) |
| `GATE+R+LA` (optional) | `GATE+R` plus a small look-ahead share of *not-yet-unlocked* harder data | Zaremba & Sutskever's (2014) "combined" strategy mixed in **harder** examples, which is not the same as replaying easier ones |

- **Gate defaults:** τ = 0.90; probe every 2% of the budget with n = 256 items per frontier skill.
- **Replay default:** ρ = 0.25, sampled uniformly over mastered skills.
- **Sensitivity grid:**
  - for `GATE+R`: τ ∈ {0.80, 0.90, 0.97} × ρ ∈ {0.10, 0.25, 0.50};
  - for `CLOCK-T+R`: the same ρ values.
- **Information parity.** The clock arm may use the same G-family probes during its tuning, so the comparison is about *contingent progression*, not about access to information (Kapoor & Narayanan, 2023). The E-family exam is never used for any decision in any arm.

### 4.3 Models and training

- **Architecture:** decoder-only transformers built from OLMo-core configs, trained from scratch:
  - **S** about 20M parameters (6 layers, d = 384);
  - **M** about 85M parameters (12 layers, d = 768).
- **Training:**
  - constant LR after a short warmup, identical across arms, since LR decay discounts late curriculum data (Luo et al., 2026);
  - a robustness check on one size using the remedies Luo et al. (2026) recommend: moderate (not near-zero) final LR decay, plus checkpoint averaging. An identical decaying schedule across arms does *not* remove the confound, and WSD also decays at the end.
- **Budget:** a matched **token** budget B per arm (counting non-padding tokens), set so that `SHUF` is still improving at B. Train past B to plateau on a subset to show where the rankings are budget-dependent. Probe compute is logged and reported both charged and uncharged (Dehghani et al., 2022).
- **Seeds:** 10 paired seeds per arm. A seed fixes the initialization and the data stream, shared across arms.

### 4.4 Metrics

- **Primary:** sealed-exam (E family) accuracy averaged over all skills at budget B.
- **Secondary:**
  - tokens-to-threshold on the deepest skill, with censoring;
  - a precisely defined AUC (skill, metric, checkpoint grid and normalization fixed in advance);
  - per-skill retention;
  - unlock times;
  - G-family accuracy;
  - depth extrapolation (T1);
  - accuracy at 0.5B and 0.25B.

### 4.5 Analysis

- Paired differences across seeds. Report mean, 95% CI, every per-seed value, and probability of improvement (Agarwal et al., 2021).
- Holm correction for the H1 contrasts (Holm, 1979).
- TOST equivalence with SESOI = 3 points.
- Mixed-effects model with seed and skill as random effects for per-skill outcomes.
- Power: the Phase 0 seed variance decides whether 10 seeds suffice. With n = 10 paired seeds, a paired test detects d_z ≈ 1.0 at 80% power; increase *n* if the observed variance requires it (Lakens, 2022).

### 4.6 Compute estimate

| | Runs | Notes |
|---|---|---|
| Main grid | 2 tasks × 9 arms × 10 seeds × 2 sizes = 360 | Each run is about 5 × 10⁷ tokens (the scale of preliminary Experiment II) |
| Sensitivity | (9 gate + 3 clock settings) × 10 seeds × 1 size × 1 task = 120 | |
| **Total** | ~480 short runs | At ~85M parameters, a run is ~2.6 × 10¹⁶ training FLOPs (~1–2 GPU-minutes on one H100 at realistic utilisation) plus probing. **Phase 1 ≈ 20–50 GPU-hours.** Replace these estimates with measured throughput in Phase 0. |

## 5. Phase 2: scale and realism (weeks 5–8; run regardless of the Phase 1 outcome)

1. **Size ladder (from scratch, T1 only):** 20M → 85M → 300M, adding 1B if the effect changes with size. Order effects in LMs can *emerge* between 160M and 410M (Yang et al., 2024, in research/notes/06), so do not extrapolate from the smallest models.
2. **Continued training of open pretrained models** on a synthetic math skill graph with real prerequisites:
   - **Models:** Pythia 410M / 1B / 1.4B (open data, intermediate checkpoints) and OLMo 2 1B. These match the team's target of at most a few billion parameters (see research/notes/06 for the suites).
   - **Setup:** full fine-tuning and LoRA, 5 paired seeds, the 6 primary arms.
3. **Report results at both scales** with the same primary metric and decision rules.

## 6. Phase 3 (deferred): natural domains

Competitive programming and competition math with an explicit prerequisite graph, at 1–3B parameters, as proposed in the final report. **Only after Phase 1–2.** The attribution-graph extension in the report is optional follow-up work and should not appear in the main paper.

## 7. Deliverables and acceptance criteria

- [ ] Code, generators, configs and logs on a public branch; every reported number traceable to a results file.
- [ ] A timestamped preregistration before the Phase 1 main runs.
- [ ] A Phase 1 report covering: H1–H4 with CIs and per-seed values; per-skill tables; unlock-time distributions; the sensitivity analysis; and a token-accounting table (per arm and per skill).
- [ ] A Phase 2 report covering the size ladder and the pretrained-model results.
- [ ] Updated Test 03 text that describes only what was run. Move cost analysis, competitive programming and attribution graphs to future work, and remove the informal remarks in Limitations.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The gate never fires (τ too high) or fires on noise | Wilson lower-bound rule; τ sensitivity; report unlock times per seed |
| Probe leakage into the exam | Disjoint G and E families (names, lengths, distractor counts); E is never consulted during training |
| Ceiling or floor effects | Calibrate so that `SHUF` reaches 30–80% at B |
| The "win" is token reallocation, not gating | `SHUF-MM` and `PACE-RAND` controls; per-skill token accounting |
| The baselines are strawmen | Equal tuning budget for `CLOCK-T+R`; yoked clock; `MIX` arm |
| Results depend on the budget | Report curves to plateau and accuracy at three budgets |
| LR-schedule confound | Constant LR as the primary schedule; robustness check with moderate decay plus checkpoint averaging (Luo et al., 2026) |
| Grokking-style delayed generalization makes unlock times erratic | Gate on G-family (OOD) accuracy rather than training accuracy; log curves (Power et al., 2022; Nanda et al., 2023) |

## 9. Out of scope

Synthetic-student or human-classroom claims; claims about from-scratch pretraining at the 7B+ scale; attribution-graph analysis (deferred).

## References

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *Advances in Neural Information Processing Systems, 34*, 29304–29320. https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html
- Bloom, B. S. (1968). Learning for mastery. *Evaluation Comment, 1*(2), 1–12. https://eric.ed.gov/?id=ED053419
- Church, R. M. (1964). Systematic effect of random error in the yoked control design. *Psychological Bulletin, 62*(2), 122–131. https://doi.org/10.1037/h0042733
- Dehghani, M., Arnab, A., Beyer, L., Vaswani, A., & Tay, Y. (2022). The efficiency misnomer. In *International Conference on Learning Representations*. https://openreview.net/forum?id=iulEMLYh1uR
- Diehl Martinez, R., Goriely, Z., McGovern, H., Davis, C., Caines, A., Buttery, P., & Beinborn, L. (2023). CLIMB – Curriculum learning for infant-inspired model building. In *Proceedings of the BabyLM Challenge at CoNLL 2023* (pp. 112–127). https://doi.org/10.18653/v1/2023.conll-babylm.10
- Graves, A., Bellemare, M. G., Menick, J., Munos, R., & Kavukcuoglu, K. (2017). Automated curriculum learning for neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR 70, pp. 1311–1320). https://proceedings.mlr.press/v70/graves17a.html
- Slavin, R. E. (1987). Mastery learning reconsidered. *Review of Educational Research, 57*(2), 175–213. https://doi.org/10.3102/00346543057002175
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70. https://www.jstor.org/stable/4615733
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), Article 100804. https://doi.org/10.1016/j.patter.2023.100804
- Kohli, H., Parthasarathy, S., Sun, H., & Yao, Y. (2026). Loop, think, & generalize: Implicit reasoning in recurrent-depth transformers. In *Conference on Language Modeling (COLM 2026)*. https://arxiv.org/abs/2604.07822
- Kulik, C.-L. C., Kulik, J. A., & Bangert-Drowns, R. L. (1990). Effectiveness of mastery learning programs: A meta-analysis. *Review of Educational Research, 60*(2), 265–299. https://doi.org/10.3102/00346543060002265
- Lakens, D. (2022). Sample size justification. *Collabra: Psychology, 8*(1), Article 33267. https://doi.org/10.1525/collabra.33267
- Lee, N., Sreenivasan, K., Lee, J. D., Lee, K., & Papailiopoulos, D. (2024). Teaching arithmetic to small transformers. In *International Conference on Learning Representations*. https://openreview.net/forum?id=dsUB4bst9S
- Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *International Conference on Learning Representations*. https://openreview.net/forum?id=T5wkZJqzkz
- Matiisen, T., Oliver, A., Cohen, T., & Schulman, J. (2020). Teacher–student curriculum learning. *IEEE Transactions on Neural Networks and Learning Systems, 31*(9), 3732–3740. https://doi.org/10.1109/TNNLS.2019.2934906
- McLeish, S., Bansal, A., Stein, A., Jain, N., Kirchenbauer, J., Bartoldson, B. R., Kailkhura, B., Bhatele, A., Geiping, J., Schwarzschild, A., & Goldstein, T. (2024). Transformers can do arithmetic with the right embeddings. *Advances in Neural Information Processing Systems, 37*. https://arxiv.org/abs/2405.17399
- Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. In *International Conference on Learning Representations*. https://openreview.net/forum?id=9XFSbDPmdW
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2201.02177
- Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318
- Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. *Advances in Neural Information Processing Systems, 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html
- Wang, Z., Nichani, E., Bietti, A., Damian, A., Hsu, D., Lee, J. D., & Wu, D. (2025). Learning compositional functions with transformers from easy-to-hard data. In *Proceedings of the 38th Conference on Learning Theory* (PMLR 291, pp. 5632–5711). https://proceedings.mlr.press/v291/wang25a.html
- Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., Mosquera, R., Paranjabe, B., Williams, A., Linzen, T., & Cotterell, R. (2023). Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *Proceedings of the BabyLM Challenge at CoNLL 2023* (pp. 1–34). https://doi.org/10.18653/v1/2023.conll-babylm.1
- Wu, X., Dyer, E., & Neyshabur, B. (2021). When do curricula work? In *International Conference on Learning Representations*. https://openreview.net/forum?id=tW4QEInpni
- Yang, Y., Jones, M., Mozer, M. C., & Ren, M. (2024). Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training. In *Advances in Neural Information Processing Systems 37*. https://openreview.net/forum?id=YSs1z5udBY
- Yao, Y., Du, Y., Zhu, D., Hahn, M., & Koller, A. (2025). Language models can learn implicit multi-hop reasoning, but only if they have lots of training data. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing* (pp. 9684–9702). https://arxiv.org/abs/2505.17923
- Zaremba, W., & Sutskever, I. (2014). *Learning to execute* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1410.4615
- Zhang, Y., Backurs, A., Bubeck, S., Eldan, R., Gunasekar, S., & Wagner, T. (2022). *Unveiling transformers with LEGO: A synthetic reasoning task* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2206.04301
