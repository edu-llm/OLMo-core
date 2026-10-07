# Methods, Statistics and Mock Review: P4 Final Report, TEST 02 (Blocked vs. Interleaved) and TEST 03 (Mastery-Gated Curricula)

## TL;DR

- **TEST 02 describes an experiment that was not run.** The text promises a 195M pretrained model on arithmetic, logic, geometry and statistics, with 4 counterbalanced orders × 3 seeds. The only completed run is a 162M from-scratch model on four bracket-cued digit tasks, with one seed and one order, evaluated only at the end: blocked 27.4% vs. interleaved 61.6%. The text instead quotes a "toy MLP pilot" (52.8% vs. 79.2%) that has no code. Report the run that exists.
- **The 34-point gap is mostly forgetting plus recency, which is already known.**
  - The blocked arm scores 95% on its last-trained skill and 0–15% on the other three.
  - With no per-block evaluation, we cannot tell whether the early skills were forgotten or never learned.
  - "Interleaving prevents catastrophic forgetting" is textbook (McCloskey & Cohen, 1989; McClelland et al., 1995). It has already been shown in LLMs under a human-learning framing (Lee, Cho & Yoo, 2024; Russin et al., 2025).
- **TEST 02's fixes are cheap, and the code is already half-way there.** The completed run took 71 seconds on one GPU, and the code already generates four counterbalanced orders. Run a Williams square × 5 seeds, with per-block evaluation, a matched-lag test, a blocked + review arm and a block-length dose-response. Pick something new to claim, such as the similarity moderator or the dose-response.
- **Yes, reviewers will require the mastery-gating subteam to run real experiments.**
  - Neither preliminary experiment has code. Both appear to be single runs, and the text and table disagree (0.59 vs. 0.55).
  - Experiment I confounds gating with replay. Once the clock baseline also gets replay (Experiment II), the gap shrinks to 0.97 vs. 0.91, which is inside the single-run noise visible in Figure 3.
  - Proposal-only sections are not publishable as research papers.
- **Minimum mastery experiment** (days on one GPU):
  - Chain-depth task with 6 arms: shuffle, mixture-matched shuffle, tuned clock + replay, yoked clock + replay, mastery + replay, and mastery without replay.
  - At least 10 paired seeds, matched *tokens*, and gate probes disjoint from the sealed exam.
  - One pre-registered primary contrast.
- **Statistics:**
  - The unit of analysis is the training run, not the test item.
  - Pre-specify one primary outcome per test and Holm-correct the per-skill contrasts.
  - 20 paired runs detect d_z ≈ 0.66; 5 runs detect only d_z ≈ 1.7.
  - Drop "preregistered" from the mission statement unless a timestamped registration exists.
- **Venues:**
  - Closed: ICLR 2027, NeurIPS 2026 and its workshops, AISTATS 2027, AAAI-27. The ARR 12 Oct 2026 deadline is too soon.
  - Separate papers: CogSci 2027 or CCN 2027 for interleaving with the human comparison (both ~Feb 2027, UNVERIFIED); ICLR 2027 workshops (~1 Feb 2027) for mastery.
  - Combined paper: COLM 2027 (~late Mar 2027, UNVERIFIED), ACL 2027 via ARR (4 Jan 2027, verified) or TMLR (rolling), but only once all three tests meet the bar.

**Prepared:** 2026-10-04. This is a reviewer-side research aid; `P4_final_report.md` was not edited. TEST 01 and the synthetic-student work are out of scope.

**Material reviewed:**
- `P4_final_report.md`, TEST 02 and TEST 03, including `paper/figures/final_report_fig2_mastery_prelim1_table.png` and `paper/figures/final_report_fig3_mastery_prelim2_curve.png`.
- Public branch `p4/blocked-vs-interleaved` of https://github.com/edu-llm/OLMo-core, **read through WebFetch only** (nothing was cloned or downloaded): `README.md`, `generate_data.py`, `train_compare.py`, and `results/farmshare-olmo-162m-job1654870/RESULTS.md`.
- No code for TEST 03's preliminary experiments exists on the three known branches.

**Verification policy:**
- Same standard as `05_methods_stats_review.md`. Every new reference was checked against a primary or authoritative source using WebSearch/WebFetch, with parallel verification agents.
- Items marked "(from 05)" or "(from 03/06)" reuse entries verified in earlier files.
- **UNVERIFIED** or **PARTLY VERIFIED** marks any field that could not be confirmed.
- Citations are APA 7.
- One side effect to disclose: WebFetch automatically cached two PDFs/BibTeX files from arXiv and NeurIPS in the Claude tool-results cache. Nothing came from GitHub.

## What was run vs. what the report says (TEST 02)

| | Report text (proposal) | Report's cited pilot | Actually run (GitHub `p4/blocked-vs-interleaved`) |
|---|---|---|---|
| Model | ~195M, 24 layers, width 768, pretrained checkpoint + burn-in | "toy MLP" | 162,164,736 params, 12 layers, width 768, SwiGLU 2048, untied embeddings, **random init** |
| Tasks | Arithmetic, logic, geometry, statistics; 4-choice natural language | unknown | Rotate-left `< >`, rotate-right `[ ]`, reverse `{ }`, swap-pairs `( )` on the same 6-digit inputs; free-form exact match |
| Training | 1,280 updates (4 sessions × 4 subjects × 80), then 160 interference updates | unknown | 512 updates (4 × 128), batch 32, AdamW, constant LR 1e-4, no interference phase |
| Orders / seeds | 4 counterbalanced orders × 3 data seeds = 12 pairs | unknown | Order 0 (ABCD) only, 1 seed. The code generates 4 cyclic orders. |
| Evaluation | Immediate + post-interference macro accuracy | "terminal accuracy" | Final only; 256 held-out items per skill |
| Result | — | 52.8% blocked vs. 79.2% interleaved; "independent models: zero effect" | Blocked 27.44% vs. interleaved 61.62%. Per skill A/B/C/D: 0/15.2/0/94.5 vs. 81.6/95.3/27.3/42.2 |
| Compute | 5.7 × 10¹⁶ FLOPs/arm (projected) | — | 1 min 11 s on one L40S (`RESULTS.md`, labelled "smoke test") |

## Reviewer's quantitative checks (reproducible from the numbers above)

- **TEST 02 item-sampling CI.** Difference 34.18 pp, Wald SE 2.06 pp, 95% CI [30.1, 38.2]. This reflects only test-item sampling, not training variance.
- **Recency-excluded accuracy.** On skills A–C (not trained last), blocked averages 5.1% and interleaved 68.1%. On D (trained last), blocked beats interleaved 94.5% vs. 42.2%.
- **Power** (paired two-sided t, α = .05, 80%; Monte Carlo, 4,000 sims per step). Detectable d_z: n = 3 → 3.26; n = 5 → 1.68; n = 8 → 1.16; n = 10 → 0.99; n = 12 → 0.88; n = 16 → 0.75; n = 20 → 0.66.
- **TEST 03 internal arithmetic.** In Figure 2, the in-distribution means equal the averages of the per-depth values (e.g., (0.97 + 0.99 + 0.97 + 0.95 + 0.18)/5 = 0.81). In Experiment II, the AUCs match a normalized trapezoid over Figure 3's ~11 checkpoints (mastery + replay ≈ 0.318; shuffle ≈ 0.108, from values read off the plot). The text's "0.59" for no-replay addition retention does not match the table's 0.55.
- **Parameter split** (TEST 02 model). Transformer blocks ≈ 12 × (4·768² + 3·768·2048) ≈ 85M. The remaining ≈ 77M are input and output embeddings (implying a vocabulary of ~50k), of which the task uses ~15 tokens.

---

# Part A — Methods and statistics literature for these two designs

Each subsection explains why the design needs this literature, gives a concrete recipe, and lists sources with APA 7 citation, URL and verification status. Sources marked "(from 05)" or "(from 03/06)" were verified for earlier reviews in this folder and are reused without re-checking. All other sources were checked for this review against publisher/DOI pages, PMLR, NeurIPS proceedings, OpenReview, ACL Anthology, arXiv, JSTOR/PubMed or ERIC (via WebSearch/WebFetch only). **UNVERIFIED** or **PARTLY VERIFIED** marks any field that could not be confirmed.

## A1. Paired designs and counterbalancing (Latin squares) for order effects

**Why it matters here.** TEST 02 is a within-"subject" design: the same initialization is trained under two schedules, and the four skills themselves are ordered. The skill that comes last is favoured by recency, and a skill may be hurt more by some predecessors than others (carryover). The code already generates a cyclic 4 × 4 Latin square (ABCD, BCDA, CDAB, DABC). That balances *serial position* but not *first-order carryover* (A is always followed by B). TEST 03's mastery gate also needs a timing-matched non-contingent control, which is the "yoked" design from learning psychology.

**Recipe.**
- Use a **Williams square** (for 4 skills, one 4 × 4 square in which every skill immediately follows every other skill exactly once), crossed with ≥ 5 seeds, so order is a blocking factor and seed is the replicate.
- Pair blocked and interleaved arms within seed × order (same initialization, data stream and within-skill example order, as the code already does). Analyse paired differences, with order as a stratum.
- If more than first-order carryover matters (e.g., in block-length sweeps), generate orders with Brooks's (2012) Euler-circuit method.
- For a yoked control in TEST 03, copy switch times from several different mastery runs, not only the same seed, because yoking itself is biased when units differ (Church, 1964).

**Sources.**
1. Williams, E. J. (1949). Experimental designs balanced for the estimation of residual effects of treatments. *Australian Journal of Scientific Research, Series A: Physical Sciences, 2*(2), 149–168. https://doi.org/10.1071/CH9490149. **Verified.** Latin squares in which every condition follows every other equally often: n sequences for even n, 2n for odd n. The standard way to counterbalance first-order carryover.
2. Jones, B., & Kenward, M. G. (2014). *Design and analysis of cross-over trials* (3rd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b17537. **Verified.** Mixed models with period, sequence and carryover terms; why unmodelled carryover biases the treatment estimate.
3. Brooks, J. L. (2012). Counterbalancing for serial order carryover effects in experimental condition orders. *Psychological Methods, 17*(4), 600–614. https://doi.org/10.1037/a0029310. **Verified.** Builds counterbalanced orders from Euler circuits, allowing repeated conditions and higher-order carryover. Useful for generating interleaved schedules with balanced skill-to-skill transitions.
4. Church, R. M. (1964). Systematic effect of random error in the yoked control design. *Psychological Bulletin, 62*(2), 122–131. https://doi.org/10.1037/h0042733. **Verified.** Yoked controls produce artifactual group differences favouring the contingent group when individuals differ, so yoked comparisons need care.
5. Bouthillier et al. (2021) (from 05; full entry in A4): "in doubt, it is better to pair."

## A2. Separating interleaving from recency, serial position and spacing

**Why it matters here.** In the completed run, the blocked arm's skills are tested at lags of 0, 128, 256 and 384 updates, the interleaved arm's at 0–3 updates. Blocked accuracy is 94.5% on the last-trained skill and 0–15% on the rest. Interleaving is also confounded with *spacing*: interleaved same-skill batches are spaced every 4 updates, while blocked batches are massed. Human research separates these with designs the ML study can borrow, and continual-learning research has standard metrics that split acquisition from forgetting.

**Recipe.**
- Evaluate every skill after every block (or every 16–32 updates) and build the accuracy matrix R[i, j]. Report average accuracy, backward transfer / forgetting (best earlier accuracy minus final accuracy) and forward transfer (Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2018).
- Test both arms at **matched lag**: add a common post-training interference phase, or model accuracy as a function of updates since last exposure.
- Add a **spaced-but-blocked** arm (same-skill batches separated by filler batches of generic or held-out-skill data) to separate interleaving from spacing (Kang & Pashler, 2012; Birnbaum et al., 2013).
- Add **blocked + final mixed review** to measure how much of the gap is recency.
- Report recency-excluded accuracy (skills not trained last) as a pre-specified secondary outcome.
- Manipulate inter-skill similarity, the main human moderator (Carvalho & Goldstone, 2014; Brunmair & Richter, 2019).

**Sources.**
6. Murdock, B. B., Jr. (1962). The serial position effect of free recall. *Journal of Experimental Psychology, 64*(5), 482–488. https://doi.org/10.1037/h0045106. **Verified.** The classic primacy/recency curve; why end-of-training material looks better unless position and delay are controlled.
7. Kang, S. H. K., & Pashler, H. (2012). Learning painting styles: Spacing is advantageous when it promotes discriminative contrast. *Applied Cognitive Psychology, 26*(1), 97–103. https://doi.org/10.1002/acp.1801. **Verified.** A spaced-but-blocked condition shows the benefit comes from interleaving (contrast), not temporal spacing. A template ablation.
8. Birnbaum, M. S., Kornell, N., Bjork, E. L., & Bjork, R. A. (2013). Why interleaving enhances inductive learning: The roles of discrimination and retrieval. *Memory & Cognition, 41*(3), 392–402. https://doi.org/10.3758/s13421-012-0272-7. **Verified.** Adding gaps helped blocked but not interleaved study, separating discrimination from spacing/retrieval.
9. Carvalho, P. F., & Goldstone, R. L. (2014). Putting category learning in order: Category structure and temporal arrangement affect the benefit of interleaved over blocked study. *Memory & Cognition, 42*(3), 481–495. https://doi.org/10.3758/s13421-013-0371-0. **Verified.** Interleaving wins for similar categories; blocking wins for dissimilar ones.
10. Brunmair, M., & Richter, T. (2019). Similarity matters: A meta-analysis of interleaved learning and its moderators. *Psychological Bulletin, 145*(11), 1029–1052. https://doi.org/10.1037/bul0000209. **Verified.** g = 0.42 overall across 59 studies; larger with between-category similarity; small for math tasks; some studies strongly negative.
11. Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. In *Advances in Neural Information Processing Systems 30*. https://proceedings.neurips.cc/paper_files/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html. **Verified** (pages not listed officially; omit). Defines the R matrix, ACC, BWT and FWT; requires evaluating every task after every stage.
12. Chaudhry, A., Dokania, P. K., Ajanthan, T., & Torr, P. H. S. (2018). Riemannian walk for incremental learning: Understanding forgetting and intransigence. In V. Ferrari, M. Hebert, C. Sminchisescu, & Y. Weiss (Eds.), *Computer Vision – ECCV 2018* (LNCS Vol. 11215, pp. 556–572). Springer. https://doi.org/10.1007/978-3-030-01252-6_33. **Verified.** Forgetting measure (best earlier minus final accuracy) and intransigence (inability to learn new tasks).
13. Jagielski, M., et al. (2023). Measuring forgetting of memorized training examples. In *ICLR*. https://openreview.net/forum?id=7bJizxLKrR (from 05). Examples seen early in training are forgotten by the end, a recency effect over training order.
14. Tirumala, K., Markosyan, A. H., Zettlemoyer, L., & Aghajanyan, A. (2022). Memorization without overfitting: Analyzing the training dynamics of large language models. In *Advances in Neural Information Processing Systems 35* (pp. 38274–38290). https://arxiv.org/abs/2205.10770 (from 05). Forgetting of once-seen batches levels off at a baseline that rises with model size.

## A3. Multiple comparisons across per-skill outcomes

**Why it matters here.** TEST 02 has 4 per-skill contrasts plus macro, worst-skill, immediate, post-interference, retention, calibration and loss outcomes. TEST 03 has 4–6 arms (6–15 pairwise contrasts) × in-distribution/OOD × final/AUC. Without a designated primary outcome, some per-skill difference will "win" by chance.

**Recipe.** Pre-specify **one primary outcome and one primary contrast** per study (TEST 02: final or post-interference macro exact match, interleaved − blocked; TEST 03: sealed-exam OOD accuracy, mastery + replay − best non-contingent replay arm). Apply **Holm** to the confirmatory secondary family (e.g., the 4 per-skill contrasts). Use Benjamini–Hochberg for larger exploratory families, noting its dependence assumptions (per-skill outcomes from one model are correlated). Treat per-skill items as clustered within skill and use item-level mixed models or clustered SEs (Miller, 2024; Baayen et al., 2008; Barr et al., 2013).

**Sources.**
15. Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70. https://www.jstor.org/stable/4615733. **Verified** (via secondary records; JSTOR page blocked a bot check). Step-down Bonferroni; controls familywise error and is uniformly more powerful than Bonferroni.
16. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological), 57*(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x. **Verified.** FDR control; the original proof assumes independent or positively dependent tests.
17. Miller, E. (2024). *Adding error bars to evals: A statistical approach to language model evaluations* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2411.00640 (from 05). Clustered SEs, paired question-level differences, power analysis.
18. Baayen, R. H., Davidson, D. J., & Bates, D. M. (2008). Mixed-effects modeling with crossed random effects for subjects and items. *Journal of Memory and Language, 59*(4), 390–412. https://doi.org/10.1016/j.jml.2007.12.005 (from 05). Map runs to "subjects" and test items to "items".
19. Barr, D. J., Levy, R., Scheepers, C., & Tily, H. J. (2013). Random effects structure for confirmatory hypothesis testing: Keep it maximal. *Journal of Memory and Language, 68*(3), 255–278. https://doi.org/10.1016/j.jml.2012.11.001 (from 05).
20. Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant. *Psychological Science, 22*(11), 1359–1366. https://doi.org/10.1177/0956797611417632. **Verified.** "Researcher degrees of freedom" (stopping rules, outcome choice) inflate false positives.
21. Kerr, N. L. (1998). HARKing: Hypothesizing after the results are known. *Personality and Social Psychology Review, 2*(3), 196–217. https://doi.org/10.1207/s15327957pspr0203_4. **Verified.** Relevant to the report's post hoc explanations of per-skill patterns.

## A4. Seed variance and number of runs

**Why it matters here.** TEST 02 has one run. TEST 03's preliminary tables appear to come from one run per arm, and Figure 3's curves jump by up to 0.4 between neighbouring checkpoints. The experimental unit is a training run, not a test item. Binomial CIs over test items capture only item-sampling variance.

**Reviewer's power numbers** (paired two-sided t-test, α = .05, 80% power; Monte Carlo): n = 5 pairs detects d_z ≈ 1.68; n = 10, d_z ≈ 0.99; n = 12, d_z ≈ 0.88; n = 20, d_z ≈ 0.66. Holm correction raises these. The 34-pp TEST 02 effect will almost surely keep its sign under replication; the per-skill, dose-response and TEST 03 contrasts need ≥ 10–20 pairs. Base the confirmatory n on the pilot's paired SD (Lakens, 2022).

**Sources.**
22. Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. In *Advances in Neural Information Processing Systems 34* (pp. 29304–29320). https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html (from 05). Few-run point estimates are unreliable; use stratified bootstrap CIs, IQM, performance profiles and probability of improvement (`rliable`).
23. Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Sepah, N., Raff, E., Madan, K., Voleti, V., Kahou, S. E., Michalski, V., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems, 3*, 747–769. https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html (from 05). Randomize as many variance sources as possible (init, data order, data sampling); pair when in doubt; use P(A > B) as a decision criterion.
24. Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2018). *How many random seeds? Statistical power analysis in deep reinforcement learning experiments* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1806.08295 (from 05). With N = 5, Type II error was 51% for the observed effect; run pilots to estimate SD.
25. Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2019). *A hitchhiker's guide to statistical comparisons of reinforcement learning algorithms* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1904.06979. **Verified** (arXiv only). False-positive rate and power of t, Welch, bootstrap and permutation tests as a function of seeds and effect size.
26. Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., & Smith, N. (2020). *Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2002.06305 (from 05). Weight initialization and data order contribute comparably to large seed variance. Relevant because TEST 02's manipulation *is* data order.
27. Madaan, L., Singh, A. K., Schaeffer, R., Poulton, A., Koyejo, S., Stenetorp, P., Narang, S., & Hupkes, D. (2024). *Quantifying variance in evaluation benchmarks* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2406.10229 (from 05). Seed variance is substantial; continuous metrics (log-likelihood) have higher signal-to-noise than discrete accuracy. Supports adding per-digit accuracy and answer log-likelihood to exact match.
28. Card, D., Henderson, P., Khandelwal, U., Jia, R., Mahowald, K., & Jurafsky, D. (2020). With little power comes great responsibility. In *Proceedings of EMNLP 2020* (pp. 9263–9274). https://doi.org/10.18653/v1/2020.emnlp-main.745 (from 05). Underpowered significant results exaggerate effects or get the sign wrong.
29. Lakens, D. (2022). Sample size justification. *Collabra: Psychology, 8*(1), Article 33267. https://doi.org/10.1525/collabra.33267. **Verified.** Six justifications for n (a priori power, sensitivity, resource constraints, etc.).

## A5. Evaluating curricula fairly: matched compute, tuned baselines, the right controls

**Why it matters here.** TEST 03 claims a gated curriculum beats a fixed clock and shuffle "under a matched token budget". The curriculum literature shows that apparent curriculum gains often come from (i) weak or untuned baselines, (ii) a growing training pool (pacing) rather than ordering, (iii) reallocating data toward hard examples, or (iv) the low-compute regime. Learning-rate decay is an additional confound for late-hard curricula.

**Recipe (controls a reviewer will expect).**
- Shuffle; **mixture-matched shuffle** (IID, with the gated arm's realized skill proportions); **random-curriculum / pacing-only** control (Wu et al., 2021); **anti-curriculum** (optional); **tuned clock** with the same tuning budget as the gate; **yoked clock**; mixed-difficulty-throughout (Zaremba & Sutskever, 2014; Wang et al., 2025).
- Match and report **tokens** (not steps) per arm and per skill, plus probe compute. Report several cost measures (Dehghani et al., 2022).
- Tune hyperparameters on a split disjoint from the sealed exam and give every arm the same tuning budget (Cawley & Talbot, 2010).
- Use constant LR or WSD, identical across arms, to avoid the decay confound (Luo et al., 2026).
- Report results at more than one budget, because curricula can help at low compute and vanish at standard compute (Wu et al., 2021).

**Sources.**
30. Wu, X., Dyer, E., & Neyshabur, B. (2021). When do curricula work? In *International Conference on Learning Representations*. https://openreview.net/forum?id=tW4QEInpni. **Verified.** Curriculum, anti-curriculum and random-curriculum orderings × 180 pacing functions × 3 seeds. At standard compute no ordering beat properly searched baselines, and gains against a mean baseline were "an artifact of the large search space". Curricula helped with limited compute (1/20 and 1/100 of the standard steps) and noisy labels; at the lowest budget, pacing alone (random ordering with a growing pool) also helped.
31. Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. In *Proceedings of the 26th Annual International Conference on Machine Learning* (pp. 41–48). ACM. https://doi.org/10.1145/1553374.1553380. **Verified** (via dblp).
32. Hacohen, G., & Weinshall, D. (2019). On the power of curriculum learning in training deep networks. In *Proceedings of the 36th International Conference on Machine Learning* (PMLR 97, pp. 2535–2544). https://proceedings.mlr.press/v97/hacohen19a.html. **Verified.** Separates the scoring function from the pacing function; includes anti- and random-curriculum controls.
33. Graves, A., Bellemare, M. G., Menick, J., Munos, R., & Kavukcuoglu, K. (2017). Automated curriculum learning for neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR 70, pp. 1311–1320). https://proceedings.mlr.press/v70/graves17a.html. **Verified.** Bandit task selection driven by learning progress; the canonical adaptive alternative to a fixed threshold.
34. Matiisen, T., Oliver, A., Cohen, T., & Schulman, J. (2020). Teacher–student curriculum learning. *IEEE Transactions on Neural Networks and Learning Systems, 31*(9), 3732–3740. https://doi.org/10.1109/TNNLS.2019.2934906. **Partly verified** (DOI verified; volume/issue/pages from secondary sources). Picks the subtask with the steepest learning curve and revisits subtasks whose performance drops (built-in replay). LSTM decimal addition and Minecraft.
35. Platanios, E. A., Stretcu, O., Neubig, G., Poczos, B., & Mitchell, T. (2019). Competence-based curriculum learning for neural machine translation. In *Proceedings of NAACL-HLT 2019* (Vol. 1, pp. 1162–1172). https://doi.org/10.18653/v1/N19-1119. **Verified.** "Competence" here is a function of training time, not measured mastery. That makes it a fixed-clock schedule in TEST 03's terms, and a natural baseline.
36. Kumar, M. P., Packer, B., & Koller, D. (2010). Self-paced learning for latent variable models. In *Advances in Neural Information Processing Systems 23*. https://papers.nips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html. **Partly verified** (pages unconfirmed). "Easy" defined by the model's current loss.
37. Soviany, P., Ionescu, R. T., Rota, P., & Sebe, N. (2022). Curriculum learning: A survey. *International Journal of Computer Vision, 130*(6), 1526–1565. https://doi.org/10.1007/s11263-022-01611-x. **Verified** (issue number varies across sources).
38. Portelas, R., Colas, C., Weng, L., Hofmann, K., & Oudeyer, P.-Y. (2020). Automatic curriculum learning for deep RL: A short survey. In *Proceedings of IJCAI-20* (pp. 4819–4825). https://doi.org/10.24963/ijcai.2020/671. **Verified.**
39. Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., Mosquera, R., Paranjabe, B., Williams, A., Linzen, T., & Cotterell, R. (2023). Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning* (pp. 1–34). https://doi.org/10.18653/v1/2023.conll-babylm.1. **Verified.** Curriculum was the most popular approach (41.9% of teams) but "did not tend to produce high scores".
40. Diehl Martinez, R., Goriely, Z., McGovern, H., Davis, C., Caines, A., Buttery, P., & Beinborn, L. (2023). CLIMB – Curriculum learning for infant-inspired model building. In *Proceedings of the BabyLM Challenge at CoNLL 2023* (pp. 112–127). https://doi.org/10.18653/v1/2023.conll-babylm.10. **Verified.** Gains came from tuning the baseline; curricula only matched the tuned baseline. A key "tune the baseline" precedent.
41. Campos, D. (2021). *Curriculum learning for language modeling* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2108.02170. **Verified.** No compelling evidence that curricula improve LM pretraining.
42. Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html. **Verified.** Selection bias from tuning can be as large as differences between methods; use held-out tuning and equal tuning budgets.
43. Dehghani, M., Arnab, A., Beyer, L., Vaswani, A., & Tay, Y. (2022). The efficiency misnomer. In *International Conference on Learning Representations*. https://openreview.net/forum?id=iulEMLYh1uR. **Verified.** Parameters, FLOPs, throughput and wall-clock can disagree; report several.
44. Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *International Conference on Learning Representations*. https://openreview.net/forum?id=T5wkZJqzkz (arXiv:2511.18903). **Verified** (via mlanthology/arXiv). With constant LR, a quality curriculum clearly beats shuffling; under standard decay most of the gain vanishes.
45. Zaremba, W., & Sutskever, I. (2014). *Learning to execute* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1410.4615. **Verified.** The naive easy-to-hard curriculum "proved ineffective"; mixing all difficulty levels in throughout improved results.
46. Wang, Z., Nichani, E., Bietti, A., Damian, A., Hsu, D., Lee, J. D., & Wu, D. (2025). Learning compositional functions with transformers from easy-to-hard data. In *Proceedings of the 38th Conference on Learning Theory* (PMLR 291, pp. 5632–5711). https://proceedings.mlr.press/v291/wang25a.html. **Verified.** For k-fold composition, depth-k-only data needs exponentially many samples; a staged curriculum *or* an easy + hard mixture suffices. A theoretical reason the mixed-difficulty control is mandatory.

## A6. Measuring "mastery": OOD and length generalization, held-out compositions, leakage

**Why it matters here.** TEST 03 defines mastery as simultaneous success on several OOD dimensions and uses it to gate training. That raises two problems: how to build valid OOD tests, and how to stop the gate from leaking test-distribution information into training.

**Recipe.**
- Name each OOD axis using the Hupkes et al. (2023) taxonomy (shift type and locus). Use separate families for **gate probes** and the **sealed exam** (e.g., gate on lengths 7–9; exam on lengths 15–30 plus held-out compositions).
- For length generalization, follow train-short/test-long protocols (Anil et al., 2022; Delétang et al., 2023; Zhou et al., 2024; McLeish et al., 2024). Control positional encoding and tokenization, which strongly affect length generalization (Kazemnejad et al., 2023; N. Lee et al., 2024).
- For compositions, use systematic held-out splits (SCAN-style held-out primitives; CFQ-style maximum compound divergence; Lake & Baroni, 2018; Keysers et al., 2020; Hupkes et al., 2020). For chain depth, use LEGO-style tasks (Zhang et al., 2022).
- Make the gate a statistical rule (e.g., lower 95% bound of probe accuracy ≥ τ on n probe items) and report unlock times per seed. Remember that generalization can arrive long after the training set is fit (Power et al., 2022; Nanda et al., 2023).
- Treat the gate's access to OOD probes as potential leakage (Kapoor & Narayanan, 2023). Give baselines the same probe access for tuning or selection.

**Sources.**
47. Hupkes, D., Giulianelli, M., Dankers, V., Artetxe, M., Elazar, Y., Pimentel, T., Christodoulopoulos, C., Lasri, K., Saphra, N., Sinclair, A., Ulmer, D., Schottmann, F., Batsuren, K., Sun, K., Sinha, K., Khalatbari, L., Ryskina, M., Frieske, R., Cotterell, R., & Jin, Z. (2023). A taxonomy and review of generalization research in NLP. *Nature Machine Intelligence, 5*(10), 1161–1174. https://doi.org/10.1038/s42256-023-00729-y. **Verified.**
48. Hupkes, D., Dankers, V., Mul, M., & Bruni, E. (2020). Compositionality decomposed: How do neural networks generalise? *Journal of Artificial Intelligence Research, 67*, 757–795. https://doi.org/10.1613/jair.1.11674. **Verified.** Systematicity, productivity, substitutivity, localism and overgeneralisation tests.
49. Anil, C., Wu, Y., Andreassen, A., Lewkowycz, A., Misra, V., Ramasesh, V., Slone, A., Gur-Ari, G., Dyer, E., & Neyshabur, B. (2022). Exploring length generalization in large language models. In *Advances in Neural Information Processing Systems 35*. https://openreview.net/forum?id=zSkYVeX7bC4. **Partly verified** (pages from citing papers only).
50. Delétang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wenliang, L. K., Catt, E., Cundy, C., Hutter, M., Legg, S., Veness, J., & Ortega, P. A. (2023). Neural networks and the Chomsky hierarchy. In *International Conference on Learning Representations*. https://openreview.net/forum?id=WbxHAzkeQcn. **Verified.** Some models fit training perfectly but never generalize in length: in-distribution accuracy is not mastery.
51. Zhou, H., Bradley, A., Littwin, E., Razin, N., Saremi, O., Susskind, J., Bengio, S., & Nakkiran, P. (2024). What algorithms can transformers learn? A study in length generalization. In *International Conference on Learning Representations*. https://openreview.net/forum?id=AssIuHnmHX. **Verified.**
52. Lee, N., Sreenivasan, K., Lee, J. D., Lee, K., & Papailiopoulos, D. (2024). Teaching arithmetic to small transformers. In *International Conference on Learning Representations*. https://openreview.net/forum?id=dsUB4bst9S. **Verified.** Data format (e.g., reversed output) causes sharp phase transitions, a confound that can swamp ordering effects.
53. McLeish, S., Bansal, A., Stein, A., Jain, N., Kirchenbauer, J., Bartoldson, B. R., Kailkhura, B., Bhatele, A., Geiping, J., Schwarzschild, A., & Goldstein, T. (2024). Transformers can do arithmetic with the right embeddings. In *Advances in Neural Information Processing Systems 37*. https://arxiv.org/abs/2405.17399. **Verified.**
54. Kazemnejad, A., Padhi, I., Natesan Ramamurthy, K., Das, P., & Reddy, S. (2023). The impact of positional encoding on length generalization in transformers. In *Advances in Neural Information Processing Systems 36*. https://doi.org/10.52202/075280-1082. **Verified.** Decoder-only transformers with no positional encoding length-generalize better than APE, ALiBi or rotary encodings on reasoning tasks, so positional encoding is a confound to control.
55. Zhang-Li, D., Lin, N., Yu, J., Zhang, Z., Yao, Z., Zhang, X., Hou, L., Zhang, J., & Li, J. (2024). *Reverse that number! Decoding order matters in arithmetic learning* [Preprint]. arXiv. https://arxiv.org/abs/2403.05845. **Verified.** This is the report's "output the digits in reverse order" link.
56. Lake, B., & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In *Proceedings of the 35th International Conference on Machine Learning* (PMLR 80, pp. 2873–2882). https://proceedings.mlr.press/v80/lake18a.html. **Verified.**
57. Keysers, D., Schärli, N., Scales, N., Buisman, H., Furrer, D., Kashubin, S., Momchev, N., Sinopalnikov, D., Stafiniak, L., Tihon, T., Tsarkov, D., Wang, X., van Zee, M., & Bousquet, O. (2020). Measuring compositional generalization: A comprehensive method on realistic data. In *International Conference on Learning Representations*. https://openreview.net/forum?id=SygcCnNKwr. **Verified.**
58. Zhang, Y., Backurs, A., Bubeck, S., Eldan, R., Gunasekar, S., & Wagner, T. (2022). *Unveiling transformers with LEGO: A synthetic reasoning task* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2206.04301. **Verified.** The reference-chain task the report's variable-chain task resembles; cite it by name.
59. Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2201.02177. **Verified.**
60. Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. In *International Conference on Learning Representations*. https://openreview.net/forum?id=9XFSbDPmdW. **Verified.**
61. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), Article 100804. https://doi.org/10.1016/j.patter.2023.100804. **Verified.** An eight-type leakage taxonomy; use its "model info sheet" idea for the gate/probe/exam separation.
62. Schaeffer, R., Miranda, B., & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? In *Advances in Neural Information Processing Systems 36*. https://openreview.net/forum?id=JRdN9GcI52 (from 05). Exact-match thresholds create apparent jumps; also report continuous metrics.

## A7. Sample efficiency and area-under-curve metrics

**Why it matters here.** TEST 03's headline "efficiency (AUC)" appears (reviewer's reconstruction) to be the normalized trapezoidal area under the dependent-skill OOD curve over ~11 checkpoints. It mechanically favours arms that begin the dependent skill early. TEST 02 reports only final accuracy.

**Recipe.** Pre-define a small set of efficiency metrics and report all of them with CIs: (i) AUC of **all-skill average** accuracy over a fixed token grid (state the normalization and interpolation); (ii) tokens-to-threshold, treating runs that never reach it as censored; (iii) accuracy at 2–3 fixed budgets; (iv) the continual-learning trio of average accuracy, forgetting/BWT and Learning Curve Area (Chaudhry et al., 2019). Use the same checkpoint grid for every arm.

**Sources.**
63. Chaudhry, A., Ranzato, M., Rohrbach, M., & Elhoseiny, M. (2019). Efficient lifelong learning with A-GEM. In *International Conference on Learning Representations*. https://openreview.net/forum?id=Hkf2_sC5FX. **Verified** (via arXiv:1812.00420). Defines Learning Curve Area (LCA_β, the average accuracy over the first β+1 mini-batches) to capture learning speed; reports 5 runs with CIs.
64. Díaz-Rodríguez, N., Lomonaco, V., Filliat, D., & Maltoni, D. (2018). *Don't forget, there is more than forgetting: New metrics for continual learning* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1810.13166. **Verified** (arXiv only; workshop venue UNVERIFIED). Accuracy over time, BWT/FWT, memory and compute efficiency.
65. Agarwal et al. (2021) (A4): performance profiles and aggregate metrics over runs and tasks, applicable to learning curves.

## A8. Preregistration in ML

**Why it matters here.** The report's mission statement says the project is "preregistered". Reviewers will look for a timestamped protocol. TEST 03 already contains the right ingredients (locked hyperparameters, seed list, zero retuning). They need to be registered *before* the confirmatory runs, including the primary outcome, primary contrast, analysis, exclusion rules and stopping rule.

**Recipe.** Register on OSF (or as a dated, tagged commit plus a hash posted publicly). Use the van Miltenburg et al. (2021) NLP question set. Label every analysis in the paper as confirmatory or exploratory (Nosek et al., 2018). Consider a registered-report-style submission (see Part C).

**Sources.**
66. Bertinetto, L., Henriques, J. F., Albanie, S., Paganini, M., & Varol, G. (Eds.). (2021). *NeurIPS 2020 Workshop on Pre-registration in Machine Learning* (Proceedings of Machine Learning Research, Vol. 148). PMLR. https://proceedings.mlr.press/v148/. **Verified.** Second edition: Albanie, S., Henriques, J. F., Bertinetto, L., Hernández-García, A., Doughty, H., & Varol, G. (Eds.). (2022). PMLR Vol. 181. https://proceedings.mlr.press/v181/. **Verified.** Papers were accepted on protocol before results existed.
67. van Miltenburg, E., van der Lee, C., & Krahmer, E. (2021). Preregistering NLP research. In *Proceedings of NAACL-HLT 2021* (pp. 613–623). https://doi.org/10.18653/v1/2021.naacl-main.51 (from 05).
68. Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences, 115*(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114 (from 05).
69. Pineau, J., et al. (2021). Improving reproducibility in machine learning research (A report from the NeurIPS 2019 Reproducibility Program). *Journal of Machine Learning Research, 22*(164), 1–20. https://jmlr.org/papers/v22/20-303.html (from 05).
70. NeurIPS. (2026). *NeurIPS paper checklist guidelines*. https://neurips.cc/public/guides/PaperChecklist (from 05). Q7 (error bars: sources of variability, method) and Q8 (compute) apply directly.

## A9. Positioning literature (novelty threats and human/machine comparisons)

**Interleaving vs. blocking: established in networks and LMs.**
71. McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), *Psychology of Learning and Motivation* (Vol. 24, pp. 109–165). Academic Press. https://doi.org/10.1016/S0079-7421(08)60536-8. **Verified.**
72. Ratcliff, R. (1990). Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions. *Psychological Review, 97*(2), 285–308. https://doi.org/10.1037/0033-295X.97.2.285. **Verified.**
73. McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review, 102*(3), 419–457. https://doi.org/10.1037/0033-295X.102.3.419. **Verified.** Interleaving old with new items prevents catastrophic interference: the 30-year-old version of TEST 02's headline.
74. French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2. **Verified.**
75. Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318. **Verified.** (The report links a Stanford proxy URL; use this DOI.)
76. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS, 114*(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114. **Verified.**
77. Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html. **Verified.**
78. Flesch, T., Balaguer, J., Dekker, R., Nili, H., & Summerfield, C. (2018). Comparing continual task learning in minds and machines. *PNAS, 115*(44), E10313–E10322. https://doi.org/10.1073/pnas.1800755115. **Verified.** Blocking helped humans and hurt networks on the same task.
79. Dekker, R. B., Otto, F., & Summerfield, C. (2022). Curriculum learning for human compositional generalization. *PNAS, 119*(41), e2205582119. https://doi.org/10.1073/pnas.2205582119. **Verified.**
80. Russin, J., Pavlick, E., & Frank, M. J. (2025). Parallel trade-offs in human cognition and neural networks: The dynamic interplay between in-context and in-weight learning. *PNAS, 122*(35), e2510270122. https://doi.org/10.1073/pnas.2510270122. **Verified.** In transformers, in-weight learning shows an interleaving advantage (blocking → forgetting) while in-context learning shows the human blocking advantage.
81. Mayo, D., Scott, T. R., Ren, M., Elsayed, G., Hermann, K., Jones, M., & Mozer, M. C. (2023). Multitask learning via interleaving: A neural network investigation. In *Proceedings of the 45th Annual Conference of the Cognitive Science Society*. https://escholarship.org/uc/item/3tb956hb. **Partly verified** (volume/pages not found).
82. Lee, B. W., Cho, H., & Yoo, K. M. (2024). Instruction tuning with human curriculum. In *Findings of the Association for Computational Linguistics: NAACL 2024* (pp. 1281–1309). https://doi.org/10.18653/v1/2024.findings-naacl.82. **Verified.** **The strongest novelty threat for TEST 02:** blocked vs. interleaved subject order in LLaMA-2 13B fine-tuning, with interleaving better on all 9 benchmarks, explicitly framed with human interleaving and catastrophic forgetting.
83. Yang, Y., Bean, A. M., McCraith, R., & Mahdi, A. (2024). *Evaluating fine-tuning efficiency of human-inspired learning strategies in medical question answering* [Preprint; NeurIPS 2024 FITML workshop]. arXiv. https://arxiv.org/abs/2408.07888. **Verified.** Interleaving was best on average, but gains were small (~1%) and model- and dataset-dependent.
84. Dong, G., Yuan, H., Lu, K., Li, C., Xue, M., Liu, D., Wang, W., Yuan, Z., Zhou, C., & Zhou, J. (2024). How abilities in large language models are affected by supervised fine-tuning data composition. In *Proceedings of the 62nd Annual Meeting of the ACL (Vol. 1)* (pp. 177–198). https://doi.org/10.18653/v1/2024.acl-long.12. **Verified.** Sequential math → code → general training forgets; mixing avoids it.
85. Lu, X., Li, X., Cheng, Q., Ding, K., Huang, X., & Qiu, X. (2024). Scaling laws for fact memorization of large language models. In *Findings of EMNLP 2024* (pp. 11263–11282). https://doi.org/10.18653/v1/2024.findings-emnlp.658 (from 06). Blocked sequential fact training in a 30M model collapses the first fact type to ~0.
86. Scialom, T., Chakrabarty, T., & Muresan, S. (2022). Fine-tuned language models are continual learners. In *Proceedings of EMNLP 2022* (pp. 6107–6122). https://doi.org/10.18653/v1/2022.emnlp-main.410. **Verified.** ~1% rehearsal nearly removes forgetting in T0.
87. Ramasesh, V. V., Dyer, E., & Raghu, M. (2021). Anatomy of catastrophic forgetting: Hidden representations and task semantics. In *International Conference on Learning Representations*. https://openreview.net/forum?id=LhY8QdUGSuw. **Verified.** Forgetting is greatest at intermediate task similarity.
88. Ramasesh, V. V., Lewkowycz, A., & Dyer, E. (2022). Effect of scale on catastrophic forgetting in neural networks. In *International Conference on Learning Representations*. https://openreview.net/forum?id=GhVS8_yPeEa. **Verified.** Larger *pretrained* models forget much less; from-scratch models benefit much less.
89. Mirzadeh, S. I., Farajtabar, M., Pascanu, R., & Ghasemzadeh, H. (2020). Understanding the role of training regimes in continual learning. In *Advances in Neural Information Processing Systems 33*. https://proceedings.neurips.cc/paper/2020/hash/518a38cc9a0173d0b2dc088166981cf8-Abstract.html. **Verified** (pages uncertain). LR, decay, batch size and dropout strongly change forgetting, so they must be held fixed and reported.
90. Xiong, Y., & Wu, S. (2025). Do large language models learn like humans: Interleaved and spaced practice in morphological learning. *Acta Psychologica, 260*, 105518. https://doi.org/10.1016/j.actpsy.2025.105518 (from 03). In-context, not in-weight.

**Human interleaving (the construct the report invokes).**
91. Rohrer, D., & Taylor, K. (2007). The shuffling of mathematics problems improves learning. *Instructional Science, 35*(6), 481–498. https://doi.org/10.1007/s11251-007-9015-8. **Verified.**
92. Kornell, N., & Bjork, R. A. (2008). Learning concepts and categories: Is spacing the "enemy of induction"? *Psychological Science, 19*(6), 585–592. https://doi.org/10.1111/j.1467-9280.2008.02127.x. **Verified.**

**Mastery and curricula (TEST 03 positioning).**
93. Bloom, B. S. (1968). Learning for mastery. *Evaluation Comment, 1*(2), 1–12. https://eric.ed.gov/?id=ED053419. **Verified** (ERIC reprint).
94. Kulik, C.-L. C., Kulik, J. A., & Bangert-Drowns, R. L. (1990). Effectiveness of mastery learning programs: A meta-analysis. *Review of Educational Research, 60*(2), 265–299. https://doi.org/10.3102/00346543060002265. **Verified.** Gains came at the cost of more instructional time, which a matched-token design removes.
95. Elman, J. L. (1993). Learning and development in neural networks: The importance of starting small. *Cognition, 48*(1), 71–99. https://doi.org/10.1016/0010-0277(93)90058-4. **Verified.**
96. Mindermann, S., Brauner, J. M., Razzak, M. T., Sharma, M., Kirsch, A., Xu, W., Höltgen, B., Gomez, A. N., Morisot, A., Farquhar, S., & Gal, Y. (2022). Prioritized training on points that are learnable, worth learning, and not yet learnt. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162, pp. 15630–15649). https://proceedings.mlr.press/v162/mindermann22a.html. **Verified.**
97. Chen, M. F., Roberts, N., Bhatia, K., Wang, J., Zhang, C., Sala, F., & Ré, C. (2023). Skill-it! A data-driven skills framework for understanding and training language models. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2307.14430. **Verified** (pages 36000–36040 per ACM DL).
98. Li, M., Chen, P., Zhang, Z., Yang, T., Zhang, X., Li, H., Cao, T., Zeng, M., Wu, Z., Jiang, M., Li, H., Li, L., & Yin, B. (2025). *Mitigating lost in multi-turn conversation via curriculum RL with verifiable accuracy and abstention rewards* [Preprint]. arXiv. https://arxiv.org/abs/2510.18731. **Verified** (arXiv comment: "ACL2026, camera-ready"; Anthology entry not checked). This is the report's "RLAAR".
99. Li, Y., Lu, T., Li, Y., Chen, Y., Huang, W.-C., Jiang, W., Wang, H., Zheng, H.-T., & Yu, P. S. (2025). *Teaching according to talents! Instruction tuning LLMs with competence-aware curriculum learning* [Preprint]. arXiv. https://arxiv.org/abs/2509.13790. **Verified** (arXiv comment: "EMNLP 2025 Findings"). This is the report's "CAMPUS".
100. Kohli, H., Parthasarathy, S., Sun, H., & Yao, Y. (2026). *Loop, think, & generalize: Implicit reasoning in recurrent-depth transformers* [Preprint]. arXiv. https://arxiv.org/abs/2604.07822. **Verified** (arXiv abstract page: "Accepted at COLM 2026"; curriculum detail checked in the full text, §6.1, on 2026-10-04).
101. Yao, Y., Du, Y., Zhu, D., Hahn, M., & Koller, A. (2025). *Language models can learn implicit multi-hop reasoning, but only if they have lots of training data* [Preprint]. arXiv. https://arxiv.org/abs/2505.17923. **Verified** on arXiv (EMNLP 2025 per the arXiv comment; Anthology not checked).

---

# Part B — Mock peer review ("Reviewer 2"), one report per section

Both reports are written as a tough but fair reviewer at a selective ML venue (ICLR/NeurIPS/TMLR) or a cognitive-science venue (CogSci) would write them. A reviewer is entitled to read linked code, and several points below come from the public branch. Each issue has a concrete fix. Issues are ranked MAJOR (would block acceptance) or MINOR (should be fixed, would not block alone).

---

## B1. Review of TEST 02, "Blocked vs. Interleaved Subject Training"

**Reviewed material.** `P4_final_report.md` (TEST 02), plus the public branch `p4/blocked-vs-interleaved` of github.com/edu-llm/OLMo-core, read on the web only (README, `generate_data.py`, `train_compare.py`).

**Recommendation as written: Reject.** The completed experiment is a clean, honestly built pipeline, and the effect it found is large. But the paper text describes a different experiment from the one that was run, it reports pilot numbers that match neither, and the one completed run (n = 1, one order, no intermediate evaluation) cannot separate an "interleaving" effect from catastrophic forgetting plus recency. With the fixes below, a narrowed version is a plausible TMLR or workshop paper.

**Confidence:** 4/5.

### Summary of what was actually done (reviewer's reconstruction from code)

- Randomly initialized OLMo-architecture decoder, 12 layers, width 768, 162,164,736 parameters. Per `RESULTS.md`: SwiGLU intermediate size 2048 and untied input/output embeddings. By the reviewer's arithmetic, about 85M of these are transformer-block weights (12 × [4·768² + 3·768·2048] ≈ 85M) and about 77M are the two embedding matrices of a ~50k-token vocabulary, of which the task uses roughly 15 tokens.
- Four synthetic skills on the same 6-digit inputs, distinguished **only by the bracket type**: `< >` rotate-left (A), `[ ]` rotate-right (B), `{ }` reverse (C), `( )` swap-adjacent-pairs (D). The prompt never names the operation. A skeleton is kept only if all four rules give different outputs. Train and test skeletons are disjoint; every training skeleton is reused for all four skills.
- 4,096 train examples per skill, batch 32, so 128 single-skill batches per skill and 512 optimizer updates per arm (one pass). AdamW, constant LR 1e-4, no warmup, no decay, clipping at 1.0, bf16. Loss on answer tokens plus EOS.
- Blocked = AAAA…BBBB…CCCC…DDDD (128 updates each). Interleaved = deterministic round-robin ABCDABCD… at the batch level. Every batch contains one skill in both arms, and within-skill example order is identical across arms. This is good design.
- The generator already builds four cyclic orders (ABCD, BCDA, CDAB, DABC), a cyclic Latin square, and the trainer accepts `--order-indices all`. **Only order 0 was run, with one seed.**
- Evaluation: greedy decoding, exact match on 256 held-out items per skill, **only at the end of training**. No per-block evaluation, no interference phase.
- Result: blocked 27.44% (281/1024) vs. interleaved 61.62% (631/1024). Per skill (A/B/C/D): blocked 0 / 15.23 / 0 / 94.53; interleaved 81.64 / 95.31 / 27.34 / 42.19.

### Strengths

1. The manipulation is clean where it matters: identical initialization, identical examples, identical within-skill order, skill-pure batches in both arms, and identical optimizer settings. Only the order of batches differs.
2. Constant learning rate removes the "late data meets a decayed LR" confound.
3. Hashes, manifests and deterministic flags make the run auditable. The counterbalanced orders are already generated.
4. The text itself recognizes that the contrast is a "schedule-package effect" (order, transitions, lag and recency together), not a pure interleaving mechanism. That is the right framing.
5. The effect is large: +34.2 pp, item-sampling 95% CI ≈ [30.1, 38.2] pp (reviewer's Wald interval treating the 1,024 items as independent; this reflects test-item sampling only, not training variance).

### MAJOR issues

#### I-M1. The paper describes an experiment that was not run, and reports pilot numbers that match no code we can find

**Problem.** The text describes arithmetic, logic, geometry and statistics problems in a four-choice format; a ~195M, 24-layer model starting from a pretrained checkpoint with a generic-data burn-in; 1,280 updates in four sessions of 80-update blocks; 160 generic interference updates; and 12 paired comparisons (4 orders × 3 data seeds). The completed run is a from-scratch 162M, 12-layer model on four digit-manipulation tasks, with 512 updates, 128-update blocks, no interference phase, and one pair. The text then cites a "toy MLP pilot" (52.8% blocked vs. 79.2% interleaved; "independent subject models showed exactly zero order effect") that matches neither the OLMo run (27.44% vs. 61.62%) nor any code on the three known branches. The paragraph also says the planned study is "not evidence about from-scratch pretraining", but the only completed OLMo experiment *is* from scratch. A reviewer who opens the repository will see the mismatch within minutes, and it will be read as either carelessness or selective reporting.

**Fix.** (a) Report the experiment that was run, with its real numbers, as the main result, and describe the 195M natural-language design as future work or a preregistered extension, clearly labelled. (b) Either release the MLP pilot code and logs or remove the 52.8/79.2 numbers. (c) Drop "independent subject models showed exactly zero order effect" as evidence: separate models trained on one subject each cannot interfere, so a zero effect is true by construction. It is a pipeline sanity check, not a result. (d) Add a "What was run" table (model, init, data, steps, orders, seeds, evaluation points) and make every number in the text traceable to a results file.

#### I-M2. Interleaving is confounded with retention interval (recency), and with acquisition vs. forgetting

**Problem.** At the final test, each skill's *lag since last training* is 0–3 updates in the interleaved arm, but 0, 128, 256 and 384 updates for D, C, B and A in the blocked arm. The blocked arm's per-skill pattern (A 0%, B 15%, C 0%, D 95%) is exactly what recency predicts. Excluding the last-trained skill, blocked mean accuracy is 5.1% vs. 68.1% for interleaved; on the last-trained skill blocked *beats* interleaved by 52 pp (94.5 vs. 42.2). Because there is no evaluation at the end of each block, the reader cannot tell whether A and C were **learned and then forgotten** (catastrophic interference) or **never learned** in their 128 updates from a random initialization, when the model still had to learn the output format. The fact that D reaches 95% in 128 updates *after* 384 updates on the other skills, while A reaches 0% as the very first block, suggests that format learning and positive transfer also differ by serial position. So the 34-pp gap mixes at least three things: forgetting, recency, and an acquisition-order (primacy/format) effect.

**Fix.**
- Evaluate every skill at the end of every block (and every 16–32 updates) in both arms. Report a learning-and-forgetting matrix R[i, j] = accuracy on skill j after block i, and the standard continual-learning summaries derived from it: average accuracy, backward transfer / forgetting, and forward transfer (Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2018).
- Add a **matched-lag** comparison so that both arms are tested at the same retention interval for each skill. Two options: (i) a common interference phase after training (generic or held-out-skill data), as the proposal already plans; or (ii) analyze accuracy as a function of updates-since-last-exposure in both arms.
- Add a **blocked + final mixed review** arm (e.g., the last 10–20% of updates interleaved) at the same total budget. If it closes most of the gap, the effect is largely recency/forgetting.
- Report the "recency-excluded" macro accuracy (skills not trained in the final block) as a pre-specified secondary outcome.

#### I-M3. n = 1 run, one order: no inferential claim about schedules is possible yet

**Problem.** The experimental unit, as the text itself correctly says, is a paired run, not a test item. There is one pair. Binomial CIs over the 1,024 items describe test-item sampling only, not variability across seeds, data draws or orders, which is the variance that matters for "interleaving helps" (Bouthillier et al., 2021; Dodge et al., 2020). Seed effects in small transformers on algorithmic tasks are often large (grokking-like transitions vary with seed), and the per-skill pattern of the interleaved arm (C 27%, D 42%) suggests the model is still mid-learning at 512 updates, which is where seed variance is highest.

**Fix.** Run all four existing orders × at least 5 seeds (initialization and data/skeleton seed both varied) = 20 pairs per condition. At this scale this is cheap (each arm is 512 updates of a 162M model). With 20 pairs, a paired t-test has 80% power for a standardized paired effect d_z ≈ 0.66 at α = .05 (reviewer's Monte Carlo); with the proposal's 12 pairs, d_z ≈ 0.88; with 5 pairs, d_z ≈ 1.68. The overall effect is so large that its sign will almost certainly survive; the point of replication is the per-skill pattern, block-length dose-response and any interaction with order. Report every pair, the mean paired difference with a t-interval, and a probability-of-improvement estimate (Agarwal et al., 2021; Bouthillier et al., 2021). Fit an item-level logistic mixed model, e.g. `correct ~ arm * skill + (1 | order) + (1 + arm | item) + (1 | run)`, so both runs and items are treated as random (Baayen et al., 2008; Barr et al., 2013).

#### I-M4. Counterbalancing is designed but not executed, and the existing design does not balance carryover

**Problem.** The four cyclic rotations balance serial position (each skill appears once in each position), which is necessary because D-last is the most important confound. But a cyclic Latin square of size 4 is not carryover-balanced: A is always followed by B, B by C, and so on. Because the skills are pairwise related (rotate-left vs. rotate-right are inverses; reverse and rotations share structure), which skill immediately precedes another can change how much it is overwritten.

**Fix.** Use a **Williams design** (a Latin square in which every skill immediately precedes every other skill equally often; for 4 treatments, one 4 × 4 square suffices; Williams, 1949; Jones & Kenward, 2014; Brooks, 2012), crossed with seeds. Treat order as a blocking factor in the analysis. Report per-position and per-predecessor effects as exploratory.

#### I-M5. Construct validity: are four bracket-cued digit permutations "subjects"? Is this the human interleaving effect?

**Problem.** The text promises arithmetic, logic, geometry and statistics. The run uses four permutations of the *same* six digits, distinguished by a single bracket token, with conflicting outputs for the same input. This is close to the worst case for interference: identical input distribution, a minimal context cue, and incompatible mappings. In humans, the interleaving benefit is strongest for *similar, confusable* categories and can reverse for dissimilar material (Carvalho & Goldstone, 2014; Brunmair & Richter, 2019), and it is usually explained by discriminative contrast and retrieval, measured on *induction or problem-type identification* at a delay (Kornell & Bjork, 2008; Kang & Pashler, 2012; Birnbaum et al., 2013; Rohrer & Taylor, 2007). What the run shows is that sequential training of a network overwrites earlier mappings, which is catastrophic interference (McCloskey & Cohen, 1989; Ratcliff, 1990), and that interleaving prevents it (McClelland et al., 1995). Those are different constructs. Flesch et al. (2018) found that blocking *helped* humans but hurt networks on the same task, and Dekker et al. (2022) found a blocked-training advantage for human compositional generalization. Calling a network result "the interleaving effect" therefore invites the objection that the human and machine phenomena are not the same thing. Similarity matters on the network side too: Ramasesh et al. (2021) found forgetting is greatest for tasks of intermediate similarity.

**Fix.** (a) Rename the claim to what was measured: "temporal task order and catastrophic interference among cue-defined skills in a small transformer". (b) Manipulate **inter-skill similarity** directly (e.g., near pairs such as rotate-left vs. rotate-right vs. far pairs such as digit-sum vs. sort), which turns the paper into a test of the main human moderator (Brunmair & Richter, 2019). (c) Add a test that mirrors the human construct: discrimination of problem type when the cue is removed or ambiguous, or transfer to unseen compositions (e.g., rotate-then-reverse), measured before forgetting differences arise. (d) Keep the "subjects" framing only for the planned natural-language study, and say clearly that the pilot is a minimal-cue interference task.

#### I-M6. Novelty relative to prior work is not argued

**Problem.** "Interleaved (mixed) training forgets less than blocked (sequential) training" is a textbook result in connectionist and continual-learning research (McCloskey & Cohen, 1989; Ratcliff, 1990; McClelland et al., 1995; French, 1999; Robins, 1995; Kirkpatrick et al., 2017), shown in direct human-vs-network comparisons (Flesch et al., 2018), in small LMs trained on facts (Lu et al., 2024: blocked fact types collapse to ~0 without replay), and in LLM fine-tuning data-composition studies that compare sequential and mixed training of math, code and general data (Dong et al., 2024). Three papers are especially close, because they already use the *human-interleaving framing* for language models:
- **Lee, Cho & Yoo (2024, NAACL Findings)** fine-tuned LLaMA-2 13B with blocked vs. interleaved subject order (with a Bloom's-taxonomy curriculum), found interleaving better on all 9 benchmarks, and explicitly tied blocking's deficit to catastrophic forgetting and to Rohrer & Taylor-style interleaving.
- **Russin, Pavlick & Frank (2025, PNAS)** showed that in transformers, in-weight (gradient) learning shows the interleaving advantage because blocking causes forgetting, while in-context learning shows the human *blocking* advantage.
- **Mayo et al. (2023, CogSci)** studied interleaving in neural networks as a model of human multitask learning (forgetting with savings, switch costs).
Yang et al. (2024, NeurIPS workshop) also compared blocked, interleaved and curriculum orders for LLM fine-tuning and found small, inconsistent gains. The report cites only Brunmair & Richter and Rolnick et al. A reviewer will ask: "What does this paper show that we did not already know?" The verification agent found no paper that trains a small transformer from scratch on synthetic skills with blocked vs. interleaved order *and* a learning-science framing, but that gap is narrow.

**Fix.** Make the contribution something that is not already known. Candidate contributions that the current pipeline can support cheaply:
- A **block-length dose-response** curve (128, 32, 8, 1 updates per block, matched budget). The text already proposes 80/20/5/1. Where does the effect switch on, and does it follow lag since last exposure?
- A **similarity moderator** analysis (I-M5b), mapping directly onto the main human meta-analytic moderator.
- **Scale and pretraining**: whether the effect shrinks with model size or a pretrained start. Ramasesh et al. (2022) report that larger pretrained models forget less, which is a testable prediction.
- **Acquisition vs. retention decomposition** (I-M2), which human studies often cannot measure cleanly.
Position these against the continual-learning literature explicitly, and cite the network–human dissociation (Flesch et al., 2018; Xiong & Wu, 2025) as motivation.

#### I-M7. Toy scale and regime: the generality claims overreach the evidence

**Problem.** One model size, from scratch, one LR (1e-4, constant, no warmup), one pass over 16,384 examples, 512 updates, and a task whose vocabulary is ~15 tokens. The interleaved arm is itself far from ceiling on C and D (27%, 42%), so the comparison is taken mid-learning, and the size and even sign of per-skill differences could change with more training. The ~77M embedding parameters (untied input and output matrices) mostly do not contribute to this task, so "162M" overstates the effective model size.

**Fix.** Report learning curves to convergence (or to a pre-specified plateau criterion) for both arms, and report the final-budget result and the full curve. Sweep LR at least over a small grid for the IID/interleaved baseline and use the same LR for both arms (state that it was chosen on the interleaved arm, or on a third IID arm, to avoid tuning to favour either). Report non-embedding parameter count. Scope the title and conclusions to "small from-scratch transformers on synthetic cue-defined skills" unless the 195M natural-language study is completed.

### MINOR issues

- **I-m1. Primary outcome.** The proposal names post-interference macro accuracy as primary, but the run has no interference phase. Pre-specify one primary outcome for the run that exists (e.g., final macro exact match) and label per-skill accuracies as secondary. Per-skill contrasts (4 tests) should be Holm-corrected (Holm, 1979) or reported as descriptive.
- **I-m2. Metric.** Exact match on a 6-digit string is all-or-nothing. Add per-digit accuracy and answer log-likelihood, which have higher signal-to-noise (Madaan et al., 2024; Schaeffer et al., 2023), and report base-model accuracy (0/1,024 per the README) in the paper.
- **I-m2b. Separate interleaving from spacing.** In the interleaved arm, each skill's updates are also spaced (every 4th update), whereas blocked updates are massed. Human work separates the two with a "spaced but blocked" arm (Kang & Pashler, 2012; Birnbaum et al., 2013). The analogue here is blocked-by-skill with generic-data or held-out-skill fillers between same-skill batches, at matched lag.
- **I-m3. Interleaved arm is also ordered.** Round-robin ABCD means C and D are always the last two batches. The interleaved arm's lowest scores are C and D, so recency is not the whole story there; say so, and randomize the interleaved order within each cycle in the replication.
- **I-m4. "Same examples."** Every training skeleton appears with all four brackets. State this, because it makes the conflict between skills maximal (same input, different outputs) and it matters for the similarity argument.
- **I-m5. Cost/compute.** `RESULTS.md` says the completed run took 1 min 11 s on one NVIDIA L40S and calls itself a "smoke test". Put this in the paper (NeurIPS checklist Q8), next to the projected FLOPs for the proposed study. It also shows that the replication asked for in I-M3 costs well under a GPU-hour. Note too that `RESULTS.md` already lists the right caveats (single deterministic run, recency not separated, no pretrained or natural-language data), but the final report leaves them out.
- **I-m6. Proposal internal consistency.** The design paragraph says the blocked condition uses 80-update blocks in four sessions; "Session orders are counterbalanced so every subject appears equally often in each serial position" needs the actual manifests. "Neutral-checkpoint creation is not included" in the cost should be estimated, since it may dominate.
- **I-m7. Citations.** Add the continual-learning and human-interleaving references listed in Part A (A2, A9). Cite Brunmair & Richter's moderator results (similarity), not only the average g.

### Questions for the authors

1. What was the accuracy on each skill at the end of its own block in the blocked arm? Was A ever above 0%?
2. Where does the 52.8% / 79.2% MLP pilot come from, and can its code be released?
3. Why was only order 0 run when the code supports all four?
4. Was the learning rate chosen before seeing the blocked/interleaved results?
5. How do results change with 2× and 4× the training budget?

### Minimum experiment that would satisfy this reviewer (cheap, uses existing code)

1. Run the four existing orders (preferably replaced by a 4 × 4 Williams square) × 5 seeds, both arms: 40 runs of 512 updates.
2. Evaluate every skill every 32 updates; report the R[i, j] matrix, forgetting/BWT and recency-excluded accuracy.
3. Add a blocked + final-mixed-review arm and a matched-lag evaluation (a short common interference phase).
4. Add a block-length gradient (128, 32, 8, 1) at matched budget.
5. Pre-register the primary outcome (final macro exact match), the primary contrast (interleaved − blocked, paired by seed × order), the analysis (paired t with all pairs shown + item-level GLMM) and the Holm-corrected per-skill secondary contrasts, before running.

---

## B2. Review of TEST 03, "On Mastery-Gated Curriculum Training"

**Reviewed material.** `P4_final_report.md` (TEST 03), including Preliminary Experiment I (Figure 2 table) and II (table and Figure 3). No code for either preliminary experiment exists on the three known branches.

**Recommendation as written: Reject (not reviewable as an empirical paper).** The literature review is thoughtful, the hypothesis is clear, and the "gate needs replay" idea is well motivated by Zaremba & Sutskever (2014) and continual learning. But the section is mainly a proposal. Its two preliminary experiments have no code, no seeds, no specification of the gate, probes, thresholds or OOD tests, no token accounting, and at least one number in the text contradicts the table. The baselines are not shown to be tuned, and the main comparison in Experiment I confounds gating with replay.

**Confidence:** 4/5.

### Summary of the submission

The authors propose comparing shuffle, fixed-clock curriculum, mastery-gated (no replay) and mastery-gated + replay schedules under a matched token budget, with mastery defined as simultaneous accuracy on several held-out OOD dimensions. Preliminary Experiment I (variable-chain depth task, depths 1–5) reports final in-distribution accuracy of 0.81 (mastery + replay), 0.47 (shuffle), 0.32 (mastery, no replay) and 0.14 (fixed clock). Preliminary Experiment II (addition → repeated addition, five arms) reports OOD repeated-addition accuracy 0.97 (mastery + replay), 0.91 (time-gated + replay), 0.85 and 0.85 (no-replay arms), 0.67 (shuffle), with an AUC efficiency metric.

### Strengths

1. A clear, falsifiable hypothesis with a pre-stated secondary claim (gate alone is insufficient; gate + replay is best).
2. The authors report a negative result (shuffle matched mastery on plain addition) and then designed a task with real prerequisite structure. That is good practice.
3. Separating gate probes from a sealed final exam, locking hyperparameters and committing to n ≥ 5 seeds and no post hoc retuning are the right commitments.
4. Experiment II adds a time-gated + replay arm, which is exactly the control needed to separate gating from replay, and its result (0.97 vs. 0.91) is reported honestly even though it shrinks the headline effect.
5. The learning-rate-decay confound for late-hard curricula is identified in advance.
6. The internal arithmetic is consistent: the in-distribution means in Figure 2 equal the averages of the per-depth values, and the Experiment II AUCs match a normalized trapezoid under the Figure 3 curves (reviewer's check).

### MAJOR issues

#### G-M1. No code, no seeds, no specification: the preliminary results are not yet evidence

**Problem.** Neither preliminary experiment has code on any known branch. Model size, tokenizer, LR schedule, batch size, number of steps, gate threshold, probe size and frequency, replay fraction, fixed-clock quotas, the definition of "OOD" for each task, and the number of seeds are all unstated. Both tables show single numbers with no variability; Figure 3 shows single curves that move by up to 0.4 between adjacent checkpoints (the no-replay mastery curve goes 0.60 → 0.18 → 0.48). That checkpoint-to-checkpoint swing is larger than the final gap between mastery + replay and time-gated + replay (0.06), so the headline ranking in Experiment II is inside the noise of a single run.

**Fix.** Publish the code and logs. State every hyperparameter in a table. Re-run with ≥ 10 seeds per arm (these are tiny models; see the minimum experiment below), report mean ± CI per arm, per-seed values, and curves with bands. Use the paired seed structure (same initialization and data stream across arms within a seed).

#### G-M2. Experiment I confounds gating with replay; the "fixed clock" baseline is not shown to be fair

**Problem.** In Experiment I the only replay arm is the mastery arm. Fixed clock (0.14) appears to have no replay: its depth-1 accuracy (0.15) is far below shuffle's (1.00) and similar to the no-replay mastery arm. So "mastery + replay beats fixed clock by 0.67" compares replay + gating against neither. Experiment II shows what happens when replay is added to the clock: the gap shrinks to 0.06 on the dependent skill and vanishes on retention (0.99 vs. 1.00). Moreover, the fixed-clock quotas are "predetermined" with no statement of how they were chosen. A clock that advances before the prerequisite is learnable is a strawman, and Experiment I's fixed-clock arm reaches only 0.15 even on depth 1. Wu, Dyer & Neyshabur (2021) ran curriculum, anti-curriculum and random-curriculum orderings over 180 pacing functions. At standard compute, no ordering beat properly searched baselines, and gains measured against a mean baseline were "an artifact of the large search space". Curricula helped only with much less compute or noisy labels, and there part of the benefit came from the growing dataset size (pacing) rather than the ordering. Diehl Martinez et al. (2023) similarly found that BabyLM curricula only matched a well-tuned baseline. The preliminary experiments here use small budgets, which is exactly the regime where Wu et al. found curricula help, so the result may not carry over to larger budgets.

**Fix.**
- Run the full 2 × 2 (gate: mastery vs. clock) × (replay: yes vs. no) in **both** tasks, plus shuffle.
- Tune the clock fairly: give the fixed-clock arm the same tuning budget as the mastery arm's threshold, e.g., sweep stage quotas on a validation set disjoint from the sealed exam, and pick the best.
- Add a **yoked clock** control: a fixed schedule whose switch times are copied from a mastery-arm run (same seed or a different seed), so it has the same timing but the timing is not contingent on its own performance. This is the classic way to isolate contingency from timing, and it is the single most informative control for this claim. Church (1964) showed that yoked designs carry their own bias: when individuals (here, seeds) differ, a schedule tailored to one run is systematically worse for its yoked partner, which can create a spurious advantage for the contingent arm. So yoke each control to several different mastery runs (not only the same seed), and keep the tuned clock as a second, non-yoked comparison.
- Add a **random-curriculum** (pacing-only) control: the same unlock times, but each newly unlocked slice is random rather than the next skill. Following Wu et al. (2021), this separates the effect of the growing training pool from the effect of order.
- Optional: a learning-progress curriculum (Graves et al., 2017; Matiisen et al., 2020) and a competence-based schedule (Platanios et al., 2019) as stronger adaptive baselines.

#### G-M3. Matched-token claims are not substantiated, and per-skill token allocation differs across arms

**Problem.** The hypothesis is "under a matched total token/compute budget", but Experiment I says the arms trained "for the same number of steps". In a variable-chain task, deeper prompts are longer, so equal steps do not mean equal tokens, and arms that dwell on shallow depths process fewer tokens per step. Even with equal *total* tokens, the arms differ in how many tokens each *skill* receives: shuffle spreads tokens over all depths from the start, while a gated arm concentrates them on the current frontier. A curriculum "win" could therefore be a reallocation of data toward hard skills, not an ordering effect. Probe evaluations also cost compute that the shuffle arm does not use (the cost section estimates ~15% overhead), and that compute is not counted.

**Fix.** (a) Count and report tokens, not steps, per arm and per skill, from logs. (b) Add a **mixture-matched shuffle**: an IID shuffle whose marginal skill proportions equal the mastery arm's realized proportions over the run. If it matches the mastery arm, the effect is allocation, not order. (c) Report probe compute and either charge it to the gated arms' budget or report results both ways (Dehghani et al., 2022, on reporting efficiency honestly).

#### G-M4. The "no-replay" arm is a strawman for the main hypothesis, and the gate is not compared against the right alternatives

**Problem.** That a curriculum *without* replay forgets earlier skills is the same catastrophic-interference result as TEST 02, and is known (Robins, 1995; Rolnick et al., 2019; Zaremba & Sutskever, 2014). Including it as an arm is useful as a sanity check, but beating it is not evidence for gating. The scientifically interesting comparison is mastery + replay vs. the strongest non-gated alternatives at matched tokens: tuned clock + replay, yoked clock + replay, mixture-matched shuffle, and (from Zaremba & Sutskever) a "mixed difficulty throughout" schedule.

**Fix.** Make the primary contrast **mastery + replay vs. tuned/yoked clock + replay**, pre-specified. Add a **mixed-difficulty-throughout** arm (Zaremba & Sutskever, 2014; Wang et al., 2025), which theory says can be as good as a staged curriculum on chain tasks. Treat no-replay arms as secondary illustrations. Also vary the replay fraction (e.g., 10/25/50%), since a gate that only works at one replay fraction is fragile.

#### G-M5. Definition of mastery, probe leakage, and adaptive use of OOD information

**Problem.** Mastery is defined as accuracy on "several OOD dimensions at once (larger scale, longer inputs, perturbations, novel compositions)", measured on "frozen leak-free held-out dev sets". If the gate probes and the sealed exam share OOD *dimensions* (for example, both test longer inputs), the mastery arm receives feedback from the exam distribution during training and adapts its data to it, while the baselines do not. That is an information advantage and a form of test-distribution leakage even if no exam item is reused (Kapoor & Narayanan, 2023; Cawley & Talbot, 2010). It also means the threshold, probe set and probing frequency are tuned hyperparameters that the baselines lack. For Experiment I the report never says what "OOD" means; for Experiment II "Addition OOD test set (retention)" is undefined. Finally, a threshold gate can fire on a noisy probe estimate; with small probes, the unlock time is itself random and should be reported.

**Fix.**
- Separate the OOD axes: gate on one family (e.g., lengths 7–9, or one perturbation type) and evaluate on disjoint families (lengths 15–30, unseen compositions, unseen templates) that the gate never sees. Report both.
- Give the baselines equal access to the probes (e.g., the clock arm may use probes to pick its quotas during tuning, or for checkpoint selection) so the comparison is about *contingent progression*, not about information.
- Pre-specify threshold, probe size, probe frequency and the statistical rule for "mastered" (e.g., lower 95% bound of probe accuracy ≥ τ), and report unlock times per seed. Run a sensitivity analysis over τ.
- Do not let the gate fire on in-distribution accuracy alone. Small algorithmic models can fit the training distribution perfectly and still never generalize (Delétang et al., 2023), or generalize only long after fitting it (grokking; Power et al., 2022; Nanda et al., 2023). The report's own definition of mastery already says this; the gate rule must put it into practice.
- Use established length-generalization and compositional-split protocols to define OOD (Anil et al., 2022; Delétang et al., 2023; Zhou et al., 2024; Lake & Baroni, 2018; Keysers et al., 2020; Hupkes et al., 2023), and LEGO-style chain tasks for prerequisite depth (Zhang et al., 2022).

#### G-M6. Inconsistencies between text, tables and hypothesis

**Problem.**
- Experiment II text: "a lack of replay causes the model to degrade on addition (0.59, 0.67)". The table says 0.67 and **0.55**.
- Experiment I text: mastery + replay depth 5 is "low only because it was not unlocked within the budget". Then the in-distribution average (0.81) includes a skill that arm barely trained, while the no-replay gate reached depth 5 *earlier* despite the same threshold, which needs explaining (replay slows frontier progress at matched tokens).
- The hypothesis predicts gains "especially [in] OOD robustness on later/harder skills". In Experiment I, mastery + replay scores **0.18 on depth 5, below fixed clock (0.28)**, and the OOD column is not broken down by depth.
- Experiment I says shuffle on plain addition "matched mastery and was even more token-efficient", but gives no numbers.
- Arm names change ("fixed clock", "fixed curriculum", "time-gated"); Experiment II has five arms, the proposal four.
- Figure 3 curves are still rising steeply at the end of the budget for all arms. Final-point rankings are therefore budget-dependent; a longer run might equalize them.

**Fix.** Correct the 0.59/0.55 discrepancy. Report per-depth OOD and per-depth unlock times. Report Experiment I's addition numbers. Use one naming scheme. Pre-specify the budget and also report the curves to plateau. Re-state the hypothesis to match what the data can test (e.g., sample efficiency to a target on the deepest *unlocked* skill, plus final accuracy on all skills).

#### G-M7. Number of seeds and statistical plan

**Problem.** The proposal commits to n ≥ 5 seeds, but the preliminary results use what appears to be one. With 5 arms there are 10 pairwise contrasts. With n = 5 paired seeds, an α = .05 paired t-test has 80% power only for d_z ≈ 1.7 (reviewer's Monte Carlo), before any multiplicity correction. Curricula effects in the literature are often small relative to seed variance (Wu et al., 2021; BabyLM findings: Warstadt et al., 2023).

**Fix.** Pre-specify one primary contrast (G-M4) and one primary metric (G-M8). Use ≥ 10 seeds for the synthetic study (cheap), Holm-correct secondary contrasts, and report probability of improvement and stratified bootstrap intervals across tasks (Agarwal et al., 2021). Use n = 5 only for the expensive scaled study, with a power analysis based on the synthetic study's variance (Colas et al., 2018; Lakens, 2022).

#### G-M8. Metric definitions: "efficiency (AUC)" and the data-vs-performance curve need specification

**Problem.** The AUC in Experiment II appears (reviewer's reconstruction) to be the normalized area under the *dependent-skill* OOD accuracy curve over ~11 checkpoints. That metric mechanically favours arms that start the dependent skill early, and it depends on checkpoint spacing and the token range. "Sample efficiency" can also be defined as tokens-to-threshold, or as accuracy at fixed budgets; different choices can reverse rankings.

**Fix.** Define the AUC exactly (which skill, which metric, normalization, checkpoint grid, interpolation) before running; report it with CIs; and also report tokens-to-threshold (with censoring if never reached) and accuracy at two or three fixed budgets. The continual-learning "learning curve area" metric (Chaudhry et al., 2019) and the multi-metric reporting of Díaz-Rodríguez et al. (2018) are good templates. Report average accuracy over *all* skills, not only the dependent one.

#### G-M9. Most of the section is a proposal; the scaled study's novelty claim is untested

**Problem.** The literature review, scope (competitive programming), cost analysis and attribution-graph extension describe work that has not started. The novelty claim ("a strict unlock-on-probe gate over a skill graph ... will be unique") rests on experiments that do not exist. Competence-paced or progress-driven curricula already exist in NMT (Platanios et al., 2019, where "competence" is a time schedule), RL and LSTM algorithmic tasks (Graves et al., 2017; Matiisen et al., 2020, whose teacher also revisits subtasks whose performance drops, i.e., built-in replay; Portelas et al., 2020) and LLM post-training (RLAAR, Li et al., 2025/2026; CAMPUS, Li et al., 2025). BabyLM curricula mostly failed (Warstadt et al., 2023; Diehl Martinez et al., 2023). Closest of all on the synthetic side:
- **Wang et al. (2025, COLT)** prove for k-fold composition (pointer following, essentially the variable-chain task) that training only on depth-k data needs exponentially many samples, while *either* a staged curriculum *or* a simultaneous easy + hard mixture makes it learnable. That predicts a well-designed shuffle with enough shallow data should match a gated curriculum, so a mixed-difficulty control is required.
- **Kohli et al. (2026, arXiv, accepted at COLM 2026 per the arXiv comment)** reportedly use a curriculum over hop depth that advances at ≥95% held-out accuracy with joint training on earlier stages. That is a mastery gate with replay. **VERIFIED in the full text (§6.1)**: training advances to (k+1)-hop only after ≥95% accuracy on a held-out k-hop split, and all earlier stages are trained jointly "to prevent forgetting". The authors cite Yao et al. (2025, EMNLP; https://arxiv.org/abs/2505.17923) for this curriculum. It is used as a training tool, not compared against shuffle or fixed schedules.
- **Yao et al. (2025, EMNLP per arXiv)** find that curriculum mitigates, but does not remove, the exponential data cost of k-hop reasoning in small from-scratch transformers.
A reviewer will discount novelty until a controlled result exists, and will expect these papers to be discussed.

**Fix.** Narrow the paper to what is measured: "a controlled test of performance-contingent progression vs. fair non-contingent schedules on synthetic prerequisite tasks". Move competitive programming, cost analysis and attribution graphs to a short "future work" paragraph or a separate proposal document. Remove the informal remarks in the Limitations and Extension paragraphs.

### MINOR issues

- **G-m1.** The Kulik et al. (1990) effect and the human "mastery learning" framing should be stated with its caveats (effects smaller on standardized tests, larger with stricter criteria and more instructional time). Note that mastery learning in humans changes *time on task*, which the matched-token design removes.
- **G-m2.** The learning-rate-decay confound is handled by "identical LR schedule across arms", but an identical cosine schedule still gives late skills a smaller LR in gated arms only if they reach those skills late. Use constant LR or WSD for the main study, as the text suggests, and state which was used in Experiments I and II.
- **G-m3.** State the model size, parameter count and tokenization for Experiments I and II (digit-level vs. BPE matters for arithmetic; N. Lee et al., 2024; McLeish et al., 2024).
- **G-m4.** The "reverse the output digits" link (arXiv 2403.05845) is Zhang-Li et al. (2024), "Reverse that number!"; cite it by title, alongside N. Lee et al. (2024), the standard reference for reversed-output arithmetic in small transformers.
- **G-m5.** The Robins (1995) link points to a Stanford proxy (`stanford.idm.oclc.org`); replace it with the DOI.
- **G-m6.** Report probe sizes and the binomial uncertainty of each gate decision.
- **G-m7.** The mission statement says "This preregistered project". If there is a timestamped preregistration (OSF, AsPredicted, or a dated commit), link it; if not, remove the word.

### Would reviewers want the mastery-gating subteam to run actual experiments? Yes.

At any ML venue (ICLR, NeurIPS, ICML, TMLR, COLM) and at CogSci, a section whose evidence is two unreproducible single-run tables would not be accepted, and a section that is proposal-only would not be reviewed as a research paper at all. The only venues that accept proposal-only work are registered-report or "pre-registration" style tracks (which require a full methods protocol and pilot data) and some non-archival workshops (see Part C). Reviewers would therefore expect actual, reproducible experiments. The good news is that the needed experiment is small.

**Minimum experiment that would satisfy a fair reviewer** (feasible on one GPU in days, given the task sizes implied by Figure 3's ~5 × 10⁷ tokens per run):

1. **Tasks:** the variable-chain depth task (depths 1–5 or 1–6) as primary, and addition → repeated addition as secondary. Fix and publish the generators, splits and OOD definitions. Gate probes and sealed exam use **disjoint OOD families**.
2. **Arms (6):** shuffle; mixture-matched shuffle; tuned clock + replay; yoked clock + replay; mastery + replay; mastery, no replay (secondary). A tuned clock without replay is optional.
3. **Budget:** matched *tokens* (logged), constant LR (or WSD) identical across arms; probe compute reported; train every arm to the same budget and also report curves to plateau.
4. **Seeds:** ≥ 10 paired seeds per arm (initialization + data stream).
5. **Pre-registered primary outcome:** sealed-exam OOD accuracy averaged over all skills at the budget, with the primary contrast mastery + replay vs. the better of tuned/yoked clock + replay. Secondary: tokens-to-threshold, AUC (defined in advance), per-skill retention, unlock times. Holm-correct secondaries.
6. **Sensitivity:** threshold τ (3 values) × replay fraction (3 values) for the mastery arm, and the same replay fractions for the clock arm.
7. **Release:** code, configs, logs and the preregistration.

If mastery + replay beats yoked and tuned clock + replay and mixture-matched shuffle under these conditions, that is a publishable, novel result. If it does not, the rigorous null is still publishable at TMLR or a workshop, and it is informative given the BabyLM record.

---

## B3. Cross-cutting issues for the final report as a whole

- **"Preregistered" claim.** The mission statement calls the project preregistered. None of TEST 02 or TEST 03's completed runs appears to have a public, timestamped registration. Either link it or drop the word; reviewers check.
- **Text vs. code provenance.** Every number in a paper must be traceable to a script and a results file. At present TEST 02's pilot numbers and all TEST 03 results fail that test.
- **Shared mechanism.** TEST 02's blocked arm and TEST 03's no-replay arms show the same phenomenon (sequential training without replay erases earlier skills). A combined paper should say this once, cite it as known, and focus the contributions on what is new: block-length dose-response and similarity moderation (TEST 02), contingent vs. yoked progression with replay (TEST 03), and spacing shape under a fixed replay budget (TEST 01).

---

# Part C — Venue fit (status as of 2026-10-04)

The verification agent read the official pages on 2026-10-04. **UNVERIFIED** marks anything not yet posted or not readable; for those, the previous year's dates are given as a guide.

## C0. Already closed or unrealistic

- **ICLR 2027 main track.** Abstract was due 18 Sep 2026 and the paper 25 Sep 2026; 9 pages. https://iclr.cc/Conferences/2027/Dates. VERIFIED, closed.
- **NeurIPS 2026**, including workshops. The main paper was due 6 May 2026. Workshops had a suggested deadline of 29 Aug 2026, with mandatory notification by 29 Sep 2026. https://neurips.cc/Conferences/2026/Dates. VERIFIED, closed.
- **AAAI-27.** Paper due 28 Jul 2026. https://aaai.org/conference/aaai/aaai-27/submission-instructions/. VERIFIED, closed.
- **AISTATS 2027.** The abstract deadline (29 Sep 2026) has passed and author lists locked then. https://virtual.aistats.org/Conferences/2027/CallForPapers. VERIFIED, effectively closed.
- **BabyLM 2026** (EMNLP 2026). Closed. https://babylm.github.io/. VERIFIED.
- **ARR October 2026 cycle** (feeds NAACL 2027 and COLING 2027). Due 12 Oct 2026; 8-page long papers, 4-page short papers. https://aclrollingreview.org/dates, https://2027.naacl.org/calls/main_conference_papers/. VERIFIED. Eight days is not enough time to run the minimum experiments above, so this is not recommended.

## C1. A combined paper: "Which learning-science principles transfer to LM training?"

**Pitch that could work.** A "scorecard" paper with three controlled tests: spacing (TEST 01), interleaving (TEST 02) and mastery gating (TEST 03). Each would have a pre-registered primary contrast, ≥ 10 seeds (or 20 seed × order pairs), and an explicit human-vs-machine comparison of *mechanism*. The headline is which principles transfer, weaken or reverse. This matches the project's mission statement, and the Flesch (2018) / Russin (2025) dissociation gives a ready theoretical frame: in-weight learning favours interleaving because of forgetting, while humans can favour blocking.

**Risk.** Reviewers judge a combined paper by its weakest section. Right now TEST 03 has no reproducible experiment, TEST 02's text does not match its run, and TEST 01 has its own open issues (see file 05). A combined submission is only sensible once all three meet the minimum bars in Part B.

| Venue | Deadline | Format | Fit |
|---|---|---|---|
| **COLM 2027** | Not posted (UNVERIFIED). COLM 2026 had abstract 26 Mar / paper 31 Mar 2026 (https://colm.cc/Conferences/2026/CallForPapers) | 9 pages, double-blind, OpenReview, archival | **Best conference fit** for a science-of-LM-training paper; the timeline allows all three studies to be finished. |
| **ACL 2027** (ARR January cycle) | **4 Jan 2027** (VERIFIED, https://2027.aclweb.org/calls/main/); Kyoto, 17–22 Aug 2027 | 8 pages long, archival | Good if framed around language models. Three months is tight but feasible for TEST 02 and the synthetic TEST 03. |
| **ICML 2027** | Not posted (UNVERIFIED; the CFP URL returns 404). ICML 2026 had abstract 23 Jan / paper 28 Jan 2026 (https://icml.cc/Conferences/2026/CallForPapers). Location "South America" per https://icml.cc/Conferences/FutureMeetings | 8 pages, PMLR | High bar on novelty and scale. Only if results are strong and at least one test is scaled up. |
| **NeurIPS 2027** | Not posted (UNVERIFIED). NeurIPS 2026 papers were due 6 May 2026, 9 pages | archival | Would need multi-scale evidence. |
| **TMLR** | Rolling (https://jmlr.org/tmlr/author-guide.html) | No page limit | **Safest home.** The acceptance criteria are "claims supported by accurate and convincing evidence" plus audience interest; novelty is not required (https://jmlr.org/tmlr/acceptance-criteria.html). Rigorous nulls fit. A J2C certification makes a paper eligible to present at NeurIPS/ICLR/ICML. |
| **CogSci 2027** | CFP not posted (UNVERIFIED; the CogSci 2026 deadline was 2 Feb 2026, and the FAQ says "typically Feb 1"). Bilbao, 28–31 Jul 2027 (VERIFIED, https://cognitivesciencesociety.org/cogsci-2027/) | 6 pages, double-blind, **non-archival** (https://cognitivesciencesociety.org/submissions/) | Good for the human–machine framing, *if* each section discusses mechanism (forgetting vs. discriminative contrast; time-on-task in mastery learning). Non-archival status leaves the door open to a later TMLR or journal version. |
| Journals | Rolling | — | *Computational Brain & Behavior* (https://link.springer.com/journal/42113, single-blind, no registered reports) or *Cognitive Science* (https://cognitivesciencesociety.org/cognitive-science/; article types UNVERIFIED) for an extended cognitive version. *Open Mind* (MIT Press) details UNVERIFIED. |

## C2. Separate papers

**TEST 02 (interleaving).** Run the minimum experiment from B1, and add a similarity manipulation and block-length dose-response so the paper has a new claim.
- **CogSci 2027** (~1 Feb 2027, UNVERIFIED; 6 pages, non-archival). The best fit if the paper tests the *human* moderator (similarity) and argues about mechanism.
- **CCN 2027.** Edinburgh, 26–29 Jul 2027 (VERIFIED, https://2027.ccneuro.org/); deadlines not posted (UNVERIFIED). CCN 2026 had an archival 8-page track due 12 Feb and a non-archival 2-page track due 2 Apr (https://2026.ccneuro.org/call-for-papers/). The 2-page track suits the current pilot plus replication.
- **ICLR 2027 workshops.** Suggested paper deadline ~1 Feb 2027; the accepted workshop list is due 29 Nov 2026 (VERIFIED, https://iclr.cc/Conferences/2027/CallForWorkshops). Watch for continual-learning or science-of-DL workshops. Usually non-archival (check each).
- **CoLLAs 2027** (lifelong-learning community). Nothing posted (UNVERIFIED; CoLLAs 2026 papers were due 15 Apr 2026). https://lifelong-ml.cc/
- **TMLR**, after adding the dose-response and similarity analyses.

**TEST 03 (mastery gating).** This needs the minimum experiment in B2 before any submission.
- **ICLR 2027 workshop** (~1 Feb 2027). A realistic first target once the synthetic six-arm, ten-seed study exists.
- **CoLLAs 2027** (UNVERIFIED), because replay is central to the claim.
- **TMLR**, for the full synthetic study. A rigorous null is acceptable there.
- **COLM 2027 / NeurIPS 2027**, only after a scaled study (e.g., code or math with a skill graph).
- **BabyLM 2027**, only if the work moves to curriculum in developmentally plausible LM pretraining. Not announced (UNVERIFIED; past calls appeared in February).

## C3. Registered reports and preregistration tracks

- No ML venue with an active registered-report track in 2026–27 was found. The NeurIPS pre-registration workshops (PMLR 148, 181; https://preregister.science/) last ran in 2021. TMLR, *Cognitive Science*, *Open Mind*, *CB&B* and *Neural Computation* are not on the Center for Open Science registered-reports list (https://www.cos.io/initiatives/registered-reports). The agent's search was cut short, so a narrower track could have been missed.
- Practical advice: preregister on OSF before the confirmatory runs (A8), cite the registration in the paper, and label analyses as confirmatory or exploratory. Reviewers at TMLR and CogSci reward this even without a formal track.

## C4. Suggested timeline

1. **Oct–Nov 2026.**
   - TEST 02: run the minimum experiment (well under a GPU-hour) plus the similarity and block-length extensions.
   - TEST 03: release code and run the six-arm, ten-seed synthetic study.
   - Preregister both before the confirmatory runs.
2. **Jan–Feb 2027.**
   - Separate short papers: ACL 2027 via ARR (4 Jan, verified), CogSci 2027 (~1 Feb, UNVERIFIED) and/or ICLR 2027 workshops (~1 Feb).
3. **Mar 2027 onward.**
   - Combined paper to COLM 2027 (~late Mar, UNVERIFIED) or TMLR once TEST 01's fixes (file 05) are in.
   - Check dual-submission rules: CogSci and most workshops are non-archival; ACL, COLM and TMLR are archival.

---

# References

Status: V = verified for this review; (05), (03) or (06) = verified in an earlier file in this folder; PV = partly verified (see Part A for which field).

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *Advances in Neural Information Processing Systems, 34*, 29304–29320. https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html (05)
- Anil, C., Wu, Y., Andreassen, A., Lewkowycz, A., Misra, V., Ramasesh, V., Slone, A., Gur-Ari, G., Dyer, E., & Neyshabur, B. (2022). Exploring length generalization in large language models. *Advances in Neural Information Processing Systems, 35*. https://openreview.net/forum?id=zSkYVeX7bC4 (PV)
- Baayen, R. H., Davidson, D. J., & Bates, D. M. (2008). Mixed-effects modeling with crossed random effects for subjects and items. *Journal of Memory and Language, 59*(4), 390–412. https://doi.org/10.1016/j.jml.2007.12.005 (05)
- Barr, D. J., Levy, R., Scheepers, C., & Tily, H. J. (2013). Random effects structure for confirmatory hypothesis testing: Keep it maximal. *Journal of Memory and Language, 68*(3), 255–278. https://doi.org/10.1016/j.jml.2012.11.001 (05)
- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. In *Proceedings of the 26th Annual International Conference on Machine Learning* (pp. 41–48). ACM. https://doi.org/10.1145/1553374.1553380 (V)
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological), 57*(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x (V)
- Bertinetto, L., Henriques, J. F., Albanie, S., Paganini, M., & Varol, G. (Eds.). (2021). *NeurIPS 2020 Workshop on Pre-registration in Machine Learning* (PMLR Vol. 148). https://proceedings.mlr.press/v148/ (V; second edition, PMLR Vol. 181: https://proceedings.mlr.press/v181/)
- Birnbaum, M. S., Kornell, N., Bjork, E. L., & Bjork, R. A. (2013). Why interleaving enhances inductive learning: The roles of discrimination and retrieval. *Memory & Cognition, 41*(3), 392–402. https://doi.org/10.3758/s13421-012-0272-7 (V)
- Bloom, B. S. (1968). Learning for mastery. *Evaluation Comment, 1*(2), 1–12. https://eric.ed.gov/?id=ED053419 (V)
- Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Sepah, N., Raff, E., Madan, K., Voleti, V., Kahou, S. E., Michalski, V., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems, 3*, 747–769. https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html (05)
- Brooks, J. L. (2012). Counterbalancing for serial order carryover effects in experimental condition orders. *Psychological Methods, 17*(4), 600–614. https://doi.org/10.1037/a0029310 (V)
- Brunmair, M., & Richter, T. (2019). Similarity matters: A meta-analysis of interleaved learning and its moderators. *Psychological Bulletin, 145*(11), 1029–1052. https://doi.org/10.1037/bul0000209 (V)
- Campos, D. (2021). *Curriculum learning for language modeling* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2108.02170 (V)
- Card, D., Henderson, P., Khandelwal, U., Jia, R., Mahowald, K., & Jurafsky, D. (2020). With little power comes great responsibility. In *Proceedings of EMNLP 2020* (pp. 9263–9274). https://doi.org/10.18653/v1/2020.emnlp-main.745 (05)
- Carvalho, P. F., & Goldstone, R. L. (2014). Putting category learning in order: Category structure and temporal arrangement affect the benefit of interleaved over blocked study. *Memory & Cognition, 42*(3), 481–495. https://doi.org/10.3758/s13421-013-0371-0 (V)
- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html (V)
- Chaudhry, A., Dokania, P. K., Ajanthan, T., & Torr, P. H. S. (2018). Riemannian walk for incremental learning: Understanding forgetting and intransigence. In *Computer Vision – ECCV 2018* (LNCS Vol. 11215, pp. 556–572). Springer. https://doi.org/10.1007/978-3-030-01252-6_33 (V)
- Chaudhry, A., Ranzato, M., Rohrbach, M., & Elhoseiny, M. (2019). Efficient lifelong learning with A-GEM. In *International Conference on Learning Representations*. https://openreview.net/forum?id=Hkf2_sC5FX (V)
- Chen, M. F., Roberts, N., Bhatia, K., Wang, J., Zhang, C., Sala, F., & Ré, C. (2023). Skill-it! A data-driven skills framework for understanding and training language models. *Advances in Neural Information Processing Systems, 36*, 36000–36040. https://arxiv.org/abs/2307.14430 (V)
- Church, R. M. (1964). Systematic effect of random error in the yoked control design. *Psychological Bulletin, 62*(2), 122–131. https://doi.org/10.1037/h0042733 (V)
- Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2018). *How many random seeds? Statistical power analysis in deep reinforcement learning experiments* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1806.08295 (05)
- Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2019). *A hitchhiker's guide to statistical comparisons of reinforcement learning algorithms* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1904.06979 (V)
- Dehghani, M., Arnab, A., Beyer, L., Vaswani, A., & Tay, Y. (2022). The efficiency misnomer. In *International Conference on Learning Representations*. https://openreview.net/forum?id=iulEMLYh1uR (V)
- Dekker, R. B., Otto, F., & Summerfield, C. (2022). Curriculum learning for human compositional generalization. *Proceedings of the National Academy of Sciences, 119*(41), e2205582119. https://doi.org/10.1073/pnas.2205582119 (V)
- Delétang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wenliang, L. K., Catt, E., Cundy, C., Hutter, M., Legg, S., Veness, J., & Ortega, P. A. (2023). Neural networks and the Chomsky hierarchy. In *International Conference on Learning Representations*. https://openreview.net/forum?id=WbxHAzkeQcn (V)
- Díaz-Rodríguez, N., Lomonaco, V., Filliat, D., & Maltoni, D. (2018). *Don't forget, there is more than forgetting: New metrics for continual learning* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1810.13166 (V)
- Diehl Martinez, R., Goriely, Z., McGovern, H., Davis, C., Caines, A., Buttery, P., & Beinborn, L. (2023). CLIMB – Curriculum learning for infant-inspired model building. In *Proceedings of the BabyLM Challenge at CoNLL 2023* (pp. 112–127). https://doi.org/10.18653/v1/2023.conll-babylm.10 (V)
- Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., & Smith, N. (2020). *Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2002.06305 (05)
- Dong, G., Yuan, H., Lu, K., Li, C., Xue, M., Liu, D., Wang, W., Yuan, Z., Zhou, C., & Zhou, J. (2024). How abilities in large language models are affected by supervised fine-tuning data composition. In *Proceedings of the 62nd Annual Meeting of the ACL (Vol. 1)* (pp. 177–198). https://doi.org/10.18653/v1/2024.acl-long.12 (V)
- Elman, J. L. (1993). Learning and development in neural networks: The importance of starting small. *Cognition, 48*(1), 71–99. https://doi.org/10.1016/0010-0277(93)90058-4 (V)
- Flesch, T., Balaguer, J., Dekker, R., Nili, H., & Summerfield, C. (2018). Comparing continual task learning in minds and machines. *Proceedings of the National Academy of Sciences, 115*(44), E10313–E10322. https://doi.org/10.1073/pnas.1800755115 (V)
- French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2 (V)
- Graves, A., Bellemare, M. G., Menick, J., Munos, R., & Kavukcuoglu, K. (2017). Automated curriculum learning for neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR 70, pp. 1311–1320). https://proceedings.mlr.press/v70/graves17a.html (V)
- Hacohen, G., & Weinshall, D. (2019). On the power of curriculum learning in training deep networks. In *Proceedings of the 36th International Conference on Machine Learning* (PMLR 97, pp. 2535–2544). https://proceedings.mlr.press/v97/hacohen19a.html (V)
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70. https://www.jstor.org/stable/4615733 (V)
- Hupkes, D., Dankers, V., Mul, M., & Bruni, E. (2020). Compositionality decomposed: How do neural networks generalise? *Journal of Artificial Intelligence Research, 67*, 757–795. https://doi.org/10.1613/jair.1.11674 (V)
- Hupkes, D., Giulianelli, M., Dankers, V., Artetxe, M., Elazar, Y., Pimentel, T., Christodoulopoulos, C., Lasri, K., Saphra, N., Sinclair, A., Ulmer, D., Schottmann, F., Batsuren, K., Sun, K., Sinha, K., Khalatbari, L., Ryskina, M., Frieske, R., Cotterell, R., & Jin, Z. (2023). A taxonomy and review of generalization research in NLP. *Nature Machine Intelligence, 5*(10), 1161–1174. https://doi.org/10.1038/s42256-023-00729-y (V)
- Jagielski, M., Thakkar, O., Tramèr, F., Ippolito, D., Lee, K., Carlini, N., Wallace, E., Song, S., Thakurta, A., Papernot, N., & Zhang, C. (2023). Measuring forgetting of memorized training examples. In *International Conference on Learning Representations*. https://openreview.net/forum?id=7bJizxLKrR (05)
- Jones, B., & Kenward, M. G. (2014). *Design and analysis of cross-over trials* (3rd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b17537 (V)
- Kang, S. H. K., & Pashler, H. (2012). Learning painting styles: Spacing is advantageous when it promotes discriminative contrast. *Applied Cognitive Psychology, 26*(1), 97–103. https://doi.org/10.1002/acp.1801 (V)
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), Article 100804. https://doi.org/10.1016/j.patter.2023.100804 (V)
- Kazemnejad, A., Padhi, I., Natesan Ramamurthy, K., Das, P., & Reddy, S. (2023). The impact of positional encoding on length generalization in transformers. *Advances in Neural Information Processing Systems, 36*. https://doi.org/10.52202/075280-1082 (V)
- Kerr, N. L. (1998). HARKing: Hypothesizing after the results are known. *Personality and Social Psychology Review, 2*(3), 196–217. https://doi.org/10.1207/s15327957pspr0203_4 (V)
- Keysers, D., Schärli, N., Scales, N., Buisman, H., Furrer, D., Kashubin, S., Momchev, N., Sinopalnikov, D., Stafiniak, L., Tihon, T., Tsarkov, D., Wang, X., van Zee, M., & Bousquet, O. (2020). Measuring compositional generalization: A comprehensive method on realistic data. In *International Conference on Learning Representations*. https://openreview.net/forum?id=SygcCnNKwr (V)
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences, 114*(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114 (V)
- Kohli, H., Parthasarathy, S., Sun, H., & Yao, Y. (2026). *Loop, think, & generalize: Implicit reasoning in recurrent-depth transformers* [Preprint]. arXiv. https://arxiv.org/abs/2604.07822 (V)
- Kornell, N., & Bjork, R. A. (2008). Learning concepts and categories: Is spacing the "enemy of induction"? *Psychological Science, 19*(6), 585–592. https://doi.org/10.1111/j.1467-9280.2008.02127.x (V)
- Kulik, C.-L. C., Kulik, J. A., & Bangert-Drowns, R. L. (1990). Effectiveness of mastery learning programs: A meta-analysis. *Review of Educational Research, 60*(2), 265–299. https://doi.org/10.3102/00346543060002265 (V)
- Kumar, M. P., Packer, B., & Koller, D. (2010). Self-paced learning for latent variable models. *Advances in Neural Information Processing Systems, 23*. https://papers.nips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html (PV)
- Lake, B., & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In *Proceedings of the 35th International Conference on Machine Learning* (PMLR 80, pp. 2873–2882). https://proceedings.mlr.press/v80/lake18a.html (V)
- Lakens, D. (2022). Sample size justification. *Collabra: Psychology, 8*(1), Article 33267. https://doi.org/10.1525/collabra.33267 (V)
- Lee, B. W., Cho, H., & Yoo, K. M. (2024). Instruction tuning with human curriculum. In *Findings of the ACL: NAACL 2024* (pp. 1281–1309). https://doi.org/10.18653/v1/2024.findings-naacl.82 (V)
- Lee, N., Sreenivasan, K., Lee, J. D., Lee, K., & Papailiopoulos, D. (2024). Teaching arithmetic to small transformers. In *International Conference on Learning Representations*. https://openreview.net/forum?id=dsUB4bst9S (V)
- Li, M., Chen, P., Zhang, Z., Yang, T., Zhang, X., Li, H., Cao, T., Zeng, M., Wu, Z., Jiang, M., Li, H., Li, L., & Yin, B. (2025). *Mitigating lost in multi-turn conversation via curriculum RL with verifiable accuracy and abstention rewards* [Preprint; ACL 2026 per arXiv]. arXiv. https://arxiv.org/abs/2510.18731 (V, arXiv)
- Li, Y., Lu, T., Li, Y., Chen, Y., Huang, W.-C., Jiang, W., Wang, H., Zheng, H.-T., & Yu, P. S. (2025). *Teaching according to talents! Instruction tuning LLMs with competence-aware curriculum learning* [Preprint; EMNLP 2025 Findings per arXiv]. arXiv. https://arxiv.org/abs/2509.13790 (V, arXiv)
- Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *Advances in Neural Information Processing Systems, 30*. https://proceedings.neurips.cc/paper_files/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html (V)
- Lu, X., Li, X., Cheng, Q., Ding, K., Huang, X., & Qiu, X. (2024). Scaling laws for fact memorization of large language models. In *Findings of EMNLP 2024* (pp. 11263–11282). https://doi.org/10.18653/v1/2024.findings-emnlp.658 (06)
- Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *International Conference on Learning Representations*. https://openreview.net/forum?id=T5wkZJqzkz (V)
- Madaan, L., Singh, A. K., Schaeffer, R., Poulton, A., Koyejo, S., Stenetorp, P., Narang, S., & Hupkes, D. (2024). *Quantifying variance in evaluation benchmarks* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2406.10229 (05)
- Matiisen, T., Oliver, A., Cohen, T., & Schulman, J. (2020). Teacher–student curriculum learning. *IEEE Transactions on Neural Networks and Learning Systems, 31*(9), 3732–3740. https://doi.org/10.1109/TNNLS.2019.2934906 (PV)
- Mayo, D., Scott, T. R., Ren, M., Elsayed, G., Hermann, K., Jones, M., & Mozer, M. C. (2023). Multitask learning via interleaving: A neural network investigation. In *Proceedings of the 45th Annual Conference of the Cognitive Science Society*. https://escholarship.org/uc/item/3tb956hb (PV)
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review, 102*(3), 419–457. https://doi.org/10.1037/0033-295X.102.3.419 (V)
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), *Psychology of Learning and Motivation* (Vol. 24, pp. 109–165). Academic Press. https://doi.org/10.1016/S0079-7421(08)60536-8 (V)
- McLeish, S., Bansal, A., Stein, A., Jain, N., Kirchenbauer, J., Bartoldson, B. R., Kailkhura, B., Bhatele, A., Geiping, J., Schwarzschild, A., & Goldstein, T. (2024). Transformers can do arithmetic with the right embeddings. *Advances in Neural Information Processing Systems, 37*. https://arxiv.org/abs/2405.17399 (V)
- Miller, E. (2024). *Adding error bars to evals: A statistical approach to language model evaluations* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2411.00640 (05)
- Mindermann, S., Brauner, J. M., Razzak, M. T., Sharma, M., Kirsch, A., Xu, W., Höltgen, B., Gomez, A. N., Morisot, A., Farquhar, S., & Gal, Y. (2022). Prioritized training on points that are learnable, worth learning, and not yet learnt. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162, pp. 15630–15649). https://proceedings.mlr.press/v162/mindermann22a.html (V)
- Mirzadeh, S. I., Farajtabar, M., Pascanu, R., & Ghasemzadeh, H. (2020). Understanding the role of training regimes in continual learning. *Advances in Neural Information Processing Systems, 33*. https://proceedings.neurips.cc/paper/2020/hash/518a38cc9a0173d0b2dc088166981cf8-Abstract.html (V)
- Murdock, B. B., Jr. (1962). The serial position effect of free recall. *Journal of Experimental Psychology, 64*(5), 482–488. https://doi.org/10.1037/h0045106 (V)
- Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. In *International Conference on Learning Representations*. https://openreview.net/forum?id=9XFSbDPmdW (V)
- NeurIPS. (2026). *NeurIPS paper checklist guidelines*. https://neurips.cc/public/guides/PaperChecklist (05)
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences, 115*(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114 (05)
- Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). Improving reproducibility in machine learning research (A report from the NeurIPS 2019 Reproducibility Program). *Journal of Machine Learning Research, 22*(164), 1–20. https://jmlr.org/papers/v22/20-303.html (05)
- Platanios, E. A., Stretcu, O., Neubig, G., Poczos, B., & Mitchell, T. (2019). Competence-based curriculum learning for neural machine translation. In *Proceedings of NAACL-HLT 2019* (Vol. 1, pp. 1162–1172). https://doi.org/10.18653/v1/N19-1119 (V)
- Portelas, R., Colas, C., Weng, L., Hofmann, K., & Oudeyer, P.-Y. (2020). Automatic curriculum learning for deep RL: A short survey. In *Proceedings of IJCAI-20* (pp. 4819–4825). https://doi.org/10.24963/ijcai.2020/671 (V)
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2201.02177 (V)
- Ramasesh, V. V., Dyer, E., & Raghu, M. (2021). Anatomy of catastrophic forgetting: Hidden representations and task semantics. In *International Conference on Learning Representations*. https://openreview.net/forum?id=LhY8QdUGSuw (V)
- Ramasesh, V. V., Lewkowycz, A., & Dyer, E. (2022). Effect of scale on catastrophic forgetting in neural networks. In *International Conference on Learning Representations*. https://openreview.net/forum?id=GhVS8_yPeEa (V)
- Ratcliff, R. (1990). Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions. *Psychological Review, 97*(2), 285–308. https://doi.org/10.1037/0033-295X.97.2.285 (V)
- Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318 (V)
- Rohrer, D., & Taylor, K. (2007). The shuffling of mathematics problems improves learning. *Instructional Science, 35*(6), 481–498. https://doi.org/10.1007/s11251-007-9015-8 (V)
- Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. *Advances in Neural Information Processing Systems, 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html (V)
- Russin, J., Pavlick, E., & Frank, M. J. (2025). Parallel trade-offs in human cognition and neural networks: The dynamic interplay between in-context and in-weight learning. *Proceedings of the National Academy of Sciences, 122*(35), e2510270122. https://doi.org/10.1073/pnas.2510270122 (V)
- Schaeffer, R., Miranda, B., & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? *Advances in Neural Information Processing Systems, 36*, 55565–55581. https://openreview.net/forum?id=JRdN9GcI52 (05)
- Scialom, T., Chakrabarty, T., & Muresan, S. (2022). Fine-tuned language models are continual learners. In *Proceedings of EMNLP 2022* (pp. 6107–6122). https://doi.org/10.18653/v1/2022.emnlp-main.410 (V)
- Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant. *Psychological Science, 22*(11), 1359–1366. https://doi.org/10.1177/0956797611417632 (V)
- Soviany, P., Ionescu, R. T., Rota, P., & Sebe, N. (2022). Curriculum learning: A survey. *International Journal of Computer Vision, 130*(6), 1526–1565. https://doi.org/10.1007/s11263-022-01611-x (V)
- Tirumala, K., Markosyan, A. H., Zettlemoyer, L., & Aghajanyan, A. (2022). Memorization without overfitting: Analyzing the training dynamics of large language models. *Advances in Neural Information Processing Systems, 35*, 38274–38290. https://arxiv.org/abs/2205.10770 (05)
- Transactions on Machine Learning Research. (n.d.). *Acceptance criteria*. https://jmlr.org/tmlr/acceptance-criteria.html (05, re-checked)
- van Miltenburg, E., van der Lee, C., & Krahmer, E. (2021). Preregistering NLP research. In *Proceedings of NAACL-HLT 2021* (pp. 613–623). https://doi.org/10.18653/v1/2021.naacl-main.51 (05)
- Wang, Z., Nichani, E., Bietti, A., Damian, A., Hsu, D., Lee, J. D., & Wu, D. (2025). Learning compositional functions with transformers from easy-to-hard data. In *Proceedings of the 38th Conference on Learning Theory* (PMLR 291, pp. 5632–5711). https://proceedings.mlr.press/v291/wang25a.html (V)
- Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., Mosquera, R., Paranjabe, B., Williams, A., Linzen, T., & Cotterell, R. (2023). Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning* (pp. 1–34). https://doi.org/10.18653/v1/2023.conll-babylm.1 (V)
- Williams, E. J. (1949). Experimental designs balanced for the estimation of residual effects of treatments. *Australian Journal of Scientific Research, Series A: Physical Sciences, 2*(2), 149–168. https://doi.org/10.1071/CH9490149 (V)
- Wu, X., Dyer, E., & Neyshabur, B. (2021). When do curricula work? In *International Conference on Learning Representations*. https://openreview.net/forum?id=tW4QEInpni (V)
- Xiong, Y., & Wu, S. (2025). Do large language models learn like humans: Interleaved and spaced practice in morphological learning. *Acta Psychologica, 260*, 105518. https://doi.org/10.1016/j.actpsy.2025.105518 (03)
- Yang, Y., Bean, A. M., McCraith, R., & Mahdi, A. (2024). *Evaluating fine-tuning efficiency of human-inspired learning strategies in medical question answering* [Preprint]. arXiv. https://arxiv.org/abs/2408.07888 (V)
- Yao, Y., Du, Y., Zhu, D., Hahn, M., & Koller, A. (2025). *Language models can learn implicit multi-hop reasoning, but only if they have lots of training data* [Preprint]. arXiv. https://arxiv.org/abs/2505.17923 (V, arXiv)
- Zaremba, W., & Sutskever, I. (2014). *Learning to execute* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1410.4615 (V)
- Zhang, Y., Backurs, A., Bubeck, S., Eldan, R., Gunasekar, S., & Wagner, T. (2022). *Unveiling transformers with LEGO: A synthetic reasoning task* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2206.04301 (V)
- Zhang-Li, D., Lin, N., Yu, J., Zhang, Z., Yao, Z., Zhang, X., Hou, L., Zhang, J., & Li, J. (2024). *Reverse that number! Decoding order matters in arithmetic learning* [Preprint]. arXiv. https://arxiv.org/abs/2403.05845 (V)

**Venue pages cited in Part C** (fetched 2026-10-04): https://iclr.cc/Conferences/2027/Dates · https://iclr.cc/Conferences/2027/CallForWorkshops · https://icml.cc/Conferences/2026/CallForPapers · https://icml.cc/Conferences/FutureMeetings · https://neurips.cc/Conferences/2026/Dates · https://colm.cc/Conferences/2026/CallForPapers · https://aclrollingreview.org/dates · https://2027.naacl.org/calls/main_conference_papers/ · https://2027.aclweb.org/calls/main/ · https://jmlr.org/tmlr/author-guide.html · https://cognitivesciencesociety.org/cogsci-2027/ · https://cognitivesciencesociety.org/submissions/ · https://2027.ccneuro.org/ · https://2026.ccneuro.org/call-for-papers/ · https://virtual.aistats.org/Conferences/2027/CallForPapers · https://aaai.org/conference/aaai/aaai-27/submission-instructions/ · https://babylm.github.io/ · https://link.springer.com/journal/42113 · https://www.cos.io/initiatives/registered-reports
