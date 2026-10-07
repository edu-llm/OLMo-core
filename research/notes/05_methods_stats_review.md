# Methods, Statistics and Mock Review: P4 "Spaced Review" Whitepaper

## TL;DR

- **Mock "Reviewer 2" (Part B): 9 major issues.**
  1. The schedule isn't applied per fact.
  2. Cross-entropy may not measure knowledge.
  3. "Indistinguishable" needs an equivalence test (TOST).
  4. "Starting point, not forgetting rate" depends on the scale.
  5. Facts are treated as fixed, and only seeds vary.
  6. Missing controls: never-trained, massed, generic-data, and additive review.
  7. Three seeds, many metrics, no primary metric.
  8. Overgeneralised human analogy.
  9. Missing related work (FOREVER, SRT, Tirumala).
- **Good news:** the expanding-vs-uniform equivalence probably survives a proper TOST: the 90% CI is about [−0.0038, +0.0026] nats. Review is worth more than 180 updates of delay, a "horizontal" comparison the paper understates.
- **Statistics references (Part A):** equivalence testing (Lakens), mixed models with items as random effects (Clark 1973; Baayen; Barr), forgetting-curve comparison (Loftus 1985), seed variance (Agarwal; Bouthillier) and preregistration.
- **Venues:**
  - TMLR is the best fit after revision.
  - The ACL Rolling Review deadline (12 Oct 2026) is too soon.
  - CogSci or CCN 2027 fit only if the human comparison is made rigorous.
  - ICLR 2027 and NeurIPS 2026 are closed.


**Prepared:** 2026-10-02 (reviewer-side research aid; the paper itself was not edited)

**Paper reviewed:** `/Users/amylin/Amy/MIT/Internships/Alpha_AI_Engineering/projects/edu-llm/paper/source/P4_whitepaper_writeup.md` (Mago & Wang)

**Code inspected:** https://github.com/anshulmago1/olmo-fictionalqa-review (a temporary local copy used during review has since been deleted; the same code is on edu-llm/OLMo-core, branch anshulm/fictionalqa-review)
- `configs/final.yaml`
- `src/olmo_core/review_lab/schedules.py`, `trainer.py`, `fictionalqa.py`, `report.py`

**Data inspected:** HF `jwkirchenbauer/fictionalqa`, `fict_qa` config, 7,500 rows. Used to reconstruct the old/new split and the per-style pool sizes.

**Verification policy:**
- Every reference was checked against a primary or authoritative source: publisher or DOI page, ACL Anthology, PMLR, NeurIPS/MLSys/AAAI proceedings, OpenReview, arXiv or PubMed.
- Verification used parallel search agents plus direct arXiv metadata fetches.
- **UNVERIFIED** marks any field that could not be confirmed.
- Citations are in APA 7 format. Each has a working URL, a DOI where one exists, and otherwise a proceedings, OpenReview or arXiv link.
- Where the full text could not be read (old scanned PDFs), the summary relies on the abstract plus secondary descriptions, and this is flagged.

## Contents
- **Part A: Methodology literature.**
  - A1. Equivalence testing.
  - A2. Seed variance in ML.
  - A3. Item-level analysis.
  - A4. Forgetting-rate comparison.
  - A5. Metric validity.
  - A6. Reporting standards.
  - A7. Positioning literature.
- **Part B: Mock peer review (Reviewer 2)**, with venues.
- **Appendix: Quantitative checks** the reviewer ran on the paper's numbers and code.

## Key numbers reconstructed from the paper and code

Used throughout this document.

- **CI method.** The paired CIs are consistent with t-intervals with df = 2 (t₀.₉₇₅ = 4.303).
  - Expanding − uniform: SE ≈ 0.0011 nats, SD of paired differences ≈ 0.0019.
  - Uniform − no review: SE ≈ 0.0096.
  - Buffer-forgetting difference: SE ≈ 0.019.
- **Review schedules** (Stage-2 step; first and last review are matched):
  - Uniform: 8, 22, 37, 51, 65, 79, 94, 108, 122, 136, 151, 165.
  - Expanding (ratio 1.35): 8, 10, 13, 17, 22, 29, 38, 51, 68, 91, 123, 165.
- **What a review event is.** One optimizer step of 16 old-fact statements, sampled with replacement from one style. Styles are visited round-robin: blog, corporate, encyclopedia, news, social, blog, …
  - Styles therefore get 3, 3, 2, 2 and 2 reviews.
  - The style's last review in expanding vs. uniform: encyclopedia 51 vs. 108, news 68 vs. 122, social 91 vs. 136. Blog (123 vs. 151) and corporate (165 vs. 165) also differ, except corporate.
- **Old training pool:** 386 canonical statements from 20 events. By style: blog 28, corporate 54, encyclopedia 41, news 180, social 83.
- **Expected review exposures per fact:** blog 1.71, corporate 0.89, encyclopedia 0.78, news 0.18, social 0.39.
- **P(fact never reviewed):** 0.18, 0.41, 0.45, 0.84, 0.68. That averages ≈51% of the evaluated facts.
- **Stage 1:** 192 draws per style. News facts average ≈1.1 exposures, and P(never seen in Stage 1) ≈ 0.34 for news (≈10% of all evaluated facts).
- **Evaluation set:** 16 items per style = 80 old items and 80 new items, identical across seeds and conditions.
- **End-of-Stage-1 old loss** (final − "old-loss change") ≈ 4.204 (no review), 4.205 (uniform), 4.204 (expanding). Every arm finishes the buffer *below* its end-of-Stage-1 loss.
- **Probability scale.** Geometric-mean answer-token probability over the buffer: no review 0.0217 → 0.0187 (Δ = −0.0030); uniform 0.0260 → 0.0222 (Δ = −0.0039).
- **Review effect as a probability ratio:** e^0.173 ≈ 1.19× higher geometric-mean per-token probability.
- **Horizontal comparison.** Review arms at buffer step 180 (3.809) are still below no review at buffer onset (3.830), so review is worth more than 180 updates of delay.

# Part A — Methodology literature

## A1. Equivalence testing and concluding "no difference"

**Why the paper needs this.** The paper concludes that expanding and uniform review are "statistically and practically indistinguishable" from a 95% paired CI of [−0.0053, +0.0041] nats, and that forgetting rates do not differ from CIs that include zero (half-width ≈0.08 nats). A CI that spans zero is absence of evidence, not evidence of absence. To claim equivalence, the authors need a pre-specified smallest effect size of interest (SESOI) and either an equivalence test (TOST), a Bayesian ROPE analysis, or a Bayes factor.

**Worked check (reviewer's own arithmetic).** This assumes the reported CIs are t-based with df = 2 (t₀.₉₇₅ = 4.303), which is consistent with the reported numbers.
- For expanding − uniform, SE ≈ 0.0047 / 4.303 ≈ 0.0011 nats. The TOST 90% CI (t₀.₉₅ = 2.920) is ≈ [−0.0038, +0.0026].
- Equivalence would be declared for any symmetric bound Δ ≳ 0.004 nats, which is about 2–3% of the review-vs-no-review effect (0.173). **The expanding-vs-uniform equivalence claim is probably rescuable**, provided Δ is justified independently of the data.
- **The "same forgetting rate" claim is not rescuable.** Its 95% CI half-width (≈0.08 nats) is about half of the forgetting itself.
- With df = 2, the SD estimate is itself very imprecise: the 95% CI for σ is ≈ [0.52σ̂, 6.3σ̂].

**References (APA 7):**

1. Lakens, D. (2017). Equivalence tests: A practical primer for *t* tests, correlations, and meta-analyses. *Social Psychological and Personality Science, 8*(4), 355–362. https://doi.org/10.1177/1948550617697177. **Verified.**
   - Introduces TOST to psychologists. Set upper and lower equivalence bounds from a SESOI, then run two one-sided tests. Equivalence is supported when both reject, which is the same as the 90% CI lying inside the bounds.
   - Covers power for equivalence tests and provides the TOSTER package.
   - Directly applicable to the expanding − uniform contrast.

2. Lakens, D., Scheel, A. M., & Isager, P. M. (2018). Equivalence testing for psychological research: A tutorial. *Advances in Methods and Practices in Psychological Science, 1*(2), 259–269. https://doi.org/10.1177/2515245918770963. **Verified.**
   - A tutorial centred on justifying the SESOI. Options include theoretical or objective bounds, anchor-based (subjective) bounds, and resource-based bounds.
   - Also gives reporting guidance.
   - The best single citation for "pre-specify and justify Δ". Here Δ could be tied to the loss change that corresponds to a one-point change in exact match, or to a fraction of the review effect.

3. Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics, 15*(6), 657–680. https://doi.org/10.1007/BF01068419. **Verified.**
   - The origin of TOST.
   - Shows that TOST at α = .05 is equivalent to checking whether the 1 − 2α (90%) CI lies within the equivalence limits.
   - Shows that TOST is generally superior to the "power approach" (non-significant + adequate power ⇒ equivalent).

4. Anvari, F., & Lakens, D. (2021). Using anchor-based methods to determine the smallest effect size of interest. *Journal of Experimental Social Psychology, 96*, Article 104159. https://doi.org/10.1016/j.jesp.2021.104159. **Verified.**
   - Anchors the SESOI to an external criterion of meaningful change rather than to effect-size conventions.
   - In this paper the analogue is anchoring Δ to a behavioural criterion, such as the loss change that flips a fact from recalled to not recalled.

5. Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian estimation. *Advances in Methods and Practices in Psychological Science, 1*(2), 270–280. https://doi.org/10.1177/2515245918771304. **Verified.**
   - The HDI + ROPE decision rule: accept the null for practical purposes if the 95% HDI lies entirely inside a region of practical equivalence; reject it if the HDI lies entirely outside; otherwise withhold judgment.
   - A Bayesian alternative to TOST that forces an explicit ROPE.

6. Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D., & Iverson, G. (2009). Bayesian *t* tests for accepting and rejecting the null hypothesis. *Psychonomic Bulletin & Review, 16*(2), 225–237. https://doi.org/10.3758/PBR.16.2.225. **Verified.**
   - Default JZS Bayes-factor *t* test (Cauchy prior on effect size). It can quantify evidence *for* the null.
   - With n = 3, BF₀₁ will usually be modest. Only a BF clearly favouring H0 supports "no difference".

7. Altman, D. G., & Bland, J. M. (1995). Absence of evidence is not evidence of absence. *BMJ, 311*(7003), 485. https://doi.org/10.1136/bmj.311.7003.485. **Verified.**
   - The canonical short note: a non-significant result usually shows only an absence of evidence, often because the study lacks power. Readers should look at the estimate and its CI.

8. Greenland, S., Senn, S. J., Rothman, K. J., Carlin, J. B., Poole, C., Goodman, S. N., & Altman, D. G. (2016). Statistical tests, *P* values, confidence intervals, and power: A guide to misinterpretations. *European Journal of Epidemiology, 31*(4), 337–350. https://doi.org/10.1007/s10654-016-0149-3. **Verified.**
   - Catalogues 25 misinterpretations, including "a CI containing the null shows the null is true" and post hoc power reasoning.
   - Recommends interpreting the full range of effect sizes compatible with the data.
   - The exact numbering of individual items was not re-checked.

9. Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on *p*-values: Context, process, and purpose. *The American Statistician, 70*(2), 129–133. https://doi.org/10.1080/00031305.2016.1154108. **Verified.**
   - Six principles, including that larger p-values "do not imply a lack of importance or even lack of effect".

## A2. Seed variance and statistical practice in ML

**Why the paper needs this.** Three seeds, t-based CIs with df = 2, and a "no difference" claim are all exactly what the recent ML-statistics literature warns against. In this design a seed varies only LoRA initialization, data-sampling order and dropout. The fact split and the evaluation items are fixed across seeds (data seed 20260721), so the CIs capture only one of the variance sources that Bouthillier et al. (2021) recommend randomizing.

10. Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Sepah, N., Raff, E., Madan, K., Voleti, V., Kahou, S. E., Michalski, V., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems, 3*, 747–769. https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html (arXiv:2103.03098; the arXiv version also lists D. Serdyuk). **Verified.**
   - Recommends randomizing as many sources of variation as possible: initialization, data sampling and order, augmentation, and HPO.
   - Recommends a decision rule that combines statistical significance with a "meaningful" probability of improvement, P(A > B) ≥ 0.75. Detecting this reliably needs on the order of 29 runs.
   - Advises "in doubt, it is better to pair".
11. Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. In *Advances in Neural Information Processing Systems 34* (pp. 29304–29320). https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html (arXiv:2108.13264). **Verified.** NeurIPS 2021 Outstanding Paper Award.
   - Point estimates from few runs are unreliable, and claims made "with 3 or fewer runs may naturally raise eyebrows".
   - Recommends stratified-bootstrap interval estimates, performance profiles, the interquartile mean (IQM) and probability of improvement, all implemented in the `rliable` library.
   - Caveat for this paper: a bootstrap over 3 seeds is itself unreliable. The stratified bootstrap gets its power from pooling across many tasks.
12. Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep reinforcement learning that matters. *Proceedings of the AAAI Conference on Artificial Intelligence, 32*(1), 3207–3214. https://doi.org/10.1609/aaai.v32i1.11694. **Verified.**
   - Two groups of 5 seeds with identical hyperparameters produced significantly different learning curves.
   - Recommends reporting the number of trials, significance tests and bootstrap CIs.
13. Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., & Smith, N. (2020). *Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2002.06305. **Verified** (arXiv only).
   - Across 2,100 BERT fine-tuning runs, weight initialization and data order contributed comparably to large seed variance.
   - Directly relevant: here the seed controls both of these.
14. Sellam, T., Yadlowsky, S., Tenney, I., Wei, J., Saphra, N., D'Amour, A., Linzen, T., Bastings, J., Turc, I., Eisenstein, J., Das, D., & Pavlick, E. (2022). The MultiBERTs: BERT reproductions for robustness analysis. In *International Conference on Learning Representations*. https://openreview.net/forum?id=K0E_F0gFDgA (arXiv:2106.16163). **Verified.** Author order follows the ICLR version; arXiv lists Tenney eleventh.
   - Proposes the **Multi-Bootstrap**, which jointly resamples seeds *and* test examples, so the CI reflects both sources of variance.
   - The single most directly applicable fix for this paper's seed-only CIs.
15. Madaan, L., Singh, A. K., Schaeffer, R., Poulton, A., Koyejo, S., Stenetorp, P., Narang, S., & Hupkes, D. (2024). *Quantifying variance in evaluation benchmarks* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2406.10229. **Verified** (arXiv preprint, no venue).
   - Seed variance across 10 7B-model seeds is substantial, so improvements should exceed it.
   - Continuous (NLL-based) metrics have much higher signal-to-noise ratios than discrete accuracy. This supports CE as a sensitive metric, though not as a valid one on its own.
16. Miller, E. (2024). *Adding error bars to evals: A statistical approach to language model evaluations* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2411.00640. **Verified** (arXiv preprint; Anthropic).
   - Treats questions as draws from a super-population. Recommends CLT standard errors, **clustered SEs** for related questions, variance reduction via next-token probabilities, **paired question-level differences** and **power analysis**.
   - Here the clusters are FictionalQA events and styles.
17. Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). The hitchhiker's guide to testing statistical significance in natural language processing. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1383–1392). https://doi.org/10.18653/v1/P18-1128. **Verified.**
   - A decision tree for choosing tests: parametric tests such as the paired t-test when assumptions hold, permutation or bootstrap tests otherwise.
   - Documents widespread misuse of significance testing in NLP.
18. Dror, R., Shlomov, S., & Reichart, R. (2019). Deep dominance — How to properly compare deep neural models. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 2773–2785). https://doi.org/10.18653/v1/P19-1266. **Verified.**
   - Compares full score distributions across seeds with an almost-stochastic-dominance test.
19. Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2018). *How many random seeds? Statistical power analysis in deep reinforcement learning experiments* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1806.08295. **Verified** (arXiv).
    - Recommends Welch's t-test, avoiding bootstrap tests with fewer than ~20 samples, setting α below .05 for small samples, correcting for multiple comparisons, and running pilot studies with ≥20 runs to estimate SD.
    - Example: with N = 5, the Type II error rate was 51% for the observed effect.
20. Reimers, N., & Gurevych, I. (2017). Reporting score distributions makes a difference: Performance study of LSTM-networks for sequence tagging. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing* (pp. 338–348). https://doi.org/10.18653/v1/D17-1035. **Verified.**
    - Seed alone produced significant differences. Report distributions, not single scores.
21. Card, D., Henderson, P., Khandelwal, U., Jia, R., Mahowald, K., & Jurafsky, D. (2020). With little power comes great responsibility. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing* (pp. 9263–9274). https://doi.org/10.18653/v1/2020.emnlp-main.745. **Verified.**
    - Underpowered NLP comparisons are common. Significant results from underpowered studies exaggerate effects (Type M error) or get their sign wrong (Type S error).
    - Recommends prospective power analysis and warns against post hoc "observed power".
22. Bui, N., Savova, G., & Wang, L. (2025). Assessing the macro and micro effects of random seeds on fine-tuning large language models. In *Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the ACL (Volume 2: Short Papers)* (pp. 41–46). https://doi.org/10.18653/v1/2025.ijcnlp-short.3. **Verified.** The proceedings title is reconstructed from the ACL Anthology ID; check the exact wording.
    - Large seed variance in LLM fine-tuning, including LoRA. Introduces a per-prediction "consistency" measure, since runs with equal aggregate scores can differ item by item.
23. Heineman, D., Hofmann, V., Magnusson, I., Gu, Y., Smith, N. A., Hajishirzi, H., Lo, K., & Dodge, J. (2025). Signal and noise: A framework for reducing uncertainty in language model evaluation. In *Advances in Neural Information Processing Systems 38*. https://arxiv.org/abs/2508.13144. **Verified** (arXiv). The NeurIPS 2025 proceedings DOI and pages were not checked on neurips.cc.
    - Evaluation decisions are more reliable on high signal-to-noise benchmarks, and continuous metrics improve SNR.
    - Written by the DataDecide/OLMo group, so especially apt here.
24. Hochlehnert, A., Bhatnagar, H., Udandarao, V., Albanie, S., Prabhu, A., & Bethge, M. (2025). A sober look at progress in language model reasoning: Pitfalls and paths to reproducibility. In *Second Conference on Language Modeling (COLM 2025)*. https://arxiv.org/abs/2504.07086. **Verified** (arXiv comment "Accepted to COLM 2025").
    - Single-seed LLM results hide large variance from seeds, decoding and hardware.
25. Picard, D. (2021). *Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision* [Preprint]. arXiv. https://arxiv.org/abs/2109.08203. **Verified** (arXiv; informal, not peer-reviewed). Cite only as an illustration.

## A3. Item-level analysis: facts as a random factor

**Why the paper needs this.** The final evaluation uses the same 80 old-fact questions (16 per style, from 20 events) in every seed and condition. The analysis averages over them and treats 3 seeds as the only replicates. Inference therefore generalizes only to "these 80 questions". Per-item losses are already logged by the code, so crossed seed × item mixed models (or cluster bootstraps over events/items) can be fitted without retraining.

**References (APA 7):**

26. Clark, H. H. (1973). The language-as-fixed-effect fallacy: A critique of language statistics in psychological research. *Journal of Verbal Learning and Verbal Behavior, 12*(4), 335–359. https://doi.org/10.1016/S0022-5371(73)80014-3. **Verified.**
    - Generalizing from a sample of linguistic items to the population of items requires treating items as random. Clark proposes min F′, which combines by-subject and by-item tests.
    - The facts here play the role of Clark's words.

27. Baayen, R. H., Davidson, D. J., & Bates, D. M. (2008). Mixed-effects modeling with crossed random effects for subjects and items. *Journal of Memory and Language, 59*(4), 390–412. https://doi.org/10.1016/j.jml.2007.12.005. **Verified.**
    - Introduces LMMs with crossed random effects (lme4) and shows advantages over F1/F2/min F′.
    - Map seeds to "subjects" and facts to "items". For example: `loss ~ condition * time + (1 + condition | fact) + (1 | event) + (1 | seed)`.

28. Barr, D. J., Levy, R., Scheepers, C., & Tily, H. J. (2013). Random effects structure for confirmatory hypothesis testing: Keep it maximal. *Journal of Memory and Language, 68*(3), 255–278. https://doi.org/10.1016/j.jml.2012.11.001. **Verified.**
    - For confirmatory tests, include by-item and by-subject random slopes for the effect of interest.
    - Random-intercept-only models can badly inflate Type I error.

29. Matuschek, H., Kliegl, R., Vasishth, S., Baayen, H., & Bates, D. (2017). Balancing Type I error and power in linear mixed models. *Journal of Memory and Language, 94*, 305–315. https://doi.org/10.1016/j.jml.2017.01.001. **Verified.**
    - A counterpoint to Barr et al.: maximal models lose power when the data do not support them, so random-effects structure should be selected parsimoniously.
    - Relevant because by-seed random slopes are barely estimable with 3 seeds.

30. Judd, C. M., Westfall, J., & Kenny, D. A. (2012). Treating stimuli as a random factor in social psychology: A new and comprehensive solution to a pervasive but largely ignored problem. *Journal of Personality and Social Psychology, 103*(1), 54–69. https://doi.org/10.1037/a0028347. **Verified.**
    - Ignoring stimulus variance inflates Type I error, sometimes severely. Crossed-random-effects models are the fix.

31. Westfall, J., Kenny, D. A., & Judd, C. M. (2014). Statistical power and optimal design in experiments in which samples of participants respond to samples of stimuli. *Journal of Experimental Psychology: General, 143*(5), 2020–2045 (issue number not independently verified). https://doi.org/10.1037/xge0000014. **Verified** (DOI and venue confirmed by the verification agent).
    - Power depends on the number of stimuli as well as participants. Supports evaluating on all 386 old facts rather than 80.

32. Lalor, J. P., Wu, H., & Yu, H. (2016). Building an evaluation scale using item response theory. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing* (pp. 648–657). Association for Computational Linguistics. https://doi.org/10.18653/v1/D16-1062. **Verified.**
    - Uses IRT to model item difficulty and discrimination in NLP evaluation. An ML-side precedent for item-level modelling.

(Miller, 2024, and Card et al., 2020, listed in A2, are also directly relevant here. Miller covers clustered SEs for non-independent items and paired per-item differences; Card et al. cover statistical power.)

## A4. Comparing forgetting rates

**Why the paper needs this.** "Review changes the starting point, not the forgetting rate" rests on equal *absolute* loss increases over the buffer, measured between two time points.

- **Vertical comparisons are scale-dependent.** Mean answer-token NLL is a log scale, so equal absolute increases mean equal *proportional* drops in geometric-mean token probability.
  - On the probability scale, no-review drops by ≈0.0030 (0.0217 → 0.0187) and uniform by ≈0.0039 (0.0260 → 0.0222). On that scale the review arm forgets *more*.
- **A horizontal (time-to-criterion) comparison is scale-invariant.** Uniform at buffer step 180 (3.809) is still better than no-review at buffer onset (3.830), so review is worth more than 180 updates of delay.
- **A condition × time interaction is the right test.** Non-crossover interactions can be removed by monotone transforms, so conclusions must be checked across scales.

**References (APA 7):**

33. Loftus, G. R. (1985). Evaluating forgetting curves. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 11*(2), 397–406. https://doi.org/10.1037/0278-7393.11.2.397. **Verified** (bibliographic details and abstract; the full-text scan was not machine-readable, so do not quote verbatim).
    - Slamecka & McElree's "degree of learning does not affect forgetting" (parallel curves) holds only if forgetting is defined as the *vertical* drop between two delays. Vertical parallelism is not invariant to monotone rescaling of performance.
    - Loftus proposes a *horizontal* criterion: the time needed for performance to fall from one level to another. It is invariant to any monotone transform.
    - Applying it, he concluded that higher degrees of learning are forgotten more slowly.
    - The direct template for re-analysing §3.3.

34. Slamecka, N. J., & McElree, B. (1983). Normal forgetting of verbal lists as a function of their degree of learning. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 9*(3), 384–397. https://doi.org/10.1037/0278-7393.9.3.384. **Verified.**
    - More study trials raised forgetting-curve intercepts but not slopes. This is the classic human "starting point, not rate" result, which the paper's §3.3 echoes and should cite together with Loftus's critique.

35. Loftus, G. R. (1978). On interpretation of interactions. *Memory & Cognition, 6*(3), 312–319. https://doi.org/10.3758/BF03197461. **Verified.**
    - Non-crossover interactions on an observed measure can be created or removed by monotone transformations, so they are not interpretable as latent-process interactions without scale assumptions.
    - A "same/different forgetting rate" claim is a condition × time interaction.

36. Wagenmakers, E.-J., Krypotos, A.-M., Criss, A. H., & Iverson, G. (2012). On the interpretation of removable interactions: A survey of the field 33 years after Loftus. *Memory & Cognition, 40*(2), 145–160. https://doi.org/10.3758/s13421-011-0158-0. **Verified.**
    - Researchers still routinely interpret removable interactions. Check whether conclusions survive plausible monotone transforms.

37. Loftus, G. R., & Bamber, D. (1990). Learning–forgetting independence, unidimensional memory models, and feature models: Comment on Bogartz (1990). *Journal of Experimental Psychology: Learning, Memory, and Cognition, 16*(5), 916–926. https://doi.org/10.1037/0278-7393.16.5.916. **Verified** (existence and bibliographic details; summary from the abstract).
    - Formalizes when horizontally parallel forgetting curves imply learning–forgetting independence under unidimensional memory models.

38. Wixted, J. T. (1990). Analyzing the empirical course of forgetting. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 16*(5), 927–935. https://doi.org/10.1037/0278-7393.16.5.927. **Verified** (bibliographic details; the summary partly relies on secondary descriptions).
    - Shows how the choice between performance and latent-strength scales determines whether forgetting curves appear parallel.

39. Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting. *Psychological Science, 2*(6), 409–415. https://doi.org/10.1111/j.1467-9280.1991.tb00175.x. **Verified.**
    - Power functions of time fit forgetting better than exponential and other forms across three paradigms.
    - Under power-law forgetting, two-point absolute differences depend strongly on where the points fall.
    - Also relevant to the human-analogy claim: the paper's buffer curve is roughly linear in loss (≈0.0011, 0.0008, 0.0008 nats/update over the successive windows), not power-like.

40. Wixted, J. T., & Carpenter, S. K. (2007). The Wickelgren power law and the Ebbinghaus savings function. *Psychological Science, 18*(2), 133–134. https://doi.org/10.1111/j.1467-9280.2007.01862.x. **Verified.**
    - Presents P = m(1 + ht)^(−f), which separates degree of learning (m) from forgetting rate (f).
    - Fitting such a function per condition (or per fact) and comparing f is a cleaner test than a two-point difference.

41. Wixted, J. T. (2004). The psychology and neuroscience of forgetting. *Annual Review of Psychology, 55*, 235–269. https://doi.org/10.1146/annurev.psych.55.090902.141555. **Verified.**
    - Background on interference and consolidation accounts of forgetting. Useful for the human-analogy discussion. Of limited direct methodological use.

42. Bamber, D. (1979). State-trace analysis: A method of testing simple theories of causation. *Journal of Mathematical Psychology, 19*(2), 137–181. https://doi.org/10.1016/0022-2496(79)90016-6. **Verified.**
    - A scale-free method that plots one dependent variable against another to test whether one latent dimension suffices.
    - Here, plot early vs. late buffer loss across facts or checkpoints for each arm and ask whether the arms fall on one monotone curve.

43. Rivera-Lares, K., Logie, R., Baddeley, A., & Della Sala, S. (2022). Rate of forgetting is independent of initial degree of learning. *Memory & Cognition, 50*(8), 1706–1718. https://doi.org/10.3758/s13421-021-01271-1. **Verified.**
    - A modern replication that explicitly contrasts the Slamecka–McElree (vertical) and Loftus (horizontal) operationalizations.
    - Shows that the conclusion depends on how forgetting rate is operationalized.

## A5. Metric validity: cross-entropy vs. behaviour

**Why the paper needs this.** The primary outcome is answer-token CE (including EOS) on QA probes, after training only on declarative `"Fictional fact: …"` statements.

- *For* CE: it is continuous and sensitive, with high SNR (Schaeffer et al., 2023; Madaan et al., 2024; Heineman et al., 2025). Log-probability on paraphrased probes has been used to track fictional-knowledge retention in OLMo (Chang et al., 2024).
- *Against* treating CE as "retained knowledge": loss and log-probability changes need not translate into behavioural changes (Schaeffer et al., 2025; Du et al., 2024; Lyu et al., 2024; Wang et al., 2024). Knowledge stored from declarative text may not be extractable in QA form without augmentation (Allen-Zhu & Li, 2024).
- The literature therefore supports **reporting CE together with a behavioural metric**, such as exact match (already computed in the code but unreported) or MCQ accuracy (available in FictionalQA). It also supports **normalizing CE against a never-trained baseline**.
- No verified source endorses relative % reduction of CE. The reviewer's argument against it is first-principles (CE has no natural zero). Effects are better expressed as nats, as probability ratios (e^Δ), or as a fraction of the Stage-1 learning gain.

44. Schaeffer, R., Miranda, B., & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? In *Advances in Neural Information Processing Systems 36* (pp. 55565–55581). https://openreview.net/forum?id=JRdN9GcI52 (arXiv:2304.15004). **Verified.** NeurIPS 2023 Outstanding Paper.
    - Apparent emergence arises from nonlinear or discontinuous metrics such as exact match. Continuous metrics show smooth change.
    - Implication here: CE can register changes that exact match would not show. A difference in CE therefore does not imply a difference in recall.
45. Schaeffer, R., Schoelkopf, H., Miranda, B., Mukobi, G., Madan, V., Ibrahim, A., Bradley, H., Biderman, S., & Koyejo, S. (2025). Why has predicting downstream capabilities of frontier AI models with scale remained elusive? In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267, pp. 53177–53266). https://proceedings.mlr.press/v267/schaeffer25b.html (arXiv:2406.04391). **Verified.**
    - Log-probability of the correct answer tracks compute well. Each transformation toward accuracy (probability, renormalization against distractors, thresholding) degrades that relationship, because accuracy depends on the mass placed on specific alternatives.
46. Du, Z., Zeng, A., Dong, Y., & Tang, J. (2024). Understanding emergent abilities of language models from the loss perspective. In *Advances in Neural Information Processing Systems 37* (pp. 53138–53167). https://arxiv.org/abs/2403.15796. **Verified.**
    - Some abilities stay at chance until pretraining loss crosses a threshold, "regardless of the continuity of metrics". So loss changes need not translate into behavioural change.
47. Hu, J., & Levy, R. (2023). Prompting is not a substitute for probability measurements in large language models. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (pp. 5040–5060). https://doi.org/10.18653/v1/2023.emnlp-main.306. **Verified.**
    - Probability measurements reveal more than prompted outputs do. This supports using likelihood as *one* measure of knowledge.
48. Lyu, C., Wu, M., & Aji, A. F. (2024). Beyond probabilities: Unveiling the misalignment in evaluating large language models. In *Proceedings of the 1st Workshop on Towards Knowledgeable Language Models (KnowLLM 2024)* (pp. 109–131). https://doi.org/10.18653/v1/2024.knowllm-1.10. **Verified** (ACL 2024 workshop paper).
    - Probability-based evaluation is substantially misaligned with generation-based answers.
49. Wang, X., Ma, B., Hu, C., Weber-Genzel, L., Röttger, P., Kreuter, F., Hovy, D., & Plank, B. (2024). "My answer is C": First-token probabilities do not match text answers in instruction-tuned language models. In *Findings of the Association for Computational Linguistics: ACL 2024* (pp. 7407–7416). https://doi.org/10.18653/v1/2024.findings-acl.441. **Verified** (via ACL Anthology search results and arXiv metadata).
50. Tsvilodub, P., Wang, H., Grosch, S., & Franke, M. (2024). *Predictions from language models for multiple-choice tasks are not robust under variation of scoring methods* [Preprint]. arXiv. https://arxiv.org/abs/2403.00998. **Verified** (arXiv preprint).
    - Item-level conclusions change with the scoring method, which creates researcher degrees of freedom.
51. Allen-Zhu, Z., & Li, Y. (2024). Physics of language models: Part 3.1, knowledge storage and extraction. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR 235, pp. 1067–1077). https://proceedings.mlr.press/v235/allen-zhu24a.html. **Verified.**
    - Facts memorized from biographies can have ~0% QA extractability unless the pretraining data are augmented (paraphrase, shuffling).
    - Directly relevant to statement-only training evaluated with QA probes.
52. Chang, H., Park, J., Ye, S., Yang, S., Seo, Y., Chang, D.-S., & Seo, M. (2024). How do large language models acquire factual knowledge during pretraining? In *Advances in Neural Information Processing Systems 37* (pp. 60626–60668). https://arxiv.org/abs/2406.11813. **Verified.**
    - Injects fictional knowledge into OLMo checkpoints and tracks target-span log-probability on memorization, semantic (paraphrase) and compositional probes.
    - Findings: knowledge accumulates in small per-exposure increments, forgetting follows a power law in steps, duplicated exposure forgets faster than paraphrased exposure, and larger batches resist forgetting.
    - The closest methodological precedent, and its findings about duplication and paraphrase bear directly on verbatim review.
53. Ovadia, O., Brief, M., Mishaeli, M., & Elisha, O. (2024). Fine-tuning or retrieval? Comparing knowledge injection in LLMs. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 237–250). https://doi.org/10.18653/v1/2024.emnlp-main.15. **Verified.**
    - Uses MCQ log-likelihood accuracy. Training on several paraphrases per fact improved fine-tuning injection.
54. Gekhman, Z., Yona, G., Aharoni, R., Eyal, M., Feder, A., Reichart, R., & Herzig, J. (2024). Does fine-tuning LLMs on new knowledge encourage hallucinations? In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 7765–7784). https://doi.org/10.18653/v1/2024.emnlp-main.444. **Verified.**
    - Measured with exact match: new ("Unknown") facts are learned slowly and, once fitted, increase hallucination.
    - A behavioural-metric precedent for knowledge-injection studies.
55. Berglund, L., Tong, M., Kaufmann, M., Balesni, M., Stickland, A. C., Korbak, T., & Evans, O. (2024). The reversal curse: LLMs trained on "A is B" fail to learn "B is A". In *International Conference on Learning Representations*. https://arxiv.org/abs/2309.12288. **Authors and title verified on arXiv.** The ICLR 2024 venue is UNVERIFIED (the arXiv comment does not state it); confirm it on OpenReview.
    - Relevant because training on statements and probing with differently structured questions is a format/direction transfer.

## A6. Reporting standards and preregistration

56. NeurIPS. (2026). *NeurIPS paper checklist guidelines*. https://neurips.cc/public/guides/PaperChecklist. **Verified** (2026 version).
   - Leaving the checklist out means desk rejection.
   - **Q7, statistical significance:** state the sources of variability captured (split, initialization, run), how error bars or CIs were computed (closed form, library, bootstrap), the assumptions (e.g., normality), and whether bars show SD or SEM.
   - **Q8, compute:** worker type, memory, compute per run and total, including preliminary or failed runs.
   - **Q6, details:** data splits and how hyperparameters were chosen.
   - What this means for the paper: it currently fails Q7 (the CI method and the variability sources are unstated) and Q8.
57. Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). Improving reproducibility in machine learning research (A report from the NeurIPS 2019 Reproducibility Program). *Journal of Machine Learning Research, 22*(164), 1–20. https://jmlr.org/papers/v22/20-303.html (arXiv:2003.12206). **Verified.**
   - The origin and analysis of the ML Reproducibility Checklist, the code-submission policy and the reproducibility challenge.
58. Bertinetto, L., Henriques, J. F., Albanie, S., Paganini, M., & Varol, G. (Eds.). (2021). *NeurIPS 2020 Workshop on Pre-registration in Machine Learning* (Proceedings of Machine Learning Research, Vol. 148). PMLR. https://proceedings.mlr.press/v148/. **Verified.**
   - A second edition followed as PMLR Vol. 181 (2022): https://proceedings.mlr.press/v181/.
   - In this model, a paper's protocol is peer-reviewed before results exist.
59. van Miltenburg, E., van der Lee, C., & Krahmer, E. (2021). Preregistering NLP research. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp. 613–623). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.naacl-main.51. **Verified.**
   - Provides preregistration question sets for NLP experiments and argues that registered reports are feasible in NLP.
60. Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences, 115*(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114. **Verified.**
   - Preregistration separates confirmatory (prediction) from exploratory (postdiction) analyses.
   - Relevant here because the post hoc dismissal of the trajectory-metric difference is a postdiction.
61. Chambers, C. D. (2013). Registered Reports: A new publishing initiative at *Cortex*. *Cortex, 49*(3), 609–610. https://doi.org/10.1016/j.cortex.2012.12.016. **PARTLY UNVERIFIED.** The DOI resolves to *Cortex*, but the title, volume and pages are from memory and the page did not load.
62. Transactions on Machine Learning Research. (n.d.). *Acceptance criteria*. https://jmlr.org/tmlr/acceptance-criteria.html. **Verified.**
   - Two criteria: (1) claims are "supported by accurate and convincing evidence", and (2) some of TMLR's audience would be interested.
   - Novelty is explicitly not required, so a rigorous negative result fits.

## A7. Positioning literature the paper should cite

**Human spacing (expanding vs. uniform).**

63. Kang, S. H. K., Lindsey, R. V., Mozer, M. C., & Pashler, H. (2014). Retrieval practice over the long term: Should spacing be expanding or equal-interval? *Psychonomic Bulletin & Review, 21*(6), 1544–1550. https://doi.org/10.3758/s13423-014-0636-z. **Verified.**
   - Design: expanding (days 1, 3, 9, 28) vs. equal (days 1, 10, 19, 28) schedules, with final recall on day 84.
   - Result: no significant final-recall difference (.49 vs. .46), but expanding kept recall higher *during* training.
   - This is the closest human analogue to the paper's trajectory-vs-final pattern, so the paper should cite it rather than dismiss its own trajectory result as an artifact.
64. Latimier, A., Peyre, H., & Ramus, F. (2021). A meta-analytic review of the benefit of spacing out retrieval practice episodes on retention. *Educational Psychology Review, 33*(3), 959–987. https://doi.org/10.1007/s10648-020-09572-8. **Verified** (volume, pages and DOI; the issue number is from the publisher listing).
   - Spaced beat massed retrieval practice (g = 0.74).
   - Expanding vs. uniform showed no significant difference (g = 0.034).
   - This is the right citation for "the human literature shows no reliable expanding advantage", rather than a single study.
65. Cepeda, N. J., Vul, E., Rohrer, D., Wixted, J. T., & Pashler, H. (2008). Spacing effects in learning: A temporal ridgeline of optimal retention. *Psychological Science, 19*(11), 1095–1102. https://doi.org/10.1111/j.1467-9280.2008.02209.x. **Verified.**
    - The optimal gap depends on the retention interval (≈20% of the test delay for delays of weeks, falling to ≈5% at one year).
    - Implication: lag-to-test must be controlled or manipulated.
66. Balota, D. A., Duchek, J. M., & Logan, J. M. (2007). Is expanded retrieval practice a superior form of spaced retrieval? A critical review of the extant literature. In J. S. Nairne (Ed.), *The foundations of remembering: Essays in honor of Henry L. Roediger, III* (pp. 83–105). Psychology Press. http://psychnet.wustl.edu/coglab/wp-content/uploads/2015/01/2007-Is-expanded.pdf. **Verified** (some sources give the page range as 83–106).
67. Storm, B. C., Bjork, R. A., & Storm, J. C. (2010). Optimizing retrieval as a learning event: When and why expanding retrieval practice enhances long-term retention. *Memory & Cognition, 38*(2), 244–253. https://doi.org/10.3758/MC.38.2.244. **Verified.**
    - Expanding beat uniform only when the activity between retrievals was highly interfering.
    - Suggests that LLM schedule effects may appear only under strong interference, for example new facts that share entities with old ones, as in the repository's `fictionalqa_interference` variant.

**Replay, spaced repetition and forgetting in neural nets and LLMs.**

68. Amiri, H., Miller, T., & Savova, G. (2017). Repeat before forgetting: Spaced repetition for efficient and effective training of neural networks. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing* (pp. 2401–2410). Association for Computational Linguistics. https://doi.org/10.18653/v1/D17-1255. **Verified.**
69. Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html (arXiv:1811.11682). **Verified.**
70. Ibrahim, A., Thérien, B., Gupta, K., Richter, M. L., Anthony, Q., Lesort, T., Belilovsky, E., & Rish, I. (2024). Simple and scalable strategies to continually pre-train large language models. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2403.08763. **Verified** (TMLR; the OpenReview forum ID was not confirmed).
71. Biderman, D., Portes, J., Gonzalez Ortiz, J. J., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., & Cunningham, J. P. (2024). LoRA learns less and forgets less. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=aloEru2qCG. **Verified.**
    - LoRA forgets less than full fine-tuning, which may compress schedule effects.
72. Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., & Zhang, Y. (2025). An empirical study of catastrophic forgetting in large language models during continual fine-tuning. *IEEE/ACM Transactions on Audio, Speech, and Language Processing, 33*, 3776–3786. https://doi.org/10.1109/TASLPRO.2025.3606231 (preprint arXiv:2308.08747, 2023). **Verified.**
73. Yang, Y., Jones, M., Mozer, M. C., & Ren, M. (2024). Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training. In *Advances in Neural Information Processing Systems 37*. https://openreview.net/forum?id=YSs1z5udBY (arXiv:2403.09613). **Verified.**
74. Zucchet, N., Bornschein, J., Chan, S., Lampinen, A., Pascanu, R., & De, S. (2025). How do language models learn facts? Dynamics, curricula and hallucinations. In *Second Conference on Language Modeling (COLM 2025)*. https://arxiv.org/abs/2503.21676. **Verified** (arXiv comment: "Accepted at the 2nd Conference on Language Modeling (2025)").
    - Fine-tuning on new facts quickly corrupts existing parametric memories.
75. Tirumala, K., Markosyan, A. H., Zettlemoyer, L., & Aghajanyan, A. (2022). Memorization without overfitting: Analyzing the training dynamics of large language models. In *Advances in Neural Information Processing Systems 35* (pp. 38274–38290). https://arxiv.org/abs/2205.10770. **Verified.**
76. Jagielski, M., Thakkar, O., Tramèr, F., Ippolito, D., Lee, K., Carlini, N., Wallace, E., Song, S., Thakurta, A., Papernot, N., & Zhang, C. (2023). Measuring forgetting of memorized training examples. In *International Conference on Learning Representations*. https://openreview.net/forum?id=7bJizxLKrR. **Verified.**
77. M'hamdi, M., & May, J. (2024). Leitner-guided memory replay for cross-lingual continual learning. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)* (pp. 7808–7821). https://doi.org/10.18653/v1/2024.naacl-long.432. **Verified.**
78. Feng, Y., Wang, H., Li, J., Chu, X., Kang, Z., Liu, Y., Wang, Y., Yu, P. S., & Wu, X.-M. (2026). *FOREVER: Forgetting curve-inspired memory replay for language model continual learning* [Preprint]. arXiv. https://arxiv.org/abs/2601.03938. **Verified** (authors and title checked on arXiv). The arXiv comment says "ACL 2026 camera-ready", but the ACL Anthology entry is UNVERIFIED.
    - Ebbinghaus-style replay intervals measured in "model time" (optimizer-update magnitude).
    - The authors' repository already implements a `forever` condition, so the paper should cite this work and discuss it.
79. Lu, Y., He, Y., Chen, J., & Zha, H. (2026). *MSSR: Memory-aware adaptive replay for continual LLM fine-tuning* [Preprint]. arXiv. https://arxiv.org/abs/2603.09892. **Verified** (arXiv).
80. Atreya, A., Batra, D., Mantri, Y. K., Bantug, G., Cowan, G. A., & Khraishi, R. (2026). *When to review: Spaced repetition for continual pre-training of language models* [Preprint]. arXiv. https://arxiv.org/abs/2608.17530. **Verified** (arXiv).
    - Per-example SM-2 scheduling with perplexity-derived recall quality. Reports beating uniform replay.
    - **This is the most directly competing work (August 2026), and its message pulls against "schedule does not matter".**
    - Note the scope difference from the abstract: SRT is *adaptive, per-example* scheduling, compared with uniform *sampling* replay. It is not a fixed expanding-vs-equal-interval contrast.
    - A reconciliation is available and could strengthen the paper: per-item adaptive selection may matter even when fixed schedule shape does not, given that this paper's review events are not even item-specific (Part B, M1).
81. Prakriya, N., Yen, J.-N., Hsieh, C.-J., & Cong, J. (2024). *Accelerating large language model pretraining via LFR pedagogy: Learn, focus, and review* [Preprint]. arXiv. https://arxiv.org/abs/2409.06131. **Verified** (arXiv). Peer-reviewed venue UNVERIFIED.
    - A spaced-repetition-inspired data scheduler for GPT-2 pretraining. The authors' own project report calls it the closest prior work.
82. Kirchenbauer, J., Mongkolsupawan, J., Wen, Y., Goldstein, T., & Ippolito, D. (2026). FictionalQA: A dataset for studying memorization and knowledge acquisition. In *International Conference on Learning Representations* (per the arXiv v2 comment "Published at ICLR 2026"). https://arxiv.org/abs/2506.05639. **Verified on arXiv; the ICLR proceedings entry was not checked.**
    - Structure:
      - 100 fictsheets/events.
      - ~1.5k documents (15 per event across the news, social, corporate, blog and encyclopedia styles).
      - ~7.5k QA pairs (~75 per event) with `duplicate_root` and duplicate flags.
      - An MCQ configuration.
      - Blind and informed answer attempts that flag guessable questions.
    - In the original paper, models train on documents (not on `fict` statements) and are evaluated with answer NLL **and 4-way MCQ accuracy**. That is precedent for adding a behavioural metric here.
    - The paper's event-level split is the right choice; question-level splits would leak through shared events.


---

# Part B — Mock peer review (Reviewer 2)

**Paper:** "Does Spaced Review Help a Language Model Retain Facts During Continued Training? An expanding- versus uniform-interval study on OLMo with FictionalQA" (Mago & Wang).
**Reviewed version:** `P4_whitepaper_writeup.md` (converted from the .docx on 2026-10-02), plus the linked repository `github.com/anshulmago1/olmo-fictionalqa-review` (`configs/final.yaml`, `src/olmo_core/review_lab/{schedules,trainer,fictionalqa,report}.py`). A reviewer at TMLR/NeurIPS is entitled to inspect linked code, and several of the issues below come from it.
**Recommendation (as currently written):** Reject / major revision. The question is good and the endpoint-matched design is careful in places, but the central spacing manipulation is barely delivered at the level of individual facts, the metric's validity as "retained knowledge" is not established, and two of the three headline conclusions ("statistically and practically indistinguishable"; "starting point, not forgetting rate") are not supported by the analyses presented.
**Confidence:** 4/5.

## Summary of the submission

The authors fine-tune OLMo DataDecide-300M with LoRA (r=16) on declarative statements from 20 FictionalQA events (Stage 1, 60 updates). They then train on statements from 40 new events for 180 updates (Stage 2), during which two arms put in 12 single-update "review events" on old statements (uniform vs. expanding positions with first/last review matched at steps 8 and 165). A third arm has no review. After that comes a 180-update buffer of new-fact-only training. The primary outcome is answer-token cross-entropy on paraphrased closed-book questions about the old facts. With 3 seeds, review lowers final old-fact loss by about 0.173 nats (reported as 4.3%) relative to no review. Expanding minus uniform is −0.0006 nats, 95% paired CI [−0.0053, +0.0041]. The buffer increase in loss does not differ between arms. The authors conclude that review helps, that spacing does not matter, that review "changes the starting point, not the forgetting rate", and that this reproduces a human finding (Karpicke & Roediger, 2007).

## Strengths

1. A clean, pre-specified-looking question with a sensible fixed-budget comparison. Using fictional facts rules out contamination from pretraining.
2. The no-review buffer is a good design choice. It measures retention at delay rather than right after review.
3. Constant LR (no decay) avoids confounding schedule position with learning rate.
4. Paired-by-seed analysis with interval estimates, not just point estimates. The code is public and the Stage-1 checkpoint is shared across arms within a seed.
5. Per the code, uniform and expanding share their first (step 8) and last (step 165) review step. That controls the most obvious recency confound at the arm level. The paper should say this explicitly.

---

## MAJOR issues

### M1. The spacing manipulation is not delivered at the level of individual facts (construct validity of the independent variable)

**Problem.** In the code, a "review event" is **one optimizer update of 16 old-fact statements, sampled with replacement from a single FictionalQA document style**, with the 5 styles visited round-robin (`ReviewController` → `SkillSelector.choose`, `EncodedPool.sample`). The consequences:

* The style-level schedules are: uniform reviews at steps 8, 22, 37, 51, 65, 79, 94, 108, 122, 136, 151, 165, and expanding reviews at 8, 10, 13, 17, 22, 29, 38, 51, 68, 91, 123, 165 (computed with `fixed_review_steps`, ratio 1.35). With round-robin assignment (blog, corporate, encyclopedia, news, social, blog, …), each style gets only **2–3 review events** (blog 3, corporate 3, others 2).
* So each *style* is reviewed 2–3 times, and each individual *fact* is reviewed far less: 16 draws with replacement from that style's old-fact pool. Re-running the repository's split logic (data seed 20260721, 20 old events, canonical `duplicate_root` questions) against the public `fict_qa` table gives **386 old training statements** spread very unevenly across styles: blog 28, corporate 54, encyclopedia 41, news 180, social 83. Expected review exposures per fact are therefore 1.71 (blog), 0.89 (corporate), 0.78 (encyclopedia), 0.18 (news) and 0.39 (social). The probability that a given evaluated fact is **never reviewed** in a run is 0.18, 0.41, 0.45, 0.84 and 0.68 respectively, so **about half (≈51%) of the 80 evaluated old facts receive no review at all**. Stage 1 is similarly uneven: round-robin by style gives each style 192 draws, so news facts are seen ≈1.1 times on average, and ≈34% of news facts (≈10% of all evaluated old facts) are **never seen during Stage 1**. Their 'retention' is not retention. (Reviewer's reconstruction from code and dataset, assuming the dataset revision fetched today matches the pinned `131cb74…`. The authors should confirm from logs.) Which facts are reviewed, and when, is decided by the sampling RNG, not by the nominal schedule. At the item level, "expanding" and "uniform" are mostly two different random scatterings of 0–2 re-exposures.
* Endpoint matching holds **only for the corporate style**. For encyclopedia, news and social, the expanding arm's last review of that style came at steps 51, 68, 91, versus 108, 122, 136 under uniform. That is 45–57 updates *earlier*. So recency is confounded with schedule for 3 of 5 styles, in the direction that should favour uniform.

The paper describes the conditions as "reviews of the Stage 1 facts" spaced uniformly or expandingly. A reader will assume that each old fact follows the stated schedule. It does not. The null result for expanding vs. uniform may therefore reflect a **weak manipulation** rather than an absence of spacing effects. The review benefit itself may also be mostly *non-specific* (style-, format- or domain-level transfer) rather than re-consolidation of specific facts.

**Fix.**
(a) State precisely what a review event contains (number of items, sampling, style restriction, replacement vs. displacement of a new-fact batch).
(b) Log per-fact exposure counts and exposure times (the trainer currently keeps only a hash of example IDs). Report the distribution of per-fact review counts and lags per arm.
(c) Re-run with a design in which **every old fact** (or a fixed, counterbalanced subset) is reviewed on the nominal schedule. For example, give each review event its own fixed subset of facts, with item-to-schedule assignment counterbalanced across seeds via a Latin square.
(d) Add an item-level analysis inside the review arms: final loss for facts reviewed k = 0, 1, 2, … times, and as a function of lag since last exposure. If reviewed and never-reviewed facts benefit equally, the "review" effect is not item-specific memory maintenance and the framing must change.
(e) Report per-style results, since styles differ in their actual last-review time.

### M2. Is answer-token cross-entropy at ~3.8 nats/token "retained knowledge"? (construct validity of the dependent variable)

**Problem.** Several observations suggest that the primary metric mostly tracks something other than fact-specific knowledge.

* **Absolute level.** Final old-fact loss is ≈3.8–4.0 nats per answer token, which is a per-token perplexity of about 45–54. End-of-Stage-1 loss can be back-computed from Table 1 (final − "old-loss change") at about **4.20 nats in every arm**. The paper never reports the *pre-fine-tuning* loss of the base model on these questions, so readers cannot tell how much Stage 1 actually taught. 60 updates × 16 = 960 sampled statements in total.
* **Old-fact loss falls during new-fact training in all arms, including no-review.** Trajectory change is −0.305 for no-review. Even after the 180-update buffer, no-review old loss (3.981) is **0.22 nats below** its end-of-Stage-1 value. In other words, there is **no net forgetting** relative to initial learning in any arm. Training on *unrelated* fictional facts improves the metric more than the buffer later degrades it. This is a strong sign that much of the metric reflects generic adaptation (FictionalQA vocabulary, answer type and length distribution, the "Answer:" → short span → EOS pattern) rather than memory for specific facts. The old/new split is event-disjoint, which the code enforces, so leakage is not the explanation.
* **Train/test format mismatch.** Training inputs are `"Fictional fact: <statement>"` with loss on all tokens. Evaluation is `"Question: …\nAnswer:"` with loss on the answer tokens **plus EOS**. The model never sees QA format in training, so part of the evaluated loss (notably EOS placement and answer-span formatting) is pure format adaptation. The EOS token is included in what the paper calls "answer-token" loss. Declarative-to-question extraction is a known hard transfer for small LMs (Allen-Zhu & Li, 2024; Berglund et al., 2024).
* **Exact match is computed in the code but not reported.** If greedy exact match is near 0% in all arms, the "4.3% better retention" is a change in probability mass on strings the model never produces.
* **Relative % is not a meaningful scale.** Cross-entropy has no natural zero for this task (the floor is the irreducible answer entropy under paraphrase, and the ceiling is the base-model loss). "4.34% lower loss" depends on these arbitrary anchors. A 0.173-nat difference is more interpretable as a ≈1.19× increase in the geometric-mean per-token probability of the correct answer (e^0.173), or as a fraction of the Stage-1 learning gain.

**Fix.**
(a) First, a free analysis: because sampling is driven by deterministic RNGs (`random.Random(f"{seed}:{stage}:old:{skill}")`), the per-fact Stage-1 and review exposure counts can be reconstructed exactly. Roughly 10% of evaluated old facts (about a third of news-style facts) were never trained in Stage 1. Compare their trajectories with those of trained facts. Then add a **never-trained control set**: held-out FictionalQA events, never trained in any stage, evaluated with the same probe at every checkpoint. Report *specific retention* = loss(old) − loss(never-trained). If the never-trained curve shows the same U-shape (falling in Stage 2, rising in the buffer), the "forgetting" and much of the "review benefit" are format/domain effects. Note that review arms also see 12 fewer new-fact batches.
(b) Report base-model (step-0) loss and end-of-Stage-1 loss with CIs, and express effects as a fraction of the Stage-1 gain.
(c) Report behavioural metrics that the code or dataset already supports: greedy exact match, and a distractor-normalized log-likelihood margin or multiple-choice accuracy (FictionalQA includes an MCQ configuration).
(d) Either include QA-format examples in training for a disjoint set of facts (the standard way to make extraction possible; Allen-Zhu & Li, 2024), or explicitly reframe the outcome as "likelihood of a paraphrased answer after statement-only training".
(e) Drop or demote the relative-% framing. Report nats, probability ratios, and normalized retention.

### M3. "Statistically and practically indistinguishable" is not licensed by a 95% CI that spans zero

**Problem.** A non-significant difference is not evidence of equivalence (Altman & Bland, 1995). The paper never defines a smallest effect size of interest (SESOI), never runs an equivalence test (TOST; Schuirmann, 1987; Lakens, 2017), and never gives a Bayesian ROPE or Bayes-factor analysis (Kruschke, 2018; Rouder et al., 2009). The word "practically" needs an explicit practical threshold.

The reviewer's own back-calculation (assuming t-based CIs with df = 2, t₀.₉₇₅ = 4.303) gives SE ≈ 0.0011 nats for expanding − uniform. The TOST 90% CI would then be roughly [−0.0038, +0.0026]. Equivalence would be declared for any bound Δ ≥ ~0.004 nats, about 2–3% of the review effect. **The claim may well survive a proper test**, but only once Δ is justified *a priori* (or at least independently of the observed data). The authors also need to acknowledge that:

* With n = 3, the SD of the paired difference rests on 2 df. Its 95% CI runs from about 0.52× to 6.3× the point estimate (χ²₂ quantiles), so the precision of the interval is itself very uncertain.
* The interval reflects **only seed-level variability** for a single fixed set of facts. The old/new event split (data seed 20260721) and the 80 evaluated items are identical across all seeds and conditions. Fact-sampling variability is not represented at all (see M5).
* A tight CI around zero on a metric that may barely index item-specific memory (M2), under a manipulation that barely differs per item (M1), is "equivalence of a weak treatment on a weak measure".

**Fix.** Pre-specify Δ, for example as a fraction of the review-vs-no-review effect or as the change in loss that corresponds to a 1-percentage-point change in exact match. Report TOST p-values and the 90% CI, and/or a Bayesian estimate with a ROPE and the posterior mass inside it. Re-word the claim as "equivalent within ±Δ nats on this metric and fact set". Increase the number of seeds and, more importantly, the number of independent fact samples (different event splits per seed).

### M4. "Review changes the starting point, not the forgetting rate" is unsupported, for three separate reasons

**Problem.**
1. **Absence of evidence again.** The CI for uniform − no-review buffer forgetting is [−0.072, +0.091]. Its half-width (≈0.08 nats) is about **50% of the forgetting itself** (≈0.15–0.16). The data are compatible with review halving the forgetting or increasing it by half. "Indistinguishable from zero" is true. "Same rate" is not shown.
2. **Scale dependence (Loftus, 1985).** Equal *vertical* differences on one scale are unequal on another. Mean answer-token NLL is a log scale: equal absolute loss increases mean equal *proportional* drops in geometric-mean token probability. On a probability scale, the no-review arm drops from e^−3.830 ≈ 0.0217 to e^−3.981 ≈ 0.0187 (−0.0030), while uniform drops from 0.0260 to 0.0222 (−0.0039). So on that scale the review arm forgets **more**. On accuracy (bounded, nonlinear) the answer could differ again. "Same rate" is a property of the chosen scale, not of the memory.
3. **Two-point "rate".** The buffer forgetting is a difference of two noisy measurements. It ignores the 15- and 60-update checkpoints that are already collected, and it confounds degree of learning with rate. The classic human result is that degree of learning shifts the intercept but not the slope (Slamecka & McElree, 1983), and this has been debated on precisely these scaling grounds (Loftus, 1985; Wixted, 2004).

A **horizontal** comparison (Loftus, 1985) gives a different and arguably more useful reading. The review arms at 180 buffer updates (3.809) are still better than no-review at buffer onset (3.830). Review therefore "buys" more than 180 updates of delay, which is a large effect that the paper understates.

**Fix.** Fit a model of old-fact loss over all buffer checkpoints (0, 15, 60, 180) with a condition × time interaction, preferably at the item level with random effects (M5). Report the interaction estimate with its CI and an equivalence bound. Show the result on at least two scales (NLL and exact match / MC accuracy), add a horizontal "delay-equivalent" comparison, and state explicitly that rate conclusions are scale-relative. Otherwise, remove the mechanistic claim.

### M5. Unit of analysis: items are treated as fixed, so the inference does not generalize beyond 80 specific questions

**Problem.** The final evaluation uses **16 items per style = 80 old-fact items, identical across conditions and seeds**, drawn from just 20 fictional events. All statistics average over items and use only the 3 seeds as replicates. This is Clark's (1973) "language-as-fixed-effect fallacy" in ML clothing. The CIs speak to "these 80 questions, re-run with new seeds", not to "facts of this kind". Items are also clustered within events and styles. The per-item losses are already logged (`evaluate_metrics` returns `items`), so the fix needs no retraining.

**Fix.** Fit linear mixed-effects models (Baayen, Davidson & Bates, 2008) with crossed random effects for seed and item (items nested in events). Use maximal-justified random slopes for condition (Barr et al., 2013; tempered by Matuschek et al., 2017), or use a hierarchical/cluster bootstrap over events → items × seeds. Better still, re-draw the old/new event split per seed (or per replicate) so that fact sampling is a source of variance in the design. Use all of the old-fact pool for evaluation rather than 16 per style, and report the item-level variance decomposition.

### M6. Missing controls that are necessary to interpret either effect

**Problem and fix (each item is a control the paper needs):**

* **Never-trained fictional facts** (format/domain control). See M2(a). This is the single most important missing control.
* **Massed/cram control at matched budget.** The human spacing effect is fundamentally *spaced vs. massed*. Expanding vs. uniform is a second-order contrast that is known to be small and inconsistent in humans (Kang et al., 2014; Latimier et al., 2021). The repository already implements a `cramming` schedule. Add (i) all 12 reviews massed at one point (e.g., steps 154–165), and (ii) 12 extra Stage-1 updates (extended initial learning). Without these, "review helps" cannot be separated from "12 more old-fact updates help, whenever they occur".
* **Exposure vs. displacement.** Review events *replace* new-fact updates. The step budget is fixed, so review arms get 168 rather than 180 new-fact updates in Stage 2 (348 vs. 360 overall). The "plasticity cost" (§3.4) may be pure displacement. Add an arm in which review batches are *added* (e.g., 12 extra updates or mixed batches) and keep new-fact exposure constant. Report token counts (`old_content_tokens` and `new_content_tokens` are already logged).
* **Review timing alignment.** First and last review are matched (good), but the style-level last-review times are not (M1). The penultimate review is at step 123 (expanding) vs. 151 (uniform). Report and control these. Consider a design in which the *lag from last review to test* is varied orthogonally to schedule shape (Cepeda et al., 2008).
* **Review content.** Review uses the same training statements (verbatim restudy), whereas spaced-retrieval studies in humans use retrieval practice. A paraphrased-statement review arm and a QA-format review arm would test whether "review" acts as restudy or as retrieval.
* **Retention-interval manipulation.** Only one long delay is tested. Karpicke & Roediger (2007) found expanding > uniform at *short* delays and uniform > expanding at long delays. Testing the short-delay prediction requires a short-delay endpoint with matched lag.

### M7. Statistical power, reporting of uncertainty, and analytic flexibility

**Problem.**
* With n = 3 and df = 2, the 95% t multiplier is 4.30. Any effect smaller than about 3–4 seed-SDs is undetectable. No power or sensitivity analysis is given, so the reader cannot tell what size of spacing effect the study *could* have found. Recent ML guidance argues that 3 seeds are rarely enough for confident comparison claims (Colas et al., 2018; Agarwal et al., 2021; Bouthillier et al., 2021).
* The method for the CIs is never stated (t-based? bootstrap? A bootstrap over 3 seeds would be invalid). The paper reports seed SDs in Table 1 but no SD for pre-buffer loss.
* There are seven outcome metrics, three contrasts, and three buffer delays, with no designated primary contrast and no multiplicity control. The "trajectory" metric is called an expanding advantage (−0.436 vs. −0.423) without any CI, then explained away post hoc as a "front-loading artifact". Note that a short-delay expanding advantage is exactly what Karpicke & Roediger (2007) reported, so dismissing it as an artifact is selective.
* There is no preregistration and no statement of which analyses were planned.

**Fix.** Give a sensitivity analysis: the minimum detectable effect at 80% power, given the observed paired SD, together with its uncertainty. Increase seeds to ≥10 for the confirmatory comparison (cheap at 300M with LoRA). Pre-specify one primary outcome and one primary contrast, or apply Holm correction. Report CIs for every comparison, including the trajectory metric. State how CIs were computed. Show all per-seed values (n = 3 is small enough to tabulate). Consider preregistering the follow-up (van Miltenburg et al., 2021) or submitting it as a registered report.

### M8. Generality and the human-analogy claims overreach

**Problem.**
* The evidence comes from one model (300M), one adaptation method (LoRA r=16), one sequence length (64 tokens, which may truncate some prompt+answer pairs), one LR, one dataset, one split, one review budget (12), one expansion ratio (1.35), and one buffer length. "How you space it does not [matter]" (Conclusion) and "uniform review is the sensible default … for a production system" are far broader than this.
* LoRA is known to "learn less and forget less" than full fine-tuning (Biderman et al., 2024), which could compress any schedule effect.
* Calling the result "showing that this human phenomenon reproduces in LLM continued training" is not justified. (i) A null difference is not a reproduction of a phenomenon. (ii) The human finding is a *crossover* (expanding better at short delay, uniform better at long delay; Karpicke & Roediger, 2007), and this study neither tested nor found the long-delay uniform advantage. (iii) Human expanding-vs-uniform studies concern *retrieval practice*, whereas this is verbatim restudy of statements. (iv) The forgetting here is roughly linear in loss over the buffer (≈0.0011, 0.0008, 0.0008 nats/update over the 0–15, 15–60, 60–180 windows), unlike the negatively accelerated, power-like human forgetting function (Wixted & Ebbesen, 1991). The human literature also has a real meta-analytic answer that should be cited: no reliable expanding-vs-uniform difference (Latimier et al., 2021).

**Fix.** Scope the title, abstract and conclusion to the tested regime. Remove "production default" advice or label it speculative. Add at least one more model size or full fine-tuning, and vary review budget and expansion ratio, or state plainly that the result is a single-configuration pilot. Rewrite the human-comparison paragraph as "consistent with the meta-analytic null in humans", with the restudy/retrieval and crossover caveats.

### M9. Novelty and positioning

**Problem.** The related-work section is four sentences and cites no ML work on replay, rehearsal or spaced repetition. A reviewer will expect at least:
* spaced repetition for training neural nets (Amiri et al., 2017);
* replay for continual learning (Rolnick et al., 2019) and for continual LLM pretraining (Ibrahim et al., 2024);
* forgetting under continual fine-tuning of LLMs (Luo et al., 2023/2025);
* knowledge acquisition and forgetting dynamics during LM training (Chang et al., 2024; Zucchet et al., 2025);
* cyclic/structured re-exposure and anticipatory recovery (Yang et al., 2024);
* LoRA forgetting (Biderman et al., 2024);
* knowledge injection and extraction (Allen-Zhu & Li, 2024; Ovadia et al., 2024; Gekhman et al., 2024);
* spaced-repetition-inspired data schedulers for LM pretraining (Prakriya et al., 2024, "LFR", which the authors' own project report already identifies as closest prior work);
* forgetting-curve, Leitner and SM-2 replay schedulers for LLMs: M'hamdi & May (2024, NAACL); FOREVER (Feng et al., 2026, arXiv, ACL 2026 per the authors; the repository already implements a `forever` condition, so the omission is conspicuous); MSSR (Lu et al., 2026); and especially **"When to Review" (Atreya et al., 2026, arXiv:2608.17530)**. That paper reports that per-example SM-2 scheduling beats uniform replay in continual pre-training, which pulls directly against "how you space it does not [matter]".

The likely reconciliation is that *adaptive, item-level* selection matters while *fixed schedule shape* does not. This paper cannot speak to the former, because its review events are not item-specific (M1).

As written, the contribution relative to that literature (a negative result on schedule shape at tiny scale) is not articulated.

**Fix.** Add a proper related-work section and position the paper explicitly as a controlled *negative result* on schedule shape under a fixed replay budget. TMLR and workshop tracks value that if it is rigorous.

---

## MINOR issues

* **m1. Stray editorial text.** §3.2 contains the drafting note "The old-fact loss trajectory plot (…) This is the natural place for it: it visually anchors…". Delete it. Also check that the Figure 1 caption agrees with the text ("all three rising again once the buffer begins").
* **m2. Unspecified schedule positions.** Give the exact review steps for both arms (listed in M1) and the expansion ratio, and say that first/last are matched. "The final review occurred at Stage 2 step 165" should say this holds for both arms.
* **m3. Eval cadence.** State that evaluation happens every 30 updates in Stage 2, and how "trajectory change" averages over those checkpoints (it is affected by how many of each arm's reviews precede each checkpoint: 7 of 12 expanding reviews occur before step 38).
* **m4. Data description.** State the number of old/new events (20/40), the number of training statements per stage, the number of evaluation items (80 old / 80 new), deduplication (canonical `duplicate_root`), event-disjoint splitting, and the fixed data seed. Note whether any prompt+answer exceeds 64 tokens and is truncated.
* **m5. Table 1** has no SD for pre-buffer old loss. Add per-seed values, and report end-of-Stage-1 and base-model loss.
* **m6. "Around the same FLOPs".** Give the actual numbers, and old/new content tokens per arm.
* **m7. "Seeds".** Specify what the seed controls (LoRA init, sampling RNG, dropout) and what it does not (data split, eval items). Note the fp16 autocast precision.
* **m8. Terminology.** "Buffer forgetting" is a difference in loss, not a rate. Call it "buffer loss increase" unless the time course is modelled. "Plasticity" for the new-fact loss is loose.
* **m9. Joint loss.** The equal-weight average of old and new losses is arbitrary. Justify it or drop it, and give its CI.
* **m10. Reporting checklist.** Add compute (GPU type and hours), the hyperparameter selection procedure (were LR, LoRA rank, Stage-1 length or the 1.35 ratio tuned?), and a limitations section consistent with the NeurIPS checklist (error bars: what variability they capture and how they are computed).
* **m11. Model naming.** "OLMo DataDecide 300M (≈377M total parameters)". Give the exact HF ID and revision (`allenai/DataDecide-dolma1_7-300M`, revision 4b1b42ff) as in the config.
* **m12. Citations.** Hu et al. (LoRA) appeared at ICLR 2022. FictionalQA should be cited as ICLR 2026 (per arXiv v2). DataDecide should be cited at its final venue if it has one. Landauer & Bjork (1978) is cited as showing expanding schedules "greatly improve memory in humans", which overstates it. That chapter proposed expanding rehearsal, and later controlled work (Karpicke & Roediger, 2007; Kang et al., 2014) found little or no advantage over equal spacing.
* **m13. Abstract.** "Survived a long no-review buffer across all three seeds" should give the effect with its CI. "Statistically and practically indistinguishable" should be replaced with the equivalence-test wording (M3).

---

## Questions for the authors

1. What is the base-model (pre-Stage-1) answer loss and exact match on the old and new questions?
2. What is greedy exact match at each checkpoint, per arm? (It is computed in the code.)
3. How many distinct old facts does each evaluated item's statement appear with in review batches, per arm and seed? What fraction of the 80 evaluated old facts were never reviewed?
4. Does a never-trained event set show the same fall-then-rise trajectory?
5. Were any hyperparameters (Stage-1 length, LR, ratio 1.35, buffer length, 12 events) chosen after looking at results?
6. How were the 95% CIs computed?

---

## Suggested venues (status as of 2026-10-02)

Details come from official pages fetched 2026-10-02 unless marked UNVERIFIED.

* **Best immediate fit after a moderate revision: TMLR.**
  - Rolling submission, double-blind on OpenReview, no fixed page limit (length "should be justified by its content"). https://jmlr.org/tmlr/author-guide.html
  - Acceptance turns on "claims supported by accurate and convincing evidence" and audience interest. Novelty is not required (https://jmlr.org/tmlr/acceptance-criteria.html).
  - A rigorous controlled negative result fits well. The paper must first fix M1–M5, because TMLR reviewers enforce the "claims supported" criterion strictly, and the claims would need narrowing.
* **ACL Rolling Review.**
  - Next cycle deadline **12 Oct 2026**, feeding NAACL 2027 and COLING 2027. Long papers 8 pages, short papers 4, plus unlimited references; Limitations section required. https://aclrollingreview.org/dates
  - The current version could go in as a short paper, but 10 days is too little to add the controls. The following cycle (toward ACL 2027) is listed as "January 2027".
* **TACL.** Submissions on the 1st of each month. Note the 9-month embargo after ARR review. https://transacl.org/index.php/tacl/about/submissions
* **CogSci 2027** (Bilbao, 28–31 Jul 2027).
  - Deadline not yet posted (UNVERIFIED; historically early February). Page limit UNVERIFIED (historically 6 pages + references). https://cognitivesciencesociety.org/cogsci-2027/
  - A good fit only if the human-comparison framing is made rigorous (M8): retrieval vs. restudy, the crossover prediction, and the meta-analytic framing.
* **CCN 2027.** Not yet announced (UNVERIFIED).
  - CCN 2026 had archival 8-page papers (due ~Feb) and non-archival 2-page abstracts (due ~Apr). https://2026.ccneuro.org/call-for-papers/
  - The 2-page track is a low-risk venue for the current result.
* **CoLLAs 2027** (Conference on Lifelong Learning Agents). The natural replay and forgetting community. Dates and page limits UNVERIFIED (past editions published in PMLR, with deadlines ~Feb–Mar). https://lifelong-ml.cc/
* **Workshops.** NeurIPS 2026 workshop deadlines have effectively passed (suggested date 29 Aug 2026). ICLR 2027 workshops (deadlines typically ~Jan–Feb 2027; UNVERIFIED) on continual learning, memory or the science of LMs suit the current pilot-scale study.
* **Closed or not suitable now.**
  - ICLR 2027 main track: paper deadline 25 Sep 2026 has passed; 9 pages at submission. https://iclr.cc/Conferences/2027/CallForPapers
  - NeurIPS 2026, including the new Evaluations & Datasets track: deadline May 2026 has passed; 9 pages. https://neurips.cc/Conferences/2026/CallForPapers
  - COLM 2026: closed.
  - NeurIPS 2027 or ICLR 2028 main tracks would require multi-scale, full fine-tuning, item-level spacing, behavioural metrics and a mechanism.
* **Journals for an extended cognitive version:** *Computational Brain & Behavior* (Springer, rolling; https://link.springer.com/journal/42113) or *Cognitive Science* (Wiley, rolling; https://onlinelibrary.wiley.com/journal/15516709). Article-type limits are UNVERIFIED. Both would expect a substantive human-model comparison.


# Appendix — Reviewer's quantitative checks (reproducible)

1. **CI back-calculation.** Half-width / 4.303 gives the SE; SE × √3 gives the SD of the paired difference.
   - Expanding − uniform: half-width 0.0047 → SE 0.00109 → SD 0.0019.
   - Uniform − no review (final loss): 0.0413 → 0.0096 → 0.0166. The seed-level correlation between arms is ≈0.98, so pairing is doing most of the work.
   - TOST 90% CI for expanding − uniform: −0.0006 ± 2.920 × 0.00109 = [−0.0038, +0.0026].
2. **Schedules.** `fixed_review_steps(kind, events=12, first_step=8, last_step=165, expansion_ratio=1.35)` from `schedules.py` gives the positions listed in the header. Styles are assigned round-robin in sorted order by `SkillSelector.choose`.
3. **Pool sizes.** The `_event_split` and `records_from_fictionalqa_rows` logic from `fictionalqa.py` was re-run on `fict_qa` (100 events, 7,500 rows). This assumes the HF revision currently served matches the pinned revision 131cb74…; the authors should confirm.
   - Old/new canonical statements by style: blog 28/51, corporate 54/94, encyclopedia 41/70, news 180/354, social 83/120.
   - Expected exposures: 16 × (number of events for the style) / pool size. P(never sampled) = (1 − 1/n)^draws.
4. **Probability-scale and horizontal comparisons.** exp(−loss) on the Table 1 and §3.2 values. Uniform reaches 3.809 at buffer step 180, which is below no review's 3.830 at buffer step 0.
5. **Buffer slope** (no review): (3.846 − 3.830)/15 = 0.0011, (3.880 − 3.846)/45 = 0.0008, (3.981 − 3.880)/120 = 0.0008 nats/update. The curve is roughly linear, not decelerating.
6. **Stage-2 evaluation cadence:** every 30 updates (`eval_interval: 30`). Seven of the 12 expanding reviews fall before step 38, versus 3 of 12 for uniform. This explains the trajectory-metric difference mechanically, and it matches the Kang et al. (2014) during-training pattern.
