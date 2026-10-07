# 08: Mastery-gated curriculum training: human mastery learning, ML curricula, algorithmic learning in small transformers, and replay

## TL;DR

- **Humans: mastery learning works, but the effect depends on the time it is given and on how the test is chosen.**
  - Kulik et al. (1990) found an average gain of **0.52 SD** across 103 studies.
  - On standardized tests the gain falls to 0.29 SD, and it is about **0 when time is equal** (Slavin, 1987).
  - Mastery groups usually get extra corrective time. Bloom's "2 sigma" partly reflects a stricter criterion for tutees: 90% vs 80% (VanLehn, 2011).
  - For us, the open question is the human analogue of our design: does mastery gating help *at a matched token budget*, measured on *OOD* tests?
- **Humans: the best support for gate + replay comes from relearning research.** In Rawson & Dunlosky (2011), a stricter initial criterion had strong effects before relearning, but those effects shrank as spaced relearning was added. Spaced relearning bought long-term retention cheaply. ALEKS also treats mastery as provisional and re-tests it later.
- **ML: curricula mostly help when compute is limited or labels are noisy.**
  - Across more than 25,000 vision models, ordering made no difference under standard budgets (Wu et al., 2021).
  - Curricula for pretraining small LMs gave mostly null results: BabyLM 2023 and 2024, CLIMB, and Campos (2021).
  - Recent positive pretraining results (0.5–3B) were mostly obtained with a **constant learning rate**. Luo et al. (ICLR 2026) show that standard LR decay erases much of the curriculum advantage.
- **ML: performance-triggered gates without replay have a poor track record.**
  - Zaremba & Sutskever's (2014) "naive" curriculum advanced on a validation plateau, or (in the released code) at 95% accuracy. It was "sometimes" worse than no curriculum.
  - Their winning "combined" strategy mixes in samples from the *whole* difficulty range, including harder levels not yet reached. That is not the same as replaying earlier skills.
  - Teacher–student curriculum learning (Matiisen et al., 2020) re-samples tasks that are being forgotten. It is the closest precedent for gate + replay, and it beat Zaremba's combined strategy on addition.
- **No prior work does exactly what we propose.** We found no study that trains LMs with *hard unlocks on a prerequisite skill graph, gated by held-out OOD probes*, compared with shuffle and clock schedules at a matched budget, with replay as a separate factor. The nearest neighbours each lack one piece:
  - Skill-it uses soft reweighting.
  - RLAAR uses a hard gate, but on *training* reward along a single difficulty axis.
  - Degree-Curriculum gates on training loss and does not use an LM.
  - CurLL has no gate and no replay.
  - SkillGraph gates *in-context* skills for a 7B agent.
  - **Kohli et al. (COLM 2026)** is the closest precedent. On k-hop facts, they advance to (k+1)-hop only after ≥95% accuracy on a held-out k-hop split, and keep training on all earlier stages "to prevent forgetting". That is gate + replay, but they use it only as a training tool and never compare it with shuffle or clock schedules.
  - **Yao et al. (EMNLP 2025)** find that implicit k-hop reasoning needs exponentially more data in k, and that curriculum learning reduces, but does not remove, that requirement.
- **Our preliminary tasks are known tasks with known pitfalls.**
  - The chain task is an adaptation of LEGO (Zhang et al., 2023) and of variable-binding programs (Wu, Geiger & Millière, 2025). Cite them.
  - Reversed output for addition comes from Lee et al. (ICLR 2024), not arXiv 2403.05845, which fine-tunes a 13B model and tests only in-distribution.
  - Length generalization also needs positional or place-value aids (Abacus; index hints), and it is **fragile across seeds**.
  - Grokking and heuristic-then-algorithm phase transitions mean that OOD probes can flip late, which interacts with when the gate fires.
- **Corrections to the report are needed.**
  - "Combined was consistently strongest" overstates Zaremba & Sutskever.
  - The CAMPUS "~7% vs shuffle" figure is really vs curriculum baselines. Against shuffle it is +1.67 points.
  - The RLAAR numbers are for Qwen3-8B, and its gate uses training reward.
  - "CE loss pushes models toward heuristics" is not what Lindsey et al. (2025) claim. The clock and pizza models were trained with CE and still learned exact algorithms.
  - Platanios et al.'s "competence-based" curriculum is actually **clock-based**.
- **What would convince reviewers:**
  - **Design:** a 2×2 factorial, gate (clock vs mastery) × replay (no vs yes), plus a stratified shuffle.
  - **Fairness controls:**
    - count probe compute in the budget;
    - run an LR-schedule ablation (constant or WSD plus averaging);
    - sweep the threshold;
    - add a "random-order, same pacing" control;
    - add a gated + look-ahead-mixing variant (Zaremba's combined);
    - tune all baselines equally.
  - **Evaluation and scale:** ≥5 seeds; a budget sweep; sealed exams disjoint from the gate probes; and at least one scale step (e.g., 125M → 350M → 1B), since curriculum effects shrink with scale (Elgaar & Amiri, 2026).

Prepared 2026-10-04 for the "On Mastery-Gated Curriculum Training" section (TEST 03) of `P4_final_report.md`. No paper file was edited.

## How sources were checked

- Primary sources were checked through arXiv abstract, HTML and PDF pages, OpenReview, PMLR, the ACL Anthology, NeurIPS proceedings, ICLR/ICML virtual pages, Crossref, PubMed, ERIC, OpenAlex, transformer-circuits.pub, and author-hosted PDFs.
- Nothing was cloned or downloaded from GitHub. For Zaremba & Sutskever, two code files were *read* as web pages through WebFetch.
- No synthetic-student material is included.
- **Tags:**
  - **[V-full]**: full text (or the relevant sections) read this session.
  - **[V-abs]**: abstract or landing page only.
  - **[V-meta]**: bibliographic metadata only (e.g., Crossref). Content statements in these entries come from secondary sources or recollection and are flagged.
  - **UNVERIFIED**: the source or the detail could not be confirmed.
- ⚠ in "Model scale" means the evidence comes only from models >7B, of undisclosed size, or from non-LM networks.
- Search quota ran out partway through. Coverage of 2025–2026 preprints and of recent K–12 mastery meta-analyses is therefore incomplete.
- Replay sources overlap with `03_ml_spacing_replay.md`, which verified them separately.
- The report's own citations are being checked separately for file 09. Section 6 here only lists discrepancies that matter for the literature review.

---

## 0. Definitions (each with a verifiable source)

- **Mastery learning.** "An instructional approach requiring learners to achieve a defined proficiency before proceeding to the next instructional objective" (Cook et al., 2013, abstract) [V-abs].
  - In operational terms (Slavin, 1987, p. 175) [V-full], it has three parts:
    1. "the establishment of a criterion level of performance held to represent 'mastery' of a given skill or concept";
    2. "frequent assessment of student progress toward the mastery criterion";
    3. "corrective instruction to enable students who do not initially meet the mastery criterion to do so on later parallel assessments."
- **Mastery criterion / learning to criterion.**
  - A mastery criterion is "a rule that determines whether a learner has achieved mastery," and its threshold "specifies its strictness" (Pelánek & Řihák, 2017, p. 156) [V-full].
  - In retrieval-practice research, the *initial learning criterion* is the number of correct recalls required before practice stops; for example, items "practiced until they were correctly recalled from 1 to 4 times" (Rawson & Dunlosky, 2011) [V-abs].
  - Historical thresholds include 80% on a parallel test (Bloom, 1984), and a learned probability P(learned) ≥ 0.95 under Bayesian knowledge tracing (Corbett & Anderson, 1992) [V-full].
- **Curriculum learning.** "A training strategy that trains a machine learning model from easier data to harder data, which imitates the meaningful learning order in human curricula" (Wang, Chen & Zhu, 2022, abstract) [V-abs]. Bengio et al. (2009) [V-full] proposed it as a continuation method.
- **Self-paced learning.** Each iteration "simultaneously selects easy samples and learns a new parameter vector." Easiness is judged by the current model's own loss, and a pacing weight is annealed until all the data is used (Kumar, Packer & Koller, 2010) [V-abs].
- **Competence-based curriculum.**
  - Competence c(t) is "the proportion of training data [the learner] is allowed to use at time t (measured in terms of training steps)." Batches are drawn from examples whose difficulty CDF is ≤ c(t) (Platanios et al., 2019) [V-full].
  - **Note:** c(t) is a fixed function of step count (linear or square root). It is a clock-based curriculum, *not* one gated on measured competence. The authors leave held-out-performance strategies to future work.
- **Prerequisite graph / knowledge space.**
  - In knowledge space theory, the feasible knowledge states are the subsets of items a learner can master, and prerequisite structure constrains which subsets are allowed (Doignon & Falmagne, 1985) [V-meta].
  - A state's *outer fringe* is the set of items q for which K ∪ {q} is also a state, i.e., what the learner is "ready to learn" (Cosyn et al., 2021) [V-full].
  - In LMs, Skill-it estimates a directed "skills graph" from measured transfer (Chen et al., 2023) [V-full].
- **Catastrophic forgetting (interference).** Training a network on new material in sequence can abruptly and severely disrupt performance on earlier material (McCloskey & Cohen, 1989 [V-meta]; French, 1999 [V-meta]; definitions also verified in `03_ml_spacing_replay.md`).
- **Replay (rehearsal).** Old items (or generated pseudo-items, called "pseudorehearsal") are interleaved with new training data to reduce forgetting (Robins, 1995 [V-meta]; Rolnick et al., 2019 [V-abs]).
- **Length / OOD generalization.**
  - Length generalization means extrapolating from short training instances to longer ones. Perfect in-distribution accuracy does not guarantee it (Anil et al., 2022) [V-abs].
  - In arithmetic it means, for example, training on ≤20-digit operands and testing on 100-digit ones (McLeish et al., 2024) [V-full].
  - OOD generalization more broadly means correct behaviour on inputs drawn from a distribution different from training.
- **Grokking.** Validation accuracy jumps "from chance to perfect" long after the model has overfit the training set (Power et al., 2022) [V-full]. Nanda et al. (2023) [V-full] explain it as gradual circuit formation followed by cleanup of the memorizing solution.

---

## 1. Human mastery learning and related learning science

### 1.1 Bloom (1968): Learning for mastery [V-abs]
- **Citation:** Bloom, B. S. (1968). Learning for mastery. *Evaluation Comment, 1*(2), 1–12. UCLA Center for the Study of Evaluation of Instructional Programs. (ERIC Document Reproduction Service No. ED053419)
- **URL:** https://eric.ed.gov/?id=ED053419
- **Summary:**
  - Bloom argues that "most students, perhaps over 90 percent, can master what teachers have to teach them."
  - In his model, aptitude is "the amount of time required by the learner to attain mastery," and "time allowed for learning" is "the key to mastery."
- **Relation to our study:** Bloom holds the outcome fixed and lets time vary. Our fixed-clock curriculum is his "conventional" condition. Because mastery learning lets time vary by design, a matched-budget comparison changes what is being tested; it is not just a nuisance control.

### 1.2 Bloom (1984): The 2 sigma problem [V-full]
- **Citation:** Bloom, B. S. (1984). The 2 sigma problem: The search for methods of group instruction as effective as one-to-one tutoring. *Educational Researcher, 13*(6), 4–16. https://doi.org/10.3102/0013189X013006004
- **URL:** https://doi.org/10.3102/0013189X013006004
- **Summary:**
  - The data come from dissertation studies of 3-week units, 11 periods long, in grades 4, 5 and 8.
  - Mastery-learning classes scored about 1 SD above conventional classes, and tutoring plus mastery learning about 2 SD above.
  - Groups got equal instruction time *except* for corrective work.
  - Time on task was 65% (conventional), 75% (mastery) and 90%+ (tutoring).
  - The mastery standard was 80% on a parallel test.
- **Relation to our study:** The comparison was not budget-matched. An LM analogue must equalize tokens and report criterion strictness explicitly.
- **Caveat:** The OCR text gives a mastery aptitude–achievement correlation of +.85, which contradicts Bloom's own prose. Check the page image before quoting it.

### 1.3 VanLehn (2011): Relative effectiveness of tutoring [V-full]
- **Citation:** VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems. *Educational Psychologist, 46*(4), 197–221. https://doi.org/10.1080/00461520.2011.611369
- **URL:** https://doi.org/10.1080/00461520.2011.611369
- **Summary:**
  - Effect sizes: human tutoring d = 0.79; step-based intelligent tutoring systems 0.76; answer-based CAI about 0.31.
  - On Bloom's 2 sigma, VanLehn notes that "the mastery threshold for the tutoring conditions was set at 90% instead of 80%." He writes that "this alone could account for the advantage" of tutoring over mastery learning.
- **Relation to our study:** This is human evidence that the *strictness of the criterion* can drive the outcome. Sweep the probe threshold rather than fixing it at one value.

### 1.4 Keller (1968): "Good-bye, teacher…" (PSI) [V-meta]
- **Citation:** Keller, F. S. (1968). "Good-bye, teacher…". *Journal of Applied Behavior Analysis, 1*(1), 79–89. https://doi.org/10.1901/jaba.1968.1-79
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC1310979/
- **Summary:**
  - The paper introduced the Personalized System of Instruction (the "Keller Plan").
  - Slavin (1987, read) confirms its core mechanism: students "may take the test (or parallel forms of it) as many times as they wish until they achieve a passing score."
  - The usual list of five PSI features is from secondary sources (UNVERIFIED): self-pacing, a unit-perfection requirement, proctors, written materials, and lectures used for motivation.
- **Relation to our study:** PSI is the human analogue of per-skill gating with retests on parallel (held-out) forms.

### 1.5 Kulik, Kulik & Bangert-Drowns (1990): Mastery learning meta-analysis [V-full]
- **Citation:** Kulik, C.-L. C., Kulik, J. A., & Bangert-Drowns, R. L. (1990). Effectiveness of mastery learning programs: A meta-analysis. *Review of Educational Research, 60*(2), 265–299. https://doi.org/10.3102/00346543060002265
- **URL:** https://doi.org/10.3102/00346543060002265
- **Summary:**
  - **Overall:** 108 evaluations. Mean effect on end-of-instruction exams is **0.52 SD** across the **103** studies that reported them (p. 271). By program type, PSI averaged 0.48 and Bloom's Learning for Mastery 0.59.
  - **Test type:** locally developed tests 0.57 (n = 88) vs standardized tests 0.29 (n = 11).
  - **Criterion strictness:** a stricter mastery criterion predicted larger effects (91–100% criterion: 0.64; p = .016 in the regression).
  - **Feedback:** effects were inflated when control groups got less quiz feedback; with matched feedback the mean was 0.40, n = 13.
  - **Aptitude:** 0.61 for low-aptitude students vs 0.40 for high-aptitude students, but the difference was not significant.
  - **Cost:** median extra instructional time was +4% (8 studies), and self-paced PSI lowered course completion (h = −0.14).
- **Relation to our study:** This is the canonical "moderate gain" source, but every moderator maps onto a design choice for us:
  - test alignment ↔ in-distribution vs OOD evaluation;
  - criterion strictness ↔ gate threshold;
  - matched feedback ↔ equal probe and eval exposure across arms.

### 1.6 Slavin (1987): Mastery learning reconsidered [V-full]
- **Citation:** Slavin, R. E. (1987). Mastery learning reconsidered. *Review of Educational Research, 57*(2), 175–213. https://doi.org/10.3102/00346543057002175
- **URL:** https://doi.org/10.3102/00346543057002175
- **Summary:**
  - This is a best-evidence synthesis of group-based mastery learning in studies of 4 weeks or longer.
  - On standardized tests, in equal-time studies, the median effect size was "essentially zero (ES = +.04)" (7 studies).
  - On experimenter-made tests the median was +.255.
  - Extra-time studies gained at posttest, but the gains disappeared within about 4 weeks.
  - Slavin names "unequal time and unequal objectives" as the core confounds, and calls the trade-off a "coverage versus mastery dilemma."
- **Relation to our study:** This is the central critique. Gains can come from extra budget, or from narrowing teaching to the objectives that are gated. Our design therefore needs a matched budget and broad, *non-gated* OOD exams.

### 1.7 Guskey & Pigott (1988); Guskey (2007, 2010) [V-abs / V-full / V-abs]
- **Citations:**
  - Guskey, T. R., & Pigott, T. D. (1988). Research on group-based mastery learning programs: A meta-analysis. *The Journal of Educational Research, 81*(4), 197–216. https://doi.org/10.1080/00220671.1988.10885824
  - Guskey, T. R. (2007). Closing achievement gaps: Revisiting Benjamin S. Bloom's "Learning for Mastery." *Journal of Advanced Academics, 19*(1), 8–31. https://doi.org/10.4219/jaa-2007-704
  - Guskey, T. R. (2010). Lessons of mastery learning. *Educational Leadership, 68*(2), 52–57. https://eric.ed.gov/?id=EJ913779
- **Summary:**
  - **Guskey & Pigott (1988):** 46 studies with "consistently positive effects," but significant heterogeneity. The abstract gives no pooled effect size. Hattie's figure of 0.61 for this paper is secondary.
  - **Guskey (2007, author PDF read):** describes the core of mastery learning as formative assessment, then individualized correctives, then "a second formative assessment" that is "parallel … but includes slightly different problems."
  - **Guskey (2010):** adds enrichment for learners who have already mastered a unit.
- **Relation to our study:** The second parallel assessment is the human precedent for gating on *held-out* probes rather than training accuracy. Heterogeneity across studies predicts that effects will vary by skill domain.

### 1.8 Rawson & Dunlosky (2011); Rawson, Dunlosky & Sciartelli (2013): Criterion and relearning [V-abs]
- **Citations:**
  - Rawson, K. A., & Dunlosky, J. (2011). Optimizing schedules of retrieval practice for durable and efficient learning: How much is enough? *Journal of Experimental Psychology: General, 140*(3), 283–302. https://doi.org/10.1037/a0023956
  - Rawson, K. A., Dunlosky, J., & Sciartelli, S. M. (2013). The power of successive relearning: Improving performance on course exams and long-term retention. *Educational Psychology Review, 25*(4), 523–548. https://doi.org/10.1007/s10648-013-9240-4
- **URL:** https://doi.org/10.1037/a0023956 ; https://doi.org/10.1007/s10648-013-9240-4
- **Summary:**
  - **Rawson & Dunlosky (2011):** 533 students in 3 experiments, with an initial criterion of 1–4 correct recalls followed by 1–5 relearning sessions.
    - Initial criterion and relearning were "subadditive." The criterion's effects "were strong prior to relearning but then diminished as relearning increased."
    - Relearning gave large retention gains "with a relatively minimal cost."
    - Their recommendation: an initial criterion of 3, then 3 spaced relearning sessions.
  - **Rawson et al. (2013):** successive relearning in a real course improved exam scores and long-term retention. The specific percentages are UNVERIFIED.
- **Relation to our study:** This is the strongest human support for *gate + replay* over a stricter gate alone at a matched budget.

### 1.9 Knowledge space theory and ALEKS [V-meta / V-full]
- **Citations:**
  - Doignon, J.-P., & Falmagne, J.-C. (1985). Spaces for the assessment of knowledge. *International Journal of Man-Machine Studies, 23*(2), 175–196. https://doi.org/10.1016/S0020-7373(85)80031-6 [V-meta]
  - Doignon, J.-P., & Falmagne, J.-C. (1999). *Knowledge spaces*. Springer. https://doi.org/10.1007/978-3-642-58625-5 [V-meta]
  - Falmagne, J.-C., & Doignon, J.-P. (2011). *Learning spaces: Interdisciplinary applied mathematics*. Springer. https://doi.org/10.1007/978-3-642-01039-2 [V-meta]
  - Cosyn, E., Uzun, H., Doble, C., & Matayoshi, J. (2021). A practical perspective on knowledge space theory: ALEKS and its data. *Journal of Mathematical Psychology, 101*, Article 102512. https://doi.org/10.1016/j.jmp.2021.102512 [V-full, preprint]
- **Summary:**
  - Knowledge space theory models feasible knowledge states under prerequisite constraints.
  - ALEKS implements it:
    - The student practices items from the outer fringe, i.e., what they are "ready to learn."
    - If the student "performs well enough," ALEKS "temporarily" treats the item as learned.
    - Periodic progress assessments then re-test and can remove items that have been forgotten.
- **Relation to our study:** This is the formal basis for unlocking along a skill graph, and ALEKS is a deployed gate + re-assessment system. Mastery there is *provisional* until re-verified, which argues for re-probing mastered skills.

### 1.10 Corbett & Anderson (1992, 1994): Knowledge tracing [V-full / V-meta]
- **Citations:**
  - Corbett, A. T., & Anderson, J. R. (1992). Knowledge tracing in the ACT Programming Tutor. In *Proceedings of the Fourteenth Annual Conference of the Cognitive Science Society*. Erlbaum. https://escholarship.org/uc/item/0873k1nz [V-full]
  - Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction, 4*(4), 253–278. https://doi.org/10.1007/BF01099821 [V-meta; Crossref gives the issue year as 1995]
- **Summary:**
  - Bayesian knowledge tracing (BKT) maintains P(learned) for each skill.
  - The 1992 paper states: "Mastery is defined in the tutor as a learning probability of at least 0.95."
  - The number of exercises a student needed to reach the criterion still correlated with posttest performance (r = −0.52).
- **Relation to our study:** This gives a precise historical gate. Passing the gate does not equalize learners, so log *steps-to-mastery* for each skill.

### 1.11 Pelánek & Řihák (2017); Pelánek (2017); Zhang et al. (2025): Choosing mastery criteria [V-full / V-meta / V-full]
- **Citations:**
  - Pelánek, R., & Řihák, J. (2017). Experimental analysis of mastery learning criteria. In *Proceedings of the 25th Conference on User Modeling, Adaptation and Personalization* (pp. 156–163). ACM. https://doi.org/10.1145/3079628.3079667
  - Pelánek, R. (2017). Bayesian knowledge tracing, logistic models, and beyond: An overview of learner modeling techniques. *User Modeling and User-Adapted Interaction, 27*(3–5), 313–350. https://doi.org/10.1007/s11257-017-9193-2
  - Zhang, J., Vanacore, K., Baker, R. S., Ch, N., Mills, C., & Henkel, O. (2025). How much mastery is enough mastery? The relationship between mastery in a lesson and the performance on the subsequent lesson. In *Proceedings of the 18th International Conference on Educational Data Mining* (short paper). https://educationaldatamining.org/edm2025/proceedings/2025.EDM.short-papers.4/2025.EDM.short-papers.4.pdf
- **Summary:**
  - **Pelánek & Řihák (2017):**
    - Main finding: "The choice of data sources used for mastery decision and setting of thresholds are more important than the choice of a learner modeling technique."
    - They recommend an exponential-moving-average criterion, and find 0.95 a "reasonable compromise."
  - **Zhang et al. (2025), observational, BKT in a math tutor:** a stricter threshold (0.98) is associated with better performance on the *next* lesson.
- **Relation to our study:** Use a simple exponential moving average of probe accuracy as the gate, and sweep the threshold. A stricter gate may speed up learning of downstream skills, which can be tested directly.

### 1.12 Deployed mastery gating: Ritter et al. (2016) [V-abs, partial]; Israni, Sales & Pane (2018) [V-abs]
- **Citations:**
  - Ritter, S., Yudelson, M., Fancsali, S. E., & Berman, S. R. (2016). How mastery learning works at scale. In *Proceedings of the Third ACM Conference on Learning @ Scale* (pp. 71–79). ACM. https://doi.org/10.1145/2876034.2876039
  - Israni, A., Sales, A. C., & Pane, J. F. (2018). *Mastery learning in practice: A (mostly) descriptive analysis of log data from the Cognitive Tutor Algebra I effectiveness trial* (arXiv:1802.08616). arXiv. https://arxiv.org/abs/1802.08616
- **Summary:**
  - **Ritter et al. (2016):** findings UNVERIFIED (only the opening of the abstract was accessible). Cite for context only.
  - **Israni et al. (2018):**
    - In a large randomized trial of the Cognitive Tutor, students often left sections before reaching mastery or did units out of order.
    - Observationally, teachers moving students on before mastery "appears to lower posttest scores."
    - This is a preprint.
- **Relation to our study:** The force-advance condition is our clock arm. In deployment, gates are often bypassed.

### 1.13 Mastery learning in medical simulation: Cook et al. (2013) [V-abs]; McGaghie et al. (2014) [V-abs]
- **Citations:**
  - Cook, D. A., Brydges, R., Zendejas, B., Hamstra, S. J., & Hatala, R. (2013). Mastery learning for health professionals using technology-enhanced simulation: A systematic review and meta-analysis. *Academic Medicine, 88*(8), 1178–1186. https://doi.org/10.1097/ACM.0b013e31829a365d
  - McGaghie, W. C., Issenberg, S. B., Barsuk, J. H., & Wayne, D. B. (2014). A critical review of simulation-based mastery learning with translational outcomes. *Medical Education, 48*(4), 375–385. https://doi.org/10.1111/medu.12391
- **Summary:**
  - **Cook et al. (2013), 82 studies:**
    - Skills ES 1.29 vs *no intervention*.
    - ES 1.17 vs *non-mastery* training, but from only **3 studies**, which also "required more time."
  - **McGaghie et al. (2014):** a qualitative review of 23 studies with outcomes that carried over to patient care.
- **Relation to our study:** The fair comparison, mastery vs non-mastery at matched time, rests on very little evidence even in humans.
- **Coverage note:** We found no high-quality K–12 or college mastery-learning meta-analysis from 2015–2026 through API searches. Hattie's MetaX entry (16 meta-analyses, weighted ES ≈ 0.67, https://www.visiblelearningmetax.com/influences/view/mastery_learning) [V-abs] mis-codes Kulik (1990) and Cook (2013), so use it only as a pointer. The EEF Toolkit is UNVERIFIED.

### 1.14 ZPD, scaffolding, desirable difficulties, expertise reversal
- **Vygotsky (1978)** [V-meta]: Vygotsky, L. S. (1978). *Mind in society: The development of higher psychological processes* (M. Cole, V. John-Steiner, S. Scribner, & E. Souberman, Eds.). Harvard University Press. https://doi.org/10.2307/j.ctvjf9vz4
  - Summary: introduces the zone of proximal development. The exact wording on p. 86 is UNVERIFIED.
  - Relation to our study: the outer fringe of a skill graph is a formal analogue.
- **Wood, Bruner & Ross (1976)** [V-meta]: Wood, D., Bruner, J. S., & Ross, G. (1976). The role of tutoring in problem solving. *Journal of Child Psychology and Psychiatry, 17*(2), 89–100. https://doi.org/10.1111/j.1469-7610.1976.tb00381.x
  - Summary: the origin of "scaffolding." Exact wording UNVERIFIED.
- **Bjork (1994)** [V-meta]: Bjork, R. A. (1994). Memory and metamemory considerations in the training of human beings. In J. Metcalfe & A. P. Shimamura (Eds.), *Metacognition: Knowing about knowing* (pp. 185–205). MIT Press. https://doi.org/10.7551/mitpress/4561.003.0011
  - Summary: the usual source for "desirable difficulties," i.e., performance during training is an unreliable index of learning. Content not re-read this session.
  - Relation to our study: gates should use held-out, OOD or delayed probes, not training accuracy.
- **Kalyuga, Ayres, Chandler & Sweller (2003)** [V-abs]: Kalyuga, S., Ayres, P., Chandler, P., & Sweller, J. (2003). The expertise reversal effect. *Educational Psychologist, 38*(1), 23–31. https://doi.org/10.1207/S15326985EP3801_4
  - Summary: techniques "highly effective with inexperienced learners can lose their effectiveness and even have negative consequences" for experienced learners.
  - Relation to our study: replaying *easy* material after it is mastered may be wasteful. This is consistent with E2H Reasoner's finding that fading out easy tasks helps (§2.5), and it argues for keeping replay fractions small.

---

## 2. Curriculum learning in machine learning

### 2.1 Foundations

#### Elman (1993): Starting small [V-meta]
- **Citation:** Elman, J. L. (1993). Learning and development in neural networks: The importance of starting small. *Cognition, 48*(1), 71–99. https://doi.org/10.1016/0010-0277(93)90058-4
- **URL:** https://doi.org/10.1016/0010-0277(93)90058-4
- **Model scale:** ⚠ non-LM. A simple recurrent network on an artificial grammar.
- **Summary:** As described by Rohde & Plaut (full text read), a network trained on the full grammar failed. "Starting small" succeeded, either by staging data from simple to complex across fixed epoch-based stages, or by first limiting memory.
- **Relation to our study:** This is an ancestor of the *fixed* (clock-based) arm.

#### Rohde & Plaut (1999): How important is starting small? [V-abs; the authors' 2003 follow-up chapter V-full]
- **Citation:** Rohde, D. L. T., & Plaut, D. C. (1999). Language acquisition in the absence of explicit negative evidence: How important is starting small? *Cognition, 72*(1), 67–109. https://doi.org/10.1016/S0010-0277(99)00031-1
- **URL:** https://doi.org/10.1016/S0010-0277(99)00031-1 ; chapter: https://ni.cmu.edu/~plaut/papers/pdf/RohdePlaut03chap.less-is-less.pdf
- **Model scale:** ⚠ non-LM (simple recurrent networks).
- **Summary:**
  - The authors failed to replicate Elman's result. With well-chosen training parameters, the no-curriculum regimen won (divergence 0.025 vs 0.036, p < .001).
  - Elman's apparent advantage depended on his particular initial weights and optimizer settings.
- **Relation to our study:** Badly tuned baselines can create curriculum effects that are not real. Give the shuffle arm the same tuning effort as the others.

#### Bengio, Louradour, Collobert & Weston (2009): Curriculum learning [V-full]
- **Citation:** Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. In *Proceedings of the 26th Annual International Conference on Machine Learning* (pp. 41–48). ACM. https://doi.org/10.1145/1553374.1553380
- **URL:** https://icml.cc/2009/papers/119.pdf
- **Model scale:** ⚠ tiny networks.
  - A 3-hidden-layer MLP for shape recognition.
  - A window-based ranking language model: 5-word window, 50-dimensional embeddings, 100 hidden units. That is about 1M parameters (our estimate).
- **Summary:**
  - Both curricula were *fixed* schedules: an easy-to-hard switch epoch for shapes, and a vocabulary grown by 5k words per pass for the language model.
  - The language model ended at log-rank 2.78 vs 2.83 without curriculum, a small but significant gain.
- **Relation to our study:** This is the founding fixed-curriculum result, and its LM gain is small.

#### Kumar, Packer & Koller (2010): Self-paced learning [V-abs]
- **Citation:** Kumar, M. P., Packer, B., & Koller, D. (2010). Self-paced learning for latent variable models. In *Advances in Neural Information Processing Systems 23* (pp. 1189–1197). Curran Associates.
- **URL:** https://proceedings.neurips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html
- **Model scale:** ⚠ non-neural (latent structural SVM).
- **Summary:** Easy samples are chosen by the current model's loss, and the admission weight is annealed over training. It beat the standard latent structural SVM on 4 tasks.
- **Relation to our study:** This is the first curriculum driven by the learner's state. Admission is paced by an annealed hyperparameter, not decided by a mastery test.

#### Zaremba & Sutskever (2014): Learning to Execute [V-full; released code read via WebFetch, not cloned]
- **Citation:** Zaremba, W., & Sutskever, I. (2014). *Learning to execute* (arXiv:1410.4615). arXiv. https://doi.org/10.48550/arXiv.1410.4615
  - Version 3 is headed "Under review as a conference paper at ICLR 2015." Acceptance is UNVERIFIED, so cite it as arXiv.
- **URL:** https://arxiv.org/abs/1410.4615
- **Model scale:** ⚠ non-Transformer. A 2-layer LSTM with 400 cells per layer (about 2.5M parameters), character-level.
- **Summary:**
  - **Tasks:** evaluating short Python-like programs, 9-digit addition, and copying (memorization). Difficulty is set by `length` (digits) and `nesting`.
  - **Four strategies:**
    - *baseline*: train only at the target difficulty;
    - *naive*: start at length 1, nesting 1, and increase length "once learning stops making progress on the validation set." In the released code, it also advances when validation accuracy exceeds 0.95, and validation data is generated *at the current level*;
    - *mix*: length and nesting drawn uniformly over the full target range for each sample;
    - *combined*: each sample comes from naive or from mix (about 80/20 in the code; the ratio is not stated in the paper).
  - **Results:**
    - The abstract says: "conventional curriculum learning proved ineffective," while the new variant "improved our networks' performance in all experimental conditions."
    - Naive "sometime[s] perform[ed] worse than baseline."
    - Combined outperformed all other strategies "in every configuration on program evaluation" and reached 99% on 9-digit addition (teacher-forced).
    - Combined was "generally (but not always)" better than mix, and on memorization "no longer outperforms the mixed strategy in every experimental setting."
    - Train and test came from the same distribution, so there was no OOD test.
- **Relation to our study:**
  - Naive is the closest classic analogue of our *mastery-gated, no replay* arm, and it was unreliable.
  - Combined fixes this by mixing in samples from **the whole range, including harder levels not yet unlocked**. That differs from replaying mastered skills.
  - The authors offer an untested "hidden state allocation hypothesis": capacity gets committed to the easy examples and then "might be difficult" to restructure.

#### Graves, Bellemare, Menick, Munos & Kavukcuoglu (2017): Automated curriculum learning [V-full]
- **Citation:** Graves, A., Bellemare, M. G., Menick, J., Munos, R., & Kavukcuoglu, K. (2017). Automated curriculum learning for neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR Vol. 70, pp. 1311–1320).
- **URL:** https://proceedings.mlr.press/v70/graves17a.html
- **Model scale:** ⚠ small LSTMs (1–2 layers of 512 units).
- **Summary:**
  - Task selection is a nonstationary bandit (Exp3.S), rewarded by learning-progress signals such as prediction gain and complexity gain.
  - Some signals reached the target about 2× faster than uniform sampling. Others were "much worse than uniform."
  - The authors note that "uniformly sampling from all tasks is a surprisingly strong benchmark."
- **Relation to our study:** This curriculum is driven by learning progress, not a threshold. It shows that shuffle (uniform) is a hard baseline to beat.

#### Matiisen, Oliver, Cohen & Schulman (2020): Teacher–student curriculum learning [V-full]
- **Citation:** Matiisen, T., Oliver, A., Cohen, T., & Schulman, J. (2020). Teacher–student curriculum learning. *IEEE Transactions on Neural Networks and Learning Systems, 31*(9), 3732–3740. https://doi.org/10.1109/TNNLS.2019.2934906
- **URL:** https://arxiv.org/abs/1707.00183
- **Model scale:** ⚠ small. An LSTM seq2seq with 128 units for decimal addition, and a convolutional + LSTM PPO policy for Minecraft.
- **Summary:**
  - The teacher samples tasks with the highest *absolute* slope of the learning curve. When scores fall, "unlearning occurred and this task should be practiced more."
  - On 9-digit addition it beat uniform sampling and Zaremba & Sutskever's best manual curriculum (combined).
  - The authors say the absolute-value trick "was crucial," and that "uniform sampling is surprisingly efficient."
  - They list choosing a "mastery" threshold as a burden of manual curricula.
- **Relation to our study:** This is the **strongest neural precedent for gate + replay**: selection driven by performance, with replay of skills that are decaying built in.

#### Platanios, Stretcu, Neubig, Poczos & Mitchell (2019): Competence-based curriculum for NMT [V-full]
- **Citation:** Platanios, E. A., Stretcu, O., Neubig, G., Poczos, B., & Mitchell, T. (2019). Competence-based curriculum learning for neural machine translation. In *Proceedings of the 2019 Conference of the North American Chapter of the ACL: Human Language Technologies, Vol. 1* (pp. 1162–1172). ACL. https://doi.org/10.18653/v1/N19-1119
- **URL:** https://aclanthology.org/N19-1119/
- **Model scale:** ⚠ NMT encoder–decoder models: LSTMs and Transformer-base (6 layers). Datasets range from 133k to 4.5M sentence pairs.
- **Summary:**
  - Competence c(t) is a **fixed function of training step** (linear or square root, c0 = 0.01). Difficulty is sentence length or word rarity.
  - Results: "up to a 70% decrease in training time" and "up to 2.2 BLEU" (Transformer En→De: 27.95 → 30.16).
  - RNNs gained much less.
  - The authors explicitly do not consider strategies that depend "on the learner's performance on held-out data."
- **Relation to our study:** Despite its name, this belongs to our **fixed (clock-based) arm**. Our mastery gate is roughly the variant it leaves to future work.

#### Hacohen & Weinshall (2019): On the power of curriculum learning [V-full]
- **Citation:** Hacohen, G., & Weinshall, D. (2019). On the power of curriculum learning in training deep networks. In *Proceedings of the 36th International Conference on Machine Learning* (PMLR Vol. 97, pp. 2535–2544).
- **URL:** https://proceedings.mlr.press/v97/hacohen19a.html
- **Model scale:** ⚠ non-LM. Small CNNs and VGG on CIFAR and an ImageNet subset.
- **Summary:**
  - With transfer-based scoring and step-based pacing, the curriculum learned faster and ended modestly better than random or anti-curriculum.
  - *Self-paced* scoring by the current model "impairs learning."
- **Relation to our study:** This is a fixed-pacing reference. It is a caution that difficulty measured from the learner's *current* state can hurt.

#### Weinshall, Cohen & Amir (2018): Curriculum by transfer learning [V-abs]
- **Citation:** Weinshall, D., Cohen, G., & Amir, D. (2018). Curriculum learning by transfer learning: Theory and experiments with deep networks. In *Proceedings of the 35th International Conference on Machine Learning* (PMLR Vol. 80, pp. 5238–5246).
- **URL:** https://proceedings.mlr.press/v80/weinshall18a.html
- **Model scale:** ⚠ linear regression theory, plus CNNs.
- **Summary:** For linear regression trained with SGD, convergence is faster on easier examples. Empirically, the curriculum mainly speeds up the *start* of training.
- **Relation to our study:** Curriculum benefits are front-loaded, which matters under a fixed budget.

#### Wu, Dyer & Neyshabur (2021): When do curricula work? [V-full]
- **Citation:** Wu, X., Dyer, E., & Neyshabur, B. (2021). When do curricula work? In *International Conference on Learning Representations (ICLR 2021)* [oral]. https://openreview.net/forum?id=tW4QEInpni
- **URL:** https://arxiv.org/abs/2012.03107
- **Model scale:** ⚠ non-LM. More than 25,000 image classifiers (FC, VGG, ResNet, WRN, DenseNet, EfficientNet) on CIFAR-10/100 and FOOD101(N).
- **Summary:**
  - Examples are learned in a highly consistent order across architectures (an "implicit curriculum").
  - Under standard training (about 100 epochs), curricula gave "only marginal benefits." Random ordering matched curriculum and anti-curriculum, so "any benefit is entirely due to the dynamic training set size."
  - Curricula helped with **short budgets** and with **noisy labels** (20–80% noise).
  - All pacing functions were fixed schedules.
- **Relation to our study:**
  - Expect any benefit to be largest at *small* token budgets, so run a budget sweep.
  - Add a control that uses the same pacing but random ordering, to separate the effect of a growing training set from the effect of ordering.

### 2.2 Surveys

#### Soviany, Ionescu, Rota & Sebe (2022): Curriculum learning: A survey [V-abs]
- **Citation:** Soviany, P., Ionescu, R. T., Rota, P., & Sebe, N. (2022). Curriculum learning: A survey. *International Journal of Computer Vision, 130*(6), 1526–1565. https://doi.org/10.1007/s11263-022-01611-x
- **URL:** https://arxiv.org/abs/2101.10382
- **Model scale:** N/A (survey).
- **Summary:** The survey splits curriculum design into two parts, a difficulty ranking and a pacing function, and organizes methods in a multi-perspective taxonomy.
- **Relation to our study:** Provides standard vocabulary for describing our arms.

#### Wang, Chen & Zhu (2022): A survey on curriculum learning [V-abs]
- **Citation:** Wang, X., Chen, Y., & Zhu, W. (2022). A survey on curriculum learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 44*(9), 4555–4576. https://doi.org/10.1109/TPAMI.2021.3069908
- **URL:** https://arxiv.org/abs/2010.13166
- **Model scale:** N/A (survey).
- **Summary:**
  - Frames curriculum learning as "Difficulty Measurer + Training Scheduler."
  - Groups automatic curriculum learning into self-paced learning, Transfer Teacher, RL Teacher, and other methods.
  - Early access was in 2021, which explains why the year is sometimes cited as 2021.
- **Relation to our study:** Our gate is a training scheduler driven by learner feedback.

#### Portelas, Colas, Weng, Hofmann & Oudeyer (2020): ACL for deep RL [V-abs]
- **Citation:** Portelas, R., Colas, C., Weng, L., Hofmann, K., & Oudeyer, P.-Y. (2020). Automatic curriculum learning for deep RL: A short survey. In *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence* (pp. 4819–4825). https://doi.org/10.24963/ijcai.2020/671
- **URL:** https://www.ijcai.org/proceedings/2020/0671.pdf
- **Model scale:** N/A (survey).
- **Summary:** A short survey of automatic curricula in deep RL, including task selection driven by learning progress.
- **Relation to our study:** This is the lineage behind learning-progress and competence gating.

#### Narvekar, Peng, Leonetti, Sinapov, Taylor & Stone (2020): Curriculum learning for RL domains [V-abs]
- **Citation:** Narvekar, S., Peng, B., Leonetti, M., Sinapov, J., Taylor, M. E., & Stone, P. (2020). Curriculum learning for reinforcement learning domains: A framework and survey. *Journal of Machine Learning Research, 21*(181), 1–50.
- **URL:** https://jmlr.org/papers/v21/20-212.html
- **Model scale:** N/A (survey).
- **Summary:** Background on task sequencing in RL.
- **Relation to our study:** Background only.

### 2.3 Theory

#### Saglietti, Sarao Mannelli & Saxe (2022): Analytical theory of curriculum learning [V-full]
- **Citation:** Saglietti, L., Sarao Mannelli, S., & Saxe, A. (2022). An analytical theory of curriculum learning in teacher–student networks. *Journal of Statistical Mechanics: Theory and Experiment, 2022*(11), 114014. https://doi.org/10.1088/1742-5468/ac9b3c
  - arXiv says "Accepted to NeurIPS 2022"; the proceedings entry is UNVERIFIED.
- **URL:** https://arxiv.org/abs/2106.08068
- **Model scale:** ⚠ non-LM. Single-layer perceptron teacher and student in the high-dimensional limit, plus a small CIFAR-10 experiment.
- **Summary:**
  - In online learning, the curriculum speeds learning but barely changes final performance.
  - In batch learning run to convergence, it gives no generalization benefit, because the last slice of data determines the minimum.
  - Adding an elastic coupling between phases (similar to EWC) yields "a large improvement."
- **Relation to our study:** This is a theoretical argument that a curriculum needs something to protect earlier phases from being overwritten. That supports the replay component.

#### Abbe, Cornacchia & Lotfi (2023): Curriculum on parity targets with mixed inputs [V-abs]
- **Citation:** Abbe, E., Cornacchia, E., & Lotfi, A. (2023). Provable advantage of curriculum learning on parity targets with mixed inputs. In *Advances in Neural Information Processing Systems 36* (pp. 24291–24321).
- **URL:** https://papers.nips.cc/paper_files/paper/2023/hash/4c8ce3c63f6b66d6811c6d67c68e487b-Abstract-Conference.html
- **Model scale:** ⚠ non-LM (2-layer ReLU networks on Boolean inputs).
- **Summary:** Training first on sparse inputs makes parities of large degree learnable. A network trained with noisy gradient descent on unordered samples cannot learn them without many more steps.
- **Relation to our study:** Provable benefit of ordering, under a fixed schedule.

#### Cornacchia & Mossel (2023): Mathematical model of curriculum for parities [V-abs]
- **Citation:** Cornacchia, E., & Mossel, E. (2023). A mathematical model for curriculum learning for parities. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 6402–6423).
- **URL:** https://proceedings.mlr.press/v202/cornacchia23a.html
- **Model scale:** ⚠ non-LM.
- **Summary:** Sampling from two or more product distributions can sharply reduce computational cost. For "Hamming mixtures," bounded curricula give no benefit.
- **Relation to our study:** Whether a curriculum helps depends on the structure of the data.

#### Abbe, Bengio, Lotfi & Rizk (2023): Generalization on the unseen and degree curriculum [V-full]
- **Citation:** Abbe, E., Bengio, S., Lotfi, A., & Rizk, K. (2023). Generalization on the unseen, logic reasoning and degree curriculum. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 31–60).
- **URL:** https://proceedings.mlr.press/v202/abbe23a.html (arXiv:2301.13105)
- **Model scale:** ⚠ non-LM. MLPs, small Transformers and other models on Boolean functions. The curriculum experiments used MLPs.
- **Summary:**
  - Models trained with (S)GD learn a minimum-degree interpolator on unseen input regions, which explains failures of length generalization.
  - **Degree-Curriculum is performance-gated**: it advances to the next Hamming radius only when training loss ≤ ε = 0.001.
  - With it, an MLP learned a degree-30 parity that it could not learn otherwise "even up to 10^5 samples."
  - We read the extended arXiv version.
- **Relation to our study:** This is the most direct precedent for a *gated* curriculum aimed at OOD generalization. Its gate uses *training* loss; ours uses held-out OOD probes.

### 2.4 Curricula for pretraining small LMs: largely null or fragile

#### Warstadt et al. (2023): BabyLM 2023 findings [V-full]
- **Citation:** Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., Mosquera, R., Paranjape, B., Williams, A., Linzen, T., & Cotterell, R. (2023). Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning* (pp. 1–34). ACL. https://doi.org/10.18653/v1/2023.conll-babylm.1
- **URL:** https://aclanthology.org/2023.conll-babylm.1/
- **Model scale:** Small LMs (baselines up to OPT-125M and RoBERTa-base) trained on 10M or 100M words.
- **Summary:** 13 of 31 teams tried curricula. These attempts "were largely unsuccessful, though some showed modest improvements." "The majority of these attempts did not produce consistent improvements."
- **Relation to our study:** This is the main negative prior. Almost all entries used fixed-pacing data ordering, not mastery gates.

#### Hu et al. (2024): Second BabyLM Challenge findings [V-full]
- **Citation:** Hu, M. Y., Mueller, A., Ross, C., Williams, A., Linzen, T., Zhuang, C., Cotterell, R., Choshen, L., Warstadt, A., & Wilcox, E. G. (2024). Findings of the second BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning* (pp. 1–21). ACL.
- **URL:** https://aclanthology.org/2024.conll-babylm.1.pdf
- **Model scale:** Small LMs trained on 10M or 100M words.
- **Summary:** Curriculum learning "did not lead to high scores, on average." Its regression coefficient was β = −3.6 (p = 0.055). The organizers urge participants to "think beyond data order."
- **Relation to our study:** Reinforces the negative prior for data-order curricula in small LMs.

#### Diehl Martinez et al. (2023): CLIMB [V-full]
- **Citation:** Diehl Martinez, R., Goriely, Z., McGovern, H., Davis, C., Caines, A., Buttery, P., & Beinborn, L. (2023). CLIMB – Curriculum learning for infant-inspired model building. In *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning* (pp. 112–127). ACL. https://doi.org/10.18653/v1/2023.conll-babylm.10
- **URL:** https://aclanthology.org/2023.conll-babylm.10/
- **Model scale:** A small 8-layer, hidden-size-256 RoBERTa-style model (about 10M parameters by our estimate), trained on 10M words.
- **Summary:** Vocabulary, data (including model-based "self-perplexity") and objective curricula "did not yield consistent improvements" over the non-curriculum baseline.
- **Relation to our study:** A careful null result. Even a difficulty measure driven by the learner, with fixed pacing, did not help.

#### Campos (2021): Curriculum learning for language modeling [V-full]
- **Citation:** Campos, D. (2021). *Curriculum learning for language modeling* (arXiv:2108.02170). arXiv. https://doi.org/10.48550/arXiv.2108.02170
- **URL:** https://arxiv.org/abs/2108.02170
- **Model scale:** ELMo trained on WikiText-2 and WikiText-103.
- **Summary:** Using Platanios-style competence curricula, the author reports: "We do not find compelling evidence that curriculum learning methods improve language model training." Some small-corpus benefits disappeared as the corpus grew.
- **Relation to our study:** A small-LM null result for the clock-based (fixed) arm.

#### Li, Zhang & He (2022): Sequence length warmup [V-abs]
- **Citation:** Li, C., Zhang, M., & He, Y. (2022). The stability-efficiency dilemma: Investigating sequence length warmup for training GPT models. In *Advances in Neural Information Processing Systems 35*.
- **URL:** https://arxiv.org/abs/2108.06084
- **Model scale:** GPT-2 117M and 1.5B, and a GPT-3-style 125M model. This is within our range.
- **Summary:**
  - Growing sequence length during training enables "8x larger batch size and 4x larger learning rate."
  - It reaches the same zero-shot results with up to 2.2× fewer tokens.
  - For the GPT-3 125M model it retains "99% of the zero-shot accuracy … using 10x less data."
- **Relation to our study:** A positive curriculum result at our scale, but it works mainly through *optimization stability*, not skill mastery.

#### Kocmi & Bojar (2017) [V-abs] and Xu et al. (2020) [V-abs]: Early NLP curricula
- **Citations:**
  - Kocmi, T., & Bojar, O. (2017). Curriculum learning and minibatch bucketing in neural machine translation. In *Proceedings of RANLP 2017* (pp. 379–386). https://doi.org/10.26615/978-954-452-049-6_050
  - Xu, B., Zhang, L., Mao, Z., Wang, Q., Xie, H., & Zhang, Y. (2020). Curriculum learning for natural language understanding. In *Proceedings of the 58th Annual Meeting of the ACL* (pp. 6095–6104). https://doi.org/10.18653/v1/2020.acl-main.542
- **URL:** https://aclanthology.org/R17-1050/ ; https://aclanthology.org/2020.acl-main.542/
- **Model scale:** NMT (Kocmi & Bojar; sizes UNVERIFIED); fine-tuning of pretrained LMs (Xu et al.; sizes UNVERIFIED).
- **Summary:** Kocmi & Bojar report "small improvement" from some curricula. Xu et al. report gains from "Cross Review" difficulty scoring during fine-tuning (no numbers in the abstract).
- **Relation to our study:** Weak positive priors.

### 2.5 Curricula, data ordering and skill mixing for LLMs (2023–2026)

#### Chen et al. (2023): Skill-it! [V-full]
- **Citation:** Chen, M. F., Roberts, N., Bhatia, K., Wang, J., Zhang, C., Sala, F., & Ré, C. (2023). Skill-it! A data-driven skills framework for understanding and training language models. In *Advances in Neural Information Processing Systems 36* (pp. 36000–36040).
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2023/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html (arXiv:2307.14430)
- **Model scale:**
  - GPT-Neo 125M, continually pretrained (not from scratch), for LEGO, 3-digit addition and Natural Instructions.
  - GPT-Neo 1.3B in an appendix.
  - A 3B model for RedPajama.
- **Summary:**
  - The skills graph is estimated from transfer: does training on skill A lower the loss on skill B?
  - Online sampling reweights skills with exponential weights driven by *validation loss*. It is soft reweighting and never hard-gates a skill.
  - **Results:**
    - LEGO: "37.5 points higher accuracy than random sampling" (NeurIPS abstract, confirmed; arXiv v1 says 36.5).
    - Natural Instructions: 13.6% lower validation loss on the target skill, compared with training only on that skill's data.
    - RedPajama: with 1B tokens it beats uniform sampling with 3B tokens.
- **Relation to our study:**
  - This is the closest prerequisite-graph precedent, and it **uses the same LEGO and addition synthetic tasks** as our preliminary experiments.
  - It differs in three ways: soft vs hard gating; in-distribution validation loss vs OOD probes; pretrained vs from-scratch models.

#### Chen, Hu, Lourie, Cho & Ré (2025): Aioli [V-full]
- **Citation:** Chen, M. F., Hu, M. Y., Lourie, N., Cho, K., & Ré, C. (2025). Aioli: A unified optimization framework for language model data mixing. In *International Conference on Learning Representations (ICLR 2025)*.
- **URL:** https://arxiv.org/abs/2411.05735
- **Model scale:** 160M GPT-style models on SlimPajama subsets.
- **Summary:**
  - Unifies DoReMi, Skill-it and other mixers within one framework.
  - Many existing mixing methods failed to beat **stratified sampling**, some by as much as 6.9 points.
  - Aioli itself beat stratified sampling on 6 of 6 datasets (by 0.27 perplexity on average).
- **Relation to our study:** Make the shuffle baseline *stratified by skill*. It is a strong baseline.

#### Xie et al. (2023): DoReMi [V-abs] and Albalak et al. (2023): ODM [V-full]
- **Citations:**
  - Xie, S. M., Pham, H., Dong, X., Du, N., Liu, H., Lu, Y., Liang, P., Le, Q. V., Ma, T., & Yu, A. W. (2023). DoReMi: Optimizing data mixtures speeds up language model pretraining. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2305.10429
  - Albalak, A., Pan, L., Raffel, C., & Wang, W. Y. (2023). *Efficient online data mixing for language model pre-training* (arXiv:2312.02406). arXiv. https://arxiv.org/abs/2312.02406
- **Model scale:** DoReMi uses a 280M proxy model to set weights for an 8B target model (⚠ the target is only at 8B). ODM trains a 1B Pythia-configuration model from scratch.
- **Summary:**
  - **DoReMi:** learns domain weights with group-DRO, giving +6.5 points few-shot and 2.6× fewer steps.
  - **ODM:** an EXP3 bandit rewarded by *training* loss reached the next-best method's final perplexity in 19% fewer iterations.
- **Relation to our study:** Both are soft reweighting from loss signals, with no gating and no ordering.

#### Zhang, Mohamed, Abdine, Shang & Vazirgiannis (2026): Beyond random sampling [V-full (HTML)]
- **Citation:** Zhang, Y., Mohamed, A., Abdine, H., Shang, G., & Vazirgiannis, M. (2026). Beyond random sampling: Efficient language model pretraining via curriculum learning. In *Proceedings of the 19th Conference of the European Chapter of the ACL (Vol. 1: Long Papers)* (pp. 5776–5794).
- **URL:** https://aclanthology.org/2026.eacl-long.271/ (arXiv:2506.11300)
- **Model scale:** 0.5B LLaMA-style models trained from scratch (mostly 10B tokens), plus 1B and 3B runs. More than 200 runs in total.
- **Summary:**
  - Curricula reduced the steps needed to reach baseline performance by 18–45%.
  - Using the curriculum as a warmup gave up to +3.5%.
  - Compression ratio, MTLD and Flesch reading ease were the best difficulty metrics.
  - **All runs used a constant LR of 2e-4 with no decay.**
- **Relation to our study:** A clock-based curriculum at our scale. Its positive result was obtained in exactly the LR regime that Luo et al. show favours curricula.

#### Luo et al. (2026): LR decay wastes your best data [V-full]
- **Citation:** Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *International Conference on Learning Representations (ICLR 2026)*.
- **URL:** https://arxiv.org/abs/2511.18903 ; https://iclr.cc/virtual/2026/poster/10009351
- **Model scale:** Qwen2.5-1.5B architecture trained from scratch on 30B DCLM tokens.
- **Summary:**
  - Here "curriculum" means **ascending data quality**, not difficulty.
  - Under a constant LR the curriculum clearly beats shuffle. Under cosine or WSD decay the advantage shrinks.
  - Fixes: a moderate decay floor (about 1e-3 vs 1e-5) and weight averaging (CMA/CDMA). Together they give +1.64% average accuracy over shuffle with standard decay.
  - "Folding" (interleaved) curricula were fragile.
- **Relation to our study:** This is the key confound, because the mastery arms reach their hardest skills during the low-LR tail. Ablate the LR schedule.

#### Elgaar & Amiri (2026): Learning dynamics of curriculum pretraining [V-abs]
- **Citation:** Elgaar, M., & Amiri, H. (2026). *Curriculum learning for LLM pretraining: An analysis of learning dynamics* (arXiv:2601.21698). arXiv.
- **URL:** https://arxiv.org/abs/2601.21698
- **Model scale:** Pythia 14M–1B, 300B tokens.
- **Summary:**
  - Under all orderings, models pass through the same latent phases. Curricula mainly change how long each phase lasts and how stable training is at small scale.
  - Benefits shrink with scale.
- **Relation to our study:** Curriculum effects are likely largest at the low end of our size range. Test at least one scale step.

#### Zhang et al. (2025): Preference curriculum (PDPC) [V-abs]
- **Citation:** Zhang, X., Xu, L., Duan, F., Zhou, Y., Wang, S., Weng, R., Wang, J., & Cai, X. (2025). Preference curriculum: LLMs should always be pretrained on their preferred data. In *Findings of the ACL: ACL 2025* (pp. 21181–21198).
- **URL:** https://aclanthology.org/2025.findings-acl.1091/ (arXiv:2501.13126)
- **Model scale:** 1.3B and 3B trained from scratch; the 3B model is trained on 1T tokens.
- **Summary:** Data is ordered offline by the perplexity difference between a weak and a strong reference model. The 3B model gains "over 8.1%" average accuracy on MMLU and CMMLU. Whether that figure is absolute or relative is UNVERIFIED.
- **Relation to our study:** An offline, clock-based schedule with no online mastery measurement.

#### Blakeney et al. (2024): Domain upsampling at the end of training [V-abs, from a search snippet]
- **Citation:** Blakeney, C., Paul, M., Larsen, B. W., Owen, S., & Frankle, J. (2024). Does your data spark joy? Performance gains from domain upsampling at the end of training. In *Conference on Language Modeling (COLM 2024)*. https://arxiv.org/abs/2406.03476
- **URL:** https://arxiv.org/abs/2406.03476
- **Model scale:** ⚠ 7B trained on 1T tokens.
- **Summary:** Upsampling domain data in the last 10–20% of training raised MMLU by up to 6.90 points and GSM8K by up to 8.26 points.
- **Relation to our study:** Late upsampling *coincides* with LR annealing, which is the flip side of Luo et al.'s finding.

#### Dai et al. (2025): Data efficacy (DELT) [V-abs]
- **Citation:** Dai, Y., Huang, Y., Zhang, X., Wu, W., Li, C., Lu, W., Cao, S., Dong, L., & Li, S. (2025). *Data efficacy for language model training* (arXiv:2506.21545). arXiv.
- **URL:** https://arxiv.org/abs/2506.21545
- **Model scale:** UNVERIFIED.
- **Summary:** "Folding ordering" interleaves sorted chunks of data to reduce forgetting. The abstract gives no numbers.
- **Relation to our study:** Folding resembles curriculum plus replay. Luo et al. found it fragile at 1.5B.

#### Kim & Lee (2024): Strategic data ordering [V-abs]
- **Citation:** Kim, J., & Lee, J. (2024). *Strategic data ordering: Enhancing large language model performance through curriculum learning* (arXiv:2405.07490). arXiv.
- **URL:** https://arxiv.org/abs/2405.07490
- **Model scale:** ⚠ Mistral-7B and Gemma-7B, fine-tuned.
- **Summary:** Easy-to-hard ordering "slightly improves performance."
- **Relation to our study:** A weak, unreviewed result. It is only evidence that static ordering has small effects.

#### Li et al. (2025): CAMPUS [V-full]
- **Citation:** Li, Y., Lu, T., Li, Y., Chen, Y., Huang, W.-C., Jiang, W., Wang, H., Zheng, H.-T., & Yu, P. S. (2025). Teaching according to talents! Instruction tuning LLMs with competence-aware curriculum learning. In *Findings of the ACL: EMNLP 2025*.
- **URL:** https://arxiv.org/abs/2509.13790
- **Model scale:** LLaMA-7B and 13B (main results), BLOOMZ 560M, 1.7B and 3B, and LLaMA-33B (appendix), all instruction-tuned from pretrained checkpoints.
- **Summary:**
  - Several sub-curricula are each sorted by a different difficulty metric. At each step, CAMPUS picks the sub-curriculum with the lowest perplexity under the current model.
  - **LLaMA average:** CAMPUS 38.16 vs random shuffle 36.49. The "7.0%" figure is measured against the curriculum baselines Tree-Instruct and Conifer.
  - **BLOOMZ:** 20.34 vs 18.34 for shuffle.
- **Relation to our study:** Competence-aware *soft* selection among orderings. It has no skill graph, no hard unlock and no held-out probe.

#### Li et al. (2026): RLAAR [V-full]
- **Citation:** Li, M., Chen, P., Zhang, Z., Yang, T., Zhang, X., Li, H., Cao, T., Zeng, M., Wu, Z., Jiang, M., Li, H., Li, L., & Yin, B. (2026). Mitigating lost in multi-turn conversation via curriculum RL with verifiable accuracy and abstention rewards. In *Proceedings of ACL 2026*.
- **URL:** https://arxiv.org/abs/2510.18731
- **Model scale:** Qwen3-8B, Qwen3-1.7B and Qwen2.5-7B-Instruct.
- **Summary:**
  - **Gate:** the number of conversation shards K increases only when the moving-average *training* reward satisfies r̄ ≥ 0.8·r̄₀, where r̄₀ is the single-turn reward. A final stage samples K uniformly, which in effect replays easier levels.
  - **Qwen3-8B:** LiC score 62.6 → 75.1; abstention 33.5 → 73.4.
  - **Qwen3-1.7B:** LiC 58.8 → 71.9.
- **Relation to our study:** A true hard gate on measured competence. It differs from ours in using a training-reward signal and a single difficulty axis instead of a skill graph.

#### Bae et al. (2026): Online difficulty filtering for reasoning RL [V-full (HTML)]
- **Citation:** Bae, S., Hong, J., Lee, M. Y., Kim, H., Nam, J., & Kwak, D. (2026). Online difficulty filtering for reasoning oriented reinforcement learning. In *Proceedings of EACL 2026*.
- **URL:** https://arxiv.org/abs/2504.03380
- **Model scale:** Qwen2.5-3B and 7B, SFT then GRPO.
- **Summary:**
  - A prompt is kept only if its online pass rate p lies between thresholds (e.g., 0.3–0.7).
  - The theory behind it: expected improvement is lower-bounded by the variance p(1−p).
  - Results: 3B average 26.3 → 30.1; 7B 39.6 → 41.8.
- **Relation to our study:** An instance-level analogue of a mastery filter. Rollouts that get filtered out are hidden compute that budget matching must count.

#### Yu et al. (2025): DAPO [V-full (HTML)]
- **Citation:** Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., et al. (2025). *DAPO: An open-source LLM reinforcement learning system at scale* (arXiv:2503.14476). arXiv.
- **URL:** https://arxiv.org/abs/2503.14476
- **Model scale:** ⚠ Qwen2.5-32B.
- **Summary:** "Dynamic sampling" drops prompts that every rollout solves or every rollout fails. That component alone moved AIME 2024 from 42 to 50.
- **Relation to our study:** Excludes mastered items, but has no ordering and no graph.

#### Chen et al. (2025): Self-evolving curriculum (SEC) [V-full (HTML)]
- **Citation:** Chen, X., Lu, J., Kim, M., Zhang, D., Tang, J., Piché, A., Gontier, N., Bengio, Y., & Kamalloo, E. (2025). *Self-evolving curriculum for LLM reasoning* (arXiv:2505.14970). arXiv.
- **URL:** https://arxiv.org/abs/2505.14970
- **Model scale:** Qwen2.5-3B and 7B; Llama-3.2-1B.
- **Summary:**
  - A bandit over difficulty categories is rewarded by mean absolute advantage. The hardest level is held out as an OOD test.
  - **3B, OOD:** improves over a random curriculum (e.g., Countdown 0.479 → 0.542).
  - **7B:** results are mixed.
  - **Multi-task:** with a random curriculum, multi-task training collapses partway through.
- **Relation to our study:** Selection driven by learning progress, evaluated on OOD hold-outs, but still soft and not gated.

#### Shi et al. (2025): AdaRFT [V-full (HTML)]
- **Citation:** Shi, T., Wu, Y., Song, L., Zhou, T., & Zhao, J. (2025). Efficient reinforcement finetuning via adaptive curriculum learning. *Transactions on Machine Learning Research*.
- **URL:** https://arxiv.org/abs/2504.05520
- **Model scale:** Qwen2.5-MATH-1.5B and Qwen2.5-7B.
- **Summary:**
  - A target difficulty is adjusted toward 50% reward.
  - Training time is up to 2× shorter.
  - Single seed. The roughly 1.28M rollouts used for difficulty scoring are excluded from the reported cost.
- **Relation to our study:** A lesson for budget matching: count the compute spent on probes and scoring.

#### Parashar et al. (2026): E2H Reasoner [V-abs]
- **Citation:** Parashar, S., Gui, S., Li, X., Ling, H., Vemuri, S., Olson, B., Li, E., Zhang, Y., Caverlee, J., Kalathil, D., & Ji, S. (2026). Curriculum reinforcement learning from easy to hard tasks improves LLM reasoning. In *International Conference on Learning Representations (ICLR 2026)*.
- **URL:** https://arxiv.org/abs/2506.06632
- **Model scale:** 1.5B–3B models.
- **Summary:** Probabilistic clock-based schedules (cosine or Gaussian) over difficulty tiers. "Fading out" easy tasks is essential to avoid overfitting.
- **Relation to our study:** A strong clock-based RL baseline. It cautions against replaying trivial levels indefinitely.

#### Kimi Team (2025): Kimi k1.5 [V-full (sampling section)]; Wen et al. (2025): Light-R1 [V-abs]; Jia et al. (2025) [V-abs]
- **Citations:**
  - Kimi Team. (2025). *Kimi k1.5: Scaling reinforcement learning with LLMs* (arXiv:2501.12599). https://arxiv.org/abs/2501.12599
  - Wen, L., Cai, Y., Xiao, F., He, X., An, Q., Duan, Z., Du, Y., Liu, J., Tang, L., Lv, X., Zou, H., Deng, Y., Jia, S., & Zhang, X. (2025). Light-R1: Curriculum SFT, DPO and RL for long COT from scratch and beyond. In *Proceedings of ACL 2025 (Industry Track)*. https://arxiv.org/abs/2503.10460
  - Jia, Y., Zhang, C., Diao, X., Yuan, X., Ouyang, Z., Ma, C., & Vosoughi, S. (2025). *What makes a good curriculum? Disentangling the effects of data ordering on LLM mathematical reasoning* (arXiv:2510.19099). https://arxiv.org/abs/2510.19099
- **Model scale:** ⚠ Kimi: size undisclosed. ⚠ Light-R1: 7–32B. Jia et al.: 4–8B.
- **Summary:**
  - **Kimi k1.5:** combines easy-to-hard curriculum sampling with prioritized sampling ∝ (1 − success rate).
  - **Light-R1:** a staged SFT curriculum.
  - **Jia et al.:** no single ordering wins; whether forward or reverse order is better depends on the model and the task.
- **Relation to our study:** Industry practice combines clock and soft-competence signals. Fixed orderings are setting-dependent.

#### Skill graphs and gated unlocks
- **Kalyan, Mishra, Lokam & Goyal (2025): CurLL** [V-full (HTML)]
  - Citation: Kalyan, P., Mishra, S., Lokam, S., & Goyal, N. (2025). *CurLL: A developmental framework to evaluate continual learning in language models* (arXiv:2510.13008). https://arxiv.org/abs/2510.13008
  - Model scale: SmolLM2-135M architecture, **trained from scratch**.
  - Summary: 23.4B synthetic tokens covering ages 5–10, with a prerequisite graph of more than 1,300 skills. Training is independent, joint or stage-sequential, without replay. Sequential training forgets earlier stages.
  - Relation to our study: the closest from-scratch prerequisite-graph setting, with no gate and no replay.
- **Li et al. (2026): SkillGraph** [V-full (HTML)]
  - Citation: Li, X., Li, M., Bao, K., Ma, Y., Wang, W., Liu, D., & Feng, F. (2026). *SkillGraph: Skill-augmented reinforcement learning for agents via evolving skill graphs* (arXiv:2605.12039). https://arxiv.org/abs/2605.12039
  - Model scale: ⚠ Qwen2.5-7B-Instruct.
  - Summary: level-(L+1) skills are activated "when the average success rate of level-L skills exceeds an unlocking threshold" of 0.6. However, the skills are natural-language instructions placed *in the context*, not slices of training data. ALFWorld score is 90.6 vs 77.6 for GRPO.
  - Relation to our study: the only hard graph unlock we found. It gates prompts, not training data, the source of its success counts is unclear, and it has no matched shuffle or clock comparison.
- **He et al. (2026): IOA, pedagogically-inspired distillation** [V-abs; the main session reports τ_mastery gates and ZPD-bounded steps]
  - Citation: He, B., Chen, Y., Zhang, X., Kong, L., Yu, P. S., Liu, X., & Ma, C. (2026). Pedagogically-inspired data synthesis for language model knowledge distillation. In *International Conference on Learning Representations (ICLR 2026)*. https://arxiv.org/abs/2602.12172
    - The arXiv comment says "Accepted by ICLR 2026"; the main session's note listed it as a preprint.
  - Model scale: LLaMA-3.1/3.2 and Qwen2.5 students with "less than 1/10th of the parameters" of the teacher. Exact sizes UNVERIFIED.
  - Summary:
    - The Identifier–Organizer–Adapter framework explicitly applies Bloom's mastery learning: the student must approach teacher performance on prerequisites (a τ_mastery gate) before advancing.
    - It also applies Vygotsky's ZPD, introducing material in "controlled, gradual difficulty increments."
    - Students keep 94.7% of the teacher's performance on DollyEval, and improve MATH by 19.2% and HumanEval by 22.3% over state-of-the-art baselines.
  - Relation to our study:
    - This is a direct precedent for borrowing Bloom-style mastery gates for LMs, though in synthetic-data *distillation* from pretrained students.
    - Its gains are measured against other distillation methods, not against matched shuffle or clock schedules.
- **Xu et al. (2026): D³** [V-abs]
  - Citation: Xu, Y., Hao, J., Zhang, G., & Li, Z. (2026). *D³: Dynamic directional graph-constrained data scheduling for LLM training* (arXiv:2605.31164). https://arxiv.org/abs/2605.31164
  - Model scale: UNVERIFIED.
  - Summary: a loss-based influence graph constrains the data ordering. No numbers in the abstract.
- **Li et al. (2025): MASS** [V-abs]
  - Citation: Li, J., Yu, L., Cui, Q., Zhang, Z., Zhou, J., Ye, Y., & Zhang, C. (2025). *MASS: Mathematical data selection via skill graphs for pretraining large language models* (arXiv:2503.14917). https://arxiv.org/abs/2503.14917
  - Model scale: 1B and 7B.
  - Summary: uses a math skill graph to *select* data. It matches full-data performance with 50–70% fewer tokens.
  - Relation to our study: the skill graph is used for selection, not for gating.

---

## 3. Algorithmic and arithmetic learning in small transformers ("mastery = a generalizing algorithm")

### 3.1 Arithmetic formats and length generalization

#### Lee, Sreenivasan, Lee, Lee & Papailiopoulos (2024): Teaching arithmetic to small transformers [V-full]
- **Citation:** Lee, N., Sreenivasan, K., Lee, J. D., Lee, K., & Papailiopoulos, D. (2024). Teaching arithmetic to small transformers. In *International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=dsUB4bst9S
- **URL:** https://arxiv.org/abs/2307.03381
- **Model scale:**
  - NanoGPT with 6 layers, 6 heads and d = 384 (about 10.6M parameters), trained from scratch at the character level.
  - GPT-2 up to 124M.
  - ⚠ GPT-3 fine-tuning.
- **Summary:**
  - **Plain format:** 3-digit addition plateaus at about 85%.
  - **Reversed output** (least significant digit first) shows a sharp phase transition to 100% at about 1k–4k training samples.
  - **Why reversal helps:** with least-significant-digit-first output, each output digit is "a local function" of two operand digits and the carry.
  - **Limits:** length generalization essentially fails. With GPT-3 fine-tuning, the reversed format did worse than plain.
- **Relation to our study:** This is the *primary* source for output reversal. Reversal makes in-distribution addition easy but does not by itself produce OOD mastery.

#### Zhang-Li et al. (2024): Reverse that number! [V-full]
- **Citation:** Zhang-Li, D., Lin, N., Yu, J., Zhang, Z., Yao, Z., Zhang, X., Hou, L., Zhang, J., & Li, J. (2024). *Reverse that number! Decoding order matters in arithmetic learning* (arXiv:2403.05845). arXiv. https://doi.org/10.48550/arXiv.2403.05845
- **URL:** https://arxiv.org/abs/2403.05845
- **Model scale:** ⚠ Llama2-13B, fine-tuned. There are no small or from-scratch models.
- **Summary:**
  - "LEFT" (little-endian fine-tuning) reverses both inputs and outputs.
  - Overall accuracy is 94.4 vs 83.3 for a detailed scratchpad, using about one third of the tokens.
  - All evaluation is in-distribution, with no length or OOD tests.
  - The paper credits the reversal idea to Lee et al. Its peer-reviewed venue is UNVERIFIED.
- **Relation to our study:** Weak support for "improves generalization." Cite Lee et al. (2024) as the primary source.

#### Shen et al. (2023): Positional description matters [V-full]
- **Citation:** Shen, R., Bubeck, S., Eldan, R., Lee, Y. T., Li, Y., & Zhang, Y. (2023). *Positional description matters for transformers arithmetic* (arXiv:2311.14737). arXiv.
- **URL:** https://arxiv.org/abs/2311.14737
- **Model scale:** GPT-2 small (124M), partly trained from scratch.
- **Summary:**
  - Arithmetic failures are framed as failures of *positional description*.
  - Multiplication with padding and a reversed product is "essentially perfect" up to 12 digits.
  - Randomized position tags help length generalization in addition.
- **Relation to our study:** Supports the "place-value tag" idea.

#### McLeish et al. (2024): Abacus embeddings [V-full]
- **Citation:** McLeish, S., Bansal, A., Stein, A., Jain, N., Kirchenbauer, J., Bartoldson, B. R., Kailkhura, B., Bhatele, A., Geiping, J., Schwarzschild, A., & Goldstein, T. (2024). Transformers can do arithmetic with the right embeddings. In *Advances in Neural Information Processing Systems 37*.
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/c35986bc1ee29b31c1011481b77fe540-Abstract-Conference.html (arXiv:2405.17399)
- **Model scale:** 12M–122M parameters, trained from scratch, on 1 GPU for 1 day.
- **Summary:**
  - Abacus embeddings encode each digit's position within its number. Training uses a random offset (k = 100).
  - Models trained on operands up to 20 digits reach 99.1% on operands up to 100 digits (with looping and input injection), i.e., up to 6× extrapolation.
  - Output is reversed throughout.
- **Relation to our study:**
  - This is the best evidence for place-value tags at our scale.
  - The tag range limits how far the model can extrapolate.
  - Reversal and tags are confounded, so ablate them separately.

#### Zhou, Bradley, Littwin, Razin, Saremi, Susskind, Bengio & Nakkiran (2024): RASP-L and length generalization [V-full]
- **Citation:** Zhou, H., Bradley, A., Littwin, E., Razin, N., Saremi, O., Susskind, J., Bengio, S., & Nakkiran, P. (2024). What algorithms can transformers learn? A study in length generalization. In *International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=AssIuHnmHX
- **URL:** https://arxiv.org/abs/2310.16028
- **Model scale:** Small decoders trained from scratch (6 layers, d = 512 for addition).
- **Summary:**
  - The RASP-Generalization Conjecture: transformers length-generalize when a short RASP-L program solves the task at every length.
  - With index hints, reversed addition generalizes to 50 digits on hard-carry problems. Forward order does not generalize.
- **Relation to our study:**
  - Gives a principled reason for using reverse order plus position hints.
  - OOD probes must include **long carry chains**, because random operands rarely produce them.

#### Zhou, Alon, Chen, Wang, Agarwal & Zhou (2024): Length generalization, but not robustly [V-full]
- **Citation:** Zhou, Y., Alon, U., Chen, X., Wang, X., Agarwal, R., & Zhou, D. (2024). Transformers can achieve length generalization but not robustly. In *ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models*. https://openreview.net/forum?id=DWkWIh3vFJ
- **URL:** https://arxiv.org/abs/2402.09371
- **Model scale:** 2M–268M parameters, trained from scratch (the main model is about 25M).
- **Summary:**
  - Combining FIRE positional encoding, index hints and reversed output gets 100-digit addition from 40-digit training (2.5×) at over 98%. That is the *best of 10 trials*.
  - Results vary widely across weight initialization and data order.
  - The 268M model did worse than the 25M model.
  - Lengths were sampled uniformly, with no curriculum.
- **Relation to our study:** OOD-probe gates will be **seed-sensitive**. Use 5 or more seeds and report variance.

#### Anil et al. (2022): Exploring length generalization in LLMs [V-abs]
- **Citation:** Anil, C., Wu, Y., Andreassen, A., Lewkowycz, A., Misra, V., Ramasesh, V., Slone, A., Gur-Ari, G., Dyer, E., & Neyshabur, B. (2022). Exploring length generalization in large language models. *Advances in Neural Information Processing Systems, 35*, 38546–38556.
  - Page numbers came from a search snippet.
- **URL:** https://arxiv.org/abs/2207.04901
- **Model scale:** ⚠ LaMDA 244M–64B, pretrained (sizes from a search excerpt).
- **Summary:** On parity and a chained *variable-assignment* task, fine-tuning showed length-generalization deficiencies "independent of model scale." Scratchpad prompting with in-context learning helped.
- **Relation to our study:** The variable-assignment task is a direct relative of our chain task. Scale does not fix length generalization.

### 3.2 Algorithms vs heuristics; grokking

#### Power, Burda, Edwards, Babuschkin & Misra (2022): Grokking [V-full]
- **Citation:** Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets* (arXiv:2201.02177). arXiv.
- **URL:** https://arxiv.org/abs/2201.02177
- **Model scale:** A 2-layer transformer, about 4×10^5 non-embedding parameters.
- **Summary:**
  - Validation accuracy can jump from chance to perfect long after the model has overfit.
  - Smaller datasets need more optimization before this happens.
  - Weight decay helps most.
- **Relation to our study:** When the gate is checked, and with what weight decay and budget, can decide whether mastery is ever detected.

#### Nanda, Chan, Lieberum, Smith & Steinhardt (2023): Progress measures for grokking [V-full]
- **Citation:** Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. In *International Conference on Learning Representations (ICLR 2023)*.
- **URL:** https://arxiv.org/abs/2301.05217
- **Model scale:** A 1-layer transformer with fewer than 1M parameters.
- **Summary:** For modular addition (P = 113), the model implements a discrete-Fourier "rotation" algorithm. Training goes through three phases: memorization, circuit formation, and cleanup.
- **Relation to our study:** Hidden progress comes before probe success. Consider internal progress measures alongside behavioural probes.

#### Zhong, Liu, Tegmark & Andreas (2023): The clock and the pizza [V-full]
- **Citation:** Zhong, Z., Liu, Z., Tegmark, M., & Andreas, J. (2023). The clock and the pizza: Two stories in mechanistic explanation of neural networks. In *Advances in Neural Information Processing Systems 36*.
- **URL:** https://arxiv.org/abs/2306.17844
- **Model scale:** Tiny transformers of 1–4 layers, width 32–512, trained on modular addition (p = 59).
- **Summary:**
  - Depending on hyperparameters, the same task produces different exact algorithms: "clock" or "pizza."
  - The share of runs with circular algorithms falls with depth (34% at 1 layer, 6% at 4 layers).
- **Relation to our study:** These are cross-entropy-trained models that learn exact, generalizing algorithms. Behaviourally identical "mastery" can therefore rest on different mechanisms.

#### Liu et al. (2022): Towards understanding grokking [V-abs]
- **Citation:** Liu, Z., Kitouni, O., Nolte, N., Michaud, E. J., Tegmark, M., & Williams, M. (2022). Towards understanding grokking: An effective theory of representation learning. In *Advances in Neural Information Processing Systems 35*.
- **URL:** https://arxiv.org/abs/2205.10343
- **Model scale:** Toy models and small transformers.
- **Summary:** Proposes four learning phases. Generalization happens only in a hyperparameter "Goldilocks zone."
- **Relation to our study:** Optimizer settings can determine whether the gate is ever reached.

#### Lindsey et al. (2025): On the biology of a large language model [V-full, addition sections]
- **Citation:** Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce, A., Turner, N. L., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., Persic, A., Qi, Z., Thompson, T. B., Zimmerman, S., Rivoire, K., Conerly, T., Olah, C., & Batson, J. (2025). On the biology of a large language model. *Transformer Circuits Thread*.
- **URL:** https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- **Model scale:** ⚠ Claude 3.5 Haiku (size undisclosed).
- **Summary:**
  - The model adds (e.g., 36+59) through two parallel pathways: a low-precision magnitude estimate, and an exact ones-digit lookup ("_6 + _9" → "ends in 5").
  - Asked to explain itself, the model describes the textbook carry algorithm instead.
  - The addition features generalize across contexts.
  - **The paper does not attribute this to the cross-entropy objective.**
- **Relation to our study:** A large-model illustration that heuristic pathways can produce in-distribution success. It does not support a causal claim that CE loss favours heuristics.

#### Ameisen et al. (2025): Circuit tracing [V-full, summary sections]
- **Citation:** Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., Persic, A., Qi, Z., Thompson, T. B., Zimmerman, S., Rivoire, K., Conerly, T., Olah, C., & Batson, J. (2025). Circuit tracing: Revealing computational graphs in language models. *Transformer Circuits Thread*.
  - The author order follows the page; the paper's own citation block was not retrieved.
- **URL:** https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- **Model scale:** A small 18-layer model and ⚠ Claude 3.5 Haiku.
- **Summary:** Cross-layer transcoders build a replacement model, from which attribution graphs are drawn. The addition case study traces lookup-table pathways in the small model too.
- **Relation to our study:** This is the methods reference for the report's "attribution graph" extension.

#### Nikankin, Reusch, Mueller & Belinkov (2025): Arithmetic without algorithms [V-full]
- **Citation:** Nikankin, Y., Reusch, A., Mueller, A., & Belinkov, Y. (2025). Arithmetic without algorithms: Language models solve math with a bag of heuristics. In *International Conference on Learning Representations (ICLR 2025)*. https://openreview.net/forum?id=09YTt26r2P
- **URL:** https://arxiv.org/abs/2410.21272
- **Model scale:** ⚠ Llama3-8B and 70B, Pythia-6.9B and GPT-J (6B), all pretrained.
- **Summary:**
  - A sparse circuit of about 1.5% of neurons per layer explains about 96% of arithmetic performance.
  - 91% of the top neurons implement simple heuristics.
  - Across Pythia checkpoints, the heuristics appear early in training and drive accuracy.
- **Relation to our study:** The best citation for "heuristics, not algorithms." It concerns large pretrained models, not small from-scratch ones.

### 3.3 Chain, variable-binding and multi-hop tasks (templates for Preliminary Experiment I)

#### Zhang, Backurs, Bubeck, Eldan, Gunasekar & Wagner (2023): LEGO [V-full]
- **Citation:** Zhang, Y., Backurs, A., Bubeck, S., Eldan, R., Gunasekar, S., & Wagner, T. (2023). *Unveiling transformers with LEGO: A synthetic reasoning task* (arXiv:2206.04301). arXiv.
- **URL:** https://arxiv.org/abs/2206.04301
- **Model scale:** BERT-base (about 110M) and ALBERT-base, encoder-only, either pretrained or randomly initialized.
- **Summary:**
  - The task is a chain of variable clauses over the group {±1} (e.g., `a=+1; b=−a; …`), with clauses shuffled. Supervision covers only the first 6 of 12 variables.
  - Randomly initialized BERT fails to extrapolate to longer chains. ALBERT (with weight tying), pretrained BERT, or more diverse data succeed.
  - Models that do *not* extrapolate use a shortcut instead: they fill the chain in from both ends.
- **Relation to our study:**
  - Our chain task is LEGO over integers with addition, using a decoder. Cite LEGO as the template.
  - Design probes that block end-of-chain shortcuts.
  - Weight tying or looping may help deeper chains.

#### Wu, Geiger & Millière (2025): How transformers learn variable binding [V-full]
- **Citation:** Wu, Y., Geiger, A., & Millière, R. (2025). How do transformers learn variable binding in symbolic programs? In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR Vol. 267, pp. 67284–67299).
- **URL:** https://proceedings.mlr.press/v267/wu25j.html (arXiv:2505.20896)
- **Model scale:** A 37.8M-parameter decoder trained from scratch.
- **Summary:**
  - The task is 17-line programs with chains of up to 4 hops, plus distractors.
  - Training goes through three phases: chance (about 12%); a shallow "early-line" heuristic (about 56%); then a systematic dereferencing mechanism (above 99.9%), which emerges between steps 34k and 105k.
  - The systematic circuit is built *on top of* the heuristic.
  - OOD probes on hop count and position expose where the leftover heuristic is used.
- **Relation to our study:** The closest small, from-scratch analogue of our chain task. It shows heuristic-then-algorithm phases, so the timing of gate checks matters.

#### Yao, Du, Zhu, Hahn & Koller (2025): Implicit multi-hop reasoning needs lots of data [V-abs; the main session supplied the venue page numbers]
- **Citation:** Yao, Y., Du, Y., Zhu, D., Hahn, M., & Koller, A. (2025). Language models can learn implicit multi-hop reasoning, but only if they have lots of training data. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing* (pp. 9684–9702). Association for Computational Linguistics.
- **URL:** https://arxiv.org/abs/2505.17923 (arXiv comment: "Accepted at EMNLP 2025")
- **Model scale:** GPT-2-style LMs trained from scratch on controlled synthetic 2-, 3- and 4-hop data. The abstract gives no parameter counts (UNVERIFIED).
- **Summary:**
  - Models can learn implicit k-hop reasoning, which is done in a single forward pass with no chain of thought.
  - The training data required "grows exponentially in k."
  - The number of layers required grows linearly with k, and the authors give a theoretical argument for why.
  - Curriculum learning substantially reduces the data requirement but does not remove it.
- **Relation to our study:**
  - This is the closest published result to Preliminary Experiment I (variable-chain depth). It explains why shuffle fails on deep chains.
  - It gives an independent prior that a curriculum should help on this task.
  - It also warns that the gains depend on the budget and that model depth must grow with k. A fixed-depth model may never pass deep-chain gates.

#### Kohli, Parthasarathy, Sun & Yao (2026): Loop, think, & generalize [V-abs; §6.1 curriculum detail verified in full text by the main session]
- **Citation:** Kohli, H., Parthasarathy, S., Sun, H., & Yao, Y. (2026). Loop, think, & generalize: Implicit reasoning in recurrent-depth transformers. In *Conference on Language Modeling (COLM 2026)*.
- **URL:** https://arxiv.org/abs/2604.07822 (arXiv comment: "Accepted at COLM 2026")
- **Model scale:** Recurrent-depth (looped) and vanilla transformers trained from scratch in controlled studies. Sizes are UNVERIFIED (not in the abstract).
- **Summary:**
  - Vanilla transformers struggle with both systematic generalization and depth extrapolation (e.g., train up to 5 hops, test at 10). Recurrent-depth models handle both.
  - Systematic generalization emerges through a "three-stage grokking process."
  - More recurrence at inference time extends the depth the model can handle, but too many iterations cause "overthinking."
  - **Training curriculum (§6.1, per the main session's full-text check):**
    - The model moves to (k+1)-hop facts only after reaching ≥95% accuracy on a held-out k-hop split.
    - It trains jointly on all stages introduced so far "to prevent forgetting."
- **Relation to our study:**
  - **This is the most direct precedent for mastery gating + replay.** The gate is held-out accuracy at a threshold, along a depth ladder, with cumulative replay.
  - It is used as a *training tool*, not as an experimental variable. There is no comparison with shuffle or clock schedules and no budget matching.
  - So the novelty of our study lies in the controlled, budget-matched comparison and the factorial gate × replay design, not in the gate itself. The report's "will be unique" statement should be softened accordingly.
  - It also supports looped or recurrent depth for deep-chain probes.

#### Sanford, Hsu & Telgarsky (2024): Logarithmic depth and k-hop [V-abs]
- **Citation:** Sanford, C., Hsu, D., & Telgarsky, M. (2024). Transformers, parallel computation, and logarithmic depth. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235, pp. 43276–43327).
- **URL:** https://proceedings.mlr.press/v235/sanford24a.html
- **Model scale:** Theory, plus small trained transformers.
- **Summary:** For "k-hop induction heads," depth Θ(log k) is necessary and sufficient. A search summary says trained models solve k up to about 2^L with L layers (UNVERIFIED in the primary text).
- **Relation to our study:**
  - A fixed-depth model may hit a ceiling on chain depth.
  - Probes beyond that ceiling measure architecture, not mastery.
  - Size the depth to the longest chain you probe.

#### Wang, Yue, Su & Sun (2024): Grokked transformers are implicit reasoners [V-full]
- **Citation:** Wang, B., Yue, X., Su, Y., & Sun, H. (2024). Grokked transformers are implicit reasoners: A mechanistic journey to the edge of generalization. In *Advances in Neural Information Processing Systems 37*.
- **URL:** https://arxiv.org/abs/2405.15071
- **Model scale:** An 8-layer GPT-2-style model trained from scratch.
- **Summary:**
  - Two-hop composition is learned only through grokking.
  - OOD composition stays at 0% even after 2M steps.
  - Parameter sharing can "unlock" OOD composition, but slowly.
- **Relation to our study:** Mastery of *novel compositions* may never arrive in a standard transformer. Gate criteria should be achievable for the architecture used.

#### Ramesh et al. (2024): Compositional capabilities of autoregressive transformers [V-full]
- **Citation:** Ramesh, R., Lubana, E. S., Khona, M., Dick, R. P., & Tanaka, H. (2024). Compositional capabilities of autoregressive transformers: A study on synthetic, interpretable tasks. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235, pp. 42074–42103).
- **URL:** https://proceedings.mlr.press/v235/ramesh24a.html
- **Model scale:** A small nanoGPT (12 layers, d = 120), trained from scratch.
- **Summary:**
  - Training on single functions alone does not generalize to compositions.
  - Step-by-step formats generalize from 30–100 trained compositions to all 3,125.
  - Direct formats fail.
- **Relation to our study:** Mastering the components is not sufficient. The skill-(k+1) data needs composed examples.

#### Dziri et al. (2023): Faith and fate [V-abs]
- **Citation:** Dziri, N., Lu, X., Sclar, M., Li, X. L., Jiang, L., Lin, B. Y., West, P., Bhagavatula, C., Le Bras, R., Hwang, J. D., Sanyal, S., Welleck, S., Ren, X., Ettinger, A., Harchaoui, Z., & Choi, Y. (2023). Faith and fate: Limits of transformers on compositionality. *Advances in Neural Information Processing Systems, 36*.
- **URL:** https://arxiv.org/abs/2305.18654
- **Model scale:** ⚠ GPT-3, ChatGPT and GPT-4.
- **Summary:** Transformers reduce compositional tasks such as multiplication to "linearized subgraph matching." Fine-tuned models do well in-distribution but fail OOD.
- **Relation to our study:** Supports excluding in-distribution accuracy from the mastery criterion.

#### Okawa et al. (2023) [V-abs]; Arora & Goyal (2023) [V-abs]; Feng & Steinhardt (2024) [V-abs]; Biran et al. (2024) [V-full]: brief background
- **Okawa et al. (2023).** Okawa, M., Lubana, E. S., Dick, R. P., & Tanaka, H. (2023). Compositional abilities emerge multiplicatively: Exploring diffusion models on a synthetic task. *NeurIPS 36*. https://arxiv.org/abs/2310.09336
  - ⚠ Diffusion models, not LMs.
  - Composite accuracy behaves like a product of the component accuracies, so gates on component skills should require near-perfect accuracy.
- **Arora & Goyal (2023).** Arora, S., & Goyal, A. (2023). *A theory for emergence of complex skills in language models* (arXiv:2307.15936). https://arxiv.org/abs/2307.15936
  - Theory only.
- **Feng & Steinhardt (2024).** Feng, J., & Steinhardt, J. (2024). How do language models bind entities in context? *ICLR 2024*. https://arxiv.org/abs/2310.17191
  - ⚠ Pretrained Pythia and LLaMA models.
  - Reports "binding ID" vectors.
- **Biran et al. (2024).** Biran, E., Gottesman, D., Yang, S., Geva, M., & Globerson, A. (2024). Hopping too late: Exploring the limitations of large language models on multi-hop queries. *EMNLP 2024*. https://arxiv.org/abs/2406.12775
  - ⚠ 7B–70B models.
  - The second hop is resolved "too late" in the layer stack.

---

## 4. Forgetting and replay (brief; see `03_ml_spacing_replay.md` for full entries)

#### McCloskey & Cohen (1989): Catastrophic interference [V-meta]
- **Citation:** McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation, 24*, 109–165. https://doi.org/10.1016/S0079-7421(08)60536-8
- **URL:** https://doi.org/10.1016/S0079-7421(08)60536-8
- **Model scale:** ⚠ small backpropagation networks.
- **Summary:** Training on new material in sequence can catastrophically disrupt material learned earlier.
- **Relation to our study:** The founding citation for the forgetting that our naive gate showed in Preliminary Experiment I.

#### French (1999): Catastrophic forgetting in connectionist networks [V-meta]
- **Citation:** French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2
- **URL:** https://doi.org/10.1016/S1364-6613(99)01294-2
- **Model scale:** ⚠ review of connectionist networks.
- **Summary:** A review of the problem and its remedies, including rehearsal.
- **Relation to our study:** The standard review citation.

#### Robins (1995): Rehearsal and pseudorehearsal [V-meta; V-pub in file 03]
- **Citation:** Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318
- **URL:** https://doi.org/10.1080/09540099550039318
  - The report links a Stanford library proxy URL. Replace it with this DOI.
- **Model scale:** ⚠ small backpropagation networks.
- **Summary:** Interleaving a subset of old items, or generated pseudo-items, with new training data reduces forgetting.
- **Relation to our study:** The canonical citation for the replay arm.

#### Rebuffi, Kolesnikov, Sperl & Lampert (2017): iCaRL [V-meta]
- **Citation:** Rebuffi, S.-A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). iCaRL: Incremental classifier and representation learning. In *2017 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 5533–5542). IEEE. https://doi.org/10.1109/CVPR.2017.587
- **URL:** https://doi.org/10.1109/CVPR.2017.587
- **Model scale:** ⚠ CNN image classifiers.
- **Summary:** Class-incremental learning that combines a small exemplar memory with distillation.
- **Relation to our study:** A precedent for a small stored buffer of examples per skill.

#### Rolnick, Ahuja, Schwarz, Lillicrap & Wayne (2019): Experience replay for continual learning [V-abs]
- **Citation:** Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems 32*.
- **URL:** https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html (arXiv:1811.11682)
- **Model scale:** ⚠ RL agents on Atari and DMLab.
- **Summary:** CLEAR combines replay with behavioural cloning and greatly reduces forgetting. A limited buffer performs nearly as well as an unlimited one.
- **Relation to our study:** Simple replay is a strong default.

#### Chaudhry et al. (2019): Tiny episodic memories [V-abs]
- **Citation:** Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P. K., Torr, P. H. S., & Ranzato, M. (2019). *On tiny episodic memories in continual learning* (arXiv:1902.10486). arXiv.
- **URL:** https://arxiv.org/abs/1902.10486
- **Model scale:** ⚠ supervised vision benchmarks.
- **Summary:** Training jointly on current data and a tiny memory beats specialized continual-learning methods. Even one example per class gives a 7–17% gain.
- **Relation to our study:** A small replay fraction may be enough. Sweep it.

#### Kirkpatrick et al. (2017): EWC [V-meta]
- **Citation:** Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences, 114*(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114
- **URL:** https://doi.org/10.1073/pnas.1611835114
- **Model scale:** ⚠ MLPs and RL agents.
- **Summary:** Penalizes changes to weights that were important for earlier tasks (a regularization alternative to replay).
- **Relation to our study:** It is the consolidation mechanism in Saglietti et al.'s theory. It could be an alternative "protection" arm.

Prerequisite and skill-graph learning for LMs is covered in §2.5: Skill-it, Aioli, CurLL, SkillGraph, MASS and D³.

---

## 5. Brief checks of statements in the report's literature review

These are relevant to this review. A separate full citation audit goes in file 09.

1. **Zaremba & Sutskever (2014).**
   - Accurate:
     - the four-strategy description;
     - naive advanced on a validation plateau;
     - naive was "sometimes" worse than baseline.
   - Needs correcting:
     - "Combined was consistently the strongest" is overstated. It was best on program evaluation, but "generally (but not always)" better than mix, and not always on memorization.
     - The explanation is an untested hypothesis about hidden-state allocation.
     - The released code also advances naive at 95% validation accuracy.
   - The report correctly notes that combined mixes in *harder* examples. So its "replay" arm is not Zaremba's combined; a gated + look-ahead-mixing variant would be.
2. **CAMPUS.** "~7% over static curriculum and shuffle" is inaccurate. The 7.0% is relative to the curriculum baselines (Tree-Instruct, Conifer). Against shuffle it is 38.16 vs 36.49 (LLaMA average).
3. **RLAAR.** The numbers are correct for Qwen3-8B. The gate is the moving-average *training* reward relative to single-turn performance (ρ = 0.8). The paper is now ACL 2026.
4. **Skill-it.** The description is correct. Add that it uses LEGO and addition with 125M pretrained GPT-Neo models, and that the NeurIPS abstract gives 37.5 points.
5. **Luo et al. (ICLR 2026).** Correct. Note that their "curriculum" is ascending *quality*, at 1.5B / 30B tokens.
6. **Kulik et al. (1990).** "Moderate gains" is fair: 0.52 SD on local exams. Pair it with Slavin (1987), which found about 0 on standardized tests at equal time.
7. **The addition example.**
   - Lindsey et al. (2025) do not attribute heuristics to CE loss.
   - Zhong et al.'s clock and pizza models are CE-trained exact algorithms.
   - 2403.05845 is a 13B fine-tune with no OOD tests. Cite Lee et al. (2024) for reversal, and Zhou et al. (2024a, 2024b) and McLeish et al. (2024) for length generalization and place-value aids.
8. **Preliminary Experiment I** should cite LEGO (Zhang et al., 2023) and variable binding (Wu et al., 2025) as task templates. It should cite Yao et al. (2025) and Kohli et al. (2026) as prior k-hop curriculum work.
9. **"A strict unlock-on-probe gate … will be unique."** Soften this claim. Kohli et al. (2026) already use a ≥95% held-out-accuracy gate with cumulative replay on k-hop chains, and He et al. (2026) use Bloom-style mastery gates in distillation. What remains novel is the budget-matched, factorial comparison against shuffle and clock schedules, on a skill graph, with OOD probes.

---

## 6. Synthesis

### (a) What is established vs open

**Established:**
- **In humans, mastery learning improves performance on aligned tests by about 0.5 SD** (Kulik et al., 1990). Three caveats apply:
  - Effects are much smaller, or zero, on standardized tests when time is equal (Slavin, 1987).
  - They grow with criterion strictness and with extra corrective time.
  - Most "mastery vs non-mastery" comparisons are confounded by time (Cook et al., 2013: 3 studies).
- **Relearning cheaply protects what was mastered.** Gains from the initial criterion fade without spaced relearning (Rawson & Dunlosky, 2011). Deployed tutors re-assess mastered skills (ALEKS).
- **In neural networks, curriculum benefits are modest and conditional.**
  - They appear mainly with small budgets or noisy labels (Wu et al., 2021).
  - They are front-loaded (Weinshall et al., 2018; Saglietti et al., 2022).
  - They shrink with scale (Elgaar & Amiri, 2026).
  - They are often matched by uniform or stratified sampling (Graves et al., 2017; Matiisen et al., 2020; Chen et al., 2025, Aioli).
  - Small-LM pretraining curricula are mostly null (BabyLM 2023/2024; CLIMB; Campos, 2021).
- **The LR schedule is a first-order confound** for any schedule that puts hard or high-value data late (Luo et al., 2026). Positive 0.5–3B pretraining results often used a constant LR (Zhang et al., 2026).
- **Without replay, sequential or stage-based training forgets** (McCloskey & Cohen, 1989; CurLL, 2025). Small replay buffers largely fix this (Chaudhry et al., 2019; Rolnick et al., 2019).
- **Arithmetic and chain tasks have known recipes and failure modes:**
  - reversal helps in-distribution learning (Lee et al., 2024);
  - position or place-value aids are needed for length generalization (McLeish et al., 2024; Zhou et al., 2024a), and that generalization is seed-fragile (Zhou et al., 2024b);
  - learning goes through heuristic-then-algorithm phases (Wu et al., 2025; Nanda et al., 2023);
  - depth limits apply to chains (Sanford et al., 2024);
  - OOD composition fails (Wang et al., 2024).
- **In RL fine-tuning of 1.5–8B LLMs, competence signals from training rollouts are now standard** (DAPO, online filtering, AdaRFT, SEC, RLAAR).

**Open:**
- Whether a **hard, graph-structured gate on held-out OOD probes** beats shuffle and clock schedules **at a matched token budget**, from scratch, with replay as a separate factor. No paper we found tests this.
  - A held-out-accuracy gate with cumulative replay has already been *used* as a training recipe for k-hop reasoning (Kohli et al., 2026).
  - A curriculum is known to reduce, but not remove, the exponential data cost of implicit k-hop reasoning (Yao et al., 2025).
  - What is missing is the controlled comparison.
- Whether gate gains survive a standard LR decay, and whether they hold up when probe compute is counted in the budget.
- Whether benefits on synthetic ladders (chains, addition) carry over to messy, partially ordered natural domains (competitive programming, math). The human analogue (local vs standardized tests) suggests the gains may shrink.
- How to set the gate threshold and when to check it, given grokking and seed fragility.

### (b) Strongest prior evidence for and against mastery gating in neural nets

**For:**
1. **Matiisen et al. (2020), TSCL.**
   - Performance-driven task selection, plus re-sampling of skills that are being forgotten, beat uniform sampling and Zaremba's best manual curriculum on LSTM addition.
   - The replay ingredient "was crucial."
   - This is closest to our gate + replay arm.
2. **Abbe et al. (2023), Degree-Curriculum.**
   - A gate on training loss (ε = 0.001) let an MLP learn a degree-30 parity that it could not learn otherwise.
   - It is explicitly motivated by OOD ("unseen") generalization.
3. **Zaremba & Sutskever (2014), "combined."**
   - A gated progression plus about 20% full-range mixing beat baseline in every program-evaluation configuration, and enabled 99% accuracy on 9-digit addition.
4. **RLAAR (2026).**
   - A hard gate on reward thresholds, followed by mixed-level training, improved reliability at 1.7B and 8B.
5. **Skill-it (2023).**
   - Prerequisite-aware (soft) sampling gave +37.5 points on LEGO over random sampling, at 125M.
6. **Yao et al. (2025) and Kohli et al. (2026).**
   - Curricula substantially reduce the data needed for implicit k-hop reasoning in from-scratch GPT-2-style models.
   - Kohli et al. use a ≥95% held-out gate with joint replay of all earlier stages as their k-hop recipe. It is a gate + replay design in all but name, though without a controlled comparison.
7. **Saglietti et al. (2022), theory.**
   - Curricula give lasting gains only when earlier phases are protected, i.e., gate + consolidation or replay.
8. **Human analogues.**
   - Kulik et al. (1990).
   - Rawson & Dunlosky (2011), on gate + relearning.

**Against or cautionary:**
1. **Zaremba & Sutskever (2014), "naive."** The gated, easy-band-only progression was "sometimes" worse than baseline, and "conventional curriculum learning proved ineffective."
2. **Wu et al. (2021).** Ordering had no effect under standard budgets. Apparent gains came from changing the training-set size.
3. **Small-LM pretraining is mostly null:**
   - BabyLM 2023 and 2024 (curriculum β = −3.6);
   - CLIMB (2023);
   - Campos (2021), where Platanios-style curricula on ELMo did not help.
4. **Uniform or stratified baselines are strong.** Graves et al. (2017), Matiisen et al. (2020) and Aioli (2025) all report this. Hacohen & Weinshall (2019) also found that self-paced difficulty from the learner's current state "impairs learning."
5. **Weak or misleading baselines.**
   - Rohde & Plaut (1999): curriculum effects disappear when the baseline is tuned properly.
   - Slavin (1987): gains vanish at equal time on standardized tests.
6. **The LR-decay confound** (Luo et al., 2026).
7. **The learning dynamics make gating risky.**
   - Grokking or late phase transitions can delay probe success (Power et al., 2022; Wu et al., 2025).
   - Seed fragility makes a single gate crossing unreliable (Zhou et al., 2024b).

### (c) An experiment that would be novel and convincing to reviewers

1. **Design.**
   - Primary design: a **2 × 2 factorial** of schedule (clock vs mastery gate) × replay (none vs fixed fraction).
   - Add a **skill-stratified shuffle** (as recommended by Aioli).
   - Preliminary Experiment II already includes a time-gated + replay arm. Keep it in the main study; without it, gate and replay effects cannot be separated.
2. **Budget accounting.** Match *total* compute, including probe forward passes and any filtering or scoring. Report the result both ways, with and without probe compute (lesson from AdaRFT).
3. **Controls for known confounds:**
   - **LR schedule.** Run the main comparison under constant LR or WSD with a moderate floor, with checkpoint averaging. Also run a cosine-decay ablation (Luo et al., 2026).
   - **"Same pacing, random order".** Hold the dataset-size growth fixed but drop skill ordering (Wu et al., 2021).
   - **Gated + look-ahead mixing.** Add a small share of not-yet-unlocked skills (Zaremba's combined). This tells reviewers whether replay or look-ahead does the work.
   - **Equal tuning effort** for every arm (Rohde & Plaut, 1999).
4. **Gate design.**
   - Gate on an exponential moving average of accuracy on *held-out OOD* probes: longer inputs, long carry chains, deeper chains with shortcut-blocking orderings, and novel compositions.
   - **Sweep the threshold** (e.g., 0.8 / 0.9 / 0.95 / 0.98) and the probe frequency.
   - Log steps-to-mastery for each skill (Corbett & Anderson, 1992; Pelánek & Řihák, 2017; VanLehn, 2011).
   - Keep sealed exams disjoint from the gate probes, on both item sets and OOD axes.
   - Include an exam axis that the gate never probes. This is the "standardized test" analogue that guards against Slavin's narrowing critique.
5. **Seeds and budgets.**
   - Use ≥5 seeds, and report the variance of gate-crossing times as well as the variance of final scores (Zhou et al., 2024b).
   - Run a **budget sweep**, because curricula are expected to help most at small budgets (Wu et al., 2021).
6. **Tasks.**
   - Keep the synthetic ladders, described as adaptations of LEGO and variable binding, as the mechanistic testbed.
   - Ablate the format aids (reversal, place-value tags) so that "mastery" is not created by the format.
   - Size model depth to the deepest chain probed (Sanford et al., 2024).
   - Then run one natural verifiable domain with a frozen skill graph, e.g., a competitive-programming topic DAG.
7. **Mechanism.**
   - Track progress measures and attribution graphs across arms (Nanda et al., 2023; Ameisen et al., 2025), as the report proposes.
   - Pre-register the forgetting metric for mastered skills.
8. **Novelty claim to make.**
   - "First budget-matched, factorial test of hard prerequisite-graph unlocks gated on held-out OOD probes, separating gating from replay and from LR-schedule effects."
   - Do *not* claim novelty for any of these, each of which has prior work:
     - curricula in general;
     - competence-aware selection (CAMPUS, SEC);
     - reward-gated progression (RLAAR);
     - skill graphs (Skill-it, CurLL);
     - Bloom/ZPD-inspired LM training (He et al., 2026);
     - held-out-accuracy gates with cumulative replay on k-hop chains (Kohli et al., 2026).
   - Cite Yao et al. (2025) and Kohli et al. (2026) as the direct antecedents of Preliminary Experiment I.

### (d) Scale caveats for models of a few billion parameters or less

- **Much of the classic evidence comes from non-LM or tiny networks:**
  - LSTMs of about 2.5M parameters (Zaremba & Sutskever);
  - CNNs (Wu et al., Hacohen & Weinshall);
  - perceptrons and MLPs (Saglietti et al., Abbe et al.);
  - RL agents (Matiisen et al., Rolnick et al.).
  Treat these as mechanistic analogies.
- **Evidence from directly comparable scales is mostly null or mixed:**
  - BabyLM and CLIMB, at 125M or less;
  - Campos (ELMo);
  - Elgaar & Amiri (14M–1B), where benefits shrink with scale and act mainly through stability below about 160M.
- **Positive results at 0.5–3B** (Zhang et al., 2026; Preference Curriculum) come from clock-based schedules on general text, often under a constant LR.
- **At 1.5B, Luo et al. (2026) show that LR decay erases much of the curriculum benefit.**
- **The RL-curriculum evidence (1.5B–8B) starts from pretrained, instruction-tuned models.** It does not transfer directly to from-scratch training.
- **Arithmetic length generalization does not improve monotonically with size.**
  - Zhou et al. (2024b): the 268M model did worse than the 25M model.
  - Anil et al. (2022): deficiencies were "independent of model scale" up to 64B.
  - So more parameters will not, by themselves, make the OOD gates easier to pass.
- **The "heuristics, not algorithms" evidence comes from ≥6B pretrained models or undisclosed-size production models** (Nikankin et al., 2025; Lindsey et al., 2025). In small from-scratch models, the relevant evidence is Wu et al. (2025), LEGO, Nanda et al. (2023) and Zhong et al. (2023).
- **Recommended scale ladder.**
  - Run the synthetic studies at 10–125M, with many seeds.
  - Run the main natural-domain study at about 125M and about 350M.
  - Run one confirmation at about 1B, under the same LR schedule.
  - Report whether the gate × replay interaction changes sign across the ladder.

---

## 7. Not included / unverified leads

- **Not verified because the search budget ran out:** Tree-Instruct, YODA, "Teaching LLMs according to their aptitude," Prompt Curriculum Learning, ADCL, and "Data mixing laws."
- **Seen but not read, so not cited:**
  - "Progressive Mastery" (arXiv:2506.04065);
  - "Arithmetic Pedagogy for Language Models" (arXiv:2606.05106), a working paper on an 86M model;
  - Hupkes et al. (2020), "Compositionality decomposed" (JAIR 67, 757–795, https://doi.org/10.1613/jair.1.11674 [V-meta]; content UNVERIFIED);
  - the Kulik et al. (1990) reply to Slavin, with its numbers;
  - the EEF Toolkit.
- **Deng, Choi & Shieber (2024)** [V-abs], "From explicit CoT to implicit CoT" (https://arxiv.org/abs/2405.14838):
  - With a stepwise curriculum that removes chain-of-thought tokens, GPT-2 Small solves 9×9 multiplication (up to 99%).
  - Whether the schedule is fixed or adaptive is UNVERIFIED.
- **Fan, Du, Ramchandran & Lee (2025)** [V-full, main text], "Looped transformers for length generalization" (ICLR 2025; https://arxiv.org/abs/2409.15647):
  - Uses a length curriculum "for all methods," i.e., as a default rather than as the experimental variable.
  - Whether the steps are fixed or gated on accuracy is UNVERIFIED.

---

## References (APA 7)

Abbe, E., Bengio, S., Lotfi, A., & Rizk, K. (2023). Generalization on the unseen, logic reasoning and degree curriculum. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 31–60). https://proceedings.mlr.press/v202/abbe23a.html

Abbe, E., Cornacchia, E., & Lotfi, A. (2023). Provable advantage of curriculum learning on parity targets with mixed inputs. In *Advances in Neural Information Processing Systems 36* (pp. 24291–24321). https://papers.nips.cc/paper_files/paper/2023/hash/4c8ce3c63f6b66d6811c6d67c68e487b-Abstract-Conference.html

Albalak, A., Pan, L., Raffel, C., & Wang, W. Y. (2023). *Efficient online data mixing for language model pre-training* (arXiv:2312.02406). arXiv. https://arxiv.org/abs/2312.02406

Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., Persic, A., Qi, Z., Thompson, T. B., Zimmerman, S., Rivoire, K., Conerly, T., Olah, C., & Batson, J. (2025). Circuit tracing: Revealing computational graphs in language models. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Anil, C., Wu, Y., Andreassen, A., Lewkowycz, A., Misra, V., Ramasesh, V., Slone, A., Gur-Ari, G., Dyer, E., & Neyshabur, B. (2022). Exploring length generalization in large language models. *Advances in Neural Information Processing Systems, 35*, 38546–38556. https://arxiv.org/abs/2207.04901

Arora, S., & Goyal, A. (2023). *A theory for emergence of complex skills in language models* (arXiv:2307.15936). arXiv. https://doi.org/10.48550/arXiv.2307.15936

Bae, S., Hong, J., Lee, M. Y., Kim, H., Nam, J., & Kwak, D. (2026). Online difficulty filtering for reasoning oriented reinforcement learning. In *Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics*. https://arxiv.org/abs/2504.03380

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. In *Proceedings of the 26th Annual International Conference on Machine Learning* (pp. 41–48). ACM. https://doi.org/10.1145/1553374.1553380

Biran, E., Gottesman, D., Yang, S., Geva, M., & Globerson, A. (2024). Hopping too late: Exploring the limitations of large language models on multi-hop queries. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*. https://arxiv.org/abs/2406.12775

Bjork, R. A. (1994). Memory and metamemory considerations in the training of human beings. In J. Metcalfe & A. P. Shimamura (Eds.), *Metacognition: Knowing about knowing* (pp. 185–205). MIT Press. https://doi.org/10.7551/mitpress/4561.003.0011

Blakeney, C., Paul, M., Larsen, B. W., Owen, S., & Frankle, J. (2024). Does your data spark joy? Performance gains from domain upsampling at the end of training. In *Conference on Language Modeling (COLM 2024)*. https://arxiv.org/abs/2406.03476

Bloom, B. S. (1968). Learning for mastery. *Evaluation Comment, 1*(2), 1–12. UCLA Center for the Study of Evaluation of Instructional Programs. https://eric.ed.gov/?id=ED053419

Bloom, B. S. (1984). The 2 sigma problem: The search for methods of group instruction as effective as one-to-one tutoring. *Educational Researcher, 13*(6), 4–16. https://doi.org/10.3102/0013189X013006004

Campos, D. (2021). *Curriculum learning for language modeling* (arXiv:2108.02170). arXiv. https://doi.org/10.48550/arXiv.2108.02170

Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P. K., Torr, P. H. S., & Ranzato, M. (2019). *On tiny episodic memories in continual learning* (arXiv:1902.10486). arXiv. https://arxiv.org/abs/1902.10486

Chen, M. F., Hu, M. Y., Lourie, N., Cho, K., & Ré, C. (2025). Aioli: A unified optimization framework for language model data mixing. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2411.05735

Chen, M. F., Roberts, N., Bhatia, K., Wang, J., Zhang, C., Sala, F., & Ré, C. (2023). Skill-it! A data-driven skills framework for understanding and training language models. In *Advances in Neural Information Processing Systems 36* (pp. 36000–36040). https://proceedings.neurips.cc/paper_files/paper/2023/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html

Chen, X., Lu, J., Kim, M., Zhang, D., Tang, J., Piché, A., Gontier, N., Bengio, Y., & Kamalloo, E. (2025). *Self-evolving curriculum for LLM reasoning* (arXiv:2505.14970). arXiv. https://arxiv.org/abs/2505.14970

Cook, D. A., Brydges, R., Zendejas, B., Hamstra, S. J., & Hatala, R. (2013). Mastery learning for health professionals using technology-enhanced simulation: A systematic review and meta-analysis. *Academic Medicine, 88*(8), 1178–1186. https://doi.org/10.1097/ACM.0b013e31829a365d

Corbett, A. T., & Anderson, J. R. (1992). Knowledge tracing in the ACT Programming Tutor. In *Proceedings of the Fourteenth Annual Conference of the Cognitive Science Society*. Erlbaum. https://escholarship.org/uc/item/0873k1nz

Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction, 4*(4), 253–278. https://doi.org/10.1007/BF01099821

Cornacchia, E., & Mossel, E. (2023). A mathematical model for curriculum learning for parities. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 6402–6423). https://proceedings.mlr.press/v202/cornacchia23a.html

Cosyn, E., Uzun, H., Doble, C., & Matayoshi, J. (2021). A practical perspective on knowledge space theory: ALEKS and its data. *Journal of Mathematical Psychology, 101*, Article 102512. https://doi.org/10.1016/j.jmp.2021.102512

Dai, Y., Huang, Y., Zhang, X., Wu, W., Li, C., Lu, W., Cao, S., Dong, L., & Li, S. (2025). *Data efficacy for language model training* (arXiv:2506.21545). arXiv. https://arxiv.org/abs/2506.21545

Deng, Y., Choi, Y., & Shieber, S. (2024). *From explicit CoT to implicit CoT: Learning to internalize CoT step by step* (arXiv:2405.14838). arXiv. https://arxiv.org/abs/2405.14838

Diehl Martinez, R., Goriely, Z., McGovern, H., Davis, C., Caines, A., Buttery, P., & Beinborn, L. (2023). CLIMB – Curriculum learning for infant-inspired model building. In *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning* (pp. 112–127). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.conll-babylm.10

Doignon, J.-P., & Falmagne, J.-C. (1985). Spaces for the assessment of knowledge. *International Journal of Man-Machine Studies, 23*(2), 175–196. https://doi.org/10.1016/S0020-7373(85)80031-6

Doignon, J.-P., & Falmagne, J.-C. (1999). *Knowledge spaces*. Springer. https://doi.org/10.1007/978-3-642-58625-5

Dziri, N., Lu, X., Sclar, M., Li, X. L., Jiang, L., Lin, B. Y., West, P., Bhagavatula, C., Le Bras, R., Hwang, J. D., Sanyal, S., Welleck, S., Ren, X., Ettinger, A., Harchaoui, Z., & Choi, Y. (2023). Faith and fate: Limits of transformers on compositionality. *Advances in Neural Information Processing Systems, 36*. https://arxiv.org/abs/2305.18654

Elgaar, M., & Amiri, H. (2026). *Curriculum learning for LLM pretraining: An analysis of learning dynamics* (arXiv:2601.21698). arXiv. https://arxiv.org/abs/2601.21698

Elman, J. L. (1993). Learning and development in neural networks: The importance of starting small. *Cognition, 48*(1), 71–99. https://doi.org/10.1016/0010-0277(93)90058-4

Falmagne, J.-C., & Doignon, J.-P. (2011). *Learning spaces: Interdisciplinary applied mathematics*. Springer. https://doi.org/10.1007/978-3-642-01039-2

Fan, Y., Du, Y., Ramchandran, K., & Lee, K. (2025). Looped transformers for length generalization. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2409.15647

Feng, J., & Steinhardt, J. (2024). How do language models bind entities in context? In *International Conference on Learning Representations (ICLR 2024)*. https://arxiv.org/abs/2310.17191

French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2

Graves, A., Bellemare, M. G., Menick, J., Munos, R., & Kavukcuoglu, K. (2017). Automated curriculum learning for neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR Vol. 70, pp. 1311–1320). https://proceedings.mlr.press/v70/graves17a.html

Guskey, T. R. (2007). Closing achievement gaps: Revisiting Benjamin S. Bloom's "Learning for Mastery." *Journal of Advanced Academics, 19*(1), 8–31. https://doi.org/10.4219/jaa-2007-704

Guskey, T. R. (2010). Lessons of mastery learning. *Educational Leadership, 68*(2), 52–57. https://eric.ed.gov/?id=EJ913779

Guskey, T. R., & Pigott, T. D. (1988). Research on group-based mastery learning programs: A meta-analysis. *The Journal of Educational Research, 81*(4), 197–216. https://doi.org/10.1080/00220671.1988.10885824

Hacohen, G., & Weinshall, D. (2019). On the power of curriculum learning in training deep networks. In *Proceedings of the 36th International Conference on Machine Learning* (PMLR Vol. 97, pp. 2535–2544). https://proceedings.mlr.press/v97/hacohen19a.html

He, B., Chen, Y., Zhang, X., Kong, L., Yu, P. S., Liu, X., & Ma, C. (2026). Pedagogically-inspired data synthesis for language model knowledge distillation. In *International Conference on Learning Representations (ICLR 2026)*. https://arxiv.org/abs/2602.12172

Hu, M. Y., Mueller, A., Ross, C., Williams, A., Linzen, T., Zhuang, C., Cotterell, R., Choshen, L., Warstadt, A., & Wilcox, E. G. (2024). Findings of the second BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning* (pp. 1–21). Association for Computational Linguistics. https://aclanthology.org/2024.conll-babylm.1.pdf

Israni, A., Sales, A. C., & Pane, J. F. (2018). *Mastery learning in practice: A (mostly) descriptive analysis of log data from the Cognitive Tutor Algebra I effectiveness trial* (arXiv:1802.08616). arXiv. https://arxiv.org/abs/1802.08616

Jia, Y., Zhang, C., Diao, X., Yuan, X., Ouyang, Z., Ma, C., & Vosoughi, S. (2025). *What makes a good curriculum? Disentangling the effects of data ordering on LLM mathematical reasoning* (arXiv:2510.19099). arXiv. https://arxiv.org/abs/2510.19099

Kalyan, P., Mishra, S., Lokam, S., & Goyal, N. (2025). *CurLL: A developmental framework to evaluate continual learning in language models* (arXiv:2510.13008). arXiv. https://arxiv.org/abs/2510.13008

Kalyuga, S., Ayres, P., Chandler, P., & Sweller, J. (2003). The expertise reversal effect. *Educational Psychologist, 38*(1), 23–31. https://doi.org/10.1207/S15326985EP3801_4

Keller, F. S. (1968). "Good-bye, teacher…". *Journal of Applied Behavior Analysis, 1*(1), 79–89. https://doi.org/10.1901/jaba.1968.1-79

Kim, J., & Lee, J. (2024). *Strategic data ordering: Enhancing large language model performance through curriculum learning* (arXiv:2405.07490). arXiv. https://arxiv.org/abs/2405.07490

Kimi Team. (2025). *Kimi k1.5: Scaling reinforcement learning with LLMs* (arXiv:2501.12599). arXiv. https://arxiv.org/abs/2501.12599

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences, 114*(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114

Kocmi, T., & Bojar, O. (2017). Curriculum learning and minibatch bucketing in neural machine translation. In *Proceedings of the International Conference Recent Advances in Natural Language Processing (RANLP 2017)* (pp. 379–386). INCOMA Ltd. https://doi.org/10.26615/978-954-452-049-6_050

Kohli, H., Parthasarathy, S., Sun, H., & Yao, Y. (2026). Loop, think, & generalize: Implicit reasoning in recurrent-depth transformers. In *Conference on Language Modeling (COLM 2026)*. https://arxiv.org/abs/2604.07822

Kulik, C.-L. C., Kulik, J. A., & Bangert-Drowns, R. L. (1990). Effectiveness of mastery learning programs: A meta-analysis. *Review of Educational Research, 60*(2), 265–299. https://doi.org/10.3102/00346543060002265

Kumar, M. P., Packer, B., & Koller, D. (2010). Self-paced learning for latent variable models. In *Advances in Neural Information Processing Systems 23* (pp. 1189–1197). https://proceedings.neurips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html

Lee, N., Sreenivasan, K., Lee, J. D., Lee, K., & Papailiopoulos, D. (2024). Teaching arithmetic to small transformers. In *International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=dsUB4bst9S

Li, C., Zhang, M., & He, Y. (2022). The stability-efficiency dilemma: Investigating sequence length warmup for training GPT models. In *Advances in Neural Information Processing Systems 35*. https://arxiv.org/abs/2108.06084

Li, J., Yu, L., Cui, Q., Zhang, Z., Zhou, J., Ye, Y., & Zhang, C. (2025). *MASS: Mathematical data selection via skill graphs for pretraining large language models* (arXiv:2503.14917). arXiv. https://arxiv.org/abs/2503.14917

Li, M., Chen, P., Zhang, Z., Yang, T., Zhang, X., Li, H., Cao, T., Zeng, M., Wu, Z., Jiang, M., Li, H., Li, L., & Yin, B. (2026). Mitigating lost in multi-turn conversation via curriculum RL with verifiable accuracy and abstention rewards. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)*. https://arxiv.org/abs/2510.18731

Li, X., Li, M., Bao, K., Ma, Y., Wang, W., Liu, D., & Feng, F. (2026). *SkillGraph: Skill-augmented reinforcement learning for agents via evolving skill graphs* (arXiv:2605.12039). arXiv. https://arxiv.org/abs/2605.12039

Li, Y., Lu, T., Li, Y., Chen, Y., Huang, W.-C., Jiang, W., Wang, H., Zheng, H.-T., & Yu, P. S. (2025). Teaching according to talents! Instruction tuning LLMs with competence-aware curriculum learning. In *Findings of the Association for Computational Linguistics: EMNLP 2025*. https://arxiv.org/abs/2509.13790

Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce, A., Turner, N. L., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., Persic, A., Qi, Z., Thompson, T. B., Zimmerman, S., Rivoire, K., Conerly, T., Olah, C., & Batson, J. (2025). On the biology of a large language model. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/biology.html

Liu, Z., Kitouni, O., Nolte, N., Michaud, E. J., Tegmark, M., & Williams, M. (2022). Towards understanding grokking: An effective theory of representation learning. In *Advances in Neural Information Processing Systems 35*. https://arxiv.org/abs/2205.10343

Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *International Conference on Learning Representations (ICLR 2026)*. https://arxiv.org/abs/2511.18903

Matiisen, T., Oliver, A., Cohen, T., & Schulman, J. (2020). Teacher–student curriculum learning. *IEEE Transactions on Neural Networks and Learning Systems, 31*(9), 3732–3740. https://doi.org/10.1109/TNNLS.2019.2934906

McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation, 24*, 109–165. https://doi.org/10.1016/S0079-7421(08)60536-8

McGaghie, W. C., Issenberg, S. B., Barsuk, J. H., & Wayne, D. B. (2014). A critical review of simulation-based mastery learning with translational outcomes. *Medical Education, 48*(4), 375–385. https://doi.org/10.1111/medu.12391

McLeish, S., Bansal, A., Stein, A., Jain, N., Kirchenbauer, J., Bartoldson, B. R., Kailkhura, B., Bhatele, A., Geiping, J., Schwarzschild, A., & Goldstein, T. (2024). Transformers can do arithmetic with the right embeddings. In *Advances in Neural Information Processing Systems 37*. https://proceedings.neurips.cc/paper_files/paper/2024/hash/c35986bc1ee29b31c1011481b77fe540-Abstract-Conference.html

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. In *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2301.05217

Narvekar, S., Peng, B., Leonetti, M., Sinapov, J., Taylor, M. E., & Stone, P. (2020). Curriculum learning for reinforcement learning domains: A framework and survey. *Journal of Machine Learning Research, 21*(181), 1–50. https://jmlr.org/papers/v21/20-212.html

Nikankin, Y., Reusch, A., Mueller, A., & Belinkov, Y. (2025). Arithmetic without algorithms: Language models solve math with a bag of heuristics. In *International Conference on Learning Representations (ICLR 2025)*. https://openreview.net/forum?id=09YTt26r2P

Okawa, M., Lubana, E. S., Dick, R. P., & Tanaka, H. (2023). Compositional abilities emerge multiplicatively: Exploring diffusion models on a synthetic task. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2310.09336

Parashar, S., Gui, S., Li, X., Ling, H., Vemuri, S., Olson, B., Li, E., Zhang, Y., Caverlee, J., Kalathil, D., & Ji, S. (2026). Curriculum reinforcement learning from easy to hard tasks improves LLM reasoning. In *International Conference on Learning Representations (ICLR 2026)*. https://arxiv.org/abs/2506.06632

Pelánek, R. (2017). Bayesian knowledge tracing, logistic models, and beyond: An overview of learner modeling techniques. *User Modeling and User-Adapted Interaction, 27*(3–5), 313–350. https://doi.org/10.1007/s11257-017-9193-2

Pelánek, R., & Řihák, J. (2017). Experimental analysis of mastery learning criteria. In *Proceedings of the 25th Conference on User Modeling, Adaptation and Personalization* (pp. 156–163). ACM. https://doi.org/10.1145/3079628.3079667

Platanios, E. A., Stretcu, O., Neubig, G., Poczos, B., & Mitchell, T. (2019). Competence-based curriculum learning for neural machine translation. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1* (pp. 1162–1172). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1119

Portelas, R., Colas, C., Weng, L., Hofmann, K., & Oudeyer, P.-Y. (2020). Automatic curriculum learning for deep RL: A short survey. In *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence* (pp. 4819–4825). https://doi.org/10.24963/ijcai.2020/671

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets* (arXiv:2201.02177). arXiv. https://doi.org/10.48550/arXiv.2201.02177

Ramesh, R., Lubana, E. S., Khona, M., Dick, R. P., & Tanaka, H. (2024). Compositional capabilities of autoregressive transformers: A study on synthetic, interpretable tasks. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235, pp. 42074–42103). https://proceedings.mlr.press/v235/ramesh24a.html

Rawson, K. A., & Dunlosky, J. (2011). Optimizing schedules of retrieval practice for durable and efficient learning: How much is enough? *Journal of Experimental Psychology: General, 140*(3), 283–302. https://doi.org/10.1037/a0023956

Rawson, K. A., Dunlosky, J., & Sciartelli, S. M. (2013). The power of successive relearning: Improving performance on course exams and long-term retention. *Educational Psychology Review, 25*(4), 523–548. https://doi.org/10.1007/s10648-013-9240-4

Rebuffi, S.-A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). iCaRL: Incremental classifier and representation learning. In *2017 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 5533–5542). IEEE. https://doi.org/10.1109/CVPR.2017.587

Ritter, S., Yudelson, M., Fancsali, S. E., & Berman, S. R. (2016). How mastery learning works at scale. In *Proceedings of the Third ACM Conference on Learning @ Scale* (pp. 71–79). ACM. https://doi.org/10.1145/2876034.2876039

Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318

Rohde, D. L. T., & Plaut, D. C. (1999). Language acquisition in the absence of explicit negative evidence: How important is starting small? *Cognition, 72*(1), 67–109. https://doi.org/10.1016/S0010-0277(99)00031-1

Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html

Saglietti, L., Sarao Mannelli, S., & Saxe, A. (2022). An analytical theory of curriculum learning in teacher–student networks. *Journal of Statistical Mechanics: Theory and Experiment, 2022*(11), 114014. https://doi.org/10.1088/1742-5468/ac9b3c

Sanford, C., Hsu, D., & Telgarsky, M. (2024). Transformers, parallel computation, and logarithmic depth. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235, pp. 43276–43327). https://proceedings.mlr.press/v235/sanford24a.html

Shen, R., Bubeck, S., Eldan, R., Lee, Y. T., Li, Y., & Zhang, Y. (2023). *Positional description matters for transformers arithmetic* (arXiv:2311.14737). arXiv. https://doi.org/10.48550/arXiv.2311.14737

Shi, T., Wu, Y., Song, L., Zhou, T., & Zhao, J. (2025). Efficient reinforcement finetuning via adaptive curriculum learning. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2504.05520

Slavin, R. E. (1987). Mastery learning reconsidered. *Review of Educational Research, 57*(2), 175–213. https://doi.org/10.3102/00346543057002175

Soviany, P., Ionescu, R. T., Rota, P., & Sebe, N. (2022). Curriculum learning: A survey. *International Journal of Computer Vision, 130*(6), 1526–1565. https://doi.org/10.1007/s11263-022-01611-x

VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems. *Educational Psychologist, 46*(4), 197–221. https://doi.org/10.1080/00461520.2011.611369

Vygotsky, L. S. (1978). *Mind in society: The development of higher psychological processes* (M. Cole, V. John-Steiner, S. Scribner, & E. Souberman, Eds.). Harvard University Press. https://doi.org/10.2307/j.ctvjf9vz4

Wang, B., Yue, X., Su, Y., & Sun, H. (2024). Grokked transformers are implicit reasoners: A mechanistic journey to the edge of generalization. In *Advances in Neural Information Processing Systems 37*. https://arxiv.org/abs/2405.15071

Wang, X., Chen, Y., & Zhu, W. (2022). A survey on curriculum learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 44*(9), 4555–4576. https://doi.org/10.1109/TPAMI.2021.3069908

Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., Mosquera, R., Paranjape, B., Williams, A., Linzen, T., & Cotterell, R. (2023). Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning* (pp. 1–34). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.conll-babylm.1

Weinshall, D., Cohen, G., & Amir, D. (2018). Curriculum learning by transfer learning: Theory and experiments with deep networks. In *Proceedings of the 35th International Conference on Machine Learning* (PMLR Vol. 80, pp. 5238–5246). https://proceedings.mlr.press/v80/weinshall18a.html

Wen, L., Cai, Y., Xiao, F., He, X., An, Q., Duan, Z., Du, Y., Liu, J., Tang, L., Lv, X., Zou, H., Deng, Y., Jia, S., & Zhang, X. (2025). Light-R1: Curriculum SFT, DPO and RL for long COT from scratch and beyond. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Industry Track)*. https://arxiv.org/abs/2503.10460

Wood, D., Bruner, J. S., & Ross, G. (1976). The role of tutoring in problem solving. *Journal of Child Psychology and Psychiatry, 17*(2), 89–100. https://doi.org/10.1111/j.1469-7610.1976.tb00381.x

Wu, X., Dyer, E., & Neyshabur, B. (2021). When do curricula work? In *International Conference on Learning Representations (ICLR 2021)*. https://openreview.net/forum?id=tW4QEInpni

Wu, Y., Geiger, A., & Millière, R. (2025). How do transformers learn variable binding in symbolic programs? In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR Vol. 267, pp. 67284–67299). https://proceedings.mlr.press/v267/wu25j.html

Xie, S. M., Pham, H., Dong, X., Du, N., Liu, H., Lu, Y., Liang, P., Le, Q. V., Ma, T., & Yu, A. W. (2023). DoReMi: Optimizing data mixtures speeds up language model pretraining. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2305.10429

Xu, B., Zhang, L., Mao, Z., Wang, Q., Xie, H., & Zhang, Y. (2020). Curriculum learning for natural language understanding. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 6095–6104). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.542

Xu, Y., Hao, J., Zhang, G., & Li, Z. (2026). *D³: Dynamic directional graph-constrained data scheduling for LLM training* (arXiv:2605.31164). arXiv. https://arxiv.org/abs/2605.31164

Yao, Y., Du, Y., Zhu, D., Hahn, M., & Koller, A. (2025). Language models can learn implicit multi-hop reasoning, but only if they have lots of training data. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing* (pp. 9684–9702). Association for Computational Linguistics. https://arxiv.org/abs/2505.17923

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., et al. (2025). *DAPO: An open-source LLM reinforcement learning system at scale* (arXiv:2503.14476). arXiv. https://arxiv.org/abs/2503.14476

Zaremba, W., & Sutskever, I. (2014). *Learning to execute* (arXiv:1410.4615). arXiv. https://doi.org/10.48550/arXiv.1410.4615

Zhang, C., Raghu, M., Kleinberg, J., & Bengio, S. (2021). *Pointer value retrieval: A new benchmark for understanding the limits of neural network generalization* (arXiv:2107.12580). arXiv. https://doi.org/10.48550/arXiv.2107.12580

Zhang, J., Vanacore, K., Baker, R. S., Ch, N., Mills, C., & Henkel, O. (2025). How much mastery is enough mastery? The relationship between mastery in a lesson and the performance on the subsequent lesson. In *Proceedings of the 18th International Conference on Educational Data Mining* (short paper). https://educationaldatamining.org/edm2025/proceedings/2025.EDM.short-papers.4/2025.EDM.short-papers.4.pdf

Zhang, X., Xu, L., Duan, F., Zhou, Y., Wang, S., Weng, R., Wang, J., & Cai, X. (2025). Preference curriculum: LLMs should always be pretrained on their preferred data. In *Findings of the Association for Computational Linguistics: ACL 2025* (pp. 21181–21198). https://aclanthology.org/2025.findings-acl.1091/

Zhang, Y., Backurs, A., Bubeck, S., Eldan, R., Gunasekar, S., & Wagner, T. (2023). *Unveiling transformers with LEGO: A synthetic reasoning task* (arXiv:2206.04301). arXiv. https://doi.org/10.48550/arXiv.2206.04301

Zhang, Y., Mohamed, A., Abdine, H., Shang, G., & Vazirgiannis, M. (2026). Beyond random sampling: Efficient language model pretraining via curriculum learning. In *Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 5776–5794). https://aclanthology.org/2026.eacl-long.271/

Zhang-Li, D., Lin, N., Yu, J., Zhang, Z., Yao, Z., Zhang, X., Hou, L., Zhang, J., & Li, J. (2024). *Reverse that number! Decoding order matters in arithmetic learning* (arXiv:2403.05845). arXiv. https://doi.org/10.48550/arXiv.2403.05845

Zhong, Z., Liu, Z., Tegmark, M., & Andreas, J. (2023). The clock and the pizza: Two stories in mechanistic explanation of neural networks. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2306.17844

Zhou, H., Bradley, A., Littwin, E., Razin, N., Saremi, O., Susskind, J., Bengio, S., & Nakkiran, P. (2024a). What algorithms can transformers learn? A study in length generalization. In *International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=AssIuHnmHX

Zhou, Y., Alon, U., Chen, X., Wang, X., Agarwal, R., & Zhou, D. (2024b). Transformers can achieve length generalization but not robustly. In *ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models*. https://openreview.net/forum?id=DWkWIh3vFJ
