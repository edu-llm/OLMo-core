# Citation Verification: P4 Final Report, TEST 02 (Blocked vs. Interleaved) and TEST 03 (Mastery-Gated Curriculum)

## TL;DR

- **Every cited source exists, but the report has no reference list.** Bengio et al. (2009), Kulik et al. (1990), iCaRL, LEGO, OLMES and the LR-decay paper's authors appear only as names or bare links. Several entries now have archival versions:
  - RLAAR → ACL 2026 (long papers).
  - Skill-it → NeurIPS 2023.
  - Clock/Pizza → NeurIPS 2023.
  - Rolnick et al. → NeurIPS 2019.
  - LR-decay paper → ICLR 2026 (Luo et al.; arXiv:2511.18903).
  - CAMPUS → *Findings of EMNLP 2025* (the venue in the report is correct).
- **Three citations do not support the sentences they are attached to:**
  1. The Anthropic "biology" link goes to the wrong section. `#dives-tracing` is the multi-step-reasoning (Dallas→Austin) example; addition is at `#dives-addition`. Even the addition section stresses that the features *generalize* across contexts. It never says that cross-entropy loss produces a "cheap heuristic" that "collapses" out of distribution. Use Nikankin et al. (2025) for that claim.
  2. **Robins (1995) did not "introduce" replay.** His abstract says he replicates Ratcliff's (1990) rehearsal experiments. His new contributions were sweep rehearsal and *pseudo*rehearsal.
  3. **The reversed-digit paper (Zhang-Li et al., 2024) contains no OOD or length-generalization tests.** It fine-tunes Llama2-13B with reversal *plus* step-by-step reasoning. Reversal alone fails for multiplication. "Improves … generalization" is unsupported. Lee et al. (2024) reverse the output too but say that "length generalization beyond trained digit lengths [is] difficult."
- **The CAMPUS "about 7%" is mis-scoped.** The 7.0% is CAMPUS's average gain over the two *static-curriculum* baselines (Tree-Instruct, Conifer) on LLaMA. Over random shuffle the gain is only +1.67 points (38.16 vs 36.49, about 4.6% relative).
- **The RLAAR numbers are exact** (Qwen3-8B, Table 1 averages), and the gating rule is described correctly: moving-average reward ≥ ρ·(single-turn baseline), with ρ = 0.8 and a window of 5. But 62.6→75.1 and 33.5→73.4 compare the *whole RLAAR method* against the untrained base model, not the curriculum against no curriculum. The curriculum-only ablation is on Qwen3-1.7B: 63.2 → 71.9 LiC.
- **Zaremba & Sutskever is described almost perfectly** (four strategies; naive increases difficulty when validation progress stops; naive is "sometimes worse than baseline"; combined mixes in harder examples). Two problems:
  - "Consistently the strongest" holds for program evaluation only. On memorization, combined did not always beat mix.
  - Their explanation is about exposure to *harder* examples, not about retaining easier ones. Their result motivates hard-example mixing more than replay of mastered skills.
- **The LR-decay paper is about data *quality* ordering, not difficulty, and the proposed mitigation does not work.** Using an identical schedule across arms does not remove the confound; that is the setting in which the paper finds the curriculum advantage shrinks. WSD also decays at the end. The paper's fixes are moderate decay and checkpoint averaging.
- **Number and figure problems:**
  - Preliminary Experiment II text says "(0.59, 0.67)", but the table shows 0.67 and **0.55**.
  - The one-pager figure labels the mastery panel "Prerequisite-chain task", but 0.67/0.85/0.97 are **Prelim II (repeated addition)** numbers. On the actual prerequisite-chain task (Prelim I), OOD is shuffle 0.37, naive gate 0.33, gate+replay 0.75, so the naive gate does *worse* than shuffle there.
  - `P4_onepager.html` mixes the two experiments in a single sentence.

**Report checked:** `/Users/amylin/Amy/MIT/Internships/Alpha_AI_Engineering/projects/edu-llm/paper/source/P4_final_report.md`, lines 173–307 (TEST 02 and TEST 03). I did not edit it.
**Also checked:** `P4_onepager.md`, `P4_onepager.html`, `P4_onepager_quarter.html`, `P4_onepager_figure.png`, `P4_onepager_figure_compact.png`, `paper/figures/final_report_fig2_mastery_prelim1_table.png`, `paper/figures/final_report_fig3_mastery_prelim2_curve.png`.
**Date:** 2026-10-04
**Method:**
- Bibliographic data came from primary records: arXiv abstract pages, ACL Anthology, Crossref API, NeurIPS proceedings / ML Anthology, the ICLR virtual site, transformer-circuits.pub, ERIC, OpenAlex abstracts and Semantic Scholar abstracts.
- I read full text for these papers: Zaremba & Sutskever (arXiv PDF); Rolnick et al. (NeurIPS PDF); Skill-it (NeurIPS PDF); CAMPUS (ACL Anthology PDF); RLAAR (arXiv HTML v3); Zhang-Li et al. (arXiv HTML); Lee et al. (arXiv PDF, for the suggested citation).
- I downloaded nothing from GitHub. GitHub README pages were read only through WebFetch.

**Verdict labels:** ACCURATE / PARTIALLY ACCURATE / INACCURATE. **UNVERIFIED** means I could not confirm the item against a primary source.

---

## Summary table

| # | Reference (as used in the report) | Bibliographic status | Usage verdict |
|---|---|---|---|
| 1 | Brunmair & Richter (2019) | Correct DOI. No full reference given | g = 0.42 and "varied substantially by task": **ACCURATE**. "Affect retention": PARTIALLY ACCURATE (the meta-analysis is about inductive category learning) |
| 2 | Rolnick et al. (2019) | arXiv link only. **Published at NeurIPS 2019**, pp. 350–360 | **ACCURATE** (Sec. 4.1 compares sequential vs. simultaneous training with experience held constant) |
| 3 | Bengio et al. (2009) | **No reference/link given.** ICML 2009, pp. 41–48 | **ACCURATE** |
| 4 | Kulik, Kulik & Bangert-Drowns (1990) | **No reference/link given.** *Rev. Educ. Res.* 60(2) | Direction ACCURATE. "Moderate" is plausible but the magnitude is **UNVERIFIED** from the primary text. Missing caveats (heterogeneity; Slavin's critique) |
| 5 | RLAAR (arXiv:2510.18731) | **Needs update. Published at ACL 2026** (long), pp. 33364–33381 | Gating rule: **ACCURATE**. Numbers: **exact**, but they belong to the full method, not the curriculum → PARTIALLY ACCURATE. Typo "form" |
| 6 | CAMPUS ("Li et al., EMNLP Findings 2025") | Venue **correct**. Pages 11724–11741 | **PARTIALLY ACCURATE**: 7% is versus static curricula only; versus shuffle it is about 4.6% relative (LLaMA) |
| 7 | McCloskey & Cohen (1989) | DOI link only. Year, editor and volume missing | **ACCURATE** |
| 8 | French (1999) | DOI link only. Year missing | **ACCURATE** |
| 9 | Robins (1995) | **Stanford proxy URL** (inaccessible to readers). DOI 10.1080/09540099550039318 | **INACCURATE** for "introduced by Robins" (rehearsal predates him: Ratcliff, 1990) |
| 10 | iCaRL (Rebuffi et al., 2017) | **No reference/link given.** CVPR 2017 | **ACCURATE** |
| 11 | Rolnick et al. (2019), second use | as #2 | **ACCURATE** |
| 12 | Zaremba & Sutskever (2014) | arXiv only (never formally published; the PDF says "Under review … ICLR 2015") | Mostly **ACCURATE**. "Consistently the strongest" is PARTIALLY ACCURATE. Using it to justify replay is an interpretive stretch |
| 13 | Skill-it (arXiv:2307.14430) | **Needs update: NeurIPS 2023**, pp. 36000–36040 | **ACCURATE**. Minor: "never withholds anything" is slightly overstated |
| 14 | LR-decay curriculum paper (ICLR 2026) | Title and venue correct. **Authors missing** (Luo et al.). arXiv:2511.18903 | Mechanism: ACCURATE. "Best **or hardest**" and the mitigation paragraph: **PARTIALLY ACCURATE / misleading** |
| 15 | Anthropic "Biology" (Lindsey et al., 2025) | **Wrong anchor** (`#dives-tracing` is multi-step reasoning). No author/year given | **INACCURATE** as support for "CE loss → cheap heuristic that collapses OOD" |
| 16 | Clock & Pizza (Zhong et al., 2023) | PDF link. **NeurIPS 2023**, pp. 27223–27250 | **PARTIALLY ACCURATE**: these are algorithms for *modular* addition. The Clock algorithm comes from Nanda et al. (2023) |
| 17 | Reversed digits (Zhang-Li et al., 2024) | arXiv v1 only. No published version found | Accuracy and carrying: ACCURATE (narrowly). "Simply" and "generalization": **INACCURATE** |
| 18 | LEGO tasks (Zhang et al., 2022) | **Uncited** | Description: ACCURATE ("resembling"). Citation required |
| 19 | Anthropic "Methods" (Ameisen et al., 2025) | Link only, no author/year | **ACCURATE** |
| 20 | OLMo-core | GitHub link resolves (now `allenai/Olmo-core`). The README asks users to cite the OLMo 2 report | ACCURATE as a software pointer |
| 21 | OLMES ("OLMo Eval/olmes") | **Not linked/cited**. Gu et al., *Findings of NAACL 2025* | ACCURATE in substance. Checkpoint compatibility is UNVERIFIED |

---

## TEST 02: Blocked vs. Interleaved Subject Training

### 1. Brunmair & Richter (2019)

**Corrected APA 7:**
> Brunmair, M., & Richter, T. (2019). Similarity matters: A meta-analysis of interleaved learning and its moderators. *Psychological Bulletin, 145*(11), 1029–1052. https://doi.org/10.1037/bul0000209

**Bibliographic check:** Crossref confirms the title, both authors, journal, 145(11), pp. 1029–1052 and the DOI. The DOI link in the report is correct. **CORRECT**, but a full reference entry is missing.

**Sentence (line 181):** "[Brunmair and Richter's meta-analysis] found a moderate average interleaving benefit of roughly *g = 0.42*, although results varied substantially by task"

- **Verdict: ACCURATE.** Abstract, via Semantic Scholar / PsycINFO:
  - "A multilevel meta-analysis revealed a moderate overall interleaving effect (Hedges' g = 0.42)."
  - "Interleaved practice was best for studies using paintings (g = 0.67) … mathematical tasks revealed a small interleaving effect (g = 0.34), whereas results for expository texts and tastes were ambiguous with nonsignificant overall effects. An advantage of blocking compared with interleaving was found for studies based on words (g = -0.39)."
  - The analysis covers 59 studies, 238 effect sizes and 158 samples.
- **Suggested precision:** "varied substantially by *type of learning material*." Blocking was better for word materials.

**Framing caveat. The lead-in "practice order can affect retention" is PARTIALLY ACCURATE:**
- The meta-analysis concerns *inductive (category) learning*, i.e., classifying new exemplars. It does not concern retention or forgetting.
- Its metaregression found "stronger interleaving effects for learning material more similar between categories, for learning material less similar within categories, and for more complex learning material."
- Arithmetic, logic, geometry and statistics are *dissimilar* between categories. On Brunmair & Richter's own moderator, that is the condition where the human benefit is *smallest*. The report should say so, because it weakens the human-literature motivation for TEST 02.
- The authors also conclude that "interleaving should be used with caution in certain conditions."
- **Suggested addition** (a math-practice interleaving study closer to "subjects"): Rohrer, D., & Taylor, K. (2007). The shuffling of mathematics problems improves learning. *Instructional Science, 35*(6), 481–498. https://doi.org/10.1007/s11251-007-9015-8 (Crossref-verified).

### 2. Rolnick et al. (2019)

**Corrected APA 7:**
> Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems 32* (pp. 350–360). Curran Associates. https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html (arXiv:1811.11682)

**Bibliographic check:**
- arXiv v2 (26 Nov 2019) has the comment "NeurIPS 2019".
- ML Anthology and the NeurIPS proceedings give pp. 350–360.
- **Update the reference from arXiv to NeurIPS 2019.**

**Sentence (line 181):** "continual-learning work shows that replay and temporal data order can change model forgetting ([Rolnick et al., 2019])"

- **Verdict: ACCURATE.** The order point is supported, not just the replay point.
- **Order (Sec. 4.1, Fig. 1):** The paper compares three training paradigms on DMLab tasks: "(1) Training networks on the individual tasks separately, (2) training a single network examples from all tasks simultaneously …, and (3) training a single network sequentially on examples from one task, then the next task, and so on cyclically. Across all training protocols, the total amount of experience for each task was held constant." The result: "In sequential training, tasks that are not currently being learned exhibit very dramatic catastrophic forgetting."
- **Replay:** CLEAR "virtually eliminate[s]" forgetting (Fig. 3). Example cumulative scores: Sequential 17.99 vs. CLEAR 31.40 vs. Simultaneous 32.35 on explore_object_locations.
- **Nuances worth one clause:**
  1. The setting is reinforcement learning (Atari/DMLab), not language modeling.
  2. Rolnick's "simultaneous" mixes tasks *within the same minibatch*. TEST 02 deliberately uses subject-pure batches that alternate every update, which falls between Rolnick's two conditions.
  3. Rolnick's default mix is 50-50 new/replay ("performance does not appear to be very sensitive to this ratio"). This is relevant to the TEST 03 replay fraction.

### 3. OLMo-core

**Suggested citation** (the OLMo-core README's "Citing" section asks users to cite the OLMo 2 report):
> Team OLMo, Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., et al. (2024). *2 OLMo 2 furious* (arXiv:2501.00656). arXiv. https://arxiv.org/abs/2501.00656

Software: allenai/OLMo-core [Computer software]. https://github.com/allenai/OLMo-core (it now redirects to `allenai/Olmo-core`; Apache-2.0).

- In APA 7, list the first 19 authors from the arXiv record. I did not verify the full author list or whether there is a later archival venue: **UNVERIFIED**.
- **Usage (line 207): ACCURATE as a software pointer.** The README confirms `torchrun` launch scripts, a Beaker launch CLI, `--save-folder` checkpointing and `--checkpoint` resume.
- The README does not explicitly document defining arbitrary model sizes or restoring optimizer state. The report's claim that the checkpoint "restore[s] weights, optimizer moments, scheduler step, and RNG state" is a plan to verify, not a documented feature (**UNVERIFIED** from the README). I did not inspect the code, per the no-clone constraint.

### 4. OLMES ("OLMo Eval/olmes")

**Suggested citation:**
> Gu, Y., Tafjord, O., Kuehl, B., Haddad, D., Dodge, J., & Hajishirzi, H. (2025). OLMES: A standard for language model evaluations. In *Findings of the Association for Computational Linguistics: NAACL 2025*. Association for Computational Linguistics. https://arxiv.org/abs/2406.08446

- The venue comes from the arXiv comment "Findings of NAACL 2025". Pages and DOI are **UNVERIFIED** (I did not retrieve the ACL Anthology page).
- The olmes README describes it as "Reproducible, flexible LLM evaluations" (Open Language Model Evaluation System). It accepts any Hugging Face model path or local path.
- A custom OLMo-core checkpoint therefore probably needs conversion to HF format first. The report's hedge ("after confirming compatibility") is appropriate.
- "OLMo Eval" (allenai/OLMo-Eval) is a separate, older repository. Write "olmes (Gu et al., 2025)" rather than "OLMo Eval/olmes".

### Uncited claims in TEST 02

- **"Because interleaving also changes transitions, lag, and recency …"** This is the interleaving–spacing confound. Suggest Kornell, N., & Bjork, R. A. (2008). Learning concepts and categories: Is spacing the "enemy of induction"? *Psychological Science, 19*(6), 585–592. https://doi.org/10.1111/j.1467-9280.2008.02127.x (Crossref-verified; the subtitle is from my knowledge).
- **The pilot claims (52.8% → 79.2%; "exactly zero order effect")** are internal results. Cite the code or appendix.

---

## TEST 03: On Mastery-Gated Curriculum Training

### 5. Bengio et al. (2009)

**Corrected APA 7:**
> Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. In *Proceedings of the 26th Annual International Conference on Machine Learning* (pp. 41–48). ACM. https://doi.org/10.1145/1553374.1553380

**Bibliographic check:** Crossref confirms all details. The report gives only "(Bengio et al., 2009)", with no link or reference entry.

**Sentence (line 215):** "Classic curriculum learning shows that organizing training from easier to harder can improve optimization and generalization (Bengio et al., 2009)"

- **Verdict: ACCURATE.**
- Abstract: "The experiments show that significant improvements in generalization can be achieved. We hypothesize that curriculum learning has both an effect on the speed of convergence of the training process to a minimum and … on the quality of the local minima obtained."
- **Recommended balance:** add Wu, X., Dyer, E., & Neyshabur, B. (2021). When do curricula work? *ICLR 2021*. https://arxiv.org/abs/2012.03107. Its arXiv abstract says curricula gave "only marginal benefits" on standard benchmarks, but helped with limited training time or noisy data.
  - The limited-budget finding is directly relevant to a matched-token comparison.
- The opening premise "models, like students, learn poorly when harder material arrives before current competence is in place" is uncited and contested. Cite Bengio et al. and Wu et al. together, or soften it.

### 6. Kulik, Kulik, & Bangert-Drowns (1990)

**Corrected APA 7:**
> Kulik, C.-L. C., Kulik, J. A., & Bangert-Drowns, R. L. (1990). Effectiveness of mastery learning programs: A meta-analysis. *Review of Educational Research, 60*(2), 265–299. https://doi.org/10.3102/00346543060002265

**Bibliographic check:** Crossref and ERIC (EJ415887) confirm. The report has no reference entry or link.

**Sentence (line 215):** "human mastery learning finds moderate gains when advance is tied to demonstrated competence rather than a fixed clock (Kulik et al., 1990)"

- **Verdict: ACCURATE in direction. The magnitude is UNVERIFIED and caveats are missing.**
- Abstract (OpenAlex): "A meta-analysis of findings from 108 controlled evaluations showed that mastery learning programs have positive effects on the examination performance of students … The effects appear to be stronger on the weaker students in a class, and they also vary as a function of mastery procedures used, experimental designs of studies, and course content. … self-paced mastery programs often reduce the completion rates in college classes."
- The commonly quoted average effect of about 0.5 SD is not in the abstract. I could not access the full text (publisher 403), so "moderate" is plausible but **UNVERIFIED** from the primary source.
- Kulik et al. pool Keller's self-paced PSI and Bloom's group-paced Learning for Mastery. In the group-paced form the class does not advance purely on individual competence, so "rather than a fixed clock" is a simplification.
- **Recommended counterpoint:** Slavin found much smaller effects on standardized measures in longer, more rigorous studies of group-based mastery learning. He also published a reply to Kulik et al. (both Crossref-verified):
  - Slavin, R. E. (1987). Mastery learning reconsidered. *Review of Educational Research, 57*(2), 175–213. https://doi.org/10.3102/00346543057002175
  - Slavin, R. E. (1990). Mastery learning re-reconsidered. *Review of Educational Research, 60*(2), 300–302. https://doi.org/10.3102/00346543060002300
  - I recalled Slavin's specific effect sizes from memory and did not verify them: **UNVERIFIED**. Cite the papers for "effects are smaller under stricter evaluation," not for a number.

### 7. RLAAR (Li, M. et al., 2026)

**Corrected APA 7:**
> Li, M., Chen, P., Zhang, Z., Yang, T., Zhang, X., Li, H., Cao, T., Zeng, M., Wu, Z., Jiang, M., Li, H., Li, L., & Yin, B. (2026). Mitigating lost in multi-turn conversation via curriculum RL with verifiable accuracy and abstention rewards. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 33364–33381). Association for Computational Linguistics. https://doi.org/10.18653/v1/2026.acl-long.1540

**Bibliographic check:**
- ACL Anthology 2026.acl-long.1540 and Crossref confirm the details.
- arXiv v3 (29 Apr 2026) says "ACL2026, camera-ready".
- **The report cites only arXiv and should be updated.**
- Both RLAAR and CAMPUS have first authors surnamed Li. APA 7 then requires initials in-text: "M. Li et al., 2026" vs. "Y. Li et al., 2025".

**Sentence (line 217):** "RLAAR uses a competence-gated curriculum for verifiable multi-turn RL: dialogue difficulty increases only after the model's moving-average reward clears a threshold relative to an easier baseline, which stabilizes training in a sparse-reward setting and substantially improves reliability (LiC score went form 62.6% to 75.1%, and calibrated abstention went from 33.5% to 73.4%)"

**Gating rule: ACCURATE (§3.2.3, §4.1).**
- Difficulty is the number of instruction shards K.
- Stage 1 trains on unsharded K = 1 questions and computes "the moving-average reward r̄0 over a predefined window (5 by default)". It then uses "ρ×r̄0 as the real threshold for the subsequent stages".
- The model "progresses from K shards to K+1 shards only when the condition r̄ ≥ ρ×r̄0 is met". Defaults are ρ = 0.8 and Kmax = 5. After the curriculum, K is randomized (Stage 3).
- The reward is a mixed binary reward: accuracy on solvable questions, abstention on unsolvable ones.

**Stabilization claim: ACCURATE as the authors' claim, but not quantified.**
- §3.2.3 describes "a sparse reward landscape that makes the learning process unstable and inefficient".
- §4.3.2 says of the no-curriculum run: "the training process is not as stable".
- The paper shows no variance or stability metric.

**Numbers: exact, but PARTIALLY ACCURATE in attribution.**
- The abstract says RLAAR "mitigates LiC performance decay (62.6% to 75.1%) and improves calibrated abstention rates (33.5% to 73.4%)". These are the Table 1 averages for **Qwen3-8B**, untrained base vs. full RLAAR.
- They measure the whole method (multi-turn RL + abstention reward + curriculum), not the curriculum.
- The **curriculum ablation** (Table 3) is on **Qwen3-1.7B** only, with single runs:

  | Setting | LiC | Abstain |
  |---|---|---|
  | No curriculum | 63.2 | 57.8 |
  | ρ = 0.8 | 71.9 | 68.4 |
  | ρ = 0.6 | 40.9 | 47.6 |

  ρ = 0.6 is *worse* than no curriculum, which shows the threshold matters.
- "LiC score" is a ratio, Accuracy(sharded)/Accuracy(concatenated), on the Laban et al. (2025) LiC benchmarks.

**Suggested rewrite:** "…RLAAR as a whole raised Qwen3-8B's LiC score from 62.6 to 75.1 and calibrated abstention from 33.5 to 73.4. In the authors' curriculum ablation (Qwen3-1.7B), the competence gate improved LiC from 63.2 to 71.9 over training without a curriculum."

Also fix the typo "form" → "from", and drop the redundant "(arXiv:2510.18731)".

### 8. CAMPUS (Li, Y. et al., 2025)

**Corrected APA 7:**
> Li, Y., Lu, T., Li, Y., Chen, Y., Huang, W.-C., Jiang, W., Wang, H., Zheng, H.-T., & Yu, P. S. (2025). Teaching according to talents! Instruction tuning LLMs with competence-aware curriculum learning. In C. Christodoulopoulos, T. Chakraborty, C. Rose, & V. Peng (Eds.), *Findings of the Association for Computational Linguistics: EMNLP 2025* (pp. 11724–11741). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.findings-emnlp.629

**Bibliographic check:**
- ACL Anthology 2025.findings-emnlp.629 confirms the details. arXiv v2 says "EMNLP 2025 Findings".
- **The venue in the report ("EMNLP Findings 2025") is CORRECT.** The report links only arXiv; prefer the Anthology DOI.

**Sentence (line 217):** "[CAMPUS] dynamically selects instruction sub-curricula by current model competence (e.g. perplexity) and outperforms static curriculum and shuffle-style instruction-tuning baselines by about 7% on average across math, code, and chat benchmarks"

- **Competence mechanism: ACCURATE.**
  - §3.2: the scheduler "selects the training batch with the minimum PPL among the candidates".
  - Two of its four difficulty metrics (loss, and an adversarially trained scoring model) are competence-aware.
- **Benchmarks: ACCURATE.** GSM8K (math), HumanEval (code) and MT-Bench (chat), with MATH/MBPP/MMLU in an appendix.
- **"About 7%" over static-curriculum *and shuffle* baselines: PARTIALLY ACCURATE / overstated.**
  - The 7.0% is scoped in §4.2: "CAMPUS achieves an average gain of 7.0% over other curriculum instruction tuning baselines Tree-Instruct and Conifer". It is a relative gain on LLaMA-7B/13B.
  - The intro rounds this to "outperforms state-of-the-art methods by an average of 7.0%."
  - Table 1 (LLaMA) averages:

    | Method | Average |
    |---|---|
    | CAMPUS | 38.16 |
    | Conifer | 36.57 |
    | Random Shuffle | 36.49 |
    | Tree-Instruct | 34.83 |

  - Against **random shuffle** the gain is +1.67 points (about 4.6% relative).
  - On BLOOMZ (Table 2) the shuffle comparison is 20.34 vs. 18.34 (about 10.9% relative).
  - Also relevant: §4.2 notes that "existing curriculum instruction tuning did not even outperform random shuffle tuning on partial benchmarks."
- **Suggested rewrite:** "…outperforms static-curriculum baselines by about 7% (relative) and random shuffling by about 1.7 points on average on LLaMA (GSM8K, HumanEval, MT-Bench)."
- CAMPUS is *instruction tuning* (fine-tuning), not pretraining. The "Together, these results suggest … helps LLMs" sentence should say "in post-training".

### 9. McCloskey & Cohen (1989)

**Corrected APA 7:**
> McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), *Psychology of learning and motivation* (Vol. 24, pp. 109–165). Academic Press. https://doi.org/10.1016/S0079-7421(08)60536-8

- **Bibliographic check:** Crossref confirms the title, authors, year, pages and DOI. The editor and volume come from the earlier verification (`research/notes/01_citation_verification.md`). The DOI link in the report is correct, but the year is missing in-text.
- **Usage: ACCURATE.** They are the canonical source for catastrophic interference under sequential training.

### 10. French (1999)

**Corrected APA 7:**
> French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2

- **Bibliographic check:** Crossref confirms the details. The DOI link is correct; the year is missing in-text.
- **Usage: ACCURATE.** French frames forgetting as overlap in distributed representations. "Gradient updates … overwrite the parameters that supported earlier tasks" is a fair modern paraphrase.

### 11. Robins (1995)

**Corrected APA 7:**
> Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318

**Bibliographic check:**
- Crossref confirms the details.
- **Replace the Stanford proxy link** (`www-tandfonline-com.stanford.idm.oclc.org/...`) with the public DOI above. Readers outside Stanford cannot open the proxy link.

**Sentence (line 219):** "The standard and well tested fix is replay … This was introduced by [Robins] in 1995 and has been developed extensively in modern continual learning"

- **Verdict: INACCURATE ("introduced").**
- Robins's abstract (OpenAlex): "We replicate some of the experiments described by Ratcliff (1990), including those relating to a simple 'recency' based rehearsal regime. We then develop further rehearsal regimes … 'sweep rehearsal' … We describe a solution to this problem, 'pseudorehearsal' …"
- Rehearsal predates Robins. His contributions were sweep rehearsal and pseudorehearsal, plus a systematic treatment.
- **Fix:** "Rehearsal was studied by Ratcliff (1990) and systematized by Robins (1995), who also introduced pseudorehearsal."
- Add: Ratcliff, R. (1990). Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions. *Psychological Review, 97*(2), 285–308. https://doi.org/10.1037/0033-295X.97.2.285 (Crossref-verified).

### 12. iCaRL (Rebuffi et al., 2017)

**Corrected APA 7:**
> Rebuffi, S.-A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). iCaRL: Incremental classifier and representation learning. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 5533–5542). IEEE. https://doi.org/10.1109/CVPR.2017.587

- **Bibliographic check:** Crossref confirms the details. IEEE pagination is 5533–5542; the CVF open-access version is paginated differently.
- The report gives "iCaRL method of Rebuffi and colleagues in 2017" with no reference.
- **Usage: ACCURATE.** iCaRL is a class-incremental method that keeps a stored exemplar memory (rehearsal) plus distillation. The abstract: "only the training data for a small number of classes has to be present at the same time and new classes can be added progressively."

### 13. Rolnick et al. (2019), second use

"the experience replay work of Rolnick and colleagues in 2019": **ACCURATE.** See #2. CLEAR uses a 50-50 new/replay mix (§3; Fig. 5).

### 14. Zaremba & Sutskever (2014), "Learning to Execute"

**Corrected APA 7:**
> Zaremba, W., & Sutskever, I. (2014). *Learning to execute* (arXiv:1410.4615). arXiv. https://doi.org/10.48550/arXiv.1410.4615

**Bibliographic check:**
- arXiv versions: v1 17 Oct 2014; v3 19 Feb 2015. There is no journal-ref.
- The PDF header reads "Under review as a conference paper at ICLR 2015". I found no archival proceedings version, so cite it as an arXiv preprint. Whether it ever appeared in proceedings is **UNVERIFIED**; I have no evidence that it did.

**Sentences (line 221), checked one by one against the arXiv v3 PDF:**

| Report's claim | Verdict | Evidence |
|---|---|---|
| "trained small recurrent networks to read character level programs and produce the correct outputs" | ACCURATE (they are LSTMs) | Abstract: "LSTMs can learn to map the character-level representations of such programs to their correct outputs" |
| "compared four ways of building training batches" | ACCURATE | §4: baseline, naive, mix, combined |
| Baseline "trained on the hard target distribution from the very start" | ACCURATE | §4: "we generate all the training samples with length = a and nesting = b" |
| Naive "began at the easiest setting and increased difficulty only when validation performance stopped improving" | ACCURATE | §4: "We begin with length = 1 and nesting = 1. Once learning stops making progress on the validation set, we increase length by 1." |
| Mix "drew a blend of easy and hard examples throughout" | ACCURATE | §4: "pick a random length from [1, a] and a random nesting from [1, b] … a balanced mixture of easy and difficult examples" |
| Combined "interleaved the naive ladder with mixed difficulty samples so that the model was never trained on only the current easy band" | ACCURATE | §4: "every training case is obtained either by the naive strategy or by the mix strategy … always exposes the network at least to some difficult examples" |
| "The combined strategy was consistently the strongest" | **PARTIALLY ACCURATE** | Program evaluation (§6.1): "outperforms all other strategies in every configuration". But §4 says it "would generally (but not always) outperform the mix strategy", and on memorization (§6.3) "the combined strategy no longer outperforms the mixed strategy in every experimental setting." Write "the strongest overall, and best on every program-evaluation configuration." |
| "the naive gate like ladder was unreliable and sometimes performed worse than the baseline" | ACCURATE | §4: "sometimes it gives even worse performance than baseline". §6.1: "The Naive curriculum strategy was found to sometime perform worse than baseline." |
| Explanation: capacity spent on easy patterns forces painful restructuring; interleaving difficulties keeps the representation flexible | ACCURATE (it is their *hypothesis*) | §7, "Hidden state allocation hypothesis": "the harder examples would require a restructuring of its memory patterns … might be difficult to implement"; mix samples "prevent the network from utilizing all the memory on the easy examples." Note that the authors offer this as "a plausible explanation" tied to memorization-heavy tasks. |
| "Zaremba and Sutskever mixed in examples that were harder than the current band" | **ACCURATE** | The mix component samples up to the target difficulty (a, b), which is harder than the current naive level |

**Interpretive caution: the inference to replay is PARTIALLY SUPPORTED.**
- Z&S's mechanism is *exposure to harder or full-range examples* that stops the network from committing capacity to easy cases.
- They do not attribute the naive curriculum's failures to *forgetting* earlier levels.
- The report's predicted-best arm ("mastery-gated + replay") adds *easier, already-mastered* skills to the batch. That is the opposite half of Z&S's mix and was not tested by them.
- The report already parks harder-mixing as an ablation. It should state plainly that Z&S's evidence supports "never train only on the current band", most directly through *harder* examples, and that replay-of-mastered is the report's own extension.
- Note that the report's own Preliminary Experiment I already shows naive-gate forgetting, so replay is motivated by that result and by Robins/Rolnick, not mainly by Z&S.

### 15. Skill-it (Chen et al., 2023)

**Corrected APA 7:**
> Chen, M. F., Roberts, N., Bhatia, K., Wang, J., Zhang, C., Sala, F., & Ré, C. (2023). Skill-it! A data-driven skills framework for understanding and training language models. In *Advances in Neural Information Processing Systems 36* (pp. 36000–36040). Curran Associates. https://doi.org/10.52202/075280-1562

**Bibliographic check:**
- Crossref confirms NeurIPS 2023, pp. 36000–36040. arXiv has only v1 (no comment).
- **Update from arXiv to NeurIPS 2023.**
- The report spells it "Skill-it!" in the link and "Skill-It" in "What is less established". The paper styles it "Skill-it!" in the title and "SKILL-IT" for the algorithm. Pick one.

**Sentences (line 223, and line 229):**

1. "treats a model's abilities as distinct 'skills' and shows that the mix and order of training matters because skills transfer to one another": **ACCURATE.** The abstract says prerequisite training lets "more advanced skills to be learned with less data."
2. "learns a skills graph from measured transfer (whether training on skill A lowers loss on skill B)": **ACCURATE.**
   - §4.2, brute force: "If the loss on s_eval,j when trained on both s_train,i and s_eval,j is lower, there exists an edge … with edge weight proportional to the difference in loss".
   - Linear approximation: "training on each s_i for h < H steps and setting A_ij > 0 if the loss on s_j decreases over h steps".
3. "uses it to derive prerequisite-aware sampling weights, which is a soft reweighting of the data rather than a hard gate": **ACCURATE.**
   - Algorithm 1 / Eq. 4 is an online mirror-descent update, p_{t+1}^i = p_t^i · exp(η Σ_j A_ij L_eval,j).
   - §4.3.2: it "prioritizes prerequisite skills and skills that are not yet learned".
   - **Worth adding:** Skill-it *is* competence-adaptive in a soft sense, because it upweights skills whose validation loss is still high. The distinction from the report's design is "soft, loss-driven" vs. "hard, threshold-driven".
4. "nothing is ever withheld until a competence threshold is cleared" (line 223) / "never withholds anything" (line 229):
   - Line 223 is **ACCURATE**.
   - Line 229 is slightly **overstated**. Skill-it's skill-stratified sampling (§4.3.1) does "discard training skills that do not benefit the evaluation skills". That is withholding by graph relevance, not by a competence threshold.
5. **Missed connection:** Skill-it's ordered-skill demonstrations use the **LEGO** synthetic. Its §3, "LEGO skills", defines "the ith skill s_i as the model's ability to know the ith variable of the chain" and shows that training on s1, s2 helps s3.
   - That is essentially the report's Preliminary Experiment I skill structure (depth-k chains).
   - Cite it there, and soften the novelty claim at line 229 ("will be unique").

### 16. "How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining" (Luo et al., 2026)

**Corrected APA 7:**
> Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=T5wkZJqzkz (arXiv:2511.18903)

**Bibliographic check:**
- The title matches exactly.
- The ICLR virtual page (poster 10009351) lists all 8 authors and an oral session (3C) as well as the poster. ML Anthology lists it under ICLR 2026.
- arXiv:2511.18903 (v3, 14 May 2026) carries no venue comment.
- **The report names no authors and links only the ICLR virtual page.** Use the OpenReview forum link.
- The "Fourteenth" numbering is from my knowledge (ICLR 2026 = 14th): **UNVERIFIED**. "ICLR 2026" alone is safe.

**Sentences (line 225), checked one by one:**

| Report's claim | Verdict | Evidence |
|---|---|---|
| Late data barely moves the model because update size scales with LR, and pretraining decays LR to near zero | ACCURATE | Abstract: the problem is "the incompatibility between the ascending data quality order and the decaying learning rate (LR) schedule". Curriculum "substantially outperforms random shuffling when using a constant LR", but "its advantage diminishes under standard LR decay schedules." |
| "curricula that save the best **or hardest** data for last" | **PARTIALLY ACCURATE** | The paper orders by *data quality* (quality-metric scores), not difficulty or skill depth. Applying it to a skill-depth curriculum is a reasonable extrapolation; label it as one. |
| "a 'curriculum loses' result could be an artifact of decay" | ACCURATE (good point) | This follows from the paper's central result |
| "We mitigate this by holding an identical learning-rate schedule across all arms" | **INACCURATE as a mitigation** | An identical decaying schedule is exactly the setting in which the paper finds the curriculum advantage suppressed. Matching schedules keeps the comparison fair but does not remove the confound. |
| "ideally by running an ablation … (for example, a constant-learning-rate or WSD schedule)" | PARTIALLY ACCURATE | Constant LR is the paper's diagnostic setting. **WSD still ends in a decay phase**, so it decouples nothing unless hard skills are unlocked before the decay. The paper's remedies are (1) "a more moderate LR decay schedule, where the final LR is only moderately smaller than the peak LR" and (2) model averaging, "computing a weighted average of the final few checkpoints" (CMA/CDMA). They report "1.64% over random shuffling" at 1.5B parameters / 30B tokens. |

**Internal tension:** TEST 02 already uses a *constant* LR. TEST 03 says only "identical learning-rate schedule" (Preliminary Experiment I, line 270) and never says what that schedule was. State the schedule.

### 17. Anthropic, "On the Biology of a Large Language Model" (Lindsey et al., 2025)

**Corrected APA 7** (APA 7 lists the first 19 authors, then an ellipsis and the final author):
> Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce, A., Turner, N. L., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., . . . Batson, J. (2025, March 27). On the biology of a large language model. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-addition

**Bibliographic check:**
- The byline has 27 authors (Jack Lindsey is lead; Joshua Batson is the correspondence author). Dated March 27, 2025.
- **The report's anchor `#dives-tracing` points to "Introductory Example: Multi-step Reasoning"** (Dallas → Texas → Austin), **not addition**. The addition section is `#dives-addition`.

**Sentence (line 237):** "For addition, CE loss incentivizes models in training to quickly learn a [cheap heuristic] rather than actually learning a generalizable algorithm … This heuristic performs well on in-distribution examples but collapses on data outside the training distribution"

- **Verdict: INACCURATE as support.** The link is wrong, and the correct section does not make the claim.
- The addition section describes Claude 3.5 Haiku using parallel pathways: a low-precision magnitude path and a ones-digit "lookup table" path, "recombining these heuristics to get the correct answer". So "heuristic" is fair language.
- But the section emphasizes **generalization**: one addition feature is used "in a remarkably diverse set of contexts that require addition" (astronomical tables, citation years, and others).
- It says nothing about cross-entropy loss incentivizing heuristics or about collapse out of distribution.
- **Better citations:**
  - For "bag of heuristics": Nikankin, Y., Reusch, A., Mueller, A., & Belinkov, Y. (2025). Arithmetic without algorithms: Language models solve math with a bag of heuristics. In *ICLR 2025*. https://openreview.net/forum?id=O9YTt26r2P (arXiv:2410.21272). The venue comes from ML Anthology.
  - For "fails on longer numbers": Lee et al. (2024), §1: "We find length generalization beyond trained digit lengths difficult." Full reference at #19.
- Present "CE loss incentivizes" as the authors' hypothesis. None of these papers shows it causally.

### 18. Clock and Pizza (Zhong et al., 2023)

**Corrected APA 7:**
> Zhong, Z., Liu, Z., Tegmark, M., & Andreas, J. (2023). The clock and the pizza: Two stories in mechanistic explanation of neural networks. In *Advances in Neural Information Processing Systems 36* (pp. 27223–27250). Curran Associates. https://doi.org/10.52202/075280-1186

- **Bibliographic check:** Crossref and the arXiv comment ("Accepted by NeurIPS 2023") confirm. The report links the arXiv PDF; prefer the abstract page or DOI.
- **Usage ("rather than actually learning a generalizable algorithm ([clock/Fourier algorithm, pizza algorithm])"): PARTIALLY ACCURATE.**
  - Clock and Pizza are genuine algorithmic (non-memorizing) solutions, but for **modular addition** (a + b mod p) in small networks. They are not multi-digit addition and are unrelated to length generalization.
  - The paper's main point is that "qualitatively different algorithms" arise from small hyperparameter changes.
  - The **Clock/Fourier** algorithm was first reverse-engineered by Nanda et al. Cite it as the source for "clock/Fourier": Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. In *ICLR 2023*. https://openreview.net/forum?id=9XFSbDPmdW (arXiv:2301.05217).
- **Suggested wording:** "…an algorithm that generalizes, analogous to the Fourier 'clock' and 'pizza' circuits found for modular addition (Nanda et al., 2023; Zhong et al., 2023)."

### 19. Reversed-digit output (Zhang-Li et al., 2024)

**Corrected APA 7:**
> Zhang-Li, D., Lin, N., Yu, J., Zhang, Z., Yao, Z., Zhang, X., Hou, L., Zhang, J., & Li, J. (2024). *Reverse that number! Decoding order matters in arithmetic learning* (arXiv:2403.05845). arXiv. https://doi.org/10.48550/arXiv.2403.05845

- **Bibliographic check:** arXiv has v1 only (9 Mar 2024), with no comment or journal-ref. I did not find a published version. My search budget ran out before a full venue search, so the venue status is **UNVERIFIED**.

**Sentence (line 237):** "simply compelling the model to [output the digits of the answer in reverse order] incentivizes learning how to carry and improves accuracy and generalization."

- **"Improves accuracy": ACCURATE (narrowly).** The abstract reports "an overall improvement of 11.1% in accuracy while requiring only a third of the tokens."
  - The setting is **fine-tuning Llama2-13B** (§4.3) on 5–12-digit problems, with in-distribution testing (§4.2).
- **"Incentivizes learning how to carry": ACCURATE (narrowly).**
  - §1 says Little-Endian output "simplifies carry operations".
  - §5.2 says attention "suggests the fine-tuned LLM has learned to re-compute the carry".
- **"Simply": INACCURATE.** The method (LEFT) combines reversal *with* step-by-step decomposition. For multiplication, "when the use of step-by-step is removed, it becomes impossible to learn multiplication" (§5.1).
- **"Generalization": INACCURATE / unsupported.** There are no OOD or length-generalization experiments. Accuracy still drops with more digits *within* the training range (multiplication goes from 93.6 to 76.8, Table 3).
- **Better citations:**
  - For reversal in small from-scratch transformers (which is the report's setting):
    > Lee, N., Sreenivasan, K., Lee, J. D., Lee, K., & Papailiopoulos, D. (2024). Teaching arithmetic to small transformers. In *ICLR 2024*. https://openreview.net/forum?id=dsUB4bst9S (arXiv:2307.03381)

    Lee et al. (§1): "By training on samples with reversed results … we enable the model to learn a simpler function, significantly improving sample complexity." They also report that "length generalization beyond trained digit lengths [is] difficult."
  - For **"place-value tag on each token"** (currently uncited) and for actual length generalization to 30-digit tests:
    > McLeish, S., Bansal, A., Stein, A., Jain, N., Kirchenbauer, J., Bartoldson, B. R., Kailkhura, B., Bhatele, A., Geiping, J., Schwarzschild, A., & Goldstein, T. (2024). Transformers can do arithmetic with the right embeddings. In *Advances in Neural Information Processing Systems 37*. https://doi.org/10.52202/079017-3430

    The abstract reports training on 20-digit numbers and reaching up to "99% accuracy on 100 digit addition problems."

### 20. LEGO reference-following tasks (Zhang et al., 2022). Currently uncited

**APA 7:**
> Zhang, Y., Backurs, A., Bubeck, S., Eldan, R., Gunasekar, S., & Wagner, T. (2022). *Unveiling transformers with LEGO: A synthetic reasoning task* (arXiv:2206.04301). arXiv. https://doi.org/10.48550/arXiv.2206.04301

- **Bibliographic check:**
  - arXiv v1 9 Jun 2022, v3 17 Feb 2023, no journal-ref.
  - A search result reported the OpenReview record as an ICLR 2023 *submission* (`openreview.net/forum?id=1jDN-RfQfrb`); I could not open OpenReview myself (CAPTCHA). I found no acceptance record, so cite it as arXiv 2022.
- **Yes, "LEGO" refers to this paper.** LEGO = "Learning Equality and Group Operations", a task designed to capture "the problem of following a chain of reasoning".
  - In Skill-it's description, clauses take the form "a = g x", with g a negation or assertion, shuffled, and the query asks for one variable (Skill-it §3).
  - The report's task ("c:b+2;a:5;d:c+7;b:a+3;?d=") replaces sign operations with integer offsets. "Resembling" is therefore **ACCURATE**.
- **Required:** cite Zhang et al. (2022) at line 268, and cite Chen et al. (2023, Skill-it), which already used LEGO chain depth as an ordered skill set.

### 21. Anthropic, "Circuit Tracing" (methods page; Ameisen et al., 2025)

**Corrected APA 7:**
> Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., . . . Batson, J. (2025, March 27). Circuit tracing: Revealing computational graphs in language models. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/methods.html

- **Bibliographic check:** 27 authors; dated March 27, 2025. I could not retrieve the page's own BibTeX (it was truncated). The author order above is from the byline.
- **Usage (line 306): ACCURATE.** This is the right source for "attribution graphs".
- **Caveat worth adding:** building attribution graphs requires training a cross-layer transcoder replacement model for *each* model compared (methods page). That is a non-trivial cost for comparing two small models.

### Uncited or under-cited human-literature and ML claims in TEST 03

| Claim (line) | Status | Suggested citation |
|---|---|---|
| "models, like students, learn poorly when harder material arrives before current competence" (215) | Uncited and contested | Bengio et al. (2009) for; Wu et al. (2021) for the mixed evidence |
| "human mastery learning finds moderate gains" (215) | Cited, but no reference entry | Kulik et al. (1990); add Slavin (1987, 1990) as a counterpoint |
| "The standard and well tested fix is replay" (219) | OK | Ratcliff (1990); Robins (1995); Rolnick et al. (2019); Rebuffi et al. (2017) |
| "CE loss incentivizes … cheap heuristic … collapses OOD" (237) | Mis-cited | Nikankin et al. (2025); Lee et al. (2024) |
| "place-value tag on each token" (237) | Uncited | McLeish et al. (2024) |
| LEGO tasks (268) | Uncited | Zhang et al. (2022); Chen et al. (2023) |
| "A strict unlock-on-probe gate … will be unique" (229) | Novelty claim | Soften. Zaremba & Sutskever (2014) compared a validation-triggered ladder with mixes, and Skill-it studied LEGO chain-depth ordering |
| "Low training/dev loss/accuracy … can be achieved through memorization" (233) | Uncited, but standard | Optional; Lee et al. (2024) discuss memorization vs. generalization in arithmetic |

---

## Internal inconsistencies

1. **Prelim II retention numbers (line 290 vs. table, lines 280–286).**
   - The text says no-replay arms degrade to "(0.59, 0.67)".
   - The table shows Mastery-gated, no replay = **0.67** and Time-gated no replay = **0.55**.
   - 0.59 appears nowhere. Fix the text to "(0.55, 0.67)", or fix the table if 0.59 is the true value.
2. **The one-pager mastery panel is mislabeled. Confirmed.**
   - `P4_onepager_figure.png` labels panel 03 "Prerequisite-chain task", with y-axis "Multi-step OOD accuracy" and bars Shuffle 0.67, "Gate, no replay" 0.85, "Gate + replay" 0.97.
   - Those are exactly the **Preliminary Experiment II (repeated addition)** "Multi-add OOD" values: Shuffle 0.67; Mastery-gated, no replay 0.85; Mastery-gated with replay 0.97.
   - The actual prerequisite-chain task (Preliminary Experiment I, `paper/figures/final_report_fig2_mastery_prelim1_table.png`) has OOD = Shuffle **0.37**, Mastery no replay **0.33**, Mastery + replay **0.75**, Fixed clock 0.12.
   - On that task the naive gate is *below* shuffle, so the figure's monotone story does not hold there.
   - Related problems:
     - `P4_onepager.html` (line 30) says "On a task with genuine prerequisite structure, … 0.97 vs. 0.67 for shuffle — and replay prevented the catastrophic forgetting that collapsed a naive gate." This merges Prelim I's narrative (naive-gate collapse) with Prelim II's numbers. In Prelim II the naive gate did not collapse on the target skill (0.85); it lost *addition retention* (0.67).
     - `P4_onepager_figure_compact.png` labels the middle bar just "Gate", which hides that it is the no-replay arm.
     - The one-pager omits Prelim II's Time-gated-with-replay arm (0.91), which is close to 0.97.
   - **Fix:** relabel the panel "Repeated-addition task (Prelim II)" and the axis "Multi-add OOD accuracy". Or replace the panel with the Prelim I values (0.37 / 0.33 / 0.75).
3. **The arm set differs between pilots and the design.**
   - The Procedure defines four arms: Shuffle, Fixed, Mastery, Mastery+replay.
   - Prelim II has **five** (it adds "Time-gated with replay").
   - Naming varies: "Fixed curriculum" (Procedure), "Fixed clock" (Prelim I table), "Time-gated" (Prelim II table), and "fixed"/"fixed_replay" (Fig. 3 legend). Harmonize the names.
4. **The Prelim II table header** has five cells ("| Arm | Multi-add … | Addition … | Efficiency … |  |") but rows have four. Delete the empty trailing column.
5. **Prelim II text: "All models with replay (including the shuffle baseline)."** Shuffle has no replay mechanism; it simply keeps sampling addition. Write "all arms that keep sampling addition (the replay arms and shuffle)".
6. **Prelim I text: "exceeding ninety-five percent on depths one through four".** Depth 4 is 0.95, which is not "exceeding". Write "at or above 95%".
   - "Depth five low only because it was not unlocked within the budget" needs log evidence. A never-trained depth scoring 0.18 is plausible but should be shown.
   - The per-skill means match the summary column for all four arms: 0.81, 0.47, 0.32, 0.14.
7. **Prelim II curve claim** ("both mastery models move on to multi-addition … before the halfway point"): consistent with Fig. 3. The mastery curves leave zero at about 2×10⁷ of about 5×10⁷ tokens, while the fixed arms leave zero at about 3×10⁷. Grammar fixes: "the both" → "both"; "less tokens" → "fewer tokens".
8. **The LR schedule is unstated in TEST 03.** Line 270 says "identical learning-rate schedule". Given the Luo et al. confound, state whether it decayed. TEST 02 uses a constant LR.
9. **TEST 02 pilot naming:** "toy shared-network pilot" (line 181) vs. "toy MLP pilot" (line 205); and "200M result" (line 205) vs. "approximately 195M" elsewhere.
   - The one-pager calls the same pilot a "controlled pilot" and drops "toy MLP". Readers may think it was an LM result.
   - TEST 02's compute arithmetic checks out:
     - 1,280 × 32,768 ≈ 41.9M curriculum tokens; plus 160 × 32,768 → 47.2M tokens.
     - 24 × 5.7e16 ≈ 1.37e18 FLOPs.
     - Parameter count ≈ 24·12·768² + 32k·768 ≈ 195M.
10. **Typos in cited sentences:** "went form" (line 217); "the both mastery models" (line 290). The Limitations line "lmao good luck with that" (line 302) should be removed before journal submission.

---

## Prioritized list of required corrections

**Must fix (factual or citation errors):**
1. **Prelim II numbers:** change "(0.59, 0.67)" to match the table (0.55, 0.67), or correct the table.
2. **One-pager and figure:** the mastery panel label "Prerequisite-chain task" is wrong for 0.67/0.85/0.97 (these are Prelim II). Fix `P4_onepager_figure.png`, `P4_onepager_figure_compact.png` and the sentence in `P4_onepager.html`.
3. **Anthropic biology link:** change `#dives-tracing` to `#dives-addition`. Re-cite the "cheap heuristic … collapses OOD" claim to Nikankin et al. (2025) and Lee et al. (2024), and soften "CE loss incentivizes".
4. **Robins (1995):** replace "introduced by Robins" with "studied by Ratcliff (1990) and systematized by Robins (1995), who introduced pseudorehearsal". Replace the Stanford proxy URL with https://doi.org/10.1080/09540099550039318.
5. **Reversed-digit sentence:** drop "simply" and "generalization", or cite Lee et al. (2024) for reversal and McLeish et al. (2024) for place-value tags and length generalization. Say the Zhang-Li et al. result is for a fine-tuned Llama2-13B with step-by-step reasoning.
6. **CAMPUS "about 7%":** rescope to "≈7% over static curricula; ≈1.7 points over shuffle (LLaMA)", and note it is instruction tuning.
7. **RLAAR numbers:** attribute 62.6→75.1 / 33.5→73.4 to the full method (Qwen3-8B). Add the curriculum-only ablation (Qwen3-1.7B: 63.2→71.9 LiC). Cite ACL 2026. Fix "form".
8. **LR-decay paragraph:** add the authors (Luo et al., 2026). Change "best or hardest" to "highest-quality (and, by extension, hardest)". Replace "we mitigate this by holding an identical LR schedule" with constant-LR / moderate-decay / checkpoint-averaging controls, and drop or qualify WSD.

**Should fix (missing or outdated references):**

9. Add a **reference list**. Add full references for Bengio et al. (2009), Kulik et al. (1990), McCloskey & Cohen (1989), French (1999), iCaRL (2017) and the Anthropic pages, and include years in-text for McCloskey & Cohen and French.
10. Cite **LEGO** (Zhang et al., 2022) and **Skill-it's LEGO skills** in Preliminary Experiment I, and soften the "will be unique" novelty claim.
11. Update venues: Rolnick et al. → NeurIPS 2019; Skill-it → NeurIPS 2023; Clock/Pizza → NeurIPS 2023; RLAAR → ACL 2026. Use Anthology DOIs for CAMPUS and RLAAR. Disambiguate "M. Li et al." vs. "Y. Li et al.".
12. Cite Nanda et al. (2023) for the clock/Fourier algorithm and note that it concerns modular addition.
13. Cite olmes as Gu et al. (2025, *Findings of NAACL*), and cite the OLMo 2 report for OLMo-core, as its README asks.

**Recommended (balance and precision):**

14. Brunmair & Richter: say "by type of material" (words favored blocking, g = −0.39). Note that the effect concerns inductive category learning and is weakest for dissimilar categories, which describes TEST 02's four subjects. Consider adding Rohrer & Taylor (2007).
15. Zaremba & Sutskever: change "consistently the strongest" to "strongest overall; best in every program-evaluation configuration". State that their evidence supports mixing in *harder* examples, and that replay of mastered skills is the report's own extension.
16. Kulik et al.: add the heterogeneity caveat and Slavin (1987, 1990). Bengio et al.: add Wu et al. (2021, ICLR) on when curricula help.
17. Skill-it: note that it is soft-competence-adaptive (loss-driven upweighting) and that stratified sampling does drop irrelevant skills.
18. Harmonize arm names across pilots, remove the empty table column, fix "exceeding 95%", state the LR schedule, and remove informal language.

---

## Sources

- Brunmair & Richter: https://doi.org/10.1037/bul0000209 (Crossref; abstract via Semantic Scholar API)
- Rolnick et al.: https://arxiv.org/abs/1811.11682 ; https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html ; https://mlanthology.org/neurips/2019/rolnick2019neurips-experience/
- Bengio et al.: https://doi.org/10.1145/1553374.1553380 (Crossref; abstract via OpenAlex)
- Kulik et al.: https://doi.org/10.3102/00346543060002265 ; https://eric.ed.gov/?id=EJ415887 (abstract via OpenAlex)
- Slavin: https://doi.org/10.3102/00346543057002175 ; https://doi.org/10.3102/00346543060002300
- RLAAR: https://arxiv.org/abs/2510.18731 ; https://arxiv.org/html/2510.18731v3 ; https://aclanthology.org/2026.acl-long.1540/
- CAMPUS: https://arxiv.org/abs/2509.13790 ; https://aclanthology.org/2025.findings-emnlp.629/ (full-text PDF read)
- McCloskey & Cohen: https://doi.org/10.1016/S0079-7421(08)60536-8 ; French: https://doi.org/10.1016/S1364-6613(99)01294-2
- Robins: https://doi.org/10.1080/09540099550039318 (abstract via OpenAlex) ; Ratcliff: https://doi.org/10.1037/0033-295X.97.2.285
- iCaRL: https://doi.org/10.1109/CVPR.2017.587
- Zaremba & Sutskever: https://arxiv.org/abs/1410.4615 (v3 PDF read)
- Skill-it: https://arxiv.org/abs/2307.14430 ; https://doi.org/10.52202/075280-1562 (NeurIPS PDF read)
- Luo et al.: https://iclr.cc/virtual/2026/poster/10009351 ; https://arxiv.org/abs/2511.18903 ; https://mlanthology.org/iclr/2026/luo2026iclr-learning/
- Anthropic: https://transformer-circuits.pub/2025/attribution-graphs/biology.html ; https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Clock/Pizza: https://arxiv.org/abs/2306.17844 ; https://doi.org/10.52202/075280-1186 ; Nanda et al.: https://arxiv.org/abs/2301.05217 ; https://mlanthology.org/iclr/2023/nanda2023iclr-progress/
- Reversed digits: https://arxiv.org/abs/2403.05845 ; https://arxiv.org/html/2403.05845v1
- Lee et al.: https://arxiv.org/abs/2307.03381 (PDF read) ; https://mlanthology.org/iclr/2024/lee2024iclr-teaching/
- McLeish et al.: https://arxiv.org/abs/2405.17399 ; https://mlanthology.org/neurips/2024/mcleish2024neurips-transformers/
- Nikankin et al.: https://arxiv.org/abs/2410.21272 ; https://mlanthology.org/iclr/2025/nikankin2025iclr-arithmetic/
- Wu et al.: https://arxiv.org/abs/2012.03107
- LEGO: https://arxiv.org/abs/2206.04301
- OLMo-core README: https://github.com/allenai/OLMo-core ; olmes README: https://github.com/allenai/olmes ; OLMES paper: https://arxiv.org/abs/2406.08446
- Rohrer & Taylor: https://doi.org/10.1007/s11251-007-9015-8 ; Kornell & Bjork: https://doi.org/10.1111/j.1467-9280.2008.02127.x
