# Background research for the P4 papers

*Compiled 2026-10-02 (Part I) and 2026-10-04 (Parts II–III) for the P4 "Validating learning science for machine training" work:*
- *Part I: the spacing whitepaper, [P4_whitepaper_writeup.md](../paper/source/P4_whitepaper_writeup.md).*
- *Parts II–III: the blocked-vs-interleaved and mastery-gating sections of [P4_final_report.md](../paper/source/P4_final_report.md).*

*The synthetic-student work is out of scope and is not covered.*

## Overall TL;DR

- **Spacing (Part I).** The experiment is real, but the paper misdescribes key parts of the method, misuses 3 citations and lacks related work. Its expanding-vs-uniform null is what the human literature predicts. Next steps: per-fact review schedules, a never-trained control and an equivalence test, before scaling to ≤ a few B.
- **Interleaving (Part II).** The report describes a 195M study that was never run. What *was* run (one 162M run on digit skills) shows interleaving beating blocking, which is expected in networks (catastrophic forgetting) rather than novel. A publishable version needs counterbalanced orders, seeds, recency and spacing controls, and a block-length or similarity sweep.
- **Mastery gating (Part III).** There are two preliminary experiments with no code anywhere and no seeds. Reviewers will require real experiments, and a cheap controlled design is in [PRD_mastery_gated_curriculum.md](../plans/PRD_mastery_gated_curriculum.md). Gate + replay already exists as a training tool (Kohli et al., 2026), so the contribution must be the budget-matched comparison of contingent vs. yoked/tuned progression.
- **Across all three.**
  - Each test currently shows the same known mechanism: sequential training without re-exposure erases earlier learning.
  - A combined paper should state that once and focus on what is new: schedule shape (I), block structure and similarity (II), and contingency (III).
  - Every number must trace to committed code. At present, the Test 02 pilot, both Test 03 experiments and the Pythia spacing pilot do not.
- **Scale.** Most directly comparable evidence is at ≤3B. Order and re-exposure effects can emerge between 160M and 410M, so every study needs a small size sweep before its claims are generalised.

## How this document is organised

| Part | Topic | Paper(s) checked | Key companion files |
|---|---|---|---|
| I | Spaced review (expanding vs. uniform) | Whitepaper | research/notes/00–06, [methods_verification.md](methods_verification.md) Part I |
| II | Blocked vs. interleaved training | Final Report, Test 02 | research/notes/07, 09, 10, [methods_verification.md](methods_verification.md) Part II |
| III | Mastery-gated curricula | Final Report, Test 03 | research/notes/08, 09, 10, [PRD_mastery_gated_curriculum.md](../plans/PRD_mastery_gated_curriculum.md), [methods_verification.md](methods_verification.md) Part III |

Each part has its own TL;DR, top-10 papers to add, and reference list (APA 7 with URLs). Detailed notes:

| File | Contents |
|---|---|
| [research/notes/00_code_audit.md](notes/00_code_audit.md) | Spacing: what the experiment actually did, reconstructed from the code, the pinned dataset and the tokenizer |
| [research/notes/01_citation_verification.md](notes/01_citation_verification.md) | Spacing: line-by-line check of the whitepaper's 9 references |
| [research/notes/02_human_spacing.md](notes/02_human_spacing.md) | Spacing: human learning-science literature |
| [research/notes/03_ml_spacing_replay.md](notes/03_ml_spacing_replay.md) | Spacing: spaced repetition, replay and replay scheduling in neural networks and LMs |
| [research/notes/04_llm_knowledge_forgetting.md](notes/04_llm_knowledge_forgetting.md) | Spacing: how LMs acquire and forget facts; LoRA; knowledge-evaluation methodology |
| [research/notes/05_methods_stats_review.md](notes/05_methods_stats_review.md) | Spacing: statistics, mock peer review, venues |
| [research/notes/06_small_model_scale.md](notes/06_small_model_scale.md) | All: what is known at ≤ a few billion parameters |
| [research/notes/07_interleaving.md](notes/07_interleaving.md) | Interleaving: human and ML literature, design confounds |
| [research/notes/08_mastery_curriculum.md](notes/08_mastery_curriculum.md) | Mastery gating: human mastery learning, ML curricula, algorithmic learning, replay |
| [research/notes/09_final_report_citation_verification.md](notes/09_final_report_citation_verification.md) | Interleaving and mastery: check of every citation in Tests 02–03 |
| [research/notes/10_final_report_methods_review.md](notes/10_final_report_methods_review.md) | Interleaving and mastery: statistics, mock reviews, venues |
| [research/notes/11_compute_pipeline_survey.md](notes/11_compute_pipeline_survey.md) | All: how the edu-llm org ran compute (AWS platform, now retired; ORCD; FarmShare; RunPod; EC2), and what a new experiment must provide to be launchable |

Verification conventions: a source is listed only if its title, authors and venue were checked against a primary record (publisher, DOI/Crossref, arXiv, OpenReview, ACL Anthology, PMLR, NeurIPS proceedings, PubMed). Claims about a paper's *content* were checked in the full text wherever the notes say "V-content". Items we could not confirm are marked UNVERIFIED and should not be cited until they are.


---
---

# Part I: Spaced review (whitepaper)

## Part I TL;DR

- **Existing citations.** All 9 are real, but 3 are used wrongly:
  - the paper says expanding schedules "greatly improve memory", but Cepeda et al. found no reliable difference;
  - the mechanisms sentence uses theory names Cepeda et al. don't use;
  - "reproduces the human finding" overclaims.

  Four references need their published versions instead of arXiv.
- **Background is too thin for a journal.** There is no related-work section. The ten papers to add are listed in [§0](#0-the-papers-we-most-need-to-add).
- **The human research predicts our null.** When the first and last review are matched and only the shape of the schedule changes, expanding ≈ uniform at a long delay (Karpicke & Roediger 2007, Exp. 3; Latimier et al. 2021). Expanding still does better *during* training (Kang et al. 2014), which is exactly the pattern we found. The defensible claim is "consistent with the human literature", not "reproduces" it.
- **Machine-learning prior work.** That replay helps is well established, so our "review helps" result is a replication. The niche nobody has filled is fixed expanding vs. uniform review of facts, at a matched budget, with a delayed test. FOREVER (ACL 2026) reports the opposite of our result and must be addressed.
- **The design has holes reviewers will find:**
  - Reviews aren't per fact. Each is one random batch from a document style, and about half the evaluated facts are never reviewed.
  - The loss metric partly measures format and domain adaptation; there is no never-trained control.
  - There is no formal equivalence test, and the forgetting-rate claim depends on the measurement scale.
- **Scale.** Most comparable work uses models of 3B parameters or fewer, which suits the team. Some popular claims come only from 7B+ models or vision networks. Effects of review order appear between 160M and 410M, so the null must be re-tested at 0.4–3B.
- **Next.** Fix the text, reanalyse the existing logs, then run cheap 300M reruns with per-fact schedules and controls before scaling up (§10).

---

## 0. The papers we most need to add

None of these is cited in the current draft. Full citations are in §11.

### The 5 most important

| # | Paper | Why it matters for us | Link |
|---|---|---|---|
| 1 | **Feng et al. (2026). FOREVER.** ACL 2026 | The only peer-reviewed LM paper that compares expanding and uniform replay at the same budget, and it finds **expanding better** (Qwen3-0.6B, LoRA). It contradicts our null, so we must address it head-on. Their measurement sits closer to "during training", where our expanding arm also looked better. | https://arxiv.org/abs/2601.03938 |
| 2 | **Tirumala et al. (2022). Memorization without overfitting.** NeurIPS 2022 | The closest earlier LM result to ours. In a 125M model, spaced re-injection had "minimal effect" on long-run forgetting regardless of spacing length; massed repetition helped. We should present our study as a sharper, budget-matched test of schedule *shape*. | https://proceedings.neurips.cc/paper_files/paper/2022/hash/fa0509f4dab6807e2cb465715bf2d249-Abstract-Conference.html |
| 3 | **Latimier, Peyre & Ramus (2021).** Educational Psychology Review | The human meta-analysis on exactly our contrast: expanding vs. uniform retrieval practice gives *g* = 0.034, which is not significant, and retention interval doesn't moderate it. It replaces the misused Landauer & Bjork / Cepeda claim. | https://doi.org/10.1007/s10648-020-09572-8 |
| 4 | **Kang, Lindsey, Mozer & Pashler (2014).** Psychonomic Bulletin & Review | Expanding gave higher recall *during* training (.49 vs. .41) but equal recall at 8 weeks. This is the same dissociation as our §3.5 "trajectory" result, which turns our "artifact" into a known human pattern. | https://doi.org/10.3758/s13423-014-0636-z |
| 5 | **Chang et al. (2024). How do LLMs acquire factual knowledge during pretraining?** NeurIPS 2024 | The closest design to ours: fictional facts injected into OLMo-1B/7B, with forgetting following a power law. They test only one fixed interval, which is the gap our work fills. | https://arxiv.org/abs/2406.11813 |

### 5 more that are highly relevant

| # | Paper | Why it matters for us | Link |
|---|---|---|---|
| 6 | **Atreya et al. (2026). When to Review (SRT).** arXiv preprint | Concurrent work on per-example spaced repetition (SM-2) for continual pretraining at 1.1B and 3B. It beats uniform replay, which suggests that *adaptive, per-item* scheduling, not schedule shape, is what helps. | https://arxiv.org/abs/2608.17530 |
| 7 | **Karpicke & Bauernschmidt (2011).** JEP: Learning, Memory, and Cognition | In humans, *absolute* spacing (total gap) matters a lot and *relative* spacing (schedule shape) does not. This is exactly the distinction our design manipulates. | https://doi.org/10.1037/a0023436 |
| 8 | **Yang, Jones, Mozer & Ren (2024). Reawakening knowledge.** NeurIPS 2024 | Effects of re-exposure order ("anticipatory recovery") emerge between 160M and 410M parameters and grow up to 2.8B. It is the main reason our 300M null must be re-tested at larger sizes. | https://openreview.net/forum?id=YSs1z5udBY |
| 9 | **Biderman et al. (2024). LoRA learns less and forgets less.** TMLR | LoRA reduces both learning and forgetting relative to full fine-tuning, so it could compress schedule differences. Reviewers will ask about it; a full-fine-tuning check answers them. | https://openreview.net/forum?id=aloEru2qCG |
| 10 | **Loftus (1985). Evaluating forgetting curves.** JEP: Learning, Memory, and Cognition | Whether two forgetting curves "decay at the same rate" depends on the measurement scale. It is needed to state our "review changes the starting point, not the forgetting rate" claim correctly. | https://doi.org/10.1037/0278-7393.11.2.397 |

Close runners-up:
- Kirchenbauer et al. (2026): FictionalQA's own finding of "leaky" transfer across facts. It is already cited, but this finding isn't used.
- Luo et al. (2026): justifies the constant learning rate.
- O'Neill (2026): a drift control using never-trained facts.
- Karpicke & Roediger (2007), Exp. 3: already cited, but the part that matches our design is not used.

---

## 1. Summary

**Citations in the current draft.** All nine references are real, and most bibliographic details are correct. Three uses misrepresent their sources:

1. The introduction cites Landauer & Bjork (1978) and Cepeda et al. (2006) for expanding schedules being "shown to greatly improve memory in humans". Cepeda et al. found no reliable difference between expanding and fixed spacing (62.0% vs. 58.6%, *t*(42) = 0.5, *p* = .61). They also list Landauer & Bjork among claims made "with little apparent empirical backing". Landauer & Bjork themselves reported a modest advantage (about 10 percentage points) on a test given 30 minutes later.
2. The mechanisms sentence attributes "retrieval difficulty" to Cepeda et al. Their candidate theories are consolidation, study-phase retrieval and encoding variability.
3. The conclusion says our null "reproduces" the human finding of Karpicke & Roediger (2007). Their key result was a crossover: expanding was better at 10 minutes, equal spacing was better at 2 days, and the effect was driven by the delay before the first retrieval. When the first retrieval was matched (their Exp. 3), the schedule of later retrievals made no difference. That condition is the closer analogue to our design, because we also match the first and last review.

Four references should be updated to their archival versions (LoRA → ICLR 2022, OLMo → ACL 2024, DataDecide → ICML 2025, FictionalQA → ICLR 2026), and several author names need fixing. Details are in §3.

**Coverage.** The background is not yet adequate for a journal. The paper cites no machine-learning work on replay, rehearsal or spaced repetition, and it misses five papers a reviewer will expect to see discussed:

- **Tirumala et al. (2022, NeurIPS)**, the closest prior LM result to our null. In a 125M LM, spaced re-injection of a batch had "minimal effect" on long-run forgetting regardless of the spacing length.
- **Chang et al. (2024, NeurIPS)**, the closest methodological precedent: fictional facts injected into OLMo-1B/7B, with forgetting following a power law.
- **FOREVER (Feng et al., 2026, ACL)**, which reports that expanding beats uniform replay at a matched budget in a 0.6B LoRA setting. This directly contradicts our result.
- **"When to Review" / SRT (Atreya et al., 2026)** and **LFR (Prakriya et al., 2025, CoNLL)**, two spaced-repetition schedulers for LM training.
- On the human side, the expanding-vs-equal meta-analysis (**Latimier et al., 2021**: *g* = 0.034, not significant) and **Kang et al. (2014)**. Kang et al. found that expanding was better during training but tied at an 8-week test, which is the same dissociation we report in §3.5.

**Design and measurement.** The literature review and an audit of the code raise problems that matter more than the citation fixes (§8–9):

- **The schedule manipulation does not operate on individual facts.** Each review event is a single batch of 16 statements from one FictionalQA document style. About half of the 80 evaluated old facts are never reviewed, and per-style recency differs by 45–57 updates between the two arms.
- **The primary metric mixes generic adaptation with fact memory.** Old-fact loss *falls* while the model trains only on new facts. The FictionalQA authors themselves report this kind of "leaky" transfer.
- **Two of the paper's claims are not yet supported by its statistics.** "Expanding and uniform are indistinguishable" needs a formal equivalence test. "Review changes the starting point, not the forgetting rate" depends on the measurement scale.

Most of this can be fixed by reanalysing logs that already exist, plus a few cheap reruns at 300M.

**Scale.** Most directly comparable evidence comes from models of 3B parameters or fewer, which suits the team's target. Several often-cited claims rest on 7B+ models or on vision networks and should be stated as caveats: "bigger models forget less", "LoRA learns less and forgets less", and most knowledge-injection results. The relative benefit of review is expected to shrink but stay positive as models grow. Effects of re-exposure order emerge between 160M and 410M (Yang et al., 2024), so the expanding-vs-uniform null needs re-testing at 0.4–3B before it is generalised (§7).

---

## 2. Definitions of the constructs the paper relies on

The paper should define these terms once, citing a standard source, and then use them consistently. Human learning-science terms come first, followed by ML terms.

### 2.1 Learning-science constructs

*(Detailed definitions with quotations and page references are in [research/notes/02_human_spacing.md](notes/02_human_spacing.md), "Definitions".)*

| Construct | Definition | Source |
|---|---|---|
| **Distributed-practice effect** | "An effect of interstudy interval (ISI) upon learning, as measured on subsequent tests." | Cepeda et al. (2006, p. 354) |
| **Spacing effect** | "Enhanced learning during spaced as compared with massed study episodes for a given item." Dunlosky et al. use *distributed practice* to cover both spacing and lag effects. | Cepeda et al. (2006, p. 354); Dunlosky et al. (2013, p. 36) |
| **Massed vs. spaced practice** | Massed: repetitions of an item with no intervening items or time. Spaced (distributed): repetitions separated by a measurable gap. | Cepeda et al. (2006, p. 354); Carpenter et al. (2012) |
| **Lag effect** | The advantage of longer over shorter spacing, when both schedules are spaced. | Cepeda et al. (2006, pp. 354–355); Dunlosky et al. (2013, p. 36) |
| **Inter-study interval (ISI, gap)** | "The interval separating different study episodes of the same materials." | Cepeda et al. (2006, p. 354) |
| **Retention interval (RI)** | The interval between the *last* study episode and the final test. | Cepeda et al. (2006, p. 354); Carpenter et al. (2012) |
| **Expanding / equal (uniform, fixed) / contracting schedules** | With three or more episodes, ISIs "may be equal (fixed), progressively longer (expanding), or progressively shorter (contracting)". | Cepeda et al. (2006, p. 355); Latimier et al. (2021, p. 961) |
| **Expanding retrieval practice** | An immediate first test followed by progressively longer gaps between retrieval attempts. The design keeps early retrieval successful and makes later retrievals harder. | Landauer & Bjork (1978); Karpicke & Roediger (2007, p. 704) |
| **Absolute vs. relative spacing** | Absolute: the total spacing across repetitions. Relative: how the repetitions are spaced relative to one another, i.e. the shape of the schedule. | Karpicke & Bauernschmidt (2011); Latimier et al. (2021) |
| **Retrieval practice / testing effect** | Retrieving information from memory, rather than re-reading it, improves later retention. | Roediger & Karpicke (2006); Rowland (2014) |
| **Restudy vs. test-type rehearsal** | Landauer & Bjork distinguish practice in which information "is presented repeatedly for study" from practice in which a fact is presented once and later rehearsal "takes the form of 'tests'". | Landauer & Bjork (1978, p. 625) |
| **Review / relearning** | Distributed practice is a *schedule* of learning episodes. Each episode may be restudy, retrieval or skill practice. Relearning means further practice sessions after an initial criterion has been reached. | Dunlosky et al. (2013, p. 36); Rawson & Dunlosky (2011) |
| **Desirable difficulties** | Learning conditions, including spacing, interleaving and testing, that make acquisition harder but improve long-term retention, provided the learner can still respond successfully. | Bjork (1994); Bjork & Bjork (2011) |
| **Storage vs. retrieval strength** | Storage strength is how entrenched a memory is; retrieval strength is how accessible it is now. Practice produces more learning when retrieval strength is low. | Bjork & Bjork (2011) |
| **Forgetting curve / savings** | Retention declines with time since learning. Ebbinghaus measured it as the time *saved* when relearning. Forgetting is roughly power-like, with an ever-decreasing proportional rate. | Ebbinghaus (1885/1913); Wixted & Ebbesen (1991); Wixted (2004); Murre & Dros (2015) |
| **Spaced-repetition systems** | Practical expanding schedulers: Pimsleur's graduated-interval recall (1967), the Leitner box system (1972), and SuperMemo's SM-2 algorithm, where intervals grow by an item-specific easiness factor starting at 2.5. Anki's scheduler descends from SM-2. All three are *adaptive*: an item's intervals depend on whether it is recalled. | Pimsleur (1967); Leitner (1972, date UNVERIFIED); Wozniak (1990); Reddy et al. (2016) |

**Mapping to our setup.** One LM "review" is a teacher-forced gradient update on the training statement. That makes it *restudy*, not retrieval practice. The closest human comparisons are therefore restudy-spacing studies: Cepeda et al. (2006), the restudy condition in Landauer & Bjork (1978), and Gerbier & Koenig (2012). Retrieval-practice studies (Karpicke & Roediger, 2007; Latimier et al., 2021) are informative but one step removed.

### 2.2 Machine-learning constructs

- **Catastrophic forgetting (interference).** When a network trained by gradient descent is trained on new material, its performance on earlier material often degrades sharply. The weights that supported the old material are shared with, and overwritten by, the new learning (McCloskey & Cohen, 1989; Ratcliff, 1990; French, 1999).
- **Continual learning.** Learning from a non-stationary sequence of data or tasks. The goal is to acquire new knowledge (plasticity) while retaining old knowledge (stability), usually without retraining on everything seen so far (Parisi et al., 2019; van de Ven et al., 2022; Wang et al., 2024). Our setup is continued training on a domain-incremental stream: old facts, then new facts from the same distribution.
- **Rehearsal / replay.** Re-training on previously learned items while new items are being learned (Robins, 1995). *Pseudo-rehearsal* replays generated items instead. For a review linking replay in deep learning to biological replay, see Hayes et al. (2021). Our "review" is rehearsal of stored real items.
- **Experience replay.** Mixing a small stored memory of past examples into later updates (Lin, 1992; Chaudhry et al., 2019; Rolnick et al., 2019).
- **Replay scheduling.** Deciding *when* and *what* to replay under a fixed replay budget (Klasson et al., 2023). FOREVER separates this into "when to replay" and "how strongly" (Feng et al., 2026). Our expanding-vs-uniform contrast is intended to manipulate only the *when*.
- **Low-rank adaptation (LoRA).** Fine-tuning through trainable low-rank updates to frozen weight matrices (Hu et al., 2022). LoRA "learns less and forgets less" than full fine-tuning (Biderman et al., 2024), which matters for a forgetting study (§6).

**Terminology recommendation.** "Spacing effect" in the human literature means *spaced vs. massed* practice (Cepeda et al., 2006). Our design compares two spaced schedules and has no massed arm, so it tests *schedule shape*, not the spacing effect itself. Either add a massed arm (the repository already implements a `cramming` schedule) or describe the study as "expanding vs. uniform review schedules" throughout, including in the title.

---

## 3. Verification of the existing citations

Full evidence, with quotations and page references, is in [research/notes/01_citation_verification.md](notes/01_citation_verification.md).

| Reference in draft | Bibliographic status | How it is used | Verdict and fix |
|---|---|---|---|
| Cepeda et al. (2006) | Correct | (a) spacing is robust; (b) expanding schedules "greatly improve memory"; (c) mechanisms | (a) Accurate, but the superlative "one of the most robust results in cognitive psychology" is not Cepeda's wording; soften it or cite Dempster (1988). (b) **Inaccurate**: Table 8 shows no reliable expanding-vs-fixed difference. (c) **Partly accurate**: the candidate accounts are consolidation, study-phase retrieval and encoding variability. |
| Landauer & Bjork (1978) | Consistent across secondary sources (book not inspected) | Expanding schedules "greatly improve memory" | **Inaccurate.** About a 10-point advantage on a same-session test 30 minutes later. Present it as the origin of the expanding-schedule idea, not as evidence of a large long-term benefit. |
| Karpicke & Roediger (2007) | Correct (APA 7: "Roediger, H. L., III") | Introduction: equal spacing "can match or exceed" expanding at long delays. Conclusion: our null "reproduces" this. | Introduction: **partly accurate**. Add that the advantage came from delaying the first retrieval, and that "long delay" means 2 days. Conclusion: **overstated**. Use "is consistent with", note that their task was retrieval practice while ours is restudy, and point to their Exp. 3, where the first retrieval was matched. |
| McCloskey & Cohen (1989) | Incomplete | Catastrophic forgetting | Accurate. Add editor (G. H. Bower), volume and DOI. |
| French (1999) | Correct | Catastrophic forgetting | Accurate. "Representations drift" is loose; "shared weights are overwritten" is closer to the source. |
| Hu et al. (2021) LoRA | Needs update | Method citation | Cite as ICLR **2022**. "All linear layers" is our configuration choice, not the LoRA paper's default. |
| Groeneveld et al. (2024) OLMo | Needs update | Architecture | Cite the ACL 2024 version (pp. 15789–15809). Do not use "et al." inside a reference-list entry. |
| Magnusson et al. (2025) DataDecide | Needs update | Model | Cite the ICML 2025 version (PMLR 267). "Hwang, J. D." Name the model exactly: `allenai/DataDecide-dolma1_7-300M` (Dolma 1.7 recipe), final default-seed checkpoint `step45787-seed-default`, commit `4b1b42ff`. The base model has **371.5M** parameters; the paper's "377M" includes the 5.24M LoRA adapter. |
| Kirchenbauer et al. (2025) FictionalQA | Needs update | Dataset; "held-out paraphrased questions" | Cite as ICLR **2026**; the second author is "Mongkolsupawan, N." The evaluation description is **misleading**. Each evaluation item is the dataset's question form of a statement the model *was* trained on, and the answer string appears verbatim in that statement for 93% of facts. There is one question per fact, not a set of paraphrases, and no FictionalQA held-out split is used. |

Other checkable statements:
- The "≈194 updates since the last review" figure is correct.
- The relative reductions (4.34% and 4.35%) are arithmetically correct.
- The confidence intervals are consistent with paired *t*-intervals with df = 2. The paper should state that this is the method.
- "Familiar from flashcard systems such as Anki" is fine, but note that Anki's intervals are *adaptive* (SM-2/FSRS), while ours are fixed.

---

## 4. Human learning science: what the literature actually says

Details are in [research/notes/02_human_spacing.md](notes/02_human_spacing.md).

### 4.1 Spaced vs. massed practice is robust; expanding vs. uniform is not

- **Spaced vs. massed.** The spacing effect is large and reliable. Only 12 of 271 comparisons in Cepeda et al. (2006) showed no effect or a reversal. The effect holds for retrieval practice (Latimier et al., 2021: *g* = 0.74), in large naturalistic datasets (Kim et al., 2019), in second-language learning (Kim & Webb, 2022), in mathematics (Murray et al., 2025) and in classrooms (Mawson & Kang, 2025). The optimal gap grows with the retention interval, along a "temporal ridgeline" (Cepeda et al., 2008, 2009).
- **Expanding vs. uniform.**
  - Pooled meta-analytic evidence finds no reliable difference. For retrieval practice, *g* = 0.034 [−0.10, 0.17] with *I*² = 0%, and retention interval does not moderate it (Latimier et al., 2021). For restudy, Cepeda et al. (2006) report 62.0% vs. 58.6%, which is not significant.
  - The total amount of spacing matters far more than its shape. In Karpicke & Bauernschmidt (2011), absolute spacing improved retention by about 200%, while relative spacing (schedule shape) had no effect.

### 4.2 Why individual studies disagree

- **The first-interval confound.** Equal schedules usually delay the first retrieval, and delaying it is what helps. Karpicke & Roediger (2007) found equal spacing better at 2 days (.45 vs. .33 and .60 vs. .49). Once the first retrieval was matched (Exp. 3), the schedule of later retrievals had no effect (*F* < 1). Logan & Balota (2008) and Cull (2000) also found equal ≥ expanding at delays of a day or more.
- **Restudy vs. testing.**
  - Landauer & Bjork (1978) proposed expanding schedules for *test-type* rehearsal and found about a 10-point advantage on a test 30 minutes later.
  - For pure re-presentation, they predicted uniform would be better and found .62 vs. .58 in its favour. The difference was not significant on its own, but the practice-type × schedule interaction was (Exp. II, p. 631).
  - LM review is re-presentation, so this is the most relevant human result for our design, and it predicts our null.
- **Forgetting rate and initial learning.**
  - Expanding helps when forgetting is fast or interference is high, because early retrievals still succeed (Storm et al., 2010; Maddox et al., 2011).
  - For restudy over days, expanding helps when initial learning is weak (Toppino et al., 2018; Gerbier et al., 2015).
  - Over a 35-day retention interval, equal ≈ expanding (Küpper-Tetzel et al., 2014).
  - Our facts are weakly learned and subject to heavy interference, the conditions under which expanding is most likely to help. This makes our null more informative, provided the manipulation is actually applied to each fact (§8).
- **What is measured.** Expanding schedules give higher *average* accessibility during training even when final retention is equal. Kang et al. (2014) found equal recall at an 8-week test but higher average recall during training (.49 vs. .41); see also Balota et al. (2006). The draft's §3.5 calls our trajectory result "most plausibly an artifact". The human literature treats this during-training advantage as a real outcome. We should report it as such ("a during-training benefit that does not persist after a shared final review").
- **Adaptive schedules beat both fixed shapes in humans** (Pavlik & Anderson, 2008; Lindsey et al., 2014; Mettler et al., 2016). This parallels SRT and FOREVER in ML (§5.2).

### 4.3 Theories of spacing and their machine analogues

- Cepeda et al. (2006) keep three candidate theories (consolidation, study-phase retrieval, encoding variability) and reject a fourth (deficient processing). See also Hintzman (1974), Glenberg (1979), Greene (1989), Benjamin & Tullis (2010), Maddox (2016) and Smolen et al. (2016).
- **Deficient processing** has the clearest machine analogue. The gradient on an item grows with its loss, so an item reviewed while it is still well remembered produces a smaller update. This is also the intuition behind "storage vs. retrieval strength" (Bjork & Bjork, 2011).
- **Study-phase retrieval and retrieval success** have no direct analogue in teacher-forced training.
- **Consolidation** has no clear analogue in weights trained by SGD.
- The draft should name these theories correctly and say which ones plausibly transfer.

### 4.4 Degree of learning vs. rate of forgetting

- The draft's claim that review "changes the starting point, not the forgetting rate" has a human parallel. Degree of learning shifts the intercept of the forgetting curve but not its slope (Slamecka & McElree, 1983; Rivera-Lares et al., 2022, 2023, 2025).
- That literature also carries three caveats that apply to us:
  1. Whether curves look parallel depends on the scale (Loftus, 1985; Bogartz, 1990; Loftus & Bamber, 1990).
  2. Memories of different ages forget at different rates (Slamecka, 1985). At buffer onset, our review arms' memories were refreshed more recently than the no-review arm's.
  3. Human forgetting is negatively accelerated (Wixted & Ebbesen, 1991). Our buffer forgetting is roughly linear in loss (≈0.0011, 0.0008 and 0.0008 nats per update over the three buffer windows).
- These caveats need to be written into the paper (see §9).

### 4.5 Computational models

Human scheduling has formal models whose concepts the paper could borrow (e.g., "half-life", "activation"):
- ACT-R activation (Pavlik & Anderson, 2005, 2008)
- the multiscale context model (Mozer et al., 2009)
- the SAM model of spacing (Raaijmakers, 2003)
- the Predictive Performance Equation (Walsh et al., 2018)
- half-life regression (Settles & Meeder, 2016)
- spaced-repetition optimisation (Tabibian et al., 2019; Reddy et al., 2016; Ye et al., 2022; Su et al., 2023)

Walsh et al. (2018) compare three such models directly.

### 4.6 What the human literature predicts for our design

The design holds the first and last review fixed, keeps the review count and compute fixed, varies only relative spacing, and tests once after a long delay. Under those conditions, the human literature predicts **no difference** in final retention (Karpicke & Roediger, 2007, Exp. 3; Karpicke & Bauernschmidt, 2011; Latimier et al., 2021). It also predicts **a during-training advantage for expanding** (Kang et al., 2014). We found both.

The defensible claim is "consistent with the human literature", not "reproduces a human phenomenon". There are three remaining gaps:
1. Our review is restudy, not retrieval practice.
2. There is no massed arm, so "spacing" in the strict sense is untested.
3. Per-fact schedules were not actually implemented (§8).

**A note on the team's earlier pilot.** The P4 one-pager reports an expanding advantage on retention AUC in a 20-seed Pythia pilot (Shakespeare then WikiText). That is also a during-training measure, so it fits the Kang et al. pattern rather than contradicting the whitepaper. The paper should say this explicitly if the pilot is mentioned.

---

## 5. Machine learning: replay, spaced repetition and review scheduling

Details for each paper are in [research/notes/03_ml_spacing_replay.md](notes/03_ml_spacing_replay.md).

### 5.1 Already established; present as replication, not discovery

- **Replay reduces forgetting.** This is established in classic connectionist work (Robins, 1995; McClelland et al., 1995), in continual learning (Chaudhry et al., 2019; Rolnick et al., 2019) and in language models (de Masson d'Autume et al., 2019; Scialom et al., 2022; Ibrahim et al., 2024; Liao et al., 2025). Small replay budgets of around 1% are often enough. Our finding that review beats no review is a replication in a new setting, and the paper should say so.
- **Spaced-repetition ideas have been applied to network training since 2017.** Examples include Repeat-before-Forgetting (Amiri et al., 2017), LFR (Prakriya et al., 2025), Leitner-guided replay (M'hamdi & May, 2024), FOREVER (Feng et al., 2026), MSSR (Lu et al., 2026) and SRT (Atreya et al., 2026).
- **Human scheduling effects often fail to transfer to networks.** Blocked training helps networks where interleaving helps humans (Flesch et al., 2018). In context, LLMs did better with massed exemplars, the opposite of humans (Xiong & Wu, 2025). Leitner and SuperMemo schedules transfer poorly to reinforcement-learning agents (Speckmann & Eimer, 2025).

### 5.2 The closest prior work, which the paper must discuss

| Work | Scale | What it shows | Relation to us |
|---|---|---|---|
| Tirumala et al. (2022) | Spacing test at 125M (suite 125M–13B) | Massed re-injection raises the long-run "forgetting baseline". Spaced re-injection has "minimal effect … independent of the length of spacing". | The nearest LM precedent for a null on spacing. We should frame our result as a sharper, matched-budget test of schedule *shape*. |
| Chang et al. (2024) | OLMo-1B, 7B | Fictional facts injected into pretraining. Retention decays as a power law. Duplicated data is forgotten faster. | Closest design (fictional facts, OLMo), but only one fixed 100-step interval was tested. |
| Feng et al. (2026), FOREVER | Qwen3-0.6B/4B, Llama-3.1-8B, Llama-2-13B, LoRA | Forgetting-curve-inspired replay in "model time". In one ablation table, at an identical replay budget, expanding beats uniform. | **Directly contradicts our null** at a similar scale. It is a single setting with no variance reported, and it measures task performance along a task sequence. A plausible reconciliation is that measurement happens closer to "during training", where our expanding arm also looked better (§3.5 of the draft). |
| Atreya et al. (2026), SRT | TinyLlama-1.1B, Llama-3.2-3B | Per-example SM-2 review state scheduled by perplexity beats uniform replay in continual pretraining. | Adaptive and per-item. Suggests that *per-item adaptivity*, not schedule shape, is what helps. Our current design cannot test this, because reviews are not item-specific (§8). |
| Prakriya et al. (2025), LFR | GPT-2 124M–1.5B; Llama-2 120M–500M | Perplexity-driven revisiting of data blocks speeds up pretraining and reduces forgetting. | Timing is confounded with selection and exposure count. Its baseline is no review. |
| Yang et al. (2024) | Pythia 160M–2.8B | Cyclic, fixed-order re-exposure produces "anticipatory recovery". The effect emerges between 160M and 410M. | Ordering structure changes forgetting dynamics around our model size. Fixed round-robin review order could trigger it. |
| Klasson et al. (2023) | Small MLPs and CNNs | Learned replay schedules beat fixed ones at a fixed memory size. | Evidence that timing can matter, but outside language models. |
| Liao et al. (2025) | GPT-2 0.1B and 1.5B | Periodic high-intensity replay beats one-shot replay. Forgetting curves look human-like. | Supports "periodic review helps". Confounds timing with intensity. |

**Novelty.** No study we found compares *fixed* expanding and uniform review of *facts* at a matched review count and step budget, with a delayed test after a long no-review buffer and seed-paired statistics. That is our niche. To claim it, the paper must engage FOREVER and Tirumala et al. directly, and the manipulation must actually be applied at the level of individual facts (§8, item 1). As currently run, the experiment does not hold the reviewed items fixed per fact.

---

## 6. How language models acquire and forget facts: implications for our metric and design

Details are in [research/notes/04_llm_knowledge_forgetting.md](notes/04_llm_knowledge_forgetting.md).

**Old-fact loss falls during new-fact training. The literature predicts this, and it means part of our metric is not fact-specific.**
- The FictionalQA authors report "leaky" gains on questions about events that were never trained on. They conclude that measured gains reflect "some combination of distributional and atomic learning" (Kirchenbauer et al., 2026).
- O'Neill (2026, preprint, Qwen3-4B with LoRA) finds the same and subtracts it with a "drift control" of never-trained statements.
- Several mechanisms could produce generic adaptation without any fact-specific memory: format and implicit-task adaptation (Kotha et al., 2024), answer priors that do not depend on the subject (Ghosal et al., 2024), and the continued development of recall circuits (Zucchet et al., 2025).
- Without a never-trained control, the paper cannot say old facts "ended up better learned" ("old-loss change", §2.4 of the draft). The difference between arms is better identified than the within-arm trajectories, because generic adaptation is roughly balanced across arms.

**The facts are only partly learned.**
- Answer-token loss of about 3.8 nats corresponds to a geometric-mean token probability of about 2%.
- This is consistent with too few exposures per fact. Allen-Zhu & Li (2024, 2025) find that robust storage needs on the order of 100–1000 exposures with paraphrase diversity; our facts average about 3.5 exposures in Stage 1.
- It is also consistent with training on short declarative statements, which FictionalQA found leads to memorization with little transfer.
- So the study concerns retention of *partial* traces. Human spacing studies usually train to a criterion first, and spacing effects might appear only once items are actually learned.

**Cross-entropy and behaviour can diverge.**
- Continuous metrics are more sensitive (Schaeffer et al., 2023), which justifies cross-entropy as *a* metric.
- However, a loss gain can occur without any change in answers (O'Neill, 2026), and an apparent loss of performance need not mean knowledge was lost (Zheng et al., 2025).
- Exact match is already computed in the code and should be reported. FictionalQA's multiple-choice configuration (`gend_mcq_w_grades`) would give an accuracy metric with distractors.

**LoRA is not neutral.**
- LoRA "learns less and forgets less" (Biderman et al., 2024, which tested rank 16).
- It introduces "intruder dimensions" that accumulate under sequential fine-tuning (Shuttleworth et al., 2025).
- Forgetting follows a power law in rank and number of steps (Kalajdzievski, 2024).
- Most of this evidence comes from models of about 7B, and RoBERTa-base in Shuttleworth et al. A full-fine-tuning replication at 300M is cheap and would answer the obvious reviewer question.

**Keep the constant learning rate.** Luo et al. (2026, ICLR Oral) show that learning-rate decay discounts late data in curriculum studies. This justifies a constant schedule for a timing study, but it also limits the conclusion to that regime. A short cooldown at the end of every arm would serve as a robustness check (Hägele et al., 2024).

**Closest two-stage design.** Chen et al. (2026, TMLR; "REMIX") study factoid memorization across two training stages. They find that mixing *generic* data into the second stage can rival replay of the specific facts. That implies a control we lack: an arm that inserts generic or unrelated batches at the review positions, to separate "re-exposure to these facts" from "a break from the new-fact stream".

---

## 7. Scale: what holds at a few billion parameters or less

The team plans to work with models of at most a few billion parameters. Details, with the exact model sizes studied in each paper, are in [research/notes/06_small_model_scale.md](notes/06_small_model_scale.md). The ML notes in research/notes/03 and research/notes/04 also carry a "Model scale" field for every entry.

### 7.1 Findings established at ≤ a few billion parameters

- **Small replay doses reduce forgetting from about 0.1B to 10B.** Reported doses: 1% pretraining-data injection (Bethune et al., 2025; 41M–1.27B); 1% rehearsal (Scialom et al., 2022); 5–25% replay (Ibrahim et al., 2024; 405M and 10B); replay at 99M–5.7B (Abbes et al., 2025). Our "review helps" result at 300M is consistent with all of these.
- **Injected facts are forgotten following a power law.** Later, longer-trained checkpoints learn new facts worse and forget more (Chang et al., 2024; Kim et al., 2025; both on OLMo 1B and 7B). The checkpoint used therefore matters, and the paper should name it.
- **Larger models learn facts faster per exposure and retain one-off exposures better.** Tirumala et al. (2022) cover 125M–13B, Chang et al. (2024) cover 1B to 7B, and Kirchenbauer et al. (2026) roughly 1B–8B. Carlini et al. (2023) find memorization grows with size (125M–6B).
- **Replay mostly raises the level of retention rather than slowing its decay.** Zucchet et al. (2025, ≤400M) and Chang et al. (2024) both find this. It parallels our "starting point, not forgetting rate" observation, subject to the scale caveats in §9.
- **Knowledge capacity is about 2–3.6 bits per parameter** (Allen-Zhu & Li, 2025, ≤0.5B; Morris et al., 2025, ≤1.5B). Capacity does not constrain FictionalQA at ≥100M; the number of exposures does.
- **Data-ordering curricula rarely help at small scale** (BabyLM: Warstadt et al., 2023; Hu et al., 2024).

### 7.2 Commonly cited findings that come from other scales (state as caveats)

- **"Bigger models forget less."** This rests mainly on vision models (Ramasesh et al., 2022; Mirzadeh et al., 2022) and on a 405M-vs-10B loss comparison (Ibrahim et al., 2024). In the 1–7B instruction-tuning regime, *absolute* forgetting increased with size (Luo et al., 2025; BLOOMZ 1.1–7.1B, mT0, LLaMA-7B), and strictly so only for domain knowledge. Say which quantity is meant: absolute drop, or fraction of gains lost.
- **LoRA findings.** "LoRA learns less and forgets less" comes from Llama-2-7B on code and math (Biderman et al., 2024). Kalajdzievski's (2024) forgetting law uses a single 7B model and varies rank, not model size.
- **Knowledge-injection results** such as Ovadia et al. (2024), Gekhman et al. (2024) and EntiGraph (Yang et al., 2025) come from 7B–70B models (details in research/notes/04).
- **Domain-mixture scaling laws** in our size range (Que et al., 2024; Gu et al., 2024; Wang et al., 2025) model *domain loss*, not the retention of specific facts.
- **Spaced-repetition work on networks** is mostly pre-transformer or non-LM (Amiri et al., 2017; Klasson et al., 2023). The exceptions are SRT (1.1B and 3B), FOREVER (0.6B upward), LFR (≤1.5B) and Tirumala et al. (125M).

### 7.3 What to expect between 300M and a few billion parameters

- **Relative forgetting should shrink with size.** The share of progress lost fell from about 95% to 20% between 41M and 1.27B (Bethune et al., 2025). Old-data loss increase was 0.27 nats at 405M vs. 0.23 at 10B (Ibrahim et al., 2024). The tolerable mixture ratio rises with size (Gu et al., 2024).
- **Absolute forgetting may grow**, because larger models learn more in Stage 1 (Luo et al., 2025). Report both absolute and relative measures.
- **The relative benefit of review probably shrinks but stays positive.** SRT's gain on old knowledge fell from +37 to +9 points between 1.1B and 3B (Atreya et al., 2026, single runs). Replay helped at every size in Abbes et al. (2025) and Ibrahim et al. (2024).
- **Expanding vs. uniform:** there is no direct evidence at any scale. However, effects of re-exposure order (anticipatory recovery) *emerge between 160M and 410M* and strengthen up to 2.8B (Yang et al., 2024). Our 300M null may therefore not carry over to 0.4–3B. That is the strongest scientific reason to scale the experiment up, though the prior expectation is still a null or small effect, as in humans.
- **LoRA confound across sizes.** At a fixed rank of 16, the trainable fraction falls as the model grows. Compare sizes at a matched trainable fraction, or include full fine-tuning.

### 7.4 Recommended scale ladder

1. **DataDecide 150M → 300M → 530M → 750M → 1B** on the same Dolma 1.7 recipe (Magnusson et al., 2025). This holds data and tokenizer fixed and reuses the current pipeline. Only the 1B size has three fully trained pretraining seeds.
2. **Pythia 410M → 1B → 1.4B → 2.8B (→ 6.9B)** (Biderman et al., 2023). This range covers the band where anticipatory recovery emerged. PolyPythias provides 10 pretraining seeds per size up to 410M for estimating seed variance (van der Wal et al., 2025).
3. **OLMo 1B / 7B or OLMo 2 1B / 7B** (Groeneveld et al., 2024; Team OLMo et al., 2025). This allows direct comparison with Chang et al. and Kim et al., using matched-maturity intermediate checkpoints.

Design adjustments for the ladder:
- Train Stage 1 to a fixed loss criterion so every size enters the delay at a comparable starting point.
- Use ≥5 seeds per size.
- Report absolute and relative forgetting.
- Add a full-fine-tuning arm at ≤1B.
- Add a short relearning probe.

---

## 8. What the experiment actually did (code audit)

Reconstructed from `configs/final.yaml` and the experiment code, read on GitHub (no local copy is kept), together with the pinned FictionalQA data and the DataDecide tokenizer. The derivations are in [research/notes/00_code_audit.md](notes/00_code_audit.md). These points should be confirmed against the team's run logs.

1. **A review event is one optimizer update of 16 old statements sampled with replacement from one document style.** There are 5 styles, visited round-robin. With 12 events, blog and corporate get 3 review batches each and the other styles get 2.
2. **The old-fact pool is very unbalanced across styles**: 386 facts (blog 28, corporate 54, encyclopedia 41, news 180, social 83). Expected review exposures per fact range from 1.7 (blog) to 0.18 (news). **Only 31–40 of the 80 evaluated old facts are ever reviewed, and only 11–17 are reviewed twice or more.** The same facts are reviewed in both arms within a seed; only the timing differs.
3. **Stage 1 is also unbalanced.** Expected exposures per fact range from 6.9 (blog) to 1.07 (news). Between 4 and 8 of the 80 evaluated "old" facts are never seen in Stage 1 at all.
4. **First and last reviews are matched only globally (steps 8 and 165).** The last review of each style is matched only for corporate. Uniform: blog 151, encyclopedia 108, news 122, social 136. Expanding: 123, 51, 68, 91. On average, expanding reviews of a given fact end about 37 updates earlier.
5. **Train and test formats differ.**
   - Training: `"Fictional fact: <statement>"`, with loss on every token.
   - Evaluation: `"Question: …\nAnswer:"`, with loss on the answer plus the EOS token. The answer averages 3.7 tokens, so EOS is about 20% of each scored item.
   - The model never sees the question format during training.
6. **The evaluation set is small and fixed**: 16 items per style, 80 old-fact items, the same across all seeds and conditions. Seeds 17, 23 and 42 vary LoRA initialization and sampling; they do not vary the data split or the evaluated items.
7. **Sequence length 64 truncates nothing.** The longest training item is 37 tokens.
8. **Exact match and per-item losses are logged but not reported.**
9. **"Trajectory change" is not a Stage-2-only average.** It is the mean over all nine `stage == 2` evaluations, which include the three buffer checkpoints (see [methods_verification.md](methods_verification.md)).
10. **The optimizer is re-created at the start of Stage 2.** AdamW moments are reset and the LR warms up again over 10 steps, so the shared first review at step 8 happens during warmup.

**What this means for the claims.**
- The expanding-vs-uniform null may reflect a *weak manipulation* rather than an absence of a schedule effect.
- The review benefit may be partly *non-specific*: style-level or format-level transfer rather than re-consolidation of particular facts.
- Both possibilities can be checked from existing logs. Within the review arms, compare facts reviewed *k* = 0, 1, 2, … times, and relate final loss to the time since each fact's last review.

---

## 9. Statistics and methodology

Details, and a full Reviewer-2 report, are in [research/notes/05_methods_stats_review.md](notes/05_methods_stats_review.md).

- **Equivalence, not a non-significant difference.** A confidence interval that spans zero is not evidence of no effect (Altman & Bland, 1995). Pre-specify a smallest effect of interest Δ, for example a fixed fraction of the review effect or the loss change corresponding to one percentage point of exact match. Then run TOST (Schuirmann, 1987; Lakens, 2017; Lakens et al., 2018) or a Bayesian ROPE analysis (Kruschke, 2018). From the reported numbers, the 90% interval for expanding minus uniform is about [−0.0038, +0.0026] nats, so equivalence would hold for any Δ ≥ 0.004. The claim is likely to survive, but only once Δ is justified independently of the data.
- **Facts are a random factor.** Every statistic currently treats the 80 items as fixed and the 3 seeds as the only replicates. This is Clark's (1973) "language-as-fixed-effect fallacy". Fit mixed-effects models with crossed random effects for item (nested in event) and seed (Baayen et al., 2008; Barr et al., 2013; Matuschek et al., 2017), or bootstrap hierarchically. Better still, redraw the old/new event split across replicates.
- **Forgetting-rate claims depend on scale.** Equal changes in loss are unequal changes in probability. On the probability scale, the review arms actually lose *more* during the buffer (0.0039 vs. 0.0030; Loftus, 1985). The classic human result that degree of learning shifts the intercept but not the slope (Slamecka & McElree, 1983) is debated on exactly these grounds (Loftus, 1985; Wixted, 2004). Fix this by:
  - fitting condition × time over all buffer checkpoints (0, 15, 60, 180);
  - reporting results on two scales;
  - adding a "horizontal" comparison: review arms after 180 buffer updates (3.809) are still better than the no-review arm at the start of the buffer (3.830), so review is worth more than 180 updates of delay.
- **Seeds and power.** Three seeds give df = 2 and a *t* multiplier of 4.30, and the estimated SD of the paired differences is itself uncertain by a factor of roughly 0.5–6. Use at least 10 seeds for the confirmatory comparison (cheap at 300M), report a sensitivity analysis and per-seed values, and state the CI method (Colas et al., 2018; Agarwal et al., 2021; Bouthillier et al., 2021).
- **Many metrics, no primary.** There are seven outcomes and three contrasts, with no designated primary outcome. The trajectory difference is dismissed post hoc as an artefact, yet it matches the pattern Kang et al. (2014) found in humans. Designate one primary outcome and contrast, correct the others, and consider preregistering the follow-up (van Miltenburg et al., 2021).

---

## 10. Recommended next steps

**A. Text-only fixes (no new runs)**
1. Correct the three misused citations (§3), update the archival venues and names, and add the closest prior work (§5.2, §6).
2. Describe the design exactly: the review-event content, the per-style schedules, the number of facts, the evaluation set, the exact model checkpoint and the parameter count.
3. Drop "spacing effect" in favour of "expanding vs. uniform review", unless a massed arm is added.
4. Recast "reproduces the human phenomenon" as "consistent with the meta-analytic null (Latimier et al., 2021)". Separate restudy from retrieval practice. Tie the trajectory result to Kang et al. (2014).
5. Remove the stray drafting note in §3.2 ("This is the natural place for it…").

**B. Reanalyses of existing logs**

6. Report exact match. Report effects in nats and as probability ratios rather than "% loss".
7. Item-level mixed-effects analysis. Compare reviewed with never-reviewed facts and fit a dose–response on the number of reviews. Compare Stage-1-unseen facts with seen ones.
8. TOST for expanding vs. uniform. Fit condition × time over the buffer checkpoints.

**C. Cheap reruns at 300M, which should precede any scaling up**

9. Make the schedule item-level: every old fact gets exactly *k* reviews at its own expanding or uniform positions, with first and last exposure matched *per fact*, and items counterbalanced to schedules across seeds.
10. Add controls:
    - a never-trained FictionalQA event set (40 events are currently unused);
    - a massed arm;
    - a generic-data arm at the review positions;
    - an arm where review batches are added rather than replacing new-fact batches;
    - full fine-tuning or a LoRA rank sweep.
11. Use at least 10 seeds, a different event split per replicate, and all 386 old facts in the evaluation. Add a pre-Stage-1 baseline and a multiple-choice or likelihood-margin metric.
12. Test FOREVER's schedule in our setup, and an adaptive per-item arm (the SRT idea), to separate *adaptivity* from *shape*.

**D. Scaling** (see §7). Repeat the item-level design across a model-size ladder before making any claim about a few-billion-parameter model.

**Venue fit** (details in research/notes/05). TMLR (rolling submission, judged on whether claims are supported rather than on novelty) suits a rigorous controlled negative result after the fixes in A–C. CogSci or CCN 2027 make sense if the human comparison is made rigorous. The ACL Rolling Review deadline of 12 October 2026 is too soon to add the controls. ICLR 2027 and NeurIPS 2026 are closed.

---

## 11. References

Formatted in APA 7. Every source cited above is listed with a URL; DOIs are used where they exist.

- Abbes, I., Subbaraj, G., Riemer, M., Islah, N., Therien, B., Tabaru, T., Kingetsu, H., Chandar, S., & Rish, I. (2025). *Revisiting replay and gradient alignment for continual pre-training of large language models* [Preprint]. arXiv. https://arxiv.org/abs/2508.01908
- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. In *Advances in Neural Information Processing Systems 34* (pp. 29304–29320). https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html
- Allen-Zhu, Z., & Li, Y. (2025). Physics of language models: Part 3.3, knowledge capacity scaling laws. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2404.05405
- Allen-Zhu, Z., & Li, Y. (2024). Physics of language models: Part 3.1, knowledge storage and extraction. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235, pp. 1067–1077). https://proceedings.mlr.press/v235/allen-zhu24a.html
- Altman, D. G., & Bland, J. M. (1995). Absence of evidence is not evidence of absence. *BMJ, 311*(7003), 485. https://doi.org/10.1136/bmj.311.7003.485
- Amiri, H., Miller, T., & Savova, G. (2017). Repeat before forgetting: Spaced repetition for efficient and effective training of neural networks. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing* (pp. 2401–2410). Association for Computational Linguistics. https://doi.org/10.18653/v1/D17-1255
- Atreya, A., Batra, D., Mantri, Y. K., Bantug, G., Cowan, G. A., & Khraishi, R. (2026). *When to review: Spaced repetition for continual pre-training of language models* [Preprint]. arXiv. https://arxiv.org/abs/2608.17530
- Baayen, R. H., Davidson, D. J., & Bates, D. M. (2008). Mixed-effects modeling with crossed random effects for subjects and items. *Journal of Memory and Language, 59*(4), 390–412. https://doi.org/10.1016/j.jml.2007.12.005
- Balota, D. A., Duchek, J. M., Sergent-Marshall, S. D., & Roediger, H. L., III. (2006). Does expanded retrieval produce benefits over equal-interval spacing? Explorations of spacing effects in healthy aging and early stage Alzheimer's disease. *Psychology and Aging, 21*(1), 19–31. https://doi.org/10.1037/0882-7974.21.1.19
- Barr, D. J., Levy, R., Scheepers, C., & Tily, H. J. (2013). Random effects structure for confirmatory hypothesis testing: Keep it maximal. *Journal of Memory and Language, 68*(3), 255–278. https://doi.org/10.1016/j.jml.2012.11.001
- Benjamin, A. S., & Tullis, J. G. (2010). What makes distributed practice effective? *Cognitive Psychology, 61*(3), 228–247. https://doi.org/10.1016/j.cogpsych.2010.05.004
- Bethune, L., Grangier, D., Busbridge, D., Gualdoni, E., Cuturi, M., & Ablin, P. (2025). Scaling laws for forgetting during finetuning with pretraining data injection. In *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*. https://openreview.net/forum?id=vWMij23BmQ
- Biderman, D., Portes, J., Gonzalez Ortiz, J. J., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., & Cunningham, J. P. (2024). LoRA learns less and forgets less. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=aloEru2qCG
- Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). Pythia: A suite for analyzing large language models across training and scaling. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 2397–2430). https://proceedings.mlr.press/v202/biderman23a.html
- Bjork, E. L., & Bjork, R. A. (2011). Making things hard on yourself, but in a good way: Creating desirable difficulties to enhance learning. In M. A. Gernsbacher, R. W. Pew, L. M. Hough, & J. R. Pomerantz (Eds.), *Psychology and the real world: Essays illustrating fundamental contributions to society* (pp. 56–64). Worth. https://www.researchgate.net/publication/284097727 (2nd ed., 2014, pp. 59–68: https://burrell.edu/wp-content/uploads/2020/09/EBjorkRBjork_FABBSchapter2014-2nd-ed._WithCoverPage.pdf) (full editor list UNVERIFIED)
- Bjork, R. A. (1994). Memory and metamemory considerations in the training of human beings. In J. Metcalfe & A. P. Shimamura (Eds.), *Metacognition: Knowing about knowing* (pp. 185–205). MIT Press. https://www.researchgate.net/publication/305433736
- Bogartz, R. S. (1990). Evaluating forgetting curves psychologically. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 16*(1), 138–148. https://doi.org/10.1037/0278-7393.16.1.138
- Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Sepah, N., Raff, E., Madan, K., Voleti, V., Kahou, S. E., Michalski, V., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems, 3*, 747–769. https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html
- Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramèr, F., & Zhang, C. (2023). Quantifying memorization across neural language models. In *The Eleventh International Conference on Learning Representations (ICLR 2023)*. https://openreview.net/forum?id=TatRHT_1cK
- Carpenter, S. K., Cepeda, N. J., Rohrer, D., Kang, S. H. K., & Pashler, H. (2012). Using spacing to enhance diverse forms of learning: Review of recent research and implications for instruction. *Educational Psychology Review, 24*(3), 369–378. https://doi.org/10.1007/s10648-012-9205-z
- Cepeda, N. J., Coburn, N., Rohrer, D., Wixted, J. T., Mozer, M. C., & Pashler, H. (2009). Optimizing distributed practice: Theoretical analysis and practical implications. *Experimental Psychology, 56*(4), 236–246. https://doi.org/10.1027/1618-3169.56.4.236
- Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin, 132*(3), 354–380. https://doi.org/10.1037/0033-2909.132.3.354
- Cepeda, N. J., Vul, E., Rohrer, D., Wixted, J. T., & Pashler, H. (2008). Spacing effects in learning: A temporal ridgeline of optimal retention. *Psychological Science, 19*(11), 1095–1102. https://doi.org/10.1111/j.1467-9280.2008.02209.x
- Chang, H., Park, J., Ye, S., Yang, S., Seo, Y., Chang, D.-S., & Seo, M. (2024). How do large language models acquire factual knowledge during pretraining? In *Advances in Neural Information Processing Systems 37*. https://arxiv.org/abs/2406.11813
- Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P. K., Torr, P. H. S., & Ranzato, M. (2019). *On tiny episodic memories in continual learning* [Preprint]. arXiv. https://arxiv.org/abs/1902.10486
- Chen, H., Geng, J., Bhaskar, A., Friedman, D., & Chen, D. (2026). Continual memorization of factoids in language models. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2411.07175
- Clark, H. H. (1973). The language-as-fixed-effect fallacy: A critique of language statistics in psychological research. *Journal of Verbal Learning and Verbal Behavior, 12*(4), 335–359. https://doi.org/10.1016/S0022-5371(73)80014-3
- Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2018). *How many random seeds? Statistical power analysis in deep reinforcement learning experiments* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1806.08295
- Cull, W. L. (2000). Untangling the benefits of multiple study opportunities and repeated testing for cued recall. *Applied Cognitive Psychology, 14*(3), 215–235. https://doi.org/10.1002/(SICI)1099-0720(200005/06)14:3%3C215::AID-ACP640%3E3.0.CO;2-1
- de Masson d'Autume, C., Ruder, S., Kong, L., & Yogatama, D. (2019). Episodic memory in lifelong language learning. In *Advances in Neural Information Processing Systems 32*. https://proceedings.neurips.cc/paper/2019/hash/f8d2e80c1458ea2501f98a2cafadb397-Abstract.html
- Dempster, F. N. (1988). The spacing effect: A case study in the failure to apply the results of psychological research. *American Psychologist, 43*(8), 627–634. https://doi.org/10.1037/0003-066X.43.8.627
- Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham, D. T. (2013). Improving students' learning with effective learning techniques: Promising directions from cognitive and educational psychology. *Psychological Science in the Public Interest, 14*(1), 4–58. https://doi.org/10.1177/1529100612453266
- Ebbinghaus, H. (1913). *Memory: A contribution to experimental psychology* (H. A. Ruger & C. E. Bussenius, Trans.). Teachers College, Columbia University. (Original work published 1885) https://psychclassics.yorku.ca/Ebbinghaus/index.htm
- Feng, Y., Wang, H., Li, J., Chu, X., Kang, Z., Liu, Y., Wang, Y., Yu, P. S., & Wu, X.-M. (2026). FOREVER: Forgetting curve-inspired memory replay for language model continual learning. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics*. https://arxiv.org/abs/2601.03938
- Flesch, T., Balaguer, J., Dekker, R., Nili, H., & Summerfield, C. (2018). Comparing continual task learning in minds and machines. *Proceedings of the National Academy of Sciences, 115*(44), E10313–E10322. https://doi.org/10.1073/pnas.1800755115
- French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2
- Gekhman, Z., Yona, G., Aharoni, R., Eyal, M., Feder, A., Reichart, R., & Herzig, J. (2024). Does fine-tuning LLMs on new knowledge encourage hallucinations? In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 7765–7784). https://doi.org/10.18653/v1/2024.emnlp-main.444
- Gerbier, E., & Koenig, O. (2012). Influence of multiple-day temporal distribution of repetitions on memory: A comparison of uniform, expanding, and contracting schedules. *Quarterly Journal of Experimental Psychology, 65*(3), 514–525. https://doi.org/10.1080/17470218.2011.600806
- Gerbier, E., Toppino, T. C., & Koenig, O. (2015). Optimising retention through multiple study opportunities over days: The benefit of an expanding schedule of repetitions. *Memory, 23*(6), 943–954. https://doi.org/10.1080/09658211.2014.944916
- Ghosal, G. R., Hashimoto, T., & Raghunathan, A. (2024). Understanding finetuning for factual knowledge extraction. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235, pp. 15540–15558). https://proceedings.mlr.press/v235/ghosal24a.html
- Glenberg, A. M. (1979). Component-levels theory of the effects of spacing of repetitions on recall and recognition. *Memory & Cognition, 7*(2), 95–112. https://doi.org/10.3758/BF03197590
- Greene, R. L. (1989). Spacing effects in memory: Evidence for a two-process account. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 15*(3), 371–377. https://doi.org/10.1037/0278-7393.15.3.371
- Groeneveld, D., Beltagy, I., Walsh, E., Bhagia, A., Kinney, R., Tafjord, O., Jha, A. H., Ivison, H., Magnusson, I., Wang, Y., Arora, S., Atkinson, D., Authur, R., Chandu, K. R., Cohan, A., Dumas, J., Elazar, Y., Gu, Y., Hessel, J., . . . Hajishirzi, H. (2024). OLMo: Accelerating the science of language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 15789–15809). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.acl-long.841
- Gu, J., Yang, Z., Ding, C., Zhao, R., & Tan, F. (2024). CMR scaling law: Predicting critical mixture ratios for continual pre-training of language models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 16143–16162). https://doi.org/10.18653/v1/2024.emnlp-main.903
- Hägele, A., Bakouch, E., Kosson, A., Ben Allal, L., Von Werra, L., & Jaggi, M. (2024). Scaling laws and compute-optimal training beyond fixed training durations. In *Advances in Neural Information Processing Systems 37* (pp. 76232–76264). https://doi.org/10.52202/079017-2427
- Hayes, T. L., Krishnan, G. P., Bazhenov, M., Siegelmann, H. T., Sejnowski, T. J., & Kanan, C. (2021). Replay in deep learning: Current approaches and missing biological elements. *Neural Computation, 33*(11), 2908–2950. https://doi.org/10.1162/neco_a_01433
- Hintzman, D. L. (1974). Theoretical implications of the spacing effect. In R. L. Solso (Ed.), *Theories in cognitive psychology: The Loyola Symposium* (pp. 77–99). Erlbaum. https://doi.org/10.4324/9781032722375-5
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR 2022)*. https://openreview.net/forum?id=nZeVKeeFYf9
- Hu, M. Y., Mueller, A., Ross, C., Williams, A., Linzen, T., Zhuang, C., Cotterell, R., Choshen, L., Warstadt, A., & Wilcox, E. G. (2024). Findings of the second BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *The 2nd BabyLM Challenge at CoNLL 2024* (pp. 1–21). https://aclanthology.org/2024.conll-babylm.1/
- Ibrahim, A., Thérien, B., Gupta, K., Richter, M. L., Anthony, Q., Lesort, T., Belilovsky, E., & Rish, I. (2024). Simple and scalable strategies to continually pre-train large language models. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=DimPeeCxKO
- Kalajdzievski, D. (2024). *Scaling laws for forgetting when fine-tuning large language models* [Preprint]. arXiv. https://arxiv.org/abs/2401.05605
- Kang, S. H. K., Lindsey, R. V., Mozer, M. C., & Pashler, H. (2014). Retrieval practice over the long term: Should spacing be expanding or equal-interval? *Psychonomic Bulletin & Review, 21*(6), 1544–1550. https://doi.org/10.3758/s13423-014-0636-z
- Karpicke, J. D., & Bauernschmidt, A. (2011). Spaced retrieval: Absolute spacing enhances learning regardless of relative spacing. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 37*(5), 1250–1257. https://doi.org/10.1037/a0023436
- Karpicke, J. D., & Roediger, H. L., III. (2007). Expanding retrieval practice promotes short-term retention, but equally spaced retrieval enhances long-term retention. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 33*(4), 704–719. https://doi.org/10.1037/0278-7393.33.4.704
- Kim, A. S. N., Wong-Kee-You, A. M. B., Wiseheart, M., & Rosenbaum, R. S. (2019). The spacing effect stands up to big data. *Behavior Research Methods, 51*(4), 1485–1497. https://doi.org/10.3758/s13428-018-1184-7
- Kim, J., Lee, H., Cho, H., Jang, J., Hwang, H., Won, S., Ahn, Y., Lee, D., & Seo, M. (2025). Knowledge entropy decay during language model pretraining hinders new knowledge acquisition. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. https://openreview.net/forum?id=eHehzSDUFp
- Kim, S. K., & Webb, S. (2022). The effects of spaced practice on second language learning: A meta-analysis. *Language Learning, 72*(1), 269–319. https://doi.org/10.1111/lang.12479
- Kirchenbauer, J., Mongkolsupawan, N., Wen, Y., Goldstein, T., & Ippolito, D. (2026). FictionalQA: A dataset for studying memorization and knowledge acquisition. In *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=SUNC1eJGMr
- Klasson, M., Kjellström, H., & Zhang, C. (2023). Learn the time to learn: Replay scheduling in continual learning. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=Q4aAITDgdP
- Kotha, S., Springer, J. M., & Raghunathan, A. (2024). Understanding catastrophic forgetting in language models via implicit inference. In *The Twelfth International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=VrHiF2hsrm
- Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian estimation. *Advances in Methods and Practices in Psychological Science, 1*(2), 270–280. https://doi.org/10.1177/2515245918771304
- Küpper-Tetzel, C. E., Kapler, I. V., & Wiseheart, M. (2014). Contracting, equal, and expanding learning schedules: The optimal distribution of learning sessions depends on retention interval. *Memory & Cognition, 42*(5), 729–741. https://doi.org/10.3758/s13421-014-0394-1
- Lakens, D. (2017). Equivalence tests: A practical primer for *t* tests, correlations, and meta-analyses. *Social Psychological and Personality Science, 8*(4), 355–362. https://doi.org/10.1177/1948550617697177
- Lakens, D., Scheel, A. M., & Isager, P. M. (2018). Equivalence testing for psychological research: A tutorial. *Advances in Methods and Practices in Psychological Science, 1*(2), 259–269. https://doi.org/10.1177/2515245918770963
- Landauer, T. K., & Bjork, R. A. (1978). Optimum rehearsal patterns and name learning. In M. M. Gruneberg, P. E. Morris, & R. N. Sykes (Eds.), *Practical aspects of memory* (pp. 625–632). Academic Press. https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/Landauer.Bjork_.1978.pdf
- Latimier, A., Peyre, H., & Ramus, F. (2021). A meta-analytic review of the benefit of spacing out retrieval practice episodes on retention. *Educational Psychology Review, 33*(3), 959–987. https://doi.org/10.1007/s10648-020-09572-8
- Leitner, S. (1972). *So lernt man lernen* [How to learn to learn]. Herder. https://openlibrary.org/works/OL9068102W (first-edition year UNVERIFIED)
- Liao, C., Xie, R., Sun, X., Sun, H., & Kang, Z. (2025). Exploring forgetting in large language model pre-training. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 2112–2127). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.acl-long.105
- Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine Learning, 8*(3–4), 293–321. https://doi.org/10.1007/BF00992699
- Lindsey, R. V., Shroyer, J. D., Pashler, H., & Mozer, M. C. (2014). Improving students' long-term knowledge retention through personalized review. *Psychological Science, 25*(3), 639–647. https://doi.org/10.1177/0956797613504302
- Loftus, G. R., & Bamber, D. (1990). Learning–forgetting independence, unidimensional memory models, and feature models: Comment on Bogartz (1990). *Journal of Experimental Psychology: Learning, Memory, and Cognition, 16*(5), 916–926. https://doi.org/10.1037/0278-7393.16.5.916
- Loftus, G. R. (1985). Evaluating forgetting curves. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 11*(2), 397–406. https://doi.org/10.1037/0278-7393.11.2.397
- Logan, J. M., & Balota, D. A. (2008). Expanded vs. equal interval spaced retrieval practice: Exploring different schedules of spacing and retention interval in younger and older adults. *Aging, Neuropsychology, and Cognition, 15*(3), 257–280. https://doi.org/10.1080/13825580701322171
- Lu, Y., He, Y., Chen, J., & Zha, H. (2026). *MSSR: Memory-aware adaptive replay for continual LLM fine-tuning* [Preprint]. arXiv. https://arxiv.org/abs/2603.09892
- Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=T5wkZJqzkz
- Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., & Zhang, Y. (2025). An empirical study of catastrophic forgetting in large language models during continual fine-tuning. *IEEE Transactions on Audio, Speech and Language Processing, 33*, 3776–3786. https://doi.org/10.1109/TASLPRO.2025.3606231
- Maddox, G. B. (2016). Understanding the underlying mechanism of the spacing effect in verbal learning: A case for encoding variability and study-phase retrieval. *Journal of Cognitive Psychology, 28*(6), 684–706. https://doi.org/10.1080/20445911.2016.1181637
- Maddox, G. B., Balota, D. A., Coane, J. H., & Duchek, J. M. (2011). The role of forgetting rate in producing a benefit of expanded over equal spaced retrieval in young and older adults. *Psychology and Aging, 26*(3), 661–670. https://doi.org/10.1037/a0022942
- Magnusson, I., Tai, N., Bogin, B., Heineman, D., Hwang, J. D., Soldaini, L., Bhagia, A., Liu, J., Groeneveld, D., Tafjord, O., Smith, N. A., Koh, P. W., & Dodge, J. (2025). DataDecide: How to predict best pretraining data with small experiments. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR Vol. 267, pp. 42487–42502). https://proceedings.mlr.press/v267/magnusson25a.html
- Matuschek, H., Kliegl, R., Vasishth, S., Baayen, H., & Bates, D. (2017). Balancing Type I error and power in linear mixed models. *Journal of Memory and Language, 94*, 305–315. https://doi.org/10.1016/j.jml.2017.01.001
- Mawson, R. D., & Kang, S. H. K. (2025). The distributed practice effect on classroom learning: A meta-analytic review of applied research. *Behavioral Sciences, 15*(6), 771. https://doi.org/10.3390/bs15060771
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review, 102*(3), 419–457. https://doi.org/10.1037/0033-295X.102.3.419
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), *Psychology of learning and motivation* (Vol. 24, pp. 109–165). Academic Press. https://doi.org/10.1016/S0079-7421(08)60536-8
- Mettler, E., Massey, C. M., & Kellman, P. J. (2016). A comparison of adaptive and fixed schedules of practice. *Journal of Experimental Psychology: General, 145*(7), 897–917. https://doi.org/10.1037/xge0000170
- M'hamdi, M., & May, J. (2024). Leitner-guided memory replay for cross-lingual continual learning. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)* (pp. 7808–7821). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.naacl-long.432
- Mirzadeh, S. I., Chaudhry, A., Yin, D., Hu, H., Pascanu, R., Gorur, D., & Farajtabar, M. (2022). Wide neural networks forget less catastrophically. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR Vol. 162, pp. 15699–15717). https://proceedings.mlr.press/v162/mirzadeh22a.html
- Morris, J. X., Sitawarin, C., Guo, C., Kokhlikyan, N., Suh, G. E., Rush, A. M., Chaudhuri, K., & Mahloujifar, S. (2025). *How much do language models memorize?* [Preprint]. arXiv. https://arxiv.org/abs/2505.24832 (a retitled version is listed for ICML 2026: https://openreview.net/forum?id=bA6BgSbaUi)
- Mozer, M. C., Pashler, H., Cepeda, N., Lindsey, R., & Vul, E. (2009). Predicting the optimal spacing of study: A multiscale context model of memory. In *Advances in Neural Information Processing Systems 22* (pp. 1321–1329). https://proceedings.neurips.cc/paper_files/paper/2009/hash/6bc24fc1ab650b25b4114e93a98f1eba-Abstract.html
- Murray, E., Horner, A. J., & Göbel, S. M. (2025). A meta-analytic review of the effectiveness of spacing and retrieval practice for mathematics learning. *Educational Psychology Review, 37*, Article 75. https://doi.org/10.1007/s10648-025-10035-1
- Murre, J. M. J., & Dros, J. (2015). Replication and analysis of Ebbinghaus' forgetting curve. *PLOS ONE, 10*(7), e0120644. https://doi.org/10.1371/journal.pone.0120644
- O'Neill, C. (2026). *Can a language model learn facts continually in its weights?* [Preprint]. arXiv. https://arxiv.org/abs/2607.11020
- Ovadia, O., Brief, M., Mishaeli, M., & Elisha, O. (2024). Fine-tuning or retrieval? Comparing knowledge injection in LLMs. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 237–250). https://doi.org/10.18653/v1/2024.emnlp-main.15
- Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks, 113*, 54–71. https://doi.org/10.1016/j.neunet.2019.01.012
- Pavlik, P. I., Jr., & Anderson, J. R. (2005). Practice and forgetting effects on vocabulary memory: An activation-based model of the spacing effect. *Cognitive Science, 29*(4), 559–586. https://doi.org/10.1207/s15516709cog0000_14
- Pavlik, P. I., Jr., & Anderson, J. R. (2008). Using a model to compute the optimal schedule of practice. *Journal of Experimental Psychology: Applied, 14*(2), 101–117. https://doi.org/10.1037/1076-898X.14.2.101
- Pimsleur, P. (1967). A memory schedule. *The Modern Language Journal, 51*(2), 73–75. https://doi.org/10.1111/j.1540-4781.1967.tb06700.x
- Prakriya, N., Yen, J.-N., Hsieh, C.-J., & Cong, J. (2025). Accelerating large language model pretraining via LFR pedagogy: Learn, focus, and review. In *Proceedings of the 29th Conference on Computational Natural Language Learning* (pp. 268–290). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.conll-1.18
- Que, H., Liu, J., Zhang, G., Zhang, C., Qu, X., Ma, Y., Duan, F., Bai, Z., Wang, J., Zhang, Y., Tan, X., Fu, J., Su, W., Wang, J., Qu, L., & Zheng, B. (2024). D-CPT law: Domain-specific continual pre-training scaling law for large language models. In *Advances in Neural Information Processing Systems 37*. https://openreview.net/forum?id=JzKFN5fWOk
- Raaijmakers, J. G. W. (2003). Spacing and repetition effects in human memory: Application of the SAM model. *Cognitive Science, 27*(3), 431–452. https://doi.org/10.1207/s15516709cog2703_5
- Ramasesh, V. V., Lewkowycz, A., & Dyer, E. (2022). Effect of scale on catastrophic forgetting in neural networks. In *International Conference on Learning Representations (ICLR 2022)*. https://openreview.net/forum?id=GhVS8_yPeEa
- Ratcliff, R. (1990). Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions. *Psychological Review, 97*(2), 285–308. https://doi.org/10.1037/0033-295X.97.2.285
- Rawson, K. A., & Dunlosky, J. (2011). Optimizing schedules of retrieval practice for durable and efficient learning: How much is enough? *Journal of Experimental Psychology: General, 140*(3), 283–302. https://doi.org/10.1037/a0023956
- Reddy, S., Labutov, I., Banerjee, S., & Joachims, T. (2016). Unbounded human learning: Optimal scheduling for spaced repetition. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1815–1824). https://doi.org/10.1145/2939672.2939850
- Rivera-Lares, K., Baddeley, A., & Della Sala, S. (2025). Influence of degree of learning on rate of forgetting of tonal sequences. *Memory & Cognition, 53*(2), 682–691. https://doi.org/10.3758/s13421-024-01597-6
- Rivera-Lares, K., Della Sala, S., Baddeley, A., & Logie, R. (2023). Rate of forgetting is independent from initial degree of learning across different age groups. *Quarterly Journal of Experimental Psychology, 76*(7), 1672–1682. https://doi.org/10.1177/17470218221128780
- Rivera-Lares, K., Logie, R., Baddeley, A., & Della Sala, S. (2022). Rate of forgetting is independent of initial degree of learning. *Memory & Cognition, 50*(8), 1706–1718. https://doi.org/10.3758/s13421-021-01271-1
- Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318
- Roediger, H. L., III, & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science, 17*(3), 249–255. https://doi.org/10.1111/j.1467-9280.2006.01693.x
- Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html
- Rowland, C. A. (2014). The effect of testing versus restudy on retention: A meta-analytic review of the testing effect. *Psychological Bulletin, 140*(6), 1432–1463. https://doi.org/10.1037/a0037559
- Schaeffer, R., Miranda, B., & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? In *Advances in Neural Information Processing Systems 36* (pp. 55565–55581). https://proceedings.neurips.cc/paper_files/paper/2023/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics, 15*(6), 657–680. https://doi.org/10.1007/BF01068419
- Scialom, T., Chakrabarty, T., & Muresan, S. (2022). Fine-tuned language models are continual learners. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (pp. 6107–6122). Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.emnlp-main.410
- Settles, B., & Meeder, B. (2016). A trainable spaced repetition model for language learning. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1848–1858). https://doi.org/10.18653/v1/P16-1174
- Shuttleworth, R., Andreas, J., Torralba, A., & Sharma, P. (2025). LoRA vs full fine-tuning: An illusion of equivalence. In *Advances in Neural Information Processing Systems 38*. https://openreview.net/forum?id=xp7B8rkh7L
- Slamecka, N. J., & McElree, B. (1983). Normal forgetting of verbal lists as a function of their degree of learning. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 9*(3), 384–397. https://doi.org/10.1037/0278-7393.9.3.384
- Slamecka, N. J. (1985). On comparing rates of forgetting: Comment on Loftus (1985). *Journal of Experimental Psychology: Learning, Memory, and Cognition, 11*(4), 812–816. https://doi.org/10.1037/0278-7393.11.1-4.812
- Smolen, P., Zhang, Y., & Byrne, J. H. (2016). The right time to learn: Mechanisms and optimization of spaced learning. *Nature Reviews Neuroscience, 17*(2), 77–88. https://doi.org/10.1038/nrn.2015.18
- Speckmann, M., & Eimer, T. (2025). *Task scheduling & forgetting in multi-task reinforcement learning* [Extended abstract, RLDM 2025]. arXiv. https://arxiv.org/abs/2503.01941
- Storm, B. C., Bjork, R. A., & Storm, J. C. (2010). Optimizing retrieval as a learning event: When and why expanding retrieval practice enhances long-term retention. *Memory & Cognition, 38*(2), 244–253. https://doi.org/10.3758/MC.38.2.244
- Su, J., Ye, J., Nie, L., Cao, Y., & Chen, Y. (2023). Optimizing spaced repetition schedule by capturing the dynamics of memory. *IEEE Transactions on Knowledge and Data Engineering, 35*(10), 10085–10097. https://doi.org/10.1109/TKDE.2023.3251721
- Tabibian, B., Upadhyay, U., De, A., Zarezade, A., Schölkopf, B., & Gomez-Rodriguez, M. (2019). Enhancing human learning via spaced repetition optimization. *Proceedings of the National Academy of Sciences, 116*(10), 3988–3993. https://doi.org/10.1073/pnas.1815156116
- Team OLMo, Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., Bhagia, A., Gu, Y., Huang, S., Jordan, M., Lambert, N., Schwenk, D., Tafjord, O., Anderson, T., Atkinson, D., Brahman, F., Clark, C., Dasigi, P., Dziri, N., . . . Hajishirzi, H. (2025). 2 OLMo 2 Furious. In *Conference on Language Modeling (COLM 2025)*. https://arxiv.org/abs/2501.00656
- Tirumala, K., Markosyan, A., Zettlemoyer, L., & Aghajanyan, A. (2022). Memorization without overfitting: Analyzing the training dynamics of large language models. In *Advances in Neural Information Processing Systems 35* (pp. 38274–38290). https://proceedings.neurips.cc/paper_files/paper/2022/hash/fa0509f4dab6807e2cb465715bf2d249-Abstract-Conference.html
- Toppino, T. C., Phelan, H.-A., & Gerbier, E. (2018). Level of initial training moderates the effects of distributing practice over multiple days with expanding, contracting, and uniform schedules: Evidence for study-phase retrieval. *Memory & Cognition, 46*(6), 969–978. https://doi.org/10.3758/s13421-018-0815-7
- van de Ven, G. M., Tuytelaars, T., & Tolias, A. S. (2022). Three types of incremental learning. *Nature Machine Intelligence, 4*(12), 1185–1197. https://doi.org/10.1038/s42256-022-00568-3
- van der Wal, O., Lesci, P., Muller-Eberstein, M., Saphra, N., Schoelkopf, H., Zuidema, W., & Biderman, S. (2025). PolyPythias: Stability and outliers across fifty language model pre-training runs. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2503.09543
- van Miltenburg, E., van der Lee, C., & Krahmer, E. (2021). Preregistering NLP research. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp. 613–623). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.naacl-main.51
- Walsh, M. M., Gluck, K. A., Gunzelmann, G., Jastrzembski, T., Krusmark, M., Myung, J. I., Pitt, M. A., & Zhou, R. (2018). Mechanisms underlying the spacing effect in learning: A comparison of three computational models. *Journal of Experimental Psychology: General, 147*(9), 1325–1348. https://doi.org/10.1037/xge0000416
- Wang, L., Zhang, X., Su, H., & Zhu, J. (2024). A comprehensive survey of continual learning: Theory, method and application. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 46*(8), 5362–5383. https://doi.org/10.1109/TPAMI.2024.3367329
- Wang, X., Tissue, H., Wang, L., Li, L., & Zeng, D. D. (2025). Learning dynamics in continual pre-training for large language models. In *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*. https://openreview.net/forum?id=Vk1rNMl0J1
- Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., Mosquera, R., Paranjabe, B., Williams, A., Linzen, T., & Cotterell, R. (2023). Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *Proceedings of the BabyLM Challenge at CoNLL 2023* (pp. 1–34). https://doi.org/10.18653/v1/2023.conll-babylm.1
- Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting. *Psychological Science, 2*(6), 409–415. https://doi.org/10.1111/j.1467-9280.1991.tb00175.x
- Wixted, J. T. (2004). The psychology and neuroscience of forgetting. *Annual Review of Psychology, 55*, 235–269. https://doi.org/10.1146/annurev.psych.55.090902.141555
- Wozniak, P. A. (1990). *Optimization of learning* [Master's thesis, University of Technology in Poznań]. SuperMemo 2 algorithm description: https://super-memory.com/english/ol/sm2.htm
- Xiong, Y., & Wu, S. (2025). Do large language models learn like humans: Interleaved and spaced practice in morphological learning. *Acta Psychologica, 260*, 105518. https://doi.org/10.1016/j.actpsy.2025.105518
- Yang, Y., Jones, M., Mozer, M. C., & Ren, M. (2024). Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training. In *Advances in Neural Information Processing Systems 37*. https://openreview.net/forum?id=YSs1z5udBY
- Yang, Z., Band, N., Li, S., Candès, E., & Hashimoto, T. (2025). Synthetic continued pretraining. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2409.07431
- Ye, J., Su, J., & Cao, Y. (2022). A stochastic shortest path algorithm for optimizing spaced repetition scheduling. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 4381–4390). https://doi.org/10.1145/3534678.3539081
- Zheng, J., Cai, X., Qiu, S., & Ma, Q. (2025). Spurious forgetting in continual learning of language models. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. https://openreview.net/forum?id=ScI7IlKGdI
- Zucchet, N., Bornschein, J., Chan, S., Lampinen, A. K., Pascanu, R., & De, S. (2025). How do language models learn facts? Dynamics, curricula and hallucinations. In *Second Conference on Language Modeling (COLM 2025)*. https://openreview.net/forum?id=vBcGnragkr

**Data and model artifacts**
- Kirchenbauer, J., Mongkolsupawan, N., Wen, Y., Goldstein, T., & Ippolito, D. (2026). *FictionalQA* [Data set; config `fict_qa`, revision 131cb74fdc3e601b5e896ed768ad9852ea35a8f9]. Hugging Face. https://huggingface.co/datasets/jwkirchenbauer/fictionalqa
- Allen Institute for AI. (2025). *DataDecide-dolma1_7-300M* [Model; revision 4b1b42ff7c5224a077c4f5624824dd7c8bbe98d7, branch `step45787-seed-default`]. Hugging Face. https://huggingface.co/allenai/DataDecide-dolma1_7-300M

---
---

# Part II: Blocked vs. interleaved training (Final Report, Test 02)

*Detailed notes:*
- *literature: [research/notes/07_interleaving.md](notes/07_interleaving.md)*
- *citation check: [research/notes/09_final_report_citation_verification.md](notes/09_final_report_citation_verification.md)*
- *mock review: [research/notes/10_final_report_methods_review.md](notes/10_final_report_methods_review.md) §B1*
- *code check: [methods_verification.md](methods_verification.md) Part II*

## II.0 TL;DR

- **The report describes one experiment and the team ran another.** The report proposes a 195M pretrained-checkpoint study with four school subjects. What was actually run (one run, one order) is a randomly initialised 162M model on four digit-manipulation skills. The report doesn't describe this run, and its "toy pilot" numbers (52.8% → 79.2%) have no code anywhere.
- **Human research does not straightforwardly predict an interleaving benefit for our skills.** The meta-analytic average is g = 0.42, but the benefit is largest for confusable categories, small for math (g = 0.34), and reversed for word lists (g = −0.39). Blocking can win for dissimilar or rule-based categories (Brunmair & Richter, 2019; Carvalho & Goldstone, 2014a). Human interleaving is also confounded with spacing (Kang & Pashler, 2012; Birnbaum et al., 2013; Foster et al., 2019).
- **In networks, interleaving beating blocking is the expected result, not a discovery.**
  - Blocked training causes catastrophic forgetting (McCloskey & Cohen, 1989; McClelland et al., 1995).
  - Flesch et al. (2018) found that humans learned better *blocked* while networks needed *interleaving*.
  - Russin et al. (2025) found an interleaving advantage for in-weight learning but a *blocking* advantage for in-context learning within the same transformers.
  - LM evidence points the same way: Lee et al. (2024; 13B instruction tuning); sequential SFT forgetting (Dong et al., 2024).
- **A defensible contribution needs four things:**
  - a block-length sweep at matched compute;
  - a blocked-but-spaced control (to separate interleaving from spacing);
  - an explicit recency control (counterbalanced orders, with each block evaluated in turn);
  - a manipulation of skill similarity, plus a 160M → 1B size sweep.

  None of this exists together in the literature. It can be built cheaply on the existing code, which already generates four counterbalanced orders.
- **Citations:** both cited sources are real. Brunmair & Richter's g = 0.42 is right, but the moderators need stating. Rolnick et al. is accurate as written but concerns RL agents, so add LM evidence.

## II.1 Definitions

| Term | Definition | Source |
|---|---|---|
| Blocked practice | All practice on one category or skill is completed before moving to the next (AAA BBB CCC). | Kornell & Bjork (2008); Rohrer & Taylor (2007) |
| Interleaved practice | Practice on different categories or skills is intermixed (ABC ABC ABC), usually at matched total practice. | Rohrer & Taylor (2007); Taylor & Rohrer (2010); Dunlosky et al. (2013) |
| Discriminative-contrast hypothesis | Interleaving helps because juxtaposing items from different categories highlights the features that distinguish them. | Kang & Pashler (2012); Birnbaum et al. (2013) |
| Spacing confound | Interleaving automatically spaces each category's repetitions, so part of its benefit may be a spacing effect. | Kang & Pashler (2012); Foster et al. (2019) |
| Catastrophic forgetting / interleaved learning in networks | Sequential (blocked) training overwrites earlier learning; interleaving old and new items prevents it. | McCloskey & Cohen (1989); McClelland et al. (1995) |

## II.2 The papers we most need to add

**The 5 most important**

| # | Paper | Why it matters | Link |
|---|---|---|---|
| 1 | **Flesch et al. (2018)**, PNAS | Humans learned better blocked while networks needed interleaving, on the same tasks. It is the central reason a human-to-model "transfer" claim must be argued carefully. | https://doi.org/10.1073/pnas.1800755115 |
| 2 | **Russin, Pavlick & Frank (2025)**, PNAS | In one transformer, *in-weight* learning favours interleaving while *in-context* learning favours blocking. It predicts our result and gives the mechanism. | https://doi.org/10.1073/pnas.2510270122 |
| 3 | **Lee, Cho & Yoo (2024)**, Findings of NAACL | Subject-interleaved instruction tuning beat blocked and random orders (LLaMA-2-13B). It is the closest LM precedent and a novelty threat. | https://doi.org/10.18653/v1/2024.findings-naacl.82 |
| 4 | **Brunmair & Richter (2019)**, Psychological Bulletin (already cited) | The moderators matter: g = 0.42 overall, 0.34 for math, −0.39 for words. Similarity decides the direction. | https://doi.org/10.1037/bul0000209 |
| 5 | **Foster et al. (2019)**, Memory & Cognition | Spacing alone explained the math interleaving benefit. Use it to motivate a blocked-but-spaced control arm. | https://doi.org/10.3758/s13421-019-00918-4 |

**5 more that are highly relevant**

| # | Paper | Why | Link |
|---|---|---|---|
| 6 | Carvalho & Goldstone (2014a) | Category structure decides whether blocking or interleaving wins. | https://doi.org/10.3758/s13421-013-0371-0 |
| 7 | Kang & Pashler (2012) | Separated interleaving from spacing. | https://doi.org/10.1002/acp.1801 |
| 8 | McClelland, McNaughton & O'Reilly (1995) | The classic account of why networks need interleaved learning. | https://doi.org/10.1037/0033-295X.102.3.419 |
| 9 | Dong et al. (2024), ACL | Sequential SFT keeps the last skill and forgets earlier ones in LLMs. | https://doi.org/10.18653/v1/2024.acl-long.12 |
| 10 | Yang et al. (2024), NeurIPS | Order effects under cyclic (repeated blocked) training emerge between 160M and 410M, so test more than one model size. | https://openreview.net/forum?id=YSs1z5udBY |

## II.3 Design problems found (details in methods_verification.md Part II and research/notes/10 §B1)

1. **Recency.** The blocked arm knows only its last skill (94.5%), with 0–15% on the others. Without counterbalanced orders and evaluation after each block, forgetting and never-learned can't be told apart.
2. **Incomplete learning.** The interleaved arm hadn't converged (training loss 0.21 vs. 0.02).
3. **n = 1 and one order.** The code already builds four cyclic orders; a Williams square × 5 seeds costs under 1 GPU-hour.
4. **Construct validity.** The four "skills" differ only in their bracket cue over the same digit inputs. This is catastrophic interference between near-identical mappings, not the human interleaving effect.
5. **Optimizer momentum.** Adam averages about the last 10 updates, so single-skill interleaved batches behave almost like mixed batches. This point is our own analysis.
6. **Constant LR is correct** (Luo et al., 2026). Keep it.

## II.4 Scale caveats (≤ a few B)
- Most ML evidence on interleaving vs. blocking comes from small non-LM networks (Flesch et al.) or 7B+ instruction tuning (Lee et al. 2024; Dong et al. 2024 at 7B–33B).
- Re-exposure order effects emerge between 160M and 410M (Yang et al., 2024).
- Critical mixture ratios change between 460M and 3.1B (Gu et al., 2024).
- A 162M result needs a 2–3-point size sweep (for example 160M / 410M / 1B) before it is generalised.

## II.5 References (Part II)
- Birnbaum, M. S., Kornell, N., Bjork, E. L., & Bjork, R. A. (2013). Why interleaving enhances inductive learning: The roles of discrimination and retrieval. *Memory & Cognition, 41*(3), 392–402. https://doi.org/10.3758/s13421-012-0272-7
- Brunmair, M., & Richter, T. (2019). Similarity matters: A meta-analysis of interleaved learning and its moderators. *Psychological Bulletin, 145*(11), 1029–1052. https://doi.org/10.1037/bul0000209
- Carvalho, P. F., & Goldstone, R. L. (2014a). Putting category learning in order: Category structure and temporal arrangement affect the benefit of interleaved over blocked study. *Memory & Cognition, 42*(3), 481–495. https://doi.org/10.3758/s13421-013-0371-0
- Dong, G., Yuan, H., Lu, K., Li, C., Xue, M., Liu, D., Wang, W., Yuan, Z., Zhou, C., & Zhou, J. (2024). How abilities in large language models are affected by supervised fine-tuning data composition. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 177–198). https://doi.org/10.18653/v1/2024.acl-long.12
- Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham, D. T. (2013). Improving students' learning with effective learning techniques: Promising directions from cognitive and educational psychology. *Psychological Science in the Public Interest, 14*(1), 4–58. https://doi.org/10.1177/1529100612453266
- Flesch, T., Balaguer, J., Dekker, R., Nili, H., & Summerfield, C. (2018). Comparing continual task learning in minds and machines. *Proceedings of the National Academy of Sciences, 115*(44), E10313–E10322. https://doi.org/10.1073/pnas.1800755115
- Foster, N. L., Mueller, M. L., Was, C., Rawson, K. A., & Dunlosky, J. (2019). Why does interleaving improve math learning? The contributions of discriminative contrast and distributed practice. *Memory & Cognition, 47*(6), 1088–1101. https://doi.org/10.3758/s13421-019-00918-4
- Gu, J., Yang, Z., Ding, C., Zhao, R., & Tan, F. (2024). CMR scaling law: Predicting critical mixture ratios for continual pre-training of language models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 16143–16162). https://doi.org/10.18653/v1/2024.emnlp-main.903
- Kang, S. H. K., & Pashler, H. (2012). Learning painting styles: Spacing is advantageous when it promotes discriminative contrast. *Applied Cognitive Psychology, 26*(1), 97–103. https://doi.org/10.1002/acp.1801
- Kornell, N., & Bjork, R. A. (2008). Learning concepts and categories: Is spacing the "enemy of induction"? *Psychological Science, 19*(6), 585–592. https://doi.org/10.1111/j.1467-9280.2008.02127.x
- Lee, B. W., Cho, H., & Yoo, K. M. (2024). Instruction tuning with human curriculum. In *Findings of the Association for Computational Linguistics: NAACL 2024* (pp. 1281–1309). https://doi.org/10.18653/v1/2024.findings-naacl.82
- Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=T5wkZJqzkz
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review, 102*(3), 419–457. https://doi.org/10.1037/0033-295X.102.3.419
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), *Psychology of learning and motivation* (Vol. 24, pp. 109–165). Academic Press. https://doi.org/10.1016/S0079-7421(08)60536-8
- Rohrer, D., & Taylor, K. (2007). The shuffling of mathematics problems improves learning. *Instructional Science, 35*(6), 481–498. https://doi.org/10.1007/s11251-007-9015-8
- Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html
- Russin, J., Pavlick, E., & Frank, M. J. (2025). Parallel trade-offs in human cognition and neural networks: The dynamic interplay between in-context and in-weight learning. *Proceedings of the National Academy of Sciences, 122*(35), e2510270122. https://doi.org/10.1073/pnas.2510270122
- Taylor, K., & Rohrer, D. (2010). The effects of interleaved practice. *Applied Cognitive Psychology, 24*(6), 837–848. https://doi.org/10.1002/acp.1598
- Yang, Y., Jones, M., Mozer, M. C., & Ren, M. (2024). Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training. In *Advances in Neural Information Processing Systems 37*. https://openreview.net/forum?id=YSs1z5udBY

---
---

# Part III: Mastery-gated curriculum training (Final Report, Test 03)

*Detailed notes:*
- *literature: [research/notes/08_mastery_curriculum.md](notes/08_mastery_curriculum.md)*
- *citation check: [research/notes/09_final_report_citation_verification.md](notes/09_final_report_citation_verification.md)*
- *mock review: [research/notes/10_final_report_methods_review.md](notes/10_final_report_methods_review.md) §B2*
- *experiment plan: [PRD_mastery_gated_curriculum.md](../plans/PRD_mastery_gated_curriculum.md)*

## III.0 TL;DR

- **Status.** The subteam ran two small preliminary experiments but no full experiments, and **no code for either exists on any of the repository's 152 branches**. A proposal-only section isn't publishable, so reviewers will want real, reproducible experiments. The PRD designs one, about 20–50 GPU-hours for Phase 1.
- **Human evidence is weaker than the report implies.**
  - Mastery learning gives about 0.52 SD on end-of-unit exams but about 0.29 SD on standardized tests (Kulik et al., 1990), and roughly zero when time is equalized (Slavin, 1987).
  - Our matched-token design *is* the time-equalized case, so there is no strong prior for a benefit.
  - The best human support for "gate + replay" is successive relearning (Rawson & Dunlosky, 2011).
- **ML curricula mostly help only under small budgets or noisy labels** (Wu et al., 2021). Curricula for small-LM pretraining are mostly null (BabyLM; Warstadt et al., 2023). LR decay erases much of any curriculum gain (Luo et al., 2026).
- **Gate + replay already exists as a training tool.**
  - Kohli et al. (2026) unlock k+1-hop data at ≥95% held-out accuracy while replaying earlier stages.
  - Teacher–student curricula re-sample forgotten tasks (Matiisen et al., 2020).
  - Theory says an easy + hard *mixture* can match a staged curriculum on chain tasks (Wang et al., 2025).
  - **What is new:** a budget-matched comparison of *contingent* vs. *yoked/tuned* progression with replay as a separate factor. A null would also be publishable.
- **The preliminary results can't support the hypothesis yet.** Experiment I confounds gating with replay. Once the clock arm also gets replay (Experiment II), the gap shrinks to 0.97 vs. 0.91, which is within single-run noise.
- **Citation fixes:**
  - The Anthropic "biology" link and Robins (1995) don't support their sentences.
  - The reversed-digit paper (arXiv 2403.05845) is a 13B fine-tune with no OOD tests; cite Lee et al. (2024) instead.
  - CAMPUS's "7%" is against curriculum baselines, not shuffle (+1.67 points over shuffle).
  - RLAAR's numbers compare the full method to an untrained model.
  - Zaremba & Sutskever's "combined" strategy mixed in *harder* examples. That is not replay.
  - The LR-decay paper concerns data *quality* and doesn't endorse "identical schedule" as a fix.

## III.1 Definitions

| Term | Definition | Source |
|---|---|---|
| Mastery learning | Instruction divided into units; learners advance only after reaching a criterion on a unit test, with corrective instruction and retesting until they do. | Bloom (1968); Slavin (1987); Kulik et al. (1990) |
| Curriculum learning | Training on examples ordered from easier to harder rather than at random. | Bengio et al. (2009); Elman (1993) |
| Self-paced / competence-based curriculum | Difficulty is admitted according to the model's current loss (self-paced) or a competence schedule. Note that Platanios et al.'s "competence" is a *time* schedule. | Kumar et al. (2010); Platanios et al. (2019) |
| Teacher–student / progress-based curriculum | A teacher chooses tasks by learning progress and re-samples tasks that are being forgotten. | Graves et al. (2017); Matiisen et al. (2020) |
| Replay | Continuing to train on earlier skills while learning new ones. | Robins (1995); Rolnick et al. (2019) |
| Length / OOD generalization; grokking | Performance on longer or structurally novel inputs than seen in training; generalization that appears long after the training data is fit. | Anil et al. (2022); Zhou et al. (2024a); Power et al. (2022) |

## III.2 The papers we most need to add

**The 5 most important**

| # | Paper | Why it matters | Link |
|---|---|---|---|
| 1 | **Wu, Dyer & Neyshabur (2021)**, ICLR | Curricula rarely beat well-tuned random orderings except under small budgets or noisy labels. It sets the bar for baselines. | https://openreview.net/forum?id=tW4QEInpni |
| 2 | **Kohli et al. (2026)**, COLM | The closest precedent: a ≥95% held-out gate with replay on k-hop chains (verified in the full text). Used as a tool, never compared with shuffle or clock schedules. | https://arxiv.org/abs/2604.07822 |
| 3 | **Wang et al. (2025)**, COLT | Proves that a staged curriculum *or* an easy + hard mixture makes compositional (pointer-chasing) tasks learnable. So a mixed-difficulty control is required. | https://proceedings.mlr.press/v291/wang25a.html |
| 4 | **Yao et al. (2025)**, EMNLP | k-hop reasoning needs exponentially more data; a curriculum reduces but does not remove that cost. Directly relevant to the chain-depth task. | https://arxiv.org/abs/2505.17923 |
| 5 | **Matiisen et al. (2020)**, IEEE TNNLS | Teacher–student curriculum learning: performance-driven task selection plus re-sampling of forgotten tasks. The closest ML analogue of gate + replay. | https://doi.org/10.1109/TNNLS.2019.2934906 |

**5 more that are highly relevant**

| # | Paper | Why | Link |
|---|---|---|---|
| 6 | Slavin (1987) | Mastery-learning gains are about zero when time is equalized, which is our matched-token setting. | https://doi.org/10.3102/00346543057002175 |
| 7 | Rawson & Dunlosky (2011) | Learning to criterion plus spaced relearning: the best human analogue of gate + replay. | https://doi.org/10.1037/a0023956 |
| 8 | Zhang et al. (2022), LEGO | The origin of the chain-depth task. It must be cited. | https://doi.org/10.48550/arXiv.2206.04301 |
| 9 | Lee et al. (2024), *Teaching arithmetic to small transformers* | The correct citation for reversed-output arithmetic, and evidence that length generalization is hard. | https://openreview.net/forum?id=dsUB4bst9S |
| 10 | Warstadt et al. (2023), BabyLM | Curricula for small-LM pretraining were largely unsuccessful. An important negative prior. | https://doi.org/10.18653/v1/2023.conll-babylm.1 |

## III.3 Scale caveats (≤ a few B)
- Classic curriculum evidence comes from tiny LSTMs, CNNs, MLPs and RL agents (Zaremba & Sutskever; Wu et al.; Matiisen et al.).
- At ≤125M, LM curricula are mostly null (BabyLM). Positive 0.5–3B results mostly used a constant LR and clock-based schedules on general text.
- Arithmetic length generalization does not improve monotonically with size (Zhou et al., 2024b; Anil et al., 2022), so bigger models won't make OOD gates easier to pass.
- The "heuristics not algorithms" evidence comes from ≥6B or undisclosed-size models (Nikankin et al., 2025; Lindsey et al., 2025).
- **Recommended ladder:** synthetic studies at 10–125M with many seeds; then 300M; then about 1B and continued training of open 0.4–1.4B models (PRD Phase 2).

## III.4 References (Part III)
- Anil, C., Wu, Y., Andreassen, A., Lewkowycz, A., Misra, V., Ramasesh, V., Slone, A., Gur-Ari, G., Dyer, E., & Neyshabur, B. (2022). Exploring length generalization in large language models. *Advances in Neural Information Processing Systems, 35*. https://openreview.net/forum?id=zSkYVeX7bC4 (pages partly verified)
- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. In *Proceedings of the 26th Annual International Conference on Machine Learning* (pp. 41–48). https://doi.org/10.1145/1553374.1553380
- Bloom, B. S. (1968). Learning for mastery. *Evaluation Comment, 1*(2), 1–12. https://eric.ed.gov/?id=ED053419
- Elman, J. L. (1993). Learning and development in neural networks: The importance of starting small. *Cognition, 48*(1), 71–99. https://doi.org/10.1016/0010-0277(93)90058-4
- Graves, A., Bellemare, M. G., Menick, J., Munos, R., & Kavukcuoglu, K. (2017). Automated curriculum learning for neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR 70, pp. 1311–1320). https://proceedings.mlr.press/v70/graves17a.html
- Kohli, H., Parthasarathy, S., Sun, H., & Yao, Y. (2026). Loop, think, & generalize: Implicit reasoning in recurrent-depth transformers. In *Conference on Language Modeling (COLM 2026)*. https://arxiv.org/abs/2604.07822
- Kulik, C.-L. C., Kulik, J. A., & Bangert-Drowns, R. L. (1990). Effectiveness of mastery learning programs: A meta-analysis. *Review of Educational Research, 60*(2), 265–299. https://doi.org/10.3102/00346543060002265
- Kumar, M. P., Packer, B., & Koller, D. (2010). Self-paced learning for latent variable models. In *Advances in Neural Information Processing Systems 23* (pp. 1189–1197). https://proceedings.neurips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html
- Lee, N., Sreenivasan, K., Lee, J. D., Lee, K., & Papailiopoulos, D. (2024). Teaching arithmetic to small transformers. In *International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=dsUB4bst9S
- Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce, A., Turner, N. L., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., . . . Batson, J. (2025). On the biology of a large language model. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=T5wkZJqzkz
- Matiisen, T., Oliver, A., Cohen, T., & Schulman, J. (2020). Teacher–student curriculum learning. *IEEE Transactions on Neural Networks and Learning Systems, 31*(9), 3732–3740. https://doi.org/10.1109/TNNLS.2019.2934906
- Nikankin, Y., Reusch, A., Mueller, A., & Belinkov, Y. (2025). Arithmetic without algorithms: Language models solve math with a bag of heuristics. In *International Conference on Learning Representations (ICLR 2025)*. https://openreview.net/forum?id=09YTt26r2P
- Platanios, E. A., Stretcu, O., Neubig, G., Poczos, B., & Mitchell, T. (2019). Competence-based curriculum learning for neural machine translation. In *Proceedings of NAACL-HLT 2019* (Vol. 1, pp. 1162–1172). https://doi.org/10.18653/v1/N19-1119
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2201.02177
- Rawson, K. A., & Dunlosky, J. (2011). Optimizing schedules of retrieval practice for durable and efficient learning: How much is enough? *Journal of Experimental Psychology: General, 140*(3), 283–302. https://doi.org/10.1037/a0023956
- Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318
- Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html
- Slavin, R. E. (1987). Mastery learning reconsidered. *Review of Educational Research, 57*(2), 175–213. https://doi.org/10.3102/00346543057002175
- Wang, Z., Nichani, E., Bietti, A., Damian, A., Hsu, D., Lee, J. D., & Wu, D. (2025). Learning compositional functions with transformers from easy-to-hard data. In *Proceedings of the 38th Conference on Learning Theory* (PMLR 291, pp. 5632–5711). https://proceedings.mlr.press/v291/wang25a.html
- Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., Mosquera, R., Paranjabe, B., Williams, A., Linzen, T., & Cotterell, R. (2023). Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *Proceedings of the BabyLM Challenge at CoNLL 2023* (pp. 1–34). https://doi.org/10.18653/v1/2023.conll-babylm.1
- Wu, X., Dyer, E., & Neyshabur, B. (2021). When do curricula work? In *International Conference on Learning Representations (ICLR 2021)*. https://openreview.net/forum?id=tW4QEInpni
- Yao, Y., Du, Y., Zhu, D., Hahn, M., & Koller, A. (2025). Language models can learn implicit multi-hop reasoning, but only if they have lots of training data. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing* (pp. 9684–9702). https://arxiv.org/abs/2505.17923
- Zaremba, W., & Sutskever, I. (2014). *Learning to execute* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1410.4615
- Zhang, Y., Backurs, A., Bubeck, S., Eldan, R., Gunasekar, S., & Wagner, T. (2022). *Unveiling transformers with LEGO: A synthetic reasoning task* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2206.04301
- Zhou, H., Bradley, A., Littwin, E., Razin, N., Saremi, O., Susskind, J., Bengio, S., & Nakkiran, P. (2024a). What algorithms can transformers learn? A study in length generalization. In *International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=AssIuHnmHX
- Zhou, Y., Alon, U., Chen, X., Wang, X., Agarwal, R., & Zhou, D. (2024b). Transformers can achieve length generalization but not robustly. In *ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models*. https://openreview.net/forum?id=DWkWIh3vFJ
