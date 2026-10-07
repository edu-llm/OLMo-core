# Citation and Reference Verification: P4 Whitepaper ("Does Spaced Review Help a Language Model Retain Facts During Continued Training?")

## TL;DR

- **All 9 references exist. Three are used wrongly:**
  1. **"Expanding schedules greatly improve memory (Landauer & Bjork 1978; Cepeda et al. 2006)."** Cepeda et al. found no reliable expanding-vs-fixed difference (62.0% vs 58.6%, p = .61) and list Landauer & Bjork's claim as having "little apparent empirical backing". Landauer & Bjork found only about a 10-point advantage at 30 minutes.
  2. **Mechanisms "consolidation, retrieval difficulty, encoding variability".** Cepeda et al.'s candidate theories are consolidation, *study-phase retrieval* and encoding variability. "Retrieval difficulty" comes from Bjork and from Karpicke & Roediger.
  3. **Conclusion: "reproduces" Karpicke & Roediger (2007).** Their equal-spacing advantage came from delaying the first retrieval. When the first retrieval was matched (Exp. 3, which is the condition our design matches), schedule shape made no difference. Their task was retrieval practice, while ours is restudy. Use "consistent with".
- **Use archival versions:** LoRA → ICLR 2022; OLMo → ACL 2024; DataDecide → ICML 2025; FictionalQA → ICLR 2026. Name fixes: Hwang, J. D.; Mongkolsupawan, N.; Roediger, H. L., III; Walsh, E.
- **Model details:** `allenai/DataDecide-dolma1_7-300M`, checkpoint `step45787-seed-default`. The base model has 371.5M parameters; "377M" includes LoRA.
- **The evaluation-set description is misleading.** There is one question per *trained* statement, not a set of held-out paraphrases.
- **Statistics:** the CIs are consistent with paired t-intervals (df = 2) and "4.34%" is correct. The paper should state the CI method.


**Paper checked:** `/Users/amylin/Amy/MIT/Internships/Alpha_AI_Engineering/projects/edu-llm/paper/source/P4_whitepaper_writeup.md` (not edited)
**Date of check:** 2026-10-02
**How I checked:** I compared each claim against primary sources: arXiv abstract and HTML pages, the ACL Anthology, OpenReview (web and API), PMLR, the Crossref API, the Hugging Face model and dataset APIs, and the full-text PDFs of Cepeda et al. (2006) and Karpicke & Roediger (2007). I also read the experiment code (`github.com/anshulmago1/olmo-fictionalqa-review`, `configs/final.yaml`, `src/olmo_core/review_lab/*.py`) and downloaded the pinned FictionalQA `fict_qa` parquet to inspect what the model was trained and tested on.

**Page numbers:** page numbers for Cepeda and K&R are estimated from page markers in the extracted PDF text (about ±1 page). Section names are exact.

**Verdict labels:** ACCURATE / PARTIALLY ACCURATE / INACCURATE. **UNVERIFIED** means I could not confirm the item against a primary source.

---

## Summary table

| # | Reference | Bibliographic status | Citation-use status |
|---|---|---|---|
| 1 | Cepeda et al. (2006) | Correct | **INACCURATE** for "greatly improve memory" (expanding); PARTIALLY ACCURATE for mechanisms; ACCURATE for spacing being robust, but the superlative wording is not Cepeda's |
| 2 | French (1999) | Correct | ACCURATE (the word "drift" is loose) |
| 3 | Groeneveld et al. (2024) | **Needs update.** Published at ACL 2024; "et al." in the reference list is not APA 7; ACL lists the 3rd author as "Evan Walsh" | ACCURATE (DataDecide uses OLMo tooling) |
| 4 | Hu et al. (2021) | **Needs update.** Published at ICLR 2022 | ACCURATE (but "all linear layers" is your choice, not the paper's default) |
| 5 | Karpicke & Roediger (2007) | Correct (APA wants "Roediger, H. L., III") | Introduction: PARTIALLY ACCURATE. Conclusion: **overstated** |
| 6 | Kirchenbauer et al. (2025) | **Needs update.** ICLR 2026; 2nd author's first name is "Natjanan" (N.) in the ICLR record | Dataset description PARTIALLY ACCURATE; the "held-out paraphrased questions" wording is **misleading** |
| 7 | Landauer & Bjork (1978) | Consistent across many secondary sources; the book itself is UNVERIFIED | **INACCURATE** for "greatly improve" (about a 10-percentage-point gain on a 30-min test, never shown at long delays) |
| 8 | Magnusson et al. (2025) | **Needs update.** ICML 2025, PMLR 267:42487–42502; "Hwang, J." should be "Hwang, J. D." | The model needs a fuller name; "approximately 377M total parameters" **includes the LoRA adapter** (the base model is 371.49M) |
| 9 | McCloskey & Cohen (1989) | Incomplete: missing editor, book format, and DOI | ACCURATE |

---

## 1. Cepeda, Pashler, Vul, Wixted, & Rohrer (2006)

**Corrected APA 7 citation** (as given, already correct):
> Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin, 132*(3), 354–380. https://doi.org/10.1037/0033-2909.132.3.354

**Bibliographic check:** Crossref confirms the title, all 5 authors in this order, the journal, 132(3), pp. 354–380, and the DOI. **CORRECT.**

### In-text uses

**(a)** "The *spacing effect* ... is one of the most robust results in cognitive psychology (Cepeda et al., 2006)"
- **Verdict: ACCURATE in substance, but the superlative is not Cepeda's.**
- Cepeda says spacing is robust. Results, spacing-effect section (about p. 359): "Only 12 of 271 comparisons of massed and spaced performance showed no effect or a negative effect from spacing, making the spacing effect quite robust." About p. 356: "Even though distributed practice benefits are robust, temporal moderators affect distributed practice through a complex interplay of time and task."
- Cepeda never says "one of the most robust results in cognitive psychology." That phrasing is usually traced to Dempster (1988), *American Psychologist, 43*(8), 627–634, https://doi.org/10.1037/0003-066X.43.8.627. Crossref confirms Dempster's bibliographic details; I could not retrieve his exact wording (**UNVERIFIED**).
- Fix: either soften to "a highly robust finding (Cepeda et al., 2006)" or add Dempster (1988).

**(b)** "*expanding-interval* schedules ... are shown to greatly improve memory in humans (Landauer & Bjork, 1978; Cepeda et al., 2006)"
- **Verdict: INACCURATE.** Cepeda et al. say close to the opposite.
- Section "Expanding Versus Fixed ISIs" (about pp. 364–365):
  - Only "22 comparisons of retention accuracy" were available.
  - "Overall, expanding ISIs led to better performance than fixed intervals ... Unfortunately, large standard errors, indicative of large between-study variability, make conclusions drawn from expanding versus fixed interval data necessarily tentative."
  - In studies with retention intervals of at least 1 day, Tsai (1927) favored expanding, Cull (2000) favored fixed, and Clark (1928) found no difference (Table 9).
- Table 8 (about p. 368), pooled over all retention intervals: expanding 62.0% vs. fixed 58.6% correct, **t(42) = 0.5, p = .61**. The difference is not significant.
- General Discussion (about p. 366): "Some researchers have suggested, **with little apparent empirical backing**, that expanding ISIs improve long-term learning (Hollingworth, 1913; Kitson, 1921; **Landauer & Bjork, 1978**; Pyle, 1913) ... Our review of the evidence suggests that, in general, expanding intervals either benefit learning or produce effects similar to studying with fixed spacing."
- So Cepeda supports spacing over massing. It does **not** support "expanding greatly improves memory," and it names Landauer & Bjork as an example of a claim with little empirical backing.
- Cepeda is also the closest human analogue to your design. Its expanding-vs-fixed comparisons are about **restudy** spacing. Your LLM "reviews" are re-exposures to the training statements, which is restudy, not retrieval.

**(c)** "The mechanisms behind human spacing—consolidation, retrieval difficulty, encoding variability (Cepeda et al., 2006)"
- **Verdict: PARTIALLY ACCURATE.**
- "Implications for Theories of Distributed Practice" (about pp. 368–370) discusses four theories: **deficient processing (inattention)**, **encoding variability**, **consolidation**, and **study-phase retrieval**.
- The conclusion of that section (about p. 370): "study-phase retrieval, consolidation, and encoding variability theories survive as candidate distributed practice theories, whereas deficient processing theory does not readily survive."
- "Consolidation" and "encoding variability" match Cepeda's terms. "Retrieval difficulty" does not. Cepeda's term is **study-phase retrieval**. "Retrieval difficulty" (desirable difficulty) is the Bjork / Karpicke & Roediger framing. Cepeda also calls these "candidate" theories with "little consensus." It does not present them as established mechanisms.
- Fix: "Candidate accounts of spacing include consolidation, study-phase retrieval, and encoding variability (Cepeda et al., 2006; see also Karpicke & Roediger, 2007, on retrieval difficulty)."

**Sources:** https://doi.org/10.1037/0033-2909.132.3.354 ; full text https://augmentingcognition.com/assets/Cepeda2006.pdf ; author page https://www.yorku.ca/ncepeda/publications/CPVWR2006.html

---

## 2. French (1999)

**Corrected APA 7 citation** (as given, correct):
> French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2

**Bibliographic check:** Crossref and PubMed (PMID 10322466) confirm the title, 3(4), pp. 128–135, April 1999, and the DOI. **CORRECT.**

**In-text use:** "as a model updates on new data, the representations that encoded earlier knowledge drift, and performance on the old task degrades (McCloskey & Cohen, 1989; French, 1999)"
- **Verdict: ACCURATE (with a wording caveat).**
- French's abstract says catastrophic forgetting occurs in distributed connectionist networks. It says the features that let these networks generalize (overlapping distributed representations) are "the root cause of catastrophic forgetting."
- "Representations drift" is a fair paraphrase of overlapping shared weights being overwritten. However, "representational drift" is a specific neuroscience term. A more exact phrasing: "weights shared with earlier knowledge are overwritten."
- Caveat (an internal point, not a citation error): in your own Table 1, old-fact loss in the no-review arm ends **below** its end-of-Stage-1 value (old-loss change −0.223). So there was no net catastrophic forgetting relative to Stage 1. The forgetting you measure is the rise during the buffer (+0.15). Framing the study with "catastrophic forgetting" may overstate the regime you tested.

**Sources:** https://doi.org/10.1016/S1364-6613(99)01294-2 ; https://pubmed.ncbi.nlm.nih.gov/10322466/

---

## 3. Groeneveld et al. (2024): OLMo

**Corrected APA 7 citation** (archival ACL 2024 version; APA 7 lists the first 19 authors, an ellipsis, then the last author):
> Groeneveld, D., Beltagy, I., Walsh, E., Bhagia, A., Kinney, R., Tafjord, O., Jha, A. H., Ivison, H., Magnusson, I., Wang, Y., Arora, S., Atkinson, D., Authur, R., Chandu, K. R., Cohan, A., Dumas, J., Elazar, Y., Gu, Y., Hessel, J., . . . Hajishirzi, H. (2024). OLMo: Accelerating the science of language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 15789–15809). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.acl-long.841

**Bibliographic check:**
- The arXiv:2402.00838 v4 author list has **43 authors**. The ACL Anthology and Crossref confirm publication at **ACL 2024 (Long Papers), pp. 15789–15809, DOI 10.18653/v1/2024.acl-long.841**. The paper won ACL 2024's Best Theme Paper Award.
- The 3rd author is listed as "Pete Walsh" on arXiv and as **"Evan Walsh"** in the ACL Anthology. For the archival version, use the Anthology form.
- "et al." inside a reference-list entry is not APA 7.
- Cite the ACL version instead of the arXiv version.

**In-text use:** "OLMo DataDecide 300M (... Magnusson et al., 2025; Groeneveld et al., 2024)"
- **Verdict: ACCURATE.** DataDecide §2.1: "We use OLMo's model ladder." The checkpoints load with `OLMoForCausalLM` from `hf_olmo` (the `ai2-olmo` package). Citing OLMo for the architecture and codebase is appropriate.
- **"OLMo DataDecide 300M" is not an official model name.** The suite is DataDecide, and the models are named by data recipe and size. See §8 for the recommended wording.

**Sources:** https://aclanthology.org/2024.acl-long.841/ ; https://arxiv.org/abs/2402.00838

---

## 4. Hu et al. (2021): LoRA

**Corrected APA 7 citation** (archival ICLR version):
> Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR 2022)*. https://openreview.net/forum?id=nZeVKeeFYf9

(Optionally add: arXiv preprint arXiv:2106.09685, https://arxiv.org/abs/2106.09685.)

**Bibliographic check:**
- The arXiv author list (8 authors, in this order) matches the paper's reference.
- OpenReview lists it as "ICLR 2022 Poster" (published 28 Jan 2022). DBLP has key `hu2022lowrank`.
- **Change the year to 2022 and the in-text citation to (Hu et al., 2022).**

**In-text use:** "LoRA (rank 16, α = 32, applied to all linear layers; Hu et al., 2021), which left 5.24M trainable parameters"
- **Verdict: ACCURATE** as a citation for the method. Two notes:
  1. The original LoRA paper mainly adapted only the attention projections (W_q, W_v). "All linear layers" is your configuration choice and should not read as if it came from Hu et al.
  2. I independently confirmed **5.24M**. With rank 16 on the 4 linear modules in each of 16 layers (att_proj 1024→3072, attn_out 1024→1024, ff_proj 1024→8192, ff_out 4096→1024):
     - per layer: 16 × (4096 + 2048 + 9216 + 5120) = 327,680
     - × 16 layers = **5,242,880** ✓ (output head excluded)

**Sources:** https://openreview.net/forum?id=nZeVKeeFYf9 ; https://arxiv.org/abs/2106.09685

---

## 5. Karpicke & Roediger (2007)

**Corrected APA 7 citation:**
> Karpicke, J. D., & Roediger, H. L., III. (2007). Expanding retrieval practice promotes short-term retention, but equally spaced retrieval enhances long-term retention. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 33*(4), 704–719. https://doi.org/10.1037/0278-7393.33.4.704

**Bibliographic check:** Crossref and the article PDF ("2007, Vol. 33, No. 4, 704–719") confirm all details. The only change is the APA 7 suffix: "Roediger, H. L., III." **CORRECT otherwise.**

### What the paper actually found (from the full text)

- **Exps. 1–2** (vocabulary word pairs; expanding 1–5–9 vs. equal 5–5–5, with the same mean spacing):
  - At **10 min**, expanding was better: Exp. 1 .71 vs. .62. In Exp. 2 (with feedback) it was .90 vs. .87, near ceiling; the authors call this a "modest short-term benefit."
  - At **2 days**, equal was better: Exp. 1 .45 vs. .33; Exp. 2 .60 vs. .49.
  - This held both with and without feedback.
- **Exp. 3** crossed the timing of the first test (immediate 0 vs. delayed 5) with the schedule of later tests (expanding vs. equal):
  - At 2 days there was a main effect of delaying the first test, F(1, 27) = 6.26, η²p = .19 (52% vs. 44%). There was **"no effect of the schedule of repeated tests"** (Fs < 1).
  - 2-day means: 0–1–5–9 .43; 0–5–5–5 .45; 5–1–5–9 .51; 5–5–5–5 .52.
- Abstract: "The important factor for promoting long-term retention is delaying initial retrieval to make it more difficult ... Expanding the interval between repeated tests had little effect on long-term retention in 3 experiments."
- About p. 716: "all previous experiments examining long-term retention on a delayed criterial test have showed benefits of equally spaced retrieval practice over expanding retrieval (Cull, 2000; Logan & Balota, in press; our present results)."

### In-text uses

**(a) Introduction:** "equally spaced retrieval can match or exceed expanding practice when memory is tested after a long delay (Karpicke & Roediger, 2007)"
- **Verdict: PARTIALLY ACCURATE (fair, but missing the key mechanism).**
- In the typical comparison (Exps. 1–2), equal spacing **exceeded** expanding at 2 days. It matched expanding only when the first-test delay was equated (Exp. 3).
- The authors' explanation is the **delay before the first retrieval**, not the spacing of later reviews.
- Also, "long delay" here means **2 days**.
- Suggested fix: "Equally spaced retrieval outperformed expanding retrieval at a 2-day delay, an advantage attributable to delaying the first retrieval; when the first retrieval was matched, the schedule of later retrievals had little effect (Karpicke & Roediger, 2007)."

**(b) Conclusion:** "that null mirrors the human literature, where equally spaced practice matches or beats expanding retrieval once memory is tested after a long delay (Karpicke & Roediger, 2007); the contribution here is showing that this human phenomenon reproduces in LLM continued training"
- **Verdict: OVERSTATED / PARTIALLY ACCURATE.** Four problems:
  1. **The human result in the closest comparison is "equal beats expanding," not a tie.** Your result is a tie.
  2. **There is a close structural match the paper misses.** In your code (`configs/final.yaml`: `first_step: 8`, `last_step: 165`), both review arms share the **same first and last review step**:
     - uniform: [8, 22, 37, 51, 65, 79, 94, 108, 122, 136, 151, 165]
     - expanding: [8, 10, 13, 17, 22, 29, 38, 51, 68, 91, 123, 165]

     This equates the first-review delay, which is exactly the variable K&R identified. So the right analogue is **K&R Exp. 3**, where the schedule of later reviews had no effect once the first test was matched. Saying this explicitly would strengthen the paper.
  3. **The paradigms differ.** K&R studied **retrieval practice** (cued-recall tests). Your "reviews" are **re-exposure / restudy** of declarative statements, with no retrieval attempt. Cepeda et al. (2006) on restudy spacing is the better-matched human comparison, and it also found no reliable expanding advantage (t(42) = 0.5, p = .61).
  4. **"Showing that this human phenomenon reproduces" overclaims a null result.** You have n = 3 seeds, one model, delays measured in update steps rather than time, and no equivalence test. "Is consistent with" is defensible; "reproduces" is not.
- The abstract's phrasing ("a human literature in which expanding retrieval, too, shows no reliable long-delay advantage over uniform practice") **is fair**, especially with two added citations:
  - Latimier et al. (2021) meta-analysis: expanding vs. uniform retrieval practice, **g = 0.034, n.s.**
  - Kang et al. (2014): equivalent recall at an **8-week** test after a 28-day training period.

**Sources:** https://doi.org/10.1037/0278-7393.33.4.704 ; full text https://learninglab.psych.purdue.edu/downloads/2007/2007_Karpicke_Roediger_JEPLMC.pdf

---

## 6. Kirchenbauer et al. (2025): FictionalQA

**Corrected APA 7 citation** (archival ICLR 2026 version):
> Kirchenbauer, J., Mongkolsupawan, N., Wen, Y., Goldstein, T., & Ippolito, D. (2026). FictionalQA: A dataset for studying memorization and knowledge acquisition. In *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=SUNC1eJGMr

(Optionally add: arXiv preprint arXiv:2506.05639, https://arxiv.org/abs/2506.05639.)

**Also cite the dataset itself with the pinned revision** (the experiment pins it):
> Kirchenbauer, J., Mongkolsupawan, N., Wen, Y., Goldstein, T., & Ippolito, D. (2026). *FictionalQA* [Data set; config `fict_qa`, revision 131cb74fdc3e601b5e896ed768ad9852ea35a8f9]. Hugging Face. https://huggingface.co/datasets/jwkirchenbauer/fictionalqa

**Bibliographic check:**
- **Title: CORRECT.** Exactly "FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition" on arXiv and OpenReview.
- Authors and order are correct. However, the 2nd author appears as **"Janny Mongkolsupawan"** on arXiv and as **"Natjanan Mongkolsupawan"** in the OpenReview/ICLR record and its BibTeX. For the archival citation, use **"Mongkolsupawan, N."**
- **Venue:** OpenReview shows "ICLR 2026 Poster" (venueid ICLR.cc/2026/Conference). arXiv v2 (2 Mar 2026) says "Published at ICLR 2026."
- **Change the year to 2026 and the in-text citation to (Kirchenbauer et al., 2026).** The dataset revision you pinned (131cb74…, last modified 2026-03-02) is the post-review version.

### Dataset structure (verified from the HF card, the paper, and the pinned parquet)

**Generation pipeline** (all GPT-4o-2024-08-06; one later MCQ subset uses gpt-5-mini):
- 100 **seed** events, each with one **fictsheet** (entities, events, locations, times, reasons).
- Each event has 15 **fiction** documents in 5 styles: news ×5, social ×3, corporate ×3, blog ×2, encyclopedia ×2. That gives 1,500 documents.
- Each document gets 5 Q&A items, for 7,500 `fict_qa` rows. Each row has:
  - `fict`, "a declarative form of the fictional fact"
  - `question`
  - `span_answer`
  - `natural_answer`
  - deduplication fields `duplicate_relationship` ∈ {null, exact, similar} and `duplicate_root`

**Duplicates / paraphrases:**
- In the pinned parquet: 4,326 "exact" duplicates, 1,377 "similar" (embedding-near) duplicates, and only **1,797 unique root questions** (about 18 per event).
- So the same fact is often asked several times across an event's documents, but these repeats are incidental, not designed paraphrase sets.
- The experiment keeps **one root question per cluster** (`question_id == duplicate_root`). So there is **one evaluation question per fact**, not multiple paraphrased variants.

**Splits:**
- The main repo has only a single `train` split.
- The paper's experiments use Event, Doc, and Style splits from a companion dataset, `jwkirchenbauer/fictionalqa_training_splits`.
- Your experiment uses **none of those**. It makes its own event-level old/new split: 20 old and 40 new events, seed 20260721.

**Are facts interrelated?**
- **Yes, within an event.** All documents and questions for an event come from one fictsheet and share entities (the paper's example: two questions both answered "Lake Ypsilon").
- Across events there is no shared universe by design, but the authors say documents "might overlap to a larger degree than desired across seed events" (§5).
- Your event-level old/new split is therefore the right choice. The paper should state it, as the repo README does.

### In-text uses

**(a)** "FictionalQA, a dataset of invented facts that cannot appear in the pretraining corpus" / "any measured knowledge of them must have been acquired during our fine-tuning"
- **Verdict: PARTIALLY ACCURATE.**
- The documents were generated in 2025, after DataDecide's Dolma 1.7 pretraining data was fixed, so verbatim contamination is essentially impossible.
- But FictionalQA explicitly provides `blind_answer_attempts` because some questions **can be answered without the document** (guessable years, plausible-sounding answers). The authors recommend filtering out questions with high blind-answer grades.
- Your pipeline does not filter on blind grades. Fix: say "facts absent from pretraining data, though some answers may be guessable," or filter by blind grade.

**(b)** Section 2.4: "measured on a held-out evaluation set—paraphrased questions about the facts that never appear in training—so the scores reflect retained knowledge rather than memorized training text"
- **Verdict: MISLEADING / PARTIALLY ACCURATE.**
- The code (`fictionalqa.py`) trains on `"Fictional fact: {fict}"` and evaluates on `"Question: {question}\nAnswer:"`, with loss on `natural_answer`.
- Each evaluation question is the **interrogative rewording of the very same `fict` statement the model was trained on**. Example from the pinned data:
  - fict: "The Ring of Silence Protocol was developed in 2046"
  - question: "In what year was the Ring of Silence Protocol developed?"
  - answer: "2046"
- In **93.8% of the 1,797 root items**, the `natural_answer` string appears verbatim inside the training `fict`.
- So the question wording is held out, but the facts and answer tokens are not. This tests **closed-book recall of a trained statement under a question template**, which is close to the "reversal / format-transfer" kind of memorization. It does not cleanly separate knowledge from memorized text.
- "Held-out evaluation set" also wrongly suggests a FictionalQA-provided held-out split.
- Suggested wording: "Each fact is trained in its declarative form (`fict`) and evaluated with the dataset's corresponding question (`question` → `natural_answer`). The question wording never appears in training, but the answer tokens do."
- Also report the evaluation size. The code evaluates `final_eval_examples_per_skill: 16` per style × 5 styles, so **about 80 old-fact questions** per evaluation. That is a subset of the roughly 360 old-fact root items. Please confirm.

### What the FictionalQA paper found that is relevant

- Models: Llama 3.1/3.2 and Gemma 1/2, trained with about 5% fiction mixed into Dolma webtext.
- With a 100% or 50% fiction mix: "pure verbatim memorization with no observable generalization period."
- Validation loss is U-shaped (a window where generalization happens).
- **Training on concise declarative fictsheets overfit almost immediately and transferred least**, "despite the fact that the model has memorized most" of them (§4.4).
- The authors conclude that the most concise, declarative form of a fact may not lead to the fastest knowledge acquisition.
- This bears directly on your design, which trains on short declarative `fict` statements at 100% fact data (no webtext mixing). By FictionalQA's own findings, that regime favors memorization over generalizable knowledge. Raise it as a limitation, and possibly as a reason the losses stay high (about 3.8 nats).
- They also report "leaky" transfer: performance improves on questions tied to held-out documents almost as much as on training-linked ones, because facts within an event are related.

**Sources:**
- https://arxiv.org/abs/2506.05639 ; https://arxiv.org/html/2506.05639v2
- https://openreview.net/forum?id=SUNC1eJGMr
- https://huggingface.co/datasets/jwkirchenbauer/fictionalqa
- pinned parquet https://huggingface.co/datasets/jwkirchenbauer/fictionalqa/resolve/131cb74fdc3e601b5e896ed768ad9852ea35a8f9/fict_qa/train-00000-of-00001.parquet

---

## 7. Landauer & Bjork (1978)

**Corrected APA 7 citation** (as given, with APA sentence case for the book title):
> Landauer, T. K., & Bjork, R. A. (1978). Optimum rehearsal patterns and name learning. In M. M. Gruneberg, P. E. Morris, & R. N. Sykes (Eds.), *Practical aspects of memory* (pp. 625–632). Academic Press.

**Bibliographic check:**
- Editors (Gruneberg, Morris, Sykes), pages 625–632, and Academic Press are consistent across many secondary citations: Memory & Cognition, Psychological Science, European Psychologist, Applied Psycholinguistics, and OALib.
- Most give the city as London; a few say New York. APA 7 omits the city anyway.
- The chapter has no DOI. I could not inspect the book itself, so the primary-source page numbers are **UNVERIFIED**, though the secondary sources agree.

**What they found:**
- Continuous paired-associate learning of fictitious first and last names.
- Expanding schedules (e.g., 1–4–10 intervening trials) were compared with equal schedules (5–5–5) at the same mean spacing.
- Karpicke & Roediger (2007, about p. 705) summarize: "on a final test 30 min after the learning phase, expanding retrieval practice produced about a 10% advantage over equally spaced retrieval practice."
- Logan & Balota (2008) estimated from the original figures about a 15% advantage during learning and about 8% on the final test.
- The effect was a **single-session, short-retention-interval** result. Cepeda et al. (2006, General Discussion, about p. 366) list it among claims made "with little apparent empirical backing" for **long-term** benefits.

**In-text use:** "expanding-interval schedules ... are shown to greatly improve memory in humans (Landauer & Bjork, 1978; Cepeda et al., 2006)"
- **Verdict: INACCURATE.** Landauer & Bjork showed a modest benefit (about 8–10 percentage points) on a test 30 minutes later, in one session. Later work found:
  - mixed results at short delays (about half of experiments favor expanding; Karpicke & Roediger, 2007, about pp. 715–716)
  - equal spacing better at delays of 24 hours or more (Cull, 2000; Logan & Balota, 2008; Karpicke & Roediger, 2007)
  - no difference at 8 weeks (Kang et al., 2014)
  - no overall difference in meta-analysis (Latimier et al., 2021, g = 0.034)
- Suggested fix: "Expanding schedules were proposed by Landauer and Bjork (1978), who found a modest advantage on a same-session test, and are the basis of common flashcard schedulers. Later work, however, shows no reliable long-term advantage over equal spacing (Cepeda et al., 2006; Karpicke & Roediger, 2007; Kang et al., 2014; Latimier et al., 2021)."
- This also removes a tension inside your own introduction: the paper currently says expanding schedules "greatly improve memory" and, one sentence later, that the question is "contested."

**Sources:** https://www.oalib.com/references/8099152 ; Karpicke & Roediger (2007) about p. 705 ; Logan & Balota (2008) https://doi.org/10.1080/13825580701322171

---

## 8. Magnusson et al. (2025): DataDecide

**Corrected APA 7 citation** (archival ICML 2025 version):
> Magnusson, I., Tai, N., Bogin, B., Heineman, D., Hwang, J. D., Soldaini, L., Bhagia, A., Liu, J., Groeneveld, D., Tafjord, O., Smith, N. A., Koh, P. W., & Dodge, J. (2025). DataDecide: How to predict best pretraining data with small experiments. In *Proceedings of the 42nd International Conference on Machine Learning* (Proceedings of Machine Learning Research, Vol. 267, pp. 42487–42502). PMLR. https://proceedings.mlr.press/v267/magnusson25a.html

(Optionally add: arXiv:2504.11393, https://arxiv.org/abs/2504.11393.)

**Bibliographic check:**
- The 13 authors and their order match arXiv and PMLR.
- **"Hwang, J." should be "Hwang, J. D."** (Jena D. Hwang).
- arXiv v2 says "ICML 2025." The PMLR page confirms Vol. 267, pp. 42487–42502.

### Model facts (verified from the HF API, config.json, and the paper)

- DataDecide releases **14 sizes** (4M through 1B), including **"300M"**, across **25 data recipes**. Examples: Dolma1.7, Dolma1.7 no-code / no-math-code / no-Reddit / no-Flan, Dolma1.6++, C4, FineWeb-Pro, FineWeb-Edu, Falcon, Falcon+CC (several QC variants), DCLM-Baseline (several QC variants), and DCLM/Dolma mixes. All 25 "300M" repos exist on HF.
- **300M config:**
  - d_model 1024, 16 layers, 16 heads, MLP ratio 8 (SwiGLU)
  - vocab 50,280 (embedding 50,304), **untied** input/output embeddings
  - 30B tokens, 45,787 steps, batch 320, LR 3.3e-3
  - The paper and model card list "Model size 320.0M." The paper says this counts non-embedding parameters. Arithmetically, the 16 transformer blocks alone are about 268.4M, so 320M is closer to blocks plus one 51.5M vocabulary matrix.
- **Actual total parameter count** (HF safetensors metadata): **371,491,840 (about 371.5M)**.
- **The paper's "approximately 377M total parameters" equals 371.49M base + 5.24M LoRA = 376.73M.** In other words, it counts the adapter. State "about 371M parameters (nominal '300M' DataDecide size), plus 5.24M LoRA parameters," or explicitly say "377M including the adapter."
- **Seeds:** for sizes below 1B, only the **default** seed is trained to completion. The two auxiliary seeds are stopped after about 25% of compute (§2.1).
  - Branches: `stepN-seed-default` (38 checkpoints, 0 to 45,787), `stepN-seed-small-aux-2`, `stepN-seed-small-aux-3`.
  - Your seeds 17/23/42 are LoRA/fine-tuning seeds, **not** DataDecide pretraining seeds. Say so, to avoid confusion.
- **The model actually used** (from the experiment code): `allenai/DataDecide-dolma1_7-300M`, pinned revision `4b1b42ff7c5224a077c4f5624824dd7c8bbe98d7`. I verified this commit is exactly the HF branch **`step45787-seed-default`**, the final checkpoint of the default-seed run on the **Dolma 1.7** recipe. Note it is *not* the current `main` commit (`e1a90917…`).
- Suggested Methods wording: "We used the DataDecide 300M model pretrained on the Dolma 1.7 recipe (`allenai/DataDecide-dolma1_7-300M`, final default-seed checkpoint `step45787-seed-default`, commit 4b1b42ff; ~371M parameters; OLMo architecture; Magnusson et al., 2025; Groeneveld et al., 2024)."

**In-text use:** the citation for the model, "OLMo DataDecide 300M (approximately 377M total parameters; Magnusson et al., 2025; Groeneveld et al., 2024)."
- **Verdict: PARTIALLY ACCURATE.** The source is correct, but the model is not fully specified, and the parameter count mixes in the adapter.

**Sources:**
- https://arxiv.org/abs/2504.11393 ; https://proceedings.mlr.press/v267/magnusson25a.html
- https://huggingface.co/allenai/DataDecide-dolma1_7-300M ; https://huggingface.co/allenai/DataDecide-dolma1_7-300M/blob/main/config.json
- HF API: `https://huggingface.co/api/models/allenai/DataDecide-dolma1_7-300M` (safetensors total) and `/refs` (branches)

---

## 9. McCloskey & Cohen (1989)

**Corrected APA 7 citation** (a chapter in a serial book; the original lacks the editor, book format, and DOI):
> McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), *Psychology of learning and motivation* (Vol. 24, pp. 109–165). Academic Press. https://doi.org/10.1016/S0079-7421(08)60536-8

**Bibliographic check:**
- Crossref confirms the title, *Psychology of Learning and Motivation*, pp. 109–165, 1989, and DOI 10.1016/S0079-7421(08)60536-8. Semantic Scholar and Illinois Experts agree.
- Editor G. H. Bower comes from secondary records (ORDB; the reference list in French, 1999). Crossref does not list the editor.
- French (1999) gives pp. 109–164. Most sources, including Crossref, say 109–165.
- The journal-style format the paper uses is acceptable, but add the DOI.

**In-text use:** the catastrophic forgetting sentence (see §2).
- **Verdict: ACCURATE.** The chapter shows that sequential training of backprop networks makes new learning interfere catastrophically with old learning, because new learning alters weights involved in representing old learning.

**Sources:** https://doi.org/10.1016/S0079-7421(08)60536-8 ; https://experts.illinois.edu/en/publications/catastrophic-interference-in-connectionist-networks-the-sequentia/

---

## Other checkable claims

| Claim | Verdict | Evidence / fix |
|---|---|---|
| "familiar from flashcard systems such as Anki" | ACCURATE but imprecise | The Anki manual says SM-2 grows intervals by about 2.5× (starting ease 250%), and FSRS (since 23.10) sets intervals from a target retention. Both are **adaptive**: intervals depend on whether each recall succeeds. Your expanding schedule is **fixed and non-adaptive** (ratio 1.35). Note the difference and cite the Anki manual (https://docs.ankiweb.net/deck-options.html). |
| "spacing effect ... one of the most robust results in cognitive psychology" | Substance ACCURATE; wording not Cepeda's | See §1(a); add Dempster (1988) or soften. |
| "the model has gone roughly 194 updates since it last saw an old fact" | ACCURATE | The last review is at Stage 2 step 165 (0-indexed) of 180, so 14 more Stage 2 updates plus 180 buffer updates = 194. |
| "each arm used ... around the same FLOPs" | Plausible | Review updates replace new-fact updates rather than adding steps. UNVERIFIED beyond the config. |
| "± values are standard deviations across seeds" | ACCURATE | `report.py` uses `statistics.stdev` (sample SD, n−1). |
| Paired CIs | Computed outside the repo | No CI code in `report.py`. See the numerical check below. |

---

## Numerical sanity check

For a paired t-interval with n = 3 (df = 2, t₀.₉₇₅ = 4.303), half-width = t · SD_d / √3. Solving for the implied SD of the paired differences:

| Comparison | Mean | CI | CI midpoint | Half-width | Implied SD_d (t, df=2) | SD_d if z = 1.96 had been used |
|---|---|---|---|---|---|---|
| Final old: U − N | −0.1727 | [−0.2140, −0.1314] | −0.1727 ✓ | 0.0413 | **0.0166** | 0.0365 |
| Final old: E − N | −0.1733 | [−0.2100, −0.1366] | −0.1733 ✓ | 0.0367 | **0.0148** | 0.0324 |
| Final old: E − U | −0.0006 | [−0.0053, +0.0041] | −0.0006 ✓ | 0.0047 | **0.0019** | 0.0042 |
| Buffer forgetting: U − N | +0.0097 | [−0.0718, +0.0911] | +0.00965 ✓ | 0.0815 | 0.0328 | 0.0720 |
| Buffer forgetting: E − N | +0.0066 | [−0.0663, +0.0794] | +0.00655 ✓ | 0.0729 | 0.0293 | 0.0644 |
| Buffer forgetting: E − U | −0.0031 | [−0.0118, +0.0057] | −0.00305 ✓ | 0.0088 | 0.0035 | 0.0077 |
| New loss: U − N | +0.0252 | [−0.0258, +0.0762] | +0.0252 ✓ | 0.0510 | 0.0205 | 0.0451 |
| New loss: E − N | +0.0216 | [−0.0329, +0.0760] | +0.02155 ✓ | 0.0545 | 0.0219 | 0.0481 |

**Are the t-based SDs plausible given Table 1?**
- **Final old loss, U − N:** marginal SDs are 0.031 (N) and 0.015 (U). The implied SD_d of 0.0166 requires a between-arm correlation across seeds of r ≈ 0.98. That is plausible, because each seed's three arms share the same Stage 1 checkpoint and data split.
- **Buffer forgetting, U − N:** the implied SD_d of 0.0328 against marginal SDs of 0.030 and 0.009 implies r ≈ −0.17. That is within the possible range (the maximum SD_d, at r = −1, is 0.039).
- With z instead of t, the final-old-loss U − N case would need SD_d = 0.0365, which implies r ≈ 0.12. Both scenarios fall within the possible range, so the CI widths alone cannot prove which critical value was used. All intervals are centered on the reported means, and the t(df = 2) implied SDs are internally plausible.
- **Recommendation:** state the method explicitly ("paired t-interval, df = 2, t = 4.303") and give the per-seed differences in a supplementary table. The per-seed CSV is produced by `summarize`.

**Relative reductions:**
- 0.1727 / 3.981 = **4.338%** → "4.34%" ✓
- 0.1733 / 3.981 = **4.353%** → "4.35%" ✓
- Table-level differences (3.981 − 3.809 = 0.172; 3.981 − 3.808 = 0.173) agree within rounding.

**Other internal consistency:**
- Pre-buffer + buffer forgetting = final loss for all three arms (3.830 + 0.151 = 3.981; 3.648 + 0.161 = 3.809; 3.650 + 0.158 = 3.808) ✓
- Implied end-of-Stage-1 loss (final − old-loss change) ≈ 4.204–4.205 in all arms ✓ (shared Stage 1 checkpoint)
- Joint loss: (3.981 + 2.993)/2 = 3.487 ✓; the gains of 0.074 / 0.076 match the text's 0.0737 / 0.0758 ✓
- Section 3.4 new-loss differences: 3.018 − 2.993 = 0.025 ✓; 3.015 − 2.993 = 0.022 ✓

**Caveat:** with n = 3, "statistically indistinguishable" for E − U rests on an implied SD_d of about 0.002. The interval is tight, but a 3-seed CI is fragile. Consider a TOST equivalence test against a stated margin (e.g., ±0.01 nats), or more seeds.

---

## Recommended additional references (verified via Crossref / publisher)

- Kang, S. H. K., Lindsey, R. V., Mozer, M. C., & Pashler, H. (2014). Retrieval practice over the long term: Should spacing be expanding or equal-interval? *Psychonomic Bulletin & Review, 21*(6), 1544–1550. https://doi.org/10.3758/s13423-014-0636-z
  - Equivalent recall at 8 weeks. Expanding gave higher *average* recall during training (.49 vs. .41), which parallels your "trajectory" result in §3.5.
- Latimier, A., Peyre, H., & Ramus, F. (2021). A meta-analytic review of the benefit of spacing out retrieval practice episodes on retention. *Educational Psychology Review, 33*(3), 959–987. https://doi.org/10.1007/s10648-020-09572-8
  - Spaced vs. massed g = 0.74; expanding vs. uniform g = 0.034, n.s.
- Logan, J. M., & Balota, D. A. (2008). Expanded vs. equal interval spaced retrieval practice: Exploring different schedules of spacing and retention interval in younger and older adults. *Aging, Neuropsychology, and Cognition, 15*(3), 257–280. https://doi.org/10.1080/13825580701322171
- Dempster, F. N. (1988). The spacing effect: A case study in the failure to apply the results of psychological research. *American Psychologist, 43*(8), 627–634. https://doi.org/10.1037/0003-066X.43.8.627 (the "robust" quote wording is UNVERIFIED)

Kang et al. (2014) is especially relevant. Its expanding schedule kept performance higher *during* training but gave the same final retention. That is exactly the pattern your §3.5 calls an "artifact" of front-loading. Citing it turns that observation into a replication of a known human pattern.

---

## Prioritized list of required corrections

1. **Remove or rewrite "expanding-interval schedules ... are shown to greatly improve memory in humans (Landauer & Bjork, 1978; Cepeda et al., 2006)."**
   - Cepeda et al. found no reliable expanding-vs-fixed difference (t(42) = 0.5, p = .61). They describe the long-term expanding claim, citing Landauer & Bjork, as having "little apparent empirical backing."
   - Landauer & Bjork's effect was about 10 points on a 30-minute test.
   - This sentence also contradicts the paper's next sentence.
2. **Fix the description of the evaluation set in §2.4.** It is not a held-out split or a set of paraphrases. Each evaluation item is the dataset's question form of a `fict` statement that **was trained on**, and the answer string appears verbatim in the training text 93.8% of the time. Remove "rather than memorized training text," or weaken it to "rather than verbatim continuation of training text." Report the evaluation size (16 per style × 5 = 80 items) and the event-level old/new split.
3. **Temper the conclusion's "mirrors the human literature ... this human phenomenon reproduces in LLM continued training."**
   - In K&R (2007), equal spacing *beat* expanding at 2 days, and the cause was the delay before the first retrieval.
   - Your design equates the first (and last) review, which matches K&R Exp. 3, where the schedule had no effect. Say this explicitly.
   - Note that your reviews are restudy, not retrieval practice.
   - Replace "reproduces" with "is consistent with," and add Latimier et al. (2021) and Kang et al. (2014).
4. **Correct the parameter count and model identity.** "377M" includes the 5.24M LoRA adapter; the base model has 371,491,840 parameters. Name the model fully: `allenai/DataDecide-dolma1_7-300M`, Dolma 1.7 recipe, checkpoint `step45787-seed-default` (commit 4b1b42ff). Drop "OLMo DataDecide 300M" as if it were an official name. Clarify that seeds 17/23/42 are fine-tuning seeds.
5. **Update the four arXiv references to their archival venues** (journals prefer these):
   - LoRA → ICLR 2022 (Hu et al., **2022**)
   - OLMo → ACL 2024, pp. 15789–15809, DOI 10.18653/v1/2024.acl-long.841
   - DataDecide → ICML 2025, PMLR 267:42487–42502
   - FictionalQA → ICLR 2026 (Kirchenbauer et al., **2026**)

   Update the in-text years to match.
6. **Fix the mechanisms sentence:** Cepeda's candidate theories are consolidation, **study-phase retrieval**, and encoding variability (plus deficient processing, which Cepeda rejects). "Retrieval difficulty" comes from Karpicke & Roediger / Bjork. Call these "candidate accounts," not established mechanisms.
7. **Fix author-name details:**
   - OLMo: list the first 19 authors + ". . . Hajishirzi, H." (no "et al." in APA 7); 3rd author "Walsh, E." per the ACL Anthology
   - DataDecide: "Hwang, J. D."
   - FictionalQA: "Mongkolsupawan, N." per ICLR/OpenReview (arXiv shows "Janny")
   - Karpicke & Roediger: "Roediger, H. L., III"
   - McCloskey & Cohen: add editor G. H. Bower, "Vol. 24," and DOI 10.1016/S0079-7421(08)60536-8
8. **Spacing-effect superlative:** soften, or add Dempster (1988).
9. **FictionalQA caveats in Methods / Limitations:**
   - Some questions are answerable blind (the dataset ships blind-answer grades; consider filtering).
   - FictionalQA's own finding that training on concise declarative facts produces memorization and poor transfer applies directly to training on `fict` statements at 100% fact data.
10. **Anki:** note that Anki's intervals are adaptive (SM-2 / FSRS), unlike the fixed schedule tested here, and cite the Anki manual.
11. **Statistics reporting:**
    - State the CI method (paired t, df = 2) and include per-seed differences.
    - Consider an equivalence test for the expanding-vs-uniform null.
    - Reconsider the "catastrophic forgetting" framing, since the no-review arm's old-fact loss ends below its post-Stage-1 level.
12. **Cite the dataset and model artifacts with pinned revisions** (FictionalQA `fict_qa` @ 131cb74…; DataDecide @ 4b1b42ff…) for reproducibility.
