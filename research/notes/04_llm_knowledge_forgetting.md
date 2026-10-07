# Literature Review 04: How Language Models Acquire, Store, and Forget Factual Knowledge; Catastrophic Forgetting in LLMs

## TL;DR

- **Old-fact loss falling during new-fact training is expected.** The FictionalQA authors themselves report "leaky" gains on never-trained events. Part of our metric is format/domain adaptation, not fact memory. We need a never-trained control, like O'Neill's (2026) drift control.
- **The facts are only weakly learned.** About 3.8 nats per answer token is roughly a 2% token probability. Each fact gets few exposures (about 3.5 in Stage 1), while robust storage needs about 100–1000 with paraphrases (Allen-Zhu & Li). We are studying retention of *partial* traces.
- **Loss and behaviour can diverge**, so report exact match (already logged) and a multiple-choice or likelihood-margin metric.
- **LoRA "learns less and forgets less"** (Biderman et al. 2024), which may compress schedule effects. Add a full fine-tuning check.
- **The constant learning rate is justified** (Luo et al., ICLR 2026), but it limits generality. Add a short-cooldown robustness check.
- **Controls reviewers will expect:**
  - a never-trained set;
  - a pre-training baseline;
  - behavioural metrics;
  - per-fact analysis;
  - a generic-data arm (REMIX, Chen et al. 2026);
  - a relearning probe;
  - an exposure table.

  Every entry lists the model sizes studied. Many knowledge-injection results come only from 7B+ models.


Prepared for: "Does Spaced Review Help a Language Model Retain Facts During Continued Training?" (Mago & Wang). Setup: LoRA r=16 on OLMo DataDecide 300M, FictionalQA, seq len 64, batch 16, constant LR 1e-4, answer-token CE on paraphrased held-out questions. Key results: review lowers delayed old-fact loss by about 4.3%; expanding and uniform review come out the same. Two features of the results matter here. (i) Old-fact loss falls from about 4.2 to about 3.83 during Stage 2 even with no review. (ii) Absolute losses stay high, about 3.8 nats per answer token, which is a per-token probability of about e^-3.8 ≈ 2.2%.

How sources were checked (2 Oct 2026). Every entry below was checked against at least one primary source: an arXiv abstract page or the arXiv API record, OpenReview, ACL Anthology, PMLR, NeurIPS/ICLR proceedings, a publisher DOI page, or the paper's own PDF. I pulled the full text of most arXiv papers and searched it for model sizes and key claims. When a detail could not be checked, the entry says **UNVERIFIED**. Quoted numbers come from the paper's own text unless marked otherwise. "Model scale" gives the models the paper actually studies. **Scale caveat** marks findings that come only from models much larger than ours (≥7B) or from non-LM settings.

---

## 1. Fact acquisition dynamics and knowledge injection

### 1.1 Chang et al. (2024): How LLMs acquire factual knowledge during pretraining
- **Citation (APA):** Chang, H., Park, J., Ye, S., Yang, S., Seo, Y., Chang, D.-S., & Seo, M. (2024). How do large language models acquire factual knowledge during pretraining? In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. https://arxiv.org/abs/2406.11813 (secondary sources give pages 60626–60668; I did not check these against the proceedings).
- **Model scale:** Continued pretraining from intermediate OLMo-1B and OLMo-7B checkpoints. Fictional knowledge is injected into the regular pretraining batches.
- **Summary:** The authors inject fictional-knowledge passages during OLMo pretraining. Each injection is one of three types: duplicated, paraphrased, or shown once. Afterward they probe the log-probability of the target spans. Facts are acquired through small "micro-acquisitions" at each encounter, and each gain is then partly forgotten. Forgetting follows a power law in training steps, for both memorization and generalization probes. Duplicated data is forgotten faster than paraphrased data. Larger batch sizes and deduplication make knowledge more robust to forgetting. Training on more data does not significantly improve acquisition. The paper also proposes a **"learnability threshold"**: an interval between encounters beyond which a fact is never learned. Smaller batches shorten this threshold. The authors repeat key analyses at a **constant learning rate** to rule out LR decay as a confound.
- **Relation to our paper:** This is the closest LM analogue to "review interval" effects. The power-law forgetting curve implies most loss happens soon after an exposure. That fits our finding that review mostly changes the starting point, with a buffer-forgetting rate that is similar across arms. The learnability-threshold idea gives a mechanistic reason why *spacing* could matter in principle: an interval longer than the threshold should fail. It also suggests that 12 reviews at intervals of ≤~30 steps may all sit below the threshold, which would make expanding and uniform equivalent. Our batch size of 16 is small; Chang et al. predict that this hurts retention. Their probes measure log-probability *relative to the pre-injection value*, which supports adding a no-exposure baseline (see Section 6). They also used a constant-LR control, which supports our design choice.

### 1.2 Allen-Zhu & Li (2024): Physics of Language Models, Part 3.1 (Knowledge Storage and Extraction)
- **Citation (APA):** Allen-Zhu, Z., & Li, Y. (2024). Physics of language models: Part 3.1, knowledge storage and extraction. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235, pp. 1067–1077). https://proceedings.mlr.press/v235/allen-zhu24a.html (arXiv: https://arxiv.org/abs/2309.14316)
- **Model scale:** Trained from scratch. GPT-2 with rotary embeddings: 124M (12L/12H/768d) for bioS, 302M (12L/20H/1280d) for bioR, and 682M for one negative result. Llama-architecture variants were also tested.
- **Summary:** The authors train on synthetic biographies and test QA extraction on held-out individuals. Knowledge is reliably *extractable* by questions only when the pretraining data contains enough augmentation (multiple paraphrased biographies, sentence shuffling). Without augmentation the model memorizes the biography text but extracts it with near-0% accuracy, even after QA fine-tuning. Linear probing suggests that augmentation makes the attribute information be stored on the entity-name tokens.
- **Relation to our paper:** This is directly relevant to evaluating on paraphrased questions. If the training data shows each fact in one or few surface forms, cross-entropy on paraphrased questions may barely move, which matches our high absolute losses. The paper should say how many surface forms per fact appear in training (FictionalQA offers 5 document styles per event). Our model size (~300M) is in the range they show is enough to store such knowledge, so model capacity probably does not explain the high loss. Data augmentation and exposure count are more likely explanations.

### 1.3 Allen-Zhu & Li (2025): Physics of Language Models, Part 3.2 (Knowledge Manipulation)
- **Citation (APA):** Allen-Zhu, Z., & Li, Y. (2025). Physics of language models: Part 3.2, knowledge manipulation. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2309.14402 (ICLR page: https://iclr.cc/virtual/2025/poster/28369)
- **Model scale:** Small GPT-2/Llama/Mistral-architecture models trained from scratch on synthetic biographies. The paper also tests GPT-4/4o behaviorally.
- **Summary:** Models retrieve stored attributes well. Without chain-of-thought they fail at classification and comparison over stored knowledge, and inverse search is about 0% (a reversal-curse-like effect).
- **Relation to our paper:** This is mainly a caution for test design. Paraphrased questions must ask in the *same direction* as the training text (entity → attribute). Questions that require inversion or comparison would sit near floor no matter what retention is.

### 1.4 Allen-Zhu & Li (2025): Physics of Language Models, Part 3.3 (Knowledge Capacity Scaling Laws)
- **Citation (APA):** Allen-Zhu, Z., & Li, Y. (2025). Physics of language models: Part 3.3, knowledge capacity scaling laws. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2404.05405
- **Model scale:** GPT-2 (rotary), Llama, and Mistral-style models trained from scratch, from very small sizes up to roughly 1B parameters (e.g., GPT2-20-16). Llama/Mistral comparisons are included.
- **Summary:** LMs store about **2 bits of knowledge per parameter** when each fact gets **1000 exposures** (this holds under int8 quantization). With **100 exposures**, an undertrained GPT-2 reaches only about **1 bit/param**. "Junk" data lowers capacity for useful knowledge sharply; for example, a 1:7 ratio of useful to junk tokens costs a large factor. Prepending a domain/source tag mostly recovers the loss.
- **Relation to our paper:** The paper says knowledge is "acquired" after 60 Stage-1 updates × 16 sequences = 960 sequences. Unless the fact set is tiny, each fact gets far fewer than 100 exposures. The model is then deep in the under-exposed regime, where storage is weak and fragile. This plausibly explains why losses stay around 3.8 nats. Reviewers may conclude that we are studying *partially formed* traces rather than retention of learned facts. Reporting exposures per fact for each stage would address this. The capacity numbers are not a problem for us: 5.24M LoRA parameters × 2 bits is roughly 10 Mbit, far more than a FictionalQA subset needs. The bottleneck is exposures, not capacity. One caveat: their capacity results are for full-parameter training from scratch; the LoRA case is untested.

### 1.5 Zucchet et al. (2025): How do language models learn facts? Dynamics, curricula and hallucinations
- **Citation (APA):** Zucchet, N., Bornschein, J., Chan, S., Lampinen, A. K., Pascanu, R., & De, S. (2025). How do language models learn facts? Dynamics, curricula and hallucinations. In *Second Conference on Language Modeling (COLM 2025)*. https://arxiv.org/abs/2503.21676 (OpenReview: https://openreview.net/forum?id=vBcGnragkr)
- **Model scale:** 44M-parameter transformers trained from scratch on synthetic biographies. The authors say their qualitative results are robust to model size, LR, batch size, and other settings.
- **Summary:** Fact learning happens in three phases. A plateau comes before precise recall, and it coincides with the formation of attention-based recall circuits. Imbalanced (e.g., Pareto) data distributions shorten the plateau, and data-scheduling curricula help. Hallucinations appear together with knowledge. Fine-tuning on new individuals quickly corrupts existing parametric memories. **Mixing replay of old individuals into fine-tuning only partly mitigates this**: it helps the final drop but not the initial collapse.
- **Relation to our paper:** Two points. (a) Their finding that replay mitigates forgetting only partially matches our modest 4.3% effect. (b) The recall-circuit account offers one mechanism for the *falling old-fact loss during Stage 2 without review*. Continued QA-style training on new facts may keep building general fact-retrieval machinery (attention circuits and output-format priors), and that machinery also lowers loss on old facts. This is a hypothesis, not something they test on our setup. It argues for a never-trained control set (Section 6). The 44M scale is below ours, but the work is a controlled small-model study close to our regime.

### 1.6 Kandpal et al. (2023): Large language models struggle to learn long-tail knowledge
- **Citation (APA):** Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. (2023). Large language models struggle to learn long-tail knowledge. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 15696–15707). https://proceedings.mlr.press/v202/kandpal23a.html
- **Model scale:** BLOOM family 560M–176B and other public families, plus a 4.8B-parameter counterfactual retraining experiment.
- **Summary:** QA accuracy on a fact rises strongly, both correlationally and causally, with the number of pretraining documents relevant to it. Long-tail facts are learned poorly. Larger models help, but the authors estimate that orders-of-magnitude scaling would be needed for rare facts.
- **Relation to our paper:** Fictional facts are long-tail by construction: every fact has very few "relevant documents." This supports the reading that our facts are weakly learned. It also suggests review raises effective document count, which may be why review helps at all.

### 1.7 Jiang et al. (2024): Instruction-tuned language models are better knowledge learners
- **Citation (APA):** Jiang, Z., Sun, Z., Shi, W., Rodriguez, P., Zhou, C., Neubig, G., Lin, X. V., Yih, W.-t., & Iyer, S. (2024). Instruction-tuned language models are better knowledge learners. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 5421–5434). https://doi.org/10.18653/v1/2024.acl-long.296
- **Model scale:** Llama-2 7B and 70B. **Scale caveat:** nothing below 7B.
- **Summary:** Continued pretraining on new documents (Wikipedia articles on 2023 films) followed by QA tuning leaves a gap between low document perplexity and limited QA accuracy. "Pre-instruction-tuning" (PIT) trains on QA pairs *before* the documents and improves QA accuracy by about 17.8 points on Llama-2-7B (30.3% → 48.1%).
- **Relation to our paper:** The paper documents a perplexity-versus-QA gap; low document loss does not mean extractable knowledge. This supports reporting both training-text loss and QA-format metrics. Its order effect between QA-format and document data is also a confound to check: if our Stage 2 new-fact data is QA-formatted, it could teach the *access pattern*, and that would lower old-fact QA loss without any old-fact-specific learning.

### 1.8 Yang et al. (2025): Synthetic continued pretraining (EntiGraph)
- **Citation (APA):** Yang, Z., Band, N., Li, S., Candès, E., & Hashimoto, T. (2025). Synthetic continued pretraining. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)* (Oral). https://arxiv.org/abs/2409.07431 (proceedings PDF: https://proceedings.iclr.cc/paper_files/paper/2025/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf)
- **Model scale:** Llama 3 8B continually pretrained on 455M synthetic tokens (made from 1.3M real tokens from QuALITY), with replay of RedPajama. **Scale caveat:** ≥8B only.
- **Summary:** A small corpus is too little data to learn from directly. EntiGraph extracts entities and generates diverse texts about entity relations. Continued pretraining on this synthetic corpus improves closed-book QA, and the gains compound with RAG. A theoretical model explains how synthetic augmentation "rearranges" knowledge.
- **Relation to our paper:** This is more evidence that diverse restatements are needed for paraphrase-robust knowledge (it matches Part 3.1). It suggests training-data diversity per fact as a variable that may interact with review. Reviewing an *alternative phrasing* rather than the identical sequence may matter more than the schedule.

### 1.9 Ovadia et al. (2024): Fine-tuning or retrieval? Comparing knowledge injection in LLMs
- **Citation (APA):** Ovadia, O., Brief, M., Mishaeli, M., & Elisha, O. (2024). Fine-tuning or retrieval? Comparing knowledge injection in LLMs. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 237–250). https://doi.org/10.18653/v1/2024.emnlp-main.15
- **Model scale:** Llama-2-7B, Mistral-7B, Orca-2-7B. This is at the upper bound of the scale range we care about.
- **Summary:** Unsupervised fine-tuning on documents injects knowledge only modestly; RAG consistently does better, for both previously seen and new knowledge. Exposure to numerous paraphrases of the same fact during training improves fine-tuning-based injection.
- **Relation to our paper:** This supports our finding that fine-tuning injects facts weakly (high loss). It also supports paraphrase augmentation as a stronger lever than scheduling, and that is worth a sentence in the discussion.

### 1.10 Gekhman et al. (2024): Does fine-tuning LLMs on new knowledge encourage hallucinations?
- **Citation (APA):** Gekhman, Z., Yona, G., Aharoni, R., Eyal, M., Feder, A., Reichart, R., & Herzig, J. (2024). Does fine-tuning LLMs on new knowledge encourage hallucinations? In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 7765–7784). https://doi.org/10.18653/v1/2024.emnlp-main.444
- **Model scale:** PaLM 2-S base model (parameter count undisclosed; the paper's PDF says "PaLM 2-S"). **Scale caveat:** proprietary, probably much larger than 300M.
- **Summary:** In closed-book QA fine-tuning, examples carrying *new* knowledge are learned much more slowly than examples consistent with existing knowledge. Once they are learned, they linearly increase hallucination on other questions. The authors conclude that models mostly acquire facts in pretraining and that fine-tuning teaches them to use those facts.
- **Relation to our paper:** All FictionalQA facts are "Unknown" by design, so learning is expected to be slow. This fits the high losses. Their method of sorting examples by knowledge category and tracking per-example learning also points toward per-fact analysis instead of only averaged loss.

### 1.11 Berglund et al. (2024): The reversal curse
- **Citation (APA):** Berglund, L., Tong, M., Kaufmann, M., Balesni, M., Stickland, A. C., Korbak, T., & Evans, O. (2024). The reversal curse: LLMs trained on "A is B" fail to learn "B is A". In *The Twelfth International Conference on Learning Representations (ICLR 2024)*. https://arxiv.org/abs/2309.12288 (venue per citing literature and the authors' repository; I did not find a proceedings page directly, so treat the venue as **partly UNVERIFIED**)
- **Model scale:** GPT-3 (hyperparameter sweep on GPT-3-350M, then larger GPT-3 models) and Llama-1 7B fine-tuned on fictitious statements. GPT-3.5/GPT-4 evaluated on real celebrities.
- **Summary:** After fine-tuning on "A is B," models do not assign higher likelihood to A when prompted with B than to a random name. Data augmentation does not fix this.
- **Relation to our paper:** Same lesson as Part 3.2. FictionalQA questions must match the direction in which facts appear in training documents, or the paraphrase metric will sit at floor. This is a possible reason for losses around 3.8.

### 1.12 Mecklenburg et al. (2024): Injecting new knowledge into LLMs via supervised fine-tuning
- **Citation (APA):** Mecklenburg, N., Lin, Y., Li, X., Holstein, D., Nunes, L., Malvar, S., Silva, B., Chandra, R., Aski, V., Yannam, P. K. R., Aktas, T., & Hendry, T. (2024). *Injecting new knowledge into large language models via supervised fine-tuning* [Preprint]. arXiv. https://arxiv.org/abs/2404.00213
- **Model scale:** GPT-4, fine-tuned with LoRA (**rank 16**, batch size 1, 3 epochs). **Scale caveat:** a frontier-scale model.
- **Summary:** SFT on Q&A about recent sporting events improves factual accuracy. "Fact-based" data scaling (making sure each fact gets coverage) gives more uniform knowledge acquisition than "token-based" scaling.
- **Relation to our paper:** This supports reporting and controlling *per-fact coverage*. It matters for seq-len 64: if FictionalQA documents are truncated to 64 tokens, some facts may never appear in training, and their "retention" would measure only generic drift. Mecklenburg et al. also use the same LoRA rank as we do.

### 1.13 Kirchenbauer et al. (2026): FictionalQA (the dataset we use)
- **Citation (APA):** Kirchenbauer, J., Mongkolsupawan, J., Wen, Y., Goldstein, T., & Ippolito, D. (2026). FictionalQA: A dataset for studying memorization and knowledge acquisition. In *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. https://arxiv.org/abs/2506.05639 (OpenReview: https://openreview.net/forum?id=SUNC1eJGMr). *Note: our paper cites the arXiv version, but it is now published at ICLR 2026, so the citation should be updated.*
- **Model scale:** Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, and Gemma 1/2 base models. Fine-tuned on 5% fictional and 95% Dolma webtext.
- **Summary:** Fictional documents in 5 webtext styles are generated from "fictsheets," along with QA pairs and 4-way MCQs. Training splits hold out events, documents, or styles. **Key finding for us:** MCQ accuracy gains are "**leaky**". Questions about **held-out events never trained on** also improve, only slightly less than questions about trained events. The authors therefore conclude that gains "are [not] attribute-able to either purely factual or purely stylistic memorization," but to "some combination of distributional and atomic learning." Training on fictsheets gives the *least* transfer to QA even though they are memorized (near-zero loss). Diverse document styles transfer best. The authors also report MCQ artifacts: models score above chance at step 0, and some distractors are paraphrases of the answer. They also report that fictional documents overlap across seed events more than intended.
- **Relation to our paper:** This is **the single most important threat to the paper's framing.** Our paper says: "Because the facts are invented, any measured knowledge of them must have been acquired during our fine-tuning… this is what makes the retention signal clean." The dataset authors show that training on *some* FictionalQA facts lowers loss and raises accuracy on *other, never-trained* FictionalQA questions. The likely causes are style and domain adaptation plus cross-event overlap. That gives a direct explanation for **old-fact loss falling during Stage 2 with no review**: Stage-2 new facts come from the same distribution and so lower loss on old-fact questions too. It also means part of the 4.3% review effect could be generic exposure to more FictionalQA-style data. In our design, though, review arms swap new-fact updates for old-fact updates at equal steps, so this confound is partly controlled. The paper needs (i) a **never-trained FictionalQA control set** evaluated at every checkpoint, and (ii) results reported as the difference from that control. The FictionalQA authors also used a "Base Webtext only" control to rule out training artifacts; we should add the same.

### 1.14 Zhang, Li & Wu (2024): Co-occurrence is not factual association in language models
- **Citation (APA):** Zhang, X., Li, M., & Wu, J. (2024). Co-occurrence is not factual association in language models. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. https://arxiv.org/abs/2409.14057
- **Model scale:** Full fine-tuning of LLaMA 3 8B and Gemma 7B; LoRA on LLaMA 3 70B. **Scale caveat:** ≥7B.
- **Summary:** When fine-tuned on new facts, LMs tend to learn *word co-occurrence statistics* (stored in middle layers) rather than true factual associations (stored in lower layers). Co-occurrence knowledge supports simple QA but generalizes poorly. Training on implicit associations, or actively forgetting co-occurrence statistics, improves generalization.
- **Relation to our paper:** Answer-token CE on paraphrased questions can drop because answer tokens co-occur with question entities, without real factual association. A reviewer could argue that our loss changes partly reflect co-occurrence statistics. Probes that need more than co-occurrence, such as distractors that share entity overlap, would address this.

### 1.15 Kim et al. (2025): Knowledge entropy decay during pretraining hinders new knowledge acquisition
- **Citation (APA):** Kim, J., Lee, H., Cho, H., Jang, J., Hwang, H., Won, S., Ahn, Y., Lee, D., & Seo, M. (2025). Knowledge entropy decay during language model pretraining hinders new knowledge acquisition. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)* (Oral). https://arxiv.org/abs/2410.01380
- **Model scale:** OLMo 1B and 7B checkpoints across pretraining.
- **Summary:** "Knowledge entropy," a measure of how broadly a model uses its FFN memory sources, declines over pretraining. Later checkpoints acquire *and retain* new (fictional) knowledge worse. Reactivating inactive memory sources helps.
- **Relation to our paper:** Acquisition and retention depend on *which checkpoint* is fine-tuned. We should state which DataDecide 300M checkpoint and data recipe we use (final? which corpus?) and acknowledge that results may differ by checkpoint. This study also reuses the Chang et al. fictional-knowledge probes, which makes it a natural comparison point.

### 1.16 Ou et al. (2025): How do LLMs acquire new knowledge? A knowledge circuits perspective on continual pre-training
- **Citation (APA):** Ou, Y., Yao, Y., Zhang, N., Jin, H., Sun, J., Deng, S., Li, Z., & Chen, H. (2025). How do LLMs acquire new knowledge? A knowledge circuits perspective on continual pre-training. In *Findings of the Association for Computational Linguistics: ACL 2025*. https://arxiv.org/abs/2502.11196
- **Model scale:** GPT-2 Small (124M), GPT-2 Medium (355M), TinyLlama (1.1B), Phi-1.5 (1.3B). Continued pretraining on fictional biographies. **In our scale range.**
- **Summary:** New-knowledge acquisition depends on how relevant the new facts are to existing knowledge. Knowledge circuits go through a phase shift from *formation* to *optimization*, and they evolve from deep layers to shallow ones.
- **Relation to our paper:** This is a small-model, fictional-fact continued-pretraining study close to our regime. Its formation-then-optimization picture is consistent with old-fact performance continuing to improve after Stage 1, because circuits keep being optimized during Stage 2 even when the training facts differ.

### 1.17 Ghosal, Hashimoto & Raghunathan (2024): Understanding finetuning for factual knowledge extraction
- **Citation (APA):** Ghosal, G. R., Hashimoto, T., & Raghunathan, A. (2024). Understanding finetuning for factual knowledge extraction. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235, pp. 15540–15558). https://proceedings.mlr.press/v235/ghosal24a.html
- **Model scale:** Llama-2-7B, Mistral-7B, plus theory.
- **Summary:** QA fine-tuning on *lesser-known* facts makes downstream factuality worse by 5–10%, even though all facts were seen in pretraining. The model learns to ignore the subject entity and output generic plausible answers.
- **Relation to our paper:** Our facts are all "lesser-known," so the model may learn subject-agnostic answer priors. Those priors lower CE on *all* fictional questions, old and new, without fact-specific knowledge. This is a mechanism for the Stage-2 drop in old-fact loss. A never-trained control or a shuffled-subject control would detect it.

### 1.18 O'Neill (2026): Can a language model learn facts continually in its weights?
- **Citation (APA):** O'Neill, C. (2026). *Can a language model learn facts continually in its weights?* [Preprint]. arXiv. https://arxiv.org/abs/2607.11020
- **Model scale:** Qwen3-4B with LoRA adapters by default; replicated at 4B/8B and with full fine-tuning.
- **Summary:** The paper writes invented facts sequentially (20–100 later writes) and evaluates with held-out questions of five types. Bare-statement training produces recitation. Diverse "study" data narrows the recitation-to-use gap from 27.4 to 5.4 points. After 20 later writes, bare-statement facts keep 1% accuracy and study-data facts keep 46%. **Facts can be behaviorally forgotten without being erased.** Facts that fail every question still keep 57–67% of their write's log-probability lift *after a drift correction*. The correction is "a drift control, the same quantity computed on **never-written statements**, [which] corrects for the elevation that any statement receives from the surrounding writes." Supplying a forgotten fact in context recovers 77–80%.
- **Relation to our paper:** This is the most methodologically relevant recent paper, and it uses LoRA at ≤8B. (1) It directly uses the **never-trained control** we recommend, and it finds that surrounding fact-writes *raise* likelihood on statements never written. That is our Stage-2 old-fact loss drop. (2) It shows that log-probability retention and behavioral (accuracy) retention can diverge sharply. Reviewers will want accuracy or MCQ metrics alongside CE. (3) It shows that training-data diversity matters more for retention than other factors. **Status:** preprint, not peer-reviewed.

### 1.19 Lv et al. (2025): Knowledge infusion scaling law for pre-training LLMs (supplementary)
- **Citation (APA):** Lv, K., Chen, H., Yuan, Y., Liu, L., Liu, S., Wang, Y., Su, W., & Zheng, B. (2025). *How to inject knowledge efficiently? Knowledge infusion scaling law for pre-training large language models* [Preprint]. arXiv. https://arxiv.org/abs/2509.19371
- **Model scale:** Multiple sizes (exact range **UNVERIFIED**; I read only the abstract).
- **Summary:** Over-infusing domain knowledge causes "memory collapse." Each model has a critical collapse point beyond which retention degrades sharply, and that point scales with model size.
- **Relation to our paper:** Minor. Retention capacity depends on model size and on how much is injected. This supports reporting the number of facts relative to model size.

---

## 2. Memorization and forgetting of training examples

### 2.1 Tirumala et al. (2022): Memorization without overfitting
- **Citation (APA):** Tirumala, K., Markosyan, A. H., Zettlemoyer, L., & Aghajanyan, A. (2022). Memorization without overfitting: Analyzing the training dynamics of large language models. In *Advances in Neural Information Processing Systems 35* (pp. 38274–38290). https://arxiv.org/abs/2205.10770 (proceedings PDF: https://proceedings.neurips.cc/paper_files/paper/2022/file/fa0509f4dab6807e2cb465715bf2d249-Paper-Conference.pdf)
- **Model scale:** Causal and masked LMs at 125M, 355M, 1.3B, 2.7B, 6.7B, and 13B.
- **Summary:** Larger models memorize faster and can memorize more before overfitting. Forgetting curves level off at a "**forgetting baseline**" that rises with model scale, meaning bigger models forget less. Nouns and numbers are memorized first.
- **Relation to our paper:** Forgetting of memorized content has a floor and depends on scale. At 300M our baseline should be relatively low. The finding that "nouns/numbers first" also bears on answer tokens: fictional answers are often names or numbers, which may be the first tokens learned and also the most fragile. (A separate agent covers scale effects.)

### 2.2 Jagielski et al. (2023): Measuring forgetting of memorized training examples
- **Citation (APA):** Jagielski, M., Thakkar, O., Tramèr, F., Ippolito, D., Lee, K., Carlini, N., Wallace, E., Song, S., Thakurta, A., Papernot, N., & Zhang, C. (2023). Measuring forgetting of memorized training examples. In *The Eleventh International Conference on Learning Representations (ICLR 2023)*. https://openreview.net/forum?id=7bJizxLKrR (arXiv: https://arxiv.org/abs/2207.00099)
- **Model scale:** 110M-parameter decoder-only LMs (T5 codebase), plus image and speech models.
- **Summary:** Using privacy attacks as the measure, standard models empirically *forget* specific training examples over continued training, even though worst-case non-convex models could memorize forever. Examples seen early in large-data training are less exposed. The paper points to nondeterminism in training as a cause; deterministic training does not forget.
- **Relation to our paper:** Same small-model regime as ours. It supports the expected decay of old facts during the buffer, and it gives a precedent for measuring forgetting as the change in a per-example likelihood signal.

### 2.3 Biderman et al. (2023): Emergent and predictable memorization in LLMs
- **Citation (APA):** Biderman, S., Prashanth, U. S., Sutawika, L., Schoelkopf, H., Anthony, Q., Purohit, S., & Raff, E. (2023). Emergent and predictable memorization in large language models. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. https://doi.org/10.52202/075280-1219 (arXiv: https://arxiv.org/abs/2304.11158)
- **Model scale:** Pythia suite, 70M–12B.
- **Summary:** Small models or early checkpoints predict which sequences a larger model or the final checkpoint will memorize with high precision but low recall. Small models are poor at forecasting memorization at larger scale.
- **Relation to our paper:** This is a caution against extrapolating per-sequence retention from 300M to larger models, so it supports the paper's own scale caveat.

### 2.4 Carlini et al. (2023): Quantifying memorization across neural language models
- **Citation (APA):** Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramèr, F., & Zhang, C. (2023). Quantifying memorization across neural language models. In *The Eleventh International Conference on Learning Representations (ICLR 2023)*. https://openreview.net/forum?id=TatRHT_1cK (arXiv: https://arxiv.org/abs/2202.07646)
- **Model scale:** About 120M to 6B (GPT-Neo family and GPT-J-6B, with GPT-2-sized comparisons).
- **Summary:** Verbatim extractable memorization grows log-linearly with model capacity, with how often an example is duplicated, and with prompt context length.
- **Relation to our paper:** Duplication drives *verbatim* memorization. Repeated review of identical sequences may raise verbatim memorization more than paraphrase-robust knowledge. That argues for reporting training-text loss next to paraphrase loss, to show which one review actually improves.

### 2.5 Toneva et al. (2019): An empirical study of example forgetting during deep neural network learning
- **Citation (APA):** Toneva, M., Sordoni, A., Tachet des Combes, R., Trischler, A., Bengio, Y., & Gordon, G. J. (2019). An empirical study of example forgetting during deep neural network learning. In *International Conference on Learning Representations (ICLR 2019)*. https://arxiv.org/abs/1812.05159 (ICLR poster page: https://iclr.cc/virtual/2019/poster/753)
- **Model scale:** Image classifiers (MNIST, permuted MNIST, CIFAR-10/100). **Not an LM study.**
- **Summary:** The paper defines "forgetting events" as transitions from correct to incorrect during training. Some examples are forgotten repeatedly and others never are, and this is consistent across architectures.
- **Relation to our paper:** Methodological precedent for **per-item** analysis: count each fact's learned/forgotten transitions instead of averaging loss. Averages can hide heterogeneity; some facts may be "unforgettable" and others never learned.

---

## 3. Catastrophic forgetting in LLM fine-tuning and continual learning

### 3.1 Luo et al. (2025): An empirical study of catastrophic forgetting in LLMs during continual fine-tuning
- **Citation (APA):** Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., & Zhang, Y. (2025). An empirical study of catastrophic forgetting in large language models during continual fine-tuning. *IEEE Transactions on Audio, Speech and Language Processing, 33*, 3776–3786. https://doi.org/10.1109/TASLPRO.2025.3606231 (arXiv: https://arxiv.org/abs/2308.08747)
- **Model scale:** BLOOMZ 1.1B–7.1B, mT0, LLaMA-7B, and Alpaca. The journal version adds Qwen2.5-Instruct 3B–14B.
- **Summary:** Continual instruction tuning causes forgetting of domain knowledge, reasoning, and reading comprehension across the 1B–7B range. Forgetting got *worse* with scale in that range, which the authors attribute to higher starting performance. Decoder-only BLOOMZ forgets less than encoder-decoder mT0. General instruction tuning beforehand reduces forgetting.
- **Relation to our paper:** Background for the claim that forgetting occurs at small scale. The scale trend shows that conclusions from 300M may not transfer even within a few billion parameters.

### 3.2 Kotha, Springer & Raghunathan (2024): Understanding catastrophic forgetting via implicit inference
- **Citation (APA):** Kotha, S., Springer, J. M., & Raghunathan, A. (2024). Understanding catastrophic forgetting in language models via implicit inference. In *The Twelfth International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=VrHiF2hsrm (arXiv: https://arxiv.org/abs/2309.10105)
- **Model scale:** A synthetic 22.4M GPT-2-style transformer; real models LLaMA-7B, Alpaca, Vicuna-7B, and OPT-IML-1.3B.
- **Summary:** Fine-tuning shifts the model's *implicit task inference* toward the fine-tuning distribution more than it removes capabilities. Tasks "close" to the fine-tuning distribution are hurt most. "Conjugate prompting," for example translating the prompt, recovers capabilities.
- **Relation to our paper:** Loss on old-fact questions can change because the model's inferred "task/format" changes, not because stored facts change. Stage-1 and Stage-2 facts share format, so this predicts little format-driven forgetting and may even predict improvement, which matches the observed Stage-2 drop. Old-fact loss is therefore a mixture of format effects and knowledge effects.

### 3.3 Zheng et al. (2025): Spurious forgetting in continual learning of language models
- **Citation (APA):** Zheng, J., Cai, X., Qiu, S., & Ma, Q. (2025). Spurious forgetting in continual learning of language models. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. https://openreview.net/forum?id=ScI7IlKGdI (arXiv: https://arxiv.org/abs/2501.13453)
- **Model scale:** A synthetic Biography dataset (200K fictional individuals) with a GPT-style model trained from scratch (exact size **UNVERIFIED** from the text I extracted). Real settings use LLaMa-2-7B-Chat, LLaMa-3-8B-Instruct, and **Pythia-410M** (Concept-1K).
- **Summary:** Large performance drops on old tasks often reflect lost **task alignment** rather than lost knowledge. Old-task accuracy can fall to near zero within about 150 steps of new-task training, then recover after brief re-training on a small amount of old-task data. Most of the damage happens in the first optimization steps of a new task. Freezing bottom layers, or replay, aligns the two tasks.
- **Relation to our paper:** **Very relevant to interpreting loss-based retention.** (1) Some of what we call "forgetting" during the buffer, and some of what review "protects," may be alignment or format rather than fact storage. A *relearning-savings* probe (a few steps of old-fact re-exposure after the buffer, then measure recovery) would separate these. (2) The early-steps-disrupt finding suggests that review timing relative to the *start* of new-fact training matters most. Expanding review front-loads reviews, yet it did not help at the delayed test. That is worth discussing. Pythia-410M is close to our scale.

### 3.4 Biderman et al. (2024): LoRA learns less and forgets less
- **Citation (APA):** Biderman, D., Portes, J., Gonzalez Ortiz, J. J., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., & Cunningham, J. P. (2024). LoRA learns less and forgets less. *Transactions on Machine Learning Research* (Featured Certification). https://openreview.net/forum?id=aloEru2qCG (arXiv: https://arxiv.org/abs/2405.09673)
- **Model scale:** Llama-2-7B. Code and math, through instruction fine-tuning (~100K pairs) and continued pretraining (up to 20B tokens). LoRA ranks 16, 64, and 256 compared with full fine-tuning.
- **Summary:** In standard low-rank settings (**r = 16 explicitly included**), LoRA substantially underperforms full fine-tuning on the target domain but forgets less of the source domain. Full fine-tuning learns weight perturbations with rank 10–100× higher than typical LoRA configurations.
- **Relation to our paper:** **A critical caveat for the LoRA choice.** With r=16, the model (i) learns new facts less, which fits our high absolute losses, and (ii) is structurally biased toward *less* forgetting. Our measured forgetting, and so the room for review to help, may be smaller or differently shaped than under full fine-tuning. Our paper frames LoRA as "lightweight" and "well-defined." It should say directly that LoRA is a regularizer that changes the learning-forgetting tradeoff, and that ideally the main result should be replicated with full fine-tuning or a higher rank. **Scale caveat:** 7B only, but the mechanism (low-rank constraint) is not specific to scale.

### 3.5 Shuttleworth et al. (2025): LoRA vs full fine-tuning: an illusion of equivalence
- **Citation (APA):** Shuttleworth, R., Andreas, J., Torralba, A., & Sharma, P. (2025). LoRA vs full fine-tuning: An illusion of equivalence. In *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*. https://openreview.net/forum?id=xp7B8rkh7L (arXiv: https://arxiv.org/abs/2410.21228)
- **Model scale:** RoBERTa-base (~125M) and LLaMA/LLaMA2-7B.
- **Summary:** LoRA updates create "**intruder dimensions**," high-ranking singular vectors nearly orthogonal to the pretrained ones. Full fine-tuning does not. Even when task performance is matched, LoRA's forgetting is concentrated in these intruder dimensions; scaling them down reduces forgetting. LoRA forgets less than full fine-tuning "but not always." Intruder dimensions *accumulate in sequential/continual fine-tuning* and hurt performance. Intruder dimensions are most common at low ranks; for RoBERTa they appeared consistently for r ≤ 16.
- **Relation to our paper:** Our design is a sequence of LoRA updates on a single adapter across three stages, which is exactly the continual setting where intruder dimensions accumulate. Review may work partly by keeping old facts represented within the few adapter directions. Whether that carries over to full fine-tuning is unknown. This strengthens the case for a full-FT or higher-rank replication.

### 3.6 Kalajdzievski (2024): Scaling laws for forgetting when fine-tuning LLMs
- **Citation (APA):** Kalajdzievski, D. (2024). *Scaling laws for forgetting when fine-tuning large language models* [Preprint]. arXiv. https://arxiv.org/abs/2401.05605
- **Model scale:** Llama 2 7B (chat) with LoRA. **Scale caveat:** 7B.
- **Summary:** With LoRA, forgetting (measured as cross-entropy against the base model's predictions) is strongly and linearly anti-correlated with fine-tuning performance. Forgetting grows as a *shifted power law* in the number of fine-tuned parameters (∝ rank) and the number of update steps. Early stopping and fewer trainable parameters cannot avoid it.
- **Relation to our paper:** LoRA does not rule out forgetting. Forgetting depends on rank and steps, so our r=16 and 180-step buffer are design parameters that set how large any forgetting effect can be. A rank sweep would show whether the review effect scales with forgetting.

### 3.7 Ibrahim et al. (2024): Simple and scalable strategies to continually pre-train LLMs
- **Citation (APA):** Ibrahim, A., Thérien, B., Gupta, K., Richter, M. L., Anthony, Q., Belilovsky, E., Lesort, T., & Rish, I. (2024). Simple and scalable strategies to continually pre-train large language models. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=DimPeeCxKO (arXiv: https://arxiv.org/abs/2403.08763)
- **Model scale:** **405M** and 10B decoder-only models, trained on hundreds of billions of tokens.
- **Summary:** LR re-warming, LR re-decaying, and replay of a small fraction of old data (e.g., 1–25%; 5% is typical) are together enough to match full re-training on all data, for both weak (English→English) and strong (English→German) distribution shifts. The paper also discusses "infinite" LR schedules for continual training.
- **Relation to our paper:** The canonical replay result, at a scale (405M) very close to ours. It supports "review helps." It also shows that, in continual pretraining, replay is normally studied *jointly with LR schedule*. Our constant-LR choice is defensible but non-standard and should be justified against this literature. Their replay is uniform-random. Our "uniform ≈ expanding" result agrees with the practice of not scheduling replay carefully.

### 3.8 Chen et al. (2026): Continual memorization of factoids in language models (REMIX)
- **Citation (APA):** Chen, H., Geng, J., Bhaskar, A., Friedman, D., & Chen, D. (2026). Continual memorization of factoids in language models. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2411.07175 (journal-ref per arXiv record; OpenReview PDF: https://openreview.net/attachment?id=5Yd5QAKIFR&name=pdf)
- **Model scale:** Llama-3-8B and Mistral-7B. **Scale caveat:** 7–8B.
- **Summary:** The paper formalizes "continual memorization." A model memorizes factoids in Stage 1 and must keep them through Stage-2 fine-tuning. Forgetting is severe, especially when Stage 2 is *also factoid memorization*. Standard replay helps less than **REMIX**, which mixes random word sequences or generic pretraining data into the stages and often outperforms replay. Robust memorization stores factoids in earlier layers and spreads them across more layers.
- **Relation to our paper:** **The closest design analogue**: a two-stage factoid study where Stage 2 is another factoid set. It finds that fact-on-fact interference is the worst case, which is our setting. It also suggests a **strong baseline we lack**: mixing generic (non-fact) data at the *same budget* as review. That separates "re-exposure to the specific old facts" from "any interleaving that disrupts interference." Reviewers familiar with this paper may ask for a REMIX-style or generic-data control arm.

### 3.9 Bethune et al. (2025): Scaling laws for forgetting during finetuning with pretraining data injection
- **Citation (APA):** Béthune, L., Grangier, D., Busbridge, D., Gualdoni, E., Cuturi, M., & Ablin, P. (2025). Scaling laws for forgetting during finetuning with pretraining data injection. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR Vol. 267, pp. 4020–4042). https://proceedings.mlr.press/v267/bethune25a.html (arXiv: https://arxiv.org/abs/2502.06042)
- **Model scale:** GPT-style models from 41M to 1.27B (41M, 109M, 334M, 665M, 1.27B). **In our scale range.**
- **Summary:** Scaling laws predict overfitting and forgetting during unsupervised fine-tuning as functions of target data, model size, and the fraction of pretraining data injected. Injecting as little as 1% pretraining data into the fine-tuning mix prevents forgetting of the pretraining distribution.
- **Relation to our paper:** Small-scale evidence that a small replay fraction is very effective. It supports review in general and suggests reporting the *fraction* of updates spent on review. Our 12 review events across 180 Stage-2 updates is about 6.7%, if each event is one update.

### 3.10 Marek et al. (2026): Forgetting in language models: capacity, optimization, and self-generated replay
- **Citation (APA):** Marek, M., Cho, D., Qiu, S., Chunara, R., Izmailov, P., & Wilson, A. G. (2026). *Forgetting in language models: Capacity, optimization, and self-generated replay* [Preprint]. arXiv. https://arxiv.org/abs/2605.26097
- **Model scale:** Small models: 2M, 6M–46M, and a 205M LM pretrained on 30B tokens. Llama-3.2-1B-Instruct is used for self-generated replay. **In our scale range.**
- **Summary:** Self-generated samples work as replay data and nearly eliminate forgetting. Forgetting persists when the model is near capacity saturation. Where capacity is not the limit, **low learning rates reduce forgetting but need many more steps**; replay breaks this tradeoff.
- **Relation to our paper:** The LR is a first-order factor in forgetting. Our single LR (1e-4) sets where we sit on the learning-forgetting tradeoff, so the review effect size may depend on it. An LR sweep, even of two values, would strengthen the paper. **Status:** preprint.

### 3.11 Yang, Jones, Mozer & Ren (2024): Anticipatory recovery from catastrophic interference
- **Citation (APA):** Yang, Y., Jones, M., Mozer, M. C., & Ren, M. (2024). Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. https://doi.org/10.52202/079017-2621 (arXiv: https://arxiv.org/abs/2403.09613; OpenReview: https://openreview.net/forum?id=YSs1z5udBY)
- **Model scale:** Pythia 160M–2.8B (and vision models).
- **Summary:** When documents are fine-tuned *cyclically in a fixed order*, LLMs show "anticipatory recovery": loss on a document starts to drop *before* it is seen again. The effect is emergent with scale. There is a "sharp increase of average recovery score from the 160M model to the 410M model," and smaller models show little or none.
- **Relation to our paper:** Structured, repeated sequences change forgetting dynamics in non-obvious ways. Our fixed review schedules are a structured sequence, and our 300M model sits between the scales with and without the effect. If the same Stage-2 batches repeat in a fixed order across arms, this mechanism could also contribute to loss falling *without* review. It is worth checking whether data order is fixed or reshuffled per seed.

### 3.12 Kotha & Liang (2026): Replaying pre-training data improves fine-tuning
- **Citation (APA):** Kotha, S., & Liang, P. (2026). *Replaying pre-training data improves fine-tuning* [Preprint]. arXiv. https://arxiv.org/abs/2603.04964
- **Model scale:** 150M-parameter models in controlled experiments; 8B (Llama 3) models for applied fine-tuning.
- **Summary:** Replaying *generic* pretraining data during fine-tuning improves performance on the *target* task, not only retention. It makes target data up to 1.87× more efficient (2.06× in mid-training). The mid-training experiments use a single WSD schedule across stages.
- **Relation to our paper:** Interleaving "other" data can help learning of the *current* data. In our review arms, interleaving old facts might benefit both old and new learning through this regularization-like effect. A generic-data interleaving control would separate this from fact-specific review. **Status:** preprint.

### 3.13 Harmon et al. (2025): Mapping post-training forgetting in language models at scale
- **Citation (APA):** Harmon, J., Hochlehnert, A., Bethge, M., & Prabhu, A. (2025). *Mapping post-training forgetting in language models at scale* [Preprint]. arXiv. https://arxiv.org/abs/2510.17776
- **Model scale:** Open-weight families from roughly 7B to 32B (e.g., Qwen2.5 base/Math/Instruct, Llama-8B-based distills, s1.1 7B/14B/32B). **Scale caveat:** ≥7B, but the method is independent of scale.
- **Summary:** The paper proposes a **sample-wise** forgetting metric that counts 1→0 transitions (forgetting) and 0→1 transitions (backward transfer), with **chance-adjusted** variants for MCQ. Task-level averages mix up forgetting and backward transfer and hide large changes.
- **Relation to our paper:** Our "old-loss change" averages over facts and so mixes forgetting with backward transfer, which is exactly what this paper warns about. The old-fact loss drop in Stage 2 *is* backward transfer in their terms. Reporting per-fact transitions (or per-fact loss deltas) would show whether review prevents forgetting of specific facts or shifts the whole distribution.

### 3.14 Sun et al. (2025): Pseudo forgetting in LLMs (supplementary)
- **Citation (APA):** Sun, H., Yang, Y., Li, Y., Li, J., & Gao, Y. (2025). Unveiling and addressing pseudo forgetting in large language models. In *Findings of the Association for Computational Linguistics: ACL 2025*. https://arxiv.org/abs/2411.11932
- **Model scale:** **UNVERIFIED** (abstract only).
- **Summary:** Drops on previous tasks often reflect instructions failing to activate retained abilities. Supplying partial rationales or meaningless suffixes restores performance.
- **Relation to our paper:** Same theme as spurious forgetting: behavioral loss is not erased knowledge.

### 3.15 Continual-learning surveys
- **Wu et al. (2024).** Wu, T., Luo, L., Li, Y.-F., Pan, S., Vu, T.-T., & Haffari, G. (2024). *Continual learning for large language models: A survey* [Preprint]. arXiv. https://arxiv.org/abs/2402.01364. *Model scale:* survey. *Summary:* Organizes LLM continual learning into continual pretraining, instruction tuning, and alignment, and compares it with RAG and model editing. *Relation:* framing citation for placing our study (continual knowledge injection via fine-tuning) in that taxonomy.
- **Shi et al. (2025).** Shi, H., Xu, Z., Wang, H., Qin, W., Wang, W., Wang, Y., Wang, Z., Ebrahimi, S., & Wang, H. (2025). Continual learning of large language models: A comprehensive survey. *ACM Computing Surveys, 58*(5), Article 120, 1–42. https://doi.org/10.1145/3735633 (arXiv: https://arxiv.org/abs/2404.16789). *Model scale:* survey. *Summary:* Covers vertical (general→specific) and horizontal (over time) continuity, and calls for better benchmarks and an understanding of forgetting, transfer, and acquisition. *Relation:* the peer-reviewed survey citation to use. It names "fundamental understanding of knowledge forgetting/acquisition" as an open gap, which our study partly addresses.
- **Wang et al. (2024).** Wang, L., Zhang, X., Su, H., & Zhu, J. (2024). A comprehensive survey of continual learning: Theory, method and application. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 46*(8), 5362–5383. https://doi.org/10.1109/TPAMI.2024.3367329. *Model scale:* survey (general ML). *Summary:* A general CL taxonomy (regularization, replay, optimization, representation, architecture) framed around the stability–plasticity trade-off. *Relation:* our review arms are **replay-based CL**, and the "joint loss" metric is a stability–plasticity summary. Cite for terminology.

### 3.16 Classic continual-learning references
- **Kirkpatrick et al. (2017).** Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences, 114*(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114. *Model scale:* MLPs on MNIST variants and DQN agents on Atari (**non-LM**). *Summary:* Elastic Weight Consolidation (EWC) slows learning on weights important to earlier tasks, using a Fisher-information penalty. *Relation:* the standard regularization alternative to replay; a natural "non-review" baseline for future work.
- **Kemker et al. (2018).** Kemker, R., McClure, M., Abitino, A., Hayes, T., & Kanan, C. (2018). Measuring catastrophic forgetting in neural networks. *Proceedings of the AAAI Conference on Artificial Intelligence, 32*(1), 3390–3398. https://doi.org/10.1609/aaai.v32i1.11651. *Model scale:* MLP-scale networks on image and audio data (**non-LM**). *Summary:* Proposes metrics for CL evaluation (retention of base knowledge, acquisition of new, overall) normalized against an offline-trained ideal model, and shows that no method solves forgetting across paradigms. *Relation:* precedent for normalizing retention against reference models. Our metrics would gain from normalizing against (a) a no-training baseline and (b) a jointly trained upper bound (old and new facts mixed from the start).
- **Mirzadeh et al. (2020).** Mirzadeh, S. I., Farajtabar, M., Pascanu, R., & Ghasemzadeh, H. (2020). Understanding the role of training regimes in continual learning. In *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. https://arxiv.org/abs/2006.06958 (NeurIPS venue from my own knowledge; **not checked against the proceedings**). *Model scale:* small vision networks (**non-LM**). *Summary:* Training regime (learning rate, batch size, dropout, LR decay) strongly affects forgetting through the width/curvature of minima found. *Relation:* supports treating LR and batch size as confounds or moderators of forgetting.

---

## 4. Evaluation methodology for knowledge in LMs

### 4.1 Schaeffer, Miranda & Koyejo (2023): Are emergent abilities of LLMs a mirage?
- **Citation (APA):** Schaeffer, R., Miranda, B., & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? In *Advances in Neural Information Processing Systems 36* (pp. 55565–55581). https://proceedings.neurips.cc/paper_files/paper/2023/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html (arXiv: https://arxiv.org/abs/2304.15004)
- **Model scale:** InstructGPT/GPT-3 family, a BIG-Bench meta-analysis, and vision networks.
- **Summary:** Apparent "emergent" abilities come from nonlinear or discontinuous metrics such as exact match and multiple-choice grade. Continuous metrics like per-token log-likelihood show smooth, predictable change.
- **Relation to our paper:** This cuts both ways. Continuous CE is the *more sensitive* metric and justifies detecting small effects (0.17 nats) that accuracy might miss. But a CE change does not imply a behavioral change. At 3.8 nats per token, the correct answer is still very unlikely, and accuracy is probably near zero in all arms. Reviewers will ask whether the 4.3% CE difference corresponds to any difference in correctly answered facts. Report both.

### 4.2 Schaeffer et al. (2024): Why has predicting downstream capabilities of frontier AI models with scale remained elusive?
- **Citation (APA):** Schaeffer, R., Schoelkopf, H., Miranda, B., Mukobi, G., Madan, V., Ibrahim, A., Bradley, H., Biderman, S., & Koyejo, S. (2024). *Why has predicting downstream capabilities of frontier AI models with scale remained elusive?* [Preprint]. arXiv. https://arxiv.org/abs/2406.04391 (I did not confirm a peer-reviewed venue.)
- **Model scale:** Pythia (70M–12B), Cerebras-GPT (111M–), OLMo, and other families. Five families in total.
- **Summary:** MCQ accuracy is computed from NLLs through a chain of transformations. The step that degrades predictability is the one that depends on the probability mass on *incorrect* choices, not only on the correct one.
- **Relation to our paper:** This motivates a **log-likelihood margin versus distractors** metric. Answer-token CE ignores whether incorrect answers, such as other fictional entities, gain probability at the same time. Generic "fictional-answer" priors would lower CE for correct and incorrect answers alike, and a margin metric would cancel that out. FictionalQA ships MCQ distractors that can be used for this, keeping in mind the artifact caveats noted by Kirchenbauer et al.

### 4.3 Hu et al. (2024): Predicting emergent abilities with infinite resolution evaluation (supplementary)
- **Citation (APA):** Hu, S., Liu, X., Han, X., Zhang, X., He, C., Zhao, W., Lin, Y., Ding, N., Ou, Z., Zeng, G., Liu, Z., & Sun, M. (2024). Predicting emergent abilities with infinite resolution evaluation. In *The Twelfth International Conference on Learning Representations (ICLR 2024)*. https://arxiv.org/abs/2310.03262 (venue from the authors' record; **proceedings page not checked**)
- **Model scale:** Small models; exact range **UNVERIFIED**.
- **Summary:** "PassUntil" massive sampling shows that small models improve consistently in ways that standard accuracy cannot resolve.
- **Relation to our paper:** An alternative high-resolution behavioral metric for small models with near-zero accuracy.

### 4.4 Elazar et al. (2021): Measuring and improving consistency in pretrained language models (ParaRel)
- **Citation (APA):** Elazar, Y., Kassner, N., Ravfogel, S., Ravichander, A., Hovy, E., Schütze, H., & Goldberg, Y. (2021). Measuring and improving consistency in pretrained language models. *Transactions of the Association for Computational Linguistics, 9*, 1012–1031. https://doi.org/10.1162/tacl_a_00410
- **Model scale:** BERT, RoBERTa, and ALBERT, base and large variants (encoder MLMs, ≤~340M).
- **Summary:** ParaRel (328 paraphrases across 38 relations) shows that PLMs are poorly *consistent* across meaning-preserving paraphrases of factual queries.
- **Relation to our paper:** Supports evaluating on paraphrases, which we do. It also implies paraphrase-level variance is large, so CIs should account for clustering by fact (several paraphrases per fact), not only seeds.

### 4.5 Hase et al. (2023): Methods for measuring, updating, and visualizing factual beliefs in language models
- **Citation (APA):** Hase, P., Diab, M., Celikyilmaz, A., Li, X., Kozareva, Z., Stoyanov, V., Bansal, M., & Iyer, S. (2023). Methods for measuring, updating, and visualizing factual beliefs in language models. In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics* (pp. 2714–2731). https://doi.org/10.18653/v1/2023.eacl-main.199
- **Model scale:** **UNVERIFIED** in detail (medium-sized pretrained LMs; I did not extract it from the text).
- **Summary:** Proposes measuring model "beliefs" through consistency across paraphrases and logically related statements, and metrics for whether belief updates generalize and stay local.
- **Relation to our paper:** Supports measuring knowledge through paraphrase consistency and *locality* (did training on new facts change unrelated beliefs?). The never-trained control measures exactly this locality.

### 4.6 Ren & Sutherland (2025): Learning dynamics of LLM finetuning
- **Citation (APA):** Ren, Y., & Sutherland, D. J. (2025). Learning dynamics of LLM finetuning. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2407.10490
- **Model scale:** Pythia-410M/1B/1.4B/2.8B and Qwen1.5-0.5B/1.8B. **In our scale range.**
- **Summary:** Breaks down how a gradient step on one example changes predictions on other examples and responses. It explains post-fine-tuning hallucinations in which the model uses phrases or facts from answer B to answer question A.
- **Relation to our paper:** Gives a formal mechanism for **cross-fact influence**. Training on new-fact (question, answer) pairs moves probability mass toward answer-like tokens and phrases in general, which can lower CE on old-fact answers that share tokens or style. This supports the never-trained control and per-fact influence analysis.

### 4.7 On "unseen-fact" / never-trained control items (synthesis of the sources above)
Papers that explicitly use or call for a control to separate fact-specific learning from generic or format adaptation:
- **Kirchenbauer et al. (2026, FictionalQA):** event-split held-out questions improve during training ("leaky"). Also a "Base Webtext only" training control.
- **O'Neill (2026):** an explicit "drift control … computed on never-written statements."
- **Chang et al. (2024):** measures log-probability changes relative to the pre-injection value. They also find "paraphrase" probes, which test generalization, are acquired differently from memorization probes.
- **Allen-Zhu & Li (2024, Part 3.1):** evaluates on held-out *individuals* (P_test) to test extraction versus memorization.
- **Zheng et al. (2025):** a recovery or relearning probe to tell alignment loss from knowledge loss.

I did not find a peer-reviewed paper that formally names an "unseen-fact control" as a standard. The practice appears in the papers above, so reviewers in this area will expect it.

---

## 5. Learning-rate schedule confounds in data-order studies

### 5.1 Hu et al. (2024): MiniCPM (introduces WSD)
- **Citation (APA):** Hu, S., Tu, Y., Han, X., He, C., Cui, G., Long, X., Zheng, Z., Fang, Y., Huang, Y., Zhao, W., Zhang, X., Thai, Z. L., Zhang, K., Wang, C., Yao, Y., Zhao, C., Zhou, J., Cai, J., Zhai, Z., Ding, N., Jia, C., Zeng, G., Li, D., Liu, Z., & Sun, M. (2024). MiniCPM: Unveiling the potential of small language models with scalable training strategies. In *First Conference on Language Modeling (COLM 2024)*. https://arxiv.org/abs/2404.06395 (OpenReview PDF: https://openreview.net/pdf/4d42a839e3a9a8f5de3240905bab31151bff776d.pdf)
- **Model scale:** MiniCPM 1.2B and 2.4B non-embedding parameters, plus small "wind-tunnel" models.
- **Summary:** Introduces the Warmup-Stable-Decay (WSD) schedule. Most training uses a constant LR, followed by a short decay. Loss drops sharply during the decay phase. WSD lets intermediate checkpoints be reused for continued training and domain adaptation.
- **Relation to our paper:** Our "constant LR after warmup" is the *stable* phase of WSD. The WSD literature shows that a decay phase gives large loss drops that consolidate whatever data is seen during the decay. If a practitioner ends with LR decay, data seen near the end (e.g., the buffer's new facts) gets extra weight, and old-fact retention could look very different. Our constant LR avoids this confound but also departs from typical deployment. We should state that the results apply to the stable phase.

### 5.2 Hägele et al. (2024): Scaling laws and compute-optimal training beyond fixed training durations
- **Citation (APA):** Hägele, A., Bakouch, E., Kosson, A., Ben Allal, L., Von Werra, L., & Jaggi, M. (2024). Scaling laws and compute-optimal training beyond fixed training durations. In *Advances in Neural Information Processing Systems 37* (pp. 76232–76264). https://doi.org/10.52202/079017-2427 (arXiv: https://arxiv.org/abs/2405.18392)
- **Model scale:** Small GPT-style models. I confirmed 124M and 210M runs; the sweep spans further small sizes. **In our scale range.**
- **Summary:** Constant LR plus a short cooldown matches cosine schedules and scales predictably. Stochastic weight averaging improves models along a constant-LR trajectory at no extra training cost.
- **Relation to our paper:** Supports constant LR as a legitimate, well-studied regime. It also implies a constant-LR checkpoint is "pre-cooldown," so absolute losses are higher than after a cooldown. Our ~3.8-nat losses might fall substantially with a cooldown or weight averaging. One robustness check: apply a short cooldown at the end of the buffer in all arms and test whether the review advantage survives.

### 5.3 Luo et al. (2026): How learning rate decay wastes your best data in curriculum-based LLM pretraining
- **Citation (APA):** Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *The Fourteenth International Conference on Learning Representations (ICLR 2026)* (Oral). https://arxiv.org/abs/2511.18903 (ICLR page: https://iclr.cc/virtual/2026/poster/10009351)
- **Model scale:** 1.5B-parameter models trained on 30B tokens.
- **Summary:** Ascending-quality data curricula clearly beat random order **under constant LR**, but the advantage shrinks under standard LR decay, because the best data arrives when the LR is small. Fixes are moderate decay (final LR about 1/3 of peak, "WSMD") or replacing decay with checkpoint averaging. Combined, these give +1.64% average benchmark improvement over shuffling.
- **Relation to our paper:** **Direct support for our constant-LR design.** It shows empirically that data-order effects can be hidden or created by LR decay. It also implies our *null* (expanding ≈ uniform) is specific to constant LR. Under a decaying schedule, the front-loaded expanding schedule would place reviews at *higher* LR, which could produce a difference. Cite it to justify the design and to scope the claim.

---

## 6. Directly competing or adjacent work on review scheduling in LMs (outside the assigned list, but important)

### 6.1 Atreya et al. (2026): When to review: spaced repetition for continual pre-training of language models
- **Citation (APA):** Atreya, A., Batra, D., Mantri, Y. K., Bantug, G., Cowan, G. A., & Khraishi, R. (2026). *When to review: Spaced repetition for continual pre-training of language models* [Preprint]. arXiv. https://arxiv.org/abs/2608.17530
- **Model scale:** TinyLlama-1.1B-Chat (main) and Llama-3.2-3B-Instruct (larger).
- **Summary:** Spaced Repetition Training (SRT) schedules per-example rehearsal with SuperMemo-2, using per-example perplexity as the recall-quality signal. With the old/new exposure ratio held at 20/80 to match Uniform Replay, SRT recovers 5–37 points of old-knowledge accuracy lost by naive continual pretraining while keeping new-knowledge acquisition.
- **Relation to our paper:** **A direct competitor that reaches a different conclusion.** Here an adaptive spaced schedule beats uniform replay at matched budget. Two differences to discuss. (a) SRT is *adaptive per example* (driven by the item's own loss), while our expanding schedule is a *fixed* global schedule. (b) SRT evaluates with accuracy. Our null for fixed expanding schedules is consistent with "adaptivity, not spacing per se, is what helps." The paper must cite and position against this work. **Status:** preprint (Aug 2026).

### 6.2 Feng et al. (2026): FOREVER: forgetting curve-inspired memory replay for LM continual learning
- **Citation (APA):** Feng, Y., Wang, H., Li, J., Chu, X., Kang, Z., Liu, Y., Wang, Y., Yu, P. S., & Wu, X.-M. (2026). FOREVER: Forgetting curve-inspired memory replay for language model continual learning. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)*. https://arxiv.org/abs/2601.03938 (venue per arXiv comment "ACL 2026 Camera-ready"; Anthology entry not checked)
- **Model scale:** 0.6B–13B.
- **Summary:** Replay intervals follow an Ebbinghaus forgetting curve, but "time" is defined by the magnitude of optimizer updates rather than step count. The paper reports consistent gains over step-based replay on three CL benchmarks.
- **Relation to our paper:** Raises the question of which clock spacing should be measured on. Under constant LR, steps are a reasonable proxy for update magnitude, which suits our setup. Under LR decay, step-based expanding schedules would be distorted. FOREVER is another peer-reviewed result where forgetting-curve-based replay *helps*. We need to reconcile our null with it, for example: fixed versus adaptive, metric, scale, LoRA.

### 6.3 Prakriya et al. (2024): LFR pedagogy, learn–focus–review (supplementary)
- **Citation (APA):** Prakriya, N., Yen, J.-N., Hsieh, C.-J., & Cong, J. (2024). *Accelerating large language model pretraining via LFR pedagogy: Learn, focus, and review* [Preprint]. arXiv. https://arxiv.org/abs/2409.06131
- **Model scale:** Llama and GPT models pretrained on SlimPajama and OpenWebText (sizes **UNVERIFIED**; compared against Pythia models up to 2× their size).
- **Summary:** Spaced-repetition-inspired pretraining that revisits forgettable data blocks reaches lower perplexity using 5–19% of tokens.
- **Relation to our paper:** Another adaptive-review result. Same positioning point as SRT.

---

## 7. Synthesis

### (a) Interpreting our loss-based results, including the old-fact loss drop during new-fact training

1. **The Stage-2 drop in old-fact loss under no review is expected in this literature. It indicates that part of our metric measures generic adaptation, not fact-specific retention.** Several independent sources predict that training on *some* fictional facts lowers loss on *other* fictional-fact questions:
   - The FictionalQA authors themselves find "leaky" gains on never-trained events and conclude gains reflect "some combination of distributional and atomic learning" (Kirchenbauer et al., 2026).
   - O'Neill (2026) finds that never-written statements gain likelihood from surrounding writes and builds a drift control to subtract it.
   - Possible mechanisms: QA-format and style adaptation and implicit task inference (Kotha et al., 2024; Jiang et al., 2024); subject-agnostic answer priors learned from lesser-known facts (Ghosal et al., 2024); co-occurrence statistics (Zhang et al., 2024); cross-example influence on similar answers (Ren & Sutherland, 2025); continued formation or optimization of recall circuits (Zucchet et al., 2025; Ou et al., 2025); possibly content overlap across FictionalQA events (noted by Kirchenbauer et al.).

   **Implication:** "Old-loss change" values (−0.22 to −0.40) cannot be read as "old facts ended up better learned." Without a never-trained control, the paper cannot claim old facts became *better learned*. It can only claim that loss on old-fact questions fell.

2. **The review effect (−0.17 nats) is better identified than the absolute trajectories.** All arms run the same number of steps on the same distribution, so generic adaptation is roughly balanced across arms. The review-versus-no-review difference therefore says more about fact-specific effects than any within-arm change does. Two confounds remain. (i) Review arms see *fewer* new-fact updates, and the cost of that is not measured separately. (ii) Interleaving any different data can by itself reduce interference (REMIX: Chen et al., 2026; Kotha & Liang, 2026). A generic-data interleaving arm at equal budget would isolate re-exposure to the specific old facts.

3. **High absolute loss (~3.8 nats/token) means the facts are only weakly acquired.** Expected given the literature: few exposures per fact (Allen-Zhu & Li Part 3.3: ~100–1000 exposures are needed for robust storage; Chang et al.: learnability threshold), low surface-form diversity (Allen-Zhu & Li Part 3.1; Ovadia et al.; Yang et al.), long-tail and novel facts being slow to learn (Kandpal et al.; Gekhman et al.), and LoRA learning less (Biderman et al., 2024). So the study is mostly about retaining *partial traces*. Its conclusions may not hold once facts are actually learned (low loss, high accuracy). Spacing effects might show up only once items are learned to some criterion, which the human literature typically requires.

4. **CE and behavioral retention can diverge.** Schaeffer et al. (2023) justify continuous metrics for sensitivity. But O'Neill (2026) shows that facts can be behaviorally forgotten while keeping most of their log-probability lift. Zheng et al. (2025) and Sun et al. (2025) show performance loss without knowledge loss. A 4.3% CE gain may come with no accuracy gain at all; at ~2% per-token probability, greedy accuracy is probably near zero.

5. **"Review changes the starting point, not the forgetting rate"** fits power-law forgetting after each exposure (Chang et al., 2024) and the partial effect of replay seen by Zucchet et al. (2025). Under power-law forgetting, though, the absolute buffer loss depends on the time since the last exposure. Compare forgetting *rates* only between arms with the same time since last review, or fit the curve explicitly.

6. **Expanding ≈ uniform** is consistent with fixed-schedule replay generally being treated as uniform (Ibrahim et al., 2024; Bethune et al., 2025). It conflicts with recent *adaptive* spaced-replay results (Atreya et al., 2026, SRT; Feng et al., 2026, FOREVER). The likely way to reconcile them is that per-item adaptivity, not spacing shape, drives those gains. Also, under constant LR the two schedules differ only in timing, whereas under LR decay timing interacts with step size (Luo et al., 2026).

### (b) The LoRA, 300M, and seq-len-64 choices

- **LoRA r=16 is not neutral.** LoRA "learns less and forgets less" (Biderman et al., 2024, which includes r=16). It produces intruder dimensions that accumulate during sequential fine-tuning and concentrate forgetting (Shuttleworth et al., 2025). Forgetting still grows as a power law in rank and steps (Kalajdzievski, 2024). All three were studied at ~7B, with RoBERTa-base (~125M) in Shuttleworth et al., but the mechanism does not depend on scale. Under LoRA, both the size of the forgetting being prevented and the review effect may differ from full fine-tuning. The paper's justification ("lightweight… well-defined set of weights") should be replaced with an explicit acknowledgment. Ideally add one full-FT or r∈{64, 256} replication. O'Neill (2026) found LoRA and full FT agreed qualitatively in a related setting at 4B, which offers some reassurance.
- **300M scale.** Within the range where controlled knowledge studies work. Allen-Zhu & Li store biographies in 124M–682M models; Ou et al. 2025 use GPT-2 Small/Medium; Ibrahim et al. 2024 use 405M for continual pretraining; Bethune et al. 2025 span 41M–1.27B; Marek et al. 2026 use 205M. Scale-dependent phenomena to flag:
  - The forgetting baseline rises with scale (Tirumala et al., 2022).
  - Anticipatory recovery in cyclic training appears between 160M and 410M (Yang et al., 2024), right around our size.
  - Forgetting severity varied with scale within 1–7B (Luo et al., 2025).
  - Memorization predicted at small scale transfers poorly upward (Biderman et al., 2023).
  - Many headline knowledge-injection findings come only from ≥7B models: Jiang et al. 2024; Yang et al. 2025; Ovadia et al. 2024; Gekhman et al. 2024 (PaLM 2-S); Mecklenburg et al. 2024 (GPT-4); Zhang et al. 2024; Chen et al. 2026; Biderman et al. 2024; Kalajdzievski 2024. Treat these as **scale caveats** when citing them for a ≤ few-billion-parameter setting.

  Also state which DataDecide checkpoint is used, since knowledge acquisition depends on pretraining stage (Kim et al., 2025; Chang et al., 2024).
- **Sequence length 64.** The literature does not address this choice directly, but it interacts with fact coverage. FictionalQA documents are webtext-length, so 64-token windows may cut facts or separate the subject from the attribute. Some evaluated facts might then never appear intact in training, and their "retention" would be pure generic drift. Mecklenburg et al. (2024) show that fact-level coverage matters. Report the fraction of evaluated (question, answer) facts whose supporting span appears intact in at least one 64-token training window, or train on QA-pair or fact-sentence units. Batch size 16 is small. Chang et al. (2024) find smaller batches shorten the learnability threshold and increase forgetting, so retention results may depend on batch size.
- **Constant LR.** Strongly justified for an order or scheduling study (Luo et al., 2026, ICLR Oral; Chang et al., 2024 used a constant-LR control; Hägele et al., 2024). But it limits generalization to deployed pipelines that end with decay (WSD: Hu et al., 2024; continual pretraining with re-warm and re-decay: Ibrahim et al., 2024). Marek et al. (2026) also show that the LR level itself changes the learning-forgetting tradeoff.

### (c) Additional controls and metrics reviewers will expect

1. **A never-trained FictionalQA control set.** Held-out events, ideally an *event* split so content overlap is minimal. Evaluate it at every checkpoint in every arm, and report drift-corrected retention (old-fact loss minus control loss, or change relative to control), as in O'Neill (2026) and Kirchenbauer et al. (2026).
2. **Pre-training baseline loss** on old, new, and control questions before Stage 1. This gives an absolute anchor ("learned" relative to the base model).
3. **Behavioral metrics alongside CE.** Greedy exact match or F1, MCQ accuracy, and **log-likelihood margin against FictionalQA distractors** (Schaeffer et al., 2023, 2024; Kirchenbauer et al., 2026), with chance adjustment and the MCQ-artifact caveat (Harmon et al., 2025; Kirchenbauer et al., 2026).
4. **Per-fact analysis.** Per-fact loss deltas or 1→0 / 0→1 transitions (Toneva et al., 2019; Harmon et al., 2025). CIs clustered by fact and paraphrase, not only by the 3 seeds (Elazar et al., 2021).
5. **Training-text versus paraphrase loss.** Separates verbatim memorization, which duplication drives (Carlini et al., 2023), from extractable knowledge (Allen-Zhu & Li, 2024; Jiang et al., 2024).
6. **A generic-data interleaving arm** at the same budget and positions as review: random or webtext sequences (REMIX: Chen et al., 2026; Kotha & Liang, 2026). Isolates fact-specific re-exposure from interference reduction.
7. **A relearning or savings probe** after the buffer. A few old-fact steps, then measure recovery speed (Zheng et al., 2025; O'Neill, 2026, who found slow relearning). Separates erased from inaccessible knowledge.
8. **LoRA versus full fine-tuning (or a rank sweep)** and at least one other LR (Biderman et al., 2024; Shuttleworth et al., 2025; Kalajdzievski, 2024; Marek et al., 2026).
9. **An exposure accounting table.** Number of facts, exposures per fact per stage, and coverage under 64-token windows (Allen-Zhu & Li, 2025; Mecklenburg et al., 2024).
10. **A schedule-robustness check.** A short cooldown or checkpoint averaging at the end of all arms, to test whether the review advantage survives LR decay (Hägele et al., 2024; Luo et al., 2026).
11. **Positioning against adaptive spaced replay** (Atreya et al., 2026; Feng et al., 2026; Prakriya et al., 2024). At minimum, a per-item loss-triggered review arm would test whether *adaptivity*, not spacing shape, is what matters.
12. **Citation update.** FictionalQA is now ICLR 2026, not just arXiv. Cite Chang et al. (2024) and Chen et al. (2026) as the closest prior designs.
