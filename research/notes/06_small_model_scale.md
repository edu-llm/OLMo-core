# 06: Small-model scale (≈0.1B–7B): forgetting, replay, fact acquisition, and spacing

## TL;DR

- **Holds at ≤ a few billion parameters:**
  - small replay doses reduce forgetting from about 0.1B to 10B;
  - injected facts are forgotten following a power law (OLMo 1B/7B);
  - larger models learn facts faster per exposure;
  - replay raises the retention level more than it slows decay.
- **Caveats (other scales):**
  - "bigger models forget less" comes mostly from vision models; in 1–7B LMs, *absolute* forgetting grew with size;
  - "LoRA forgets less" and most knowledge-injection results come only from 7B+ models.
- **From 300M up to a few billion, expect:**
  - relative forgetting to shrink and absolute forgetting possibly to grow;
  - the review benefit to shrink but stay positive (SRT: +37 points at 1.1B vs +9 at 3B);
  - order effects (anticipatory recovery) to emerge between 160M and 410M, so the expanding-vs-uniform null may change at 0.4–3B.
- **Scale ladder:**
  1. DataDecide 150M → 1B (same data recipe).
  2. Pythia 410M → 2.8B/6.9B (PolyPythias for seed variance).
  3. OLMo 1B/7B, to compare with Chang et al. and Kim et al.

  Use ≥5 seeds, a fixed Stage-1 loss criterion, and a full fine-tuning arm.
- **Other P4 studies (added 2026-10-04).** Scale caveats for interleaving are in [07_interleaving.md](07_interleaving.md) (synthesis (d)), and for mastery gating in [08_mastery_curriculum.md](08_mastery_curriculum.md) §6(d). In short: order effects can emerge between 160M and 410M, LM curricula are mostly null at ≤125M, and arithmetic length generalization does not improve monotonically with size.


Prepared 2026-10-02 for the P4 white paper ("Does Spaced Review Help a Language Model Retain Facts During Continued Training?": no-review vs uniform vs expanding review; FictionalQA; LoRA r=16 on OLMo DataDecide 300M).

## How items were checked

- Metadata (title, authors, date, arXiv comments) came from the arXiv API. Venues were confirmed with OpenReview search records, Crossref DOIs, ACL Anthology BibTeX and PMLR/NeurIPS proceedings pages. Model sizes and numbers came from the full text (arXiv HTML/PDF or proceedings PDF).
- **[VERIFIED]** means the citation and the specific numbers quoted were read from the source in this session.
- **[PARTIAL]** means the citation was verified but a specific detail was not (that detail is named).
- **[UNVERIFIED]** means it could not be confirmed. Do not cite it without checking.
- The WebSearch quota was used up partway through, so discovery relied on the arXiv API, OpenReview search and Crossref. Coverage of 2025–2026 preprints that lack a distinctive title is therefore probably incomplete.
- Two numbers in the paper's draft are worth checking against the sources. DataDecide reports the "300M" configuration as **320.0M non-embedding parameters** (16 layers, d_model 1024, 30B tokens, 100 tokens/param), so the draft's "≈377M total" is plausible once embeddings are included. Second, the draft cites FictionalQA as an arXiv preprint, but it is now **published at ICLR 2026** (arXiv comment).

---

## 1. How catastrophic forgetting depends on scale

### 1.1 Ramasesh, Lewkowycz & Dyer (2022): Effect of scale on catastrophic forgetting **[PARTIAL]**
- **Citation:** Ramasesh, V. V., Lewkowycz, A., & Dyer, E. (2022). Effect of scale on catastrophic forgetting in neural networks. In *International Conference on Learning Representations (ICLR 2022)*. https://openreview.net/forum?id=GhVS8_yPeEa
- **Model scale:** Mostly vision. ResNets and Vision Transformers of increasing size, pretrained on ImageNet-21k and evaluated on two-task split CIFAR-10 / split CIFAR-100. Reviewers describe a short NLP section (§3.4) in which a pretrained language model is fine-tuned sequentially on tasks such as IMDB and English Wikipedia. **UNVERIFIED: the exact LM parameter counts.** The OpenReview PDF sits behind a bot challenge and the paper is not on arXiv.
- **Summary:** Large pretrained ResNets and Transformers resist forgetting much better than models trained from scratch. The robustness improves systematically with both model size and pretraining-data size, and class representations become more orthogonal as scale grows. Reviewers pointed out that pretraining and the downstream tasks overlap (ImageNet-21k vs CIFAR), and that pretraining is confounded with over-parameterization.
- **Relation to our paper:** This is the main source for "bigger models forget less", but the evidence is mostly from image classification. If cited for LMs, add the caveat that its LM evidence is limited.

### 1.2 Mirzadeh et al. (2022): Wide neural networks forget less catastrophically **[VERIFIED]**
- **Citation:** Mirzadeh, S. I., Chaudhry, A., Yin, D., Hu, H., Pascanu, R., Gorur, D., & Farajtabar, M. (2022). Wide neural networks forget less catastrophically. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR Vol. 162, pp. 15699–15717). https://proceedings.mlr.press/v162/mirzadeh22a.html
- **Model scale:** Not LMs. A 2-layer MLP with width 32–2048 on Rotated MNIST, and WideResNet-10 with width multipliers ×1, 2, 4, 8 on Split CIFAR-100.
- **Summary:** Making the network wider cuts forgetting substantially; making it deeper does not. Widening the MLP from 32 to 2048 reduced task-1 forgetting from 62% to 48% (after 5 tasks). Widening the WRN ×8 reduced it from 42% to 31% (after 20 tasks). The authors explain this through gradient orthogonality, sparsity, and the lazy-training regime.
- **Relation to our paper:** Mechanistic background only. These are small non-LM networks, so cite them as an analogy and not as LM evidence.

### 1.3 Luo et al. (2023/2025): Empirical study of catastrophic forgetting in LLMs during continual fine-tuning **[VERIFIED]**
- **Citation:** Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., & Zhang, Y. (2025). An empirical study of catastrophic forgetting in large language models during continual fine-tuning. *IEEE Transactions on Audio, Speech and Language Processing, 33*, 3776–3786. https://doi.org/10.1109/TASLPRO.2025.3606231 (preprint: https://arxiv.org/abs/2308.08747)
- **Model scale:** BLOOMZ 1.1B, 1.7B, 3B and 7.1B. mT0 1.2B and 3.7B. BLOOM-7.1B, LLaMA-7B and Alpaca-7B. Continual *instruction* tuning over five generation tasks (Simp → Emdg → InqQG → Exp → HGen) with lr 2e-5 and 3 epochs per task. The paper does not say whether it used full fine-tuning or LoRA; nothing mentions adapters.
- **Summary:** Forgetting occurs at every size from 1B to 7B. Measured as an absolute drop (FG), it *grows* with scale. For BLOOMZ domain knowledge, FG goes 9.54 (1.1B) → 10.72 → 14.63 → 18.37 (7.1B). For mT0 it goes 9.18 (1.2B) → 20.15 (3.7B). The trend is strictly monotone only for domain knowledge. Reasoning dips at 1.7B and reading comprehension dips at 7.1B. The authors attribute the trend to larger models' higher starting performance. Decoder-only BLOOMZ forgets less than encoder–decoder mT0. Prior general instruction tuning (Alpaca vs LLaMA) reduces forgetting.
- **Relation to our paper:** This is the main counter-example to "bigger forgets less", and it falls exactly in the team's target range. The direction depends on the metric (absolute drop vs fraction of gains lost) and on the setting (instruction tuning on general skills vs injected facts).

### 1.4 Scialom, Chakrabarty & Muresan (2022): Fine-tuned language models are continual learners **[VERIFIED]**
- **Citation:** Scialom, T., Chakrabarty, T., & Muresan, S. (2022). Fine-tuned language models are continual learners. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (pp. 6107–6122). Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.emnlp-main.410
- **Model scale:** T0_3B and T0pp (11B). The ablation uses T5-small (about 60M; the size is from general knowledge, the paper only says "T5-small"), T5-3B, and a randomly initialized T5-3B.
- **Summary:** Rehearsal with a **1% memory buffer** lets T0 learn 8 new instruction tasks while keeping performance on earlier tasks: 99.8% of the upper bound for T0pp and 98.0% for T0_3B. The ablation shows that continual-learning ability comes from pretraining and not from scale. T5-small still "mostly maintains" results relative to its upper bound, while a randomly initialized 3B model fails.
- **Relation to our paper:** Evidence from roughly 60M to 11B that even tiny replay budgets work, and that the replay benefit does not need billions of parameters. This supports the team's finding that review works at 300M.

### 1.5 Ibrahim et al. (2024): Simple and scalable strategies to continually pre-train LLMs **[VERIFIED]**
- **Citation:** Ibrahim, A., Thérien, B., Gupta, K., Richter, M. L., Anthony, Q., Lesort, T., Belilovsky, E., & Rish, I. (2024). Simple and scalable strategies to continually pre-train large language models. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=DimPeeCxKO (arXiv: https://arxiv.org/abs/2403.08763)
- **Model scale:** 405M, with both a weak shift (Pile→SlimPajama) and a strong shift (Pile→German). 10B, with the weak shift only.
- **Summary:** Re-warming the learning rate, re-decaying it, and replaying old data together match re-training from scratch. Replay levels tested: 0.5/1/5/10/50% for the weak shift and 1/5/10/25/50% for the German shift. They chose **5% for the weak shift and 25% for the strong shift**, and even 1% noticeably reduced forgetting. Without replay, forgetting was **0.27 nats at 405M vs 0.23 nats at 10B**, i.e. slightly less at the larger size. With 5% replay the figures were 0.21 vs 0.19, a difference the authors call negligible. The same replay percentage worked at both scales.
- **Relation to our paper:** This is the best LM evidence spanning sub-1B to 10B. It shows a mild reduction in forgetting with size and little change in the replay fraction needed. Note also that the team's constant-LR design sidesteps the re-warming confound this paper studies.

### 1.6 Kalajdzievski (2024): Scaling laws for forgetting when fine-tuning LLMs **[VERIFIED; arXiv preprint, no venue found]**
- **Citation:** Kalajdzievski, D. (2024). *Scaling laws for forgetting when fine-tuning large language models* (arXiv:2401.05605). arXiv. https://arxiv.org/abs/2401.05605
- **Model scale:** One base model, **Llama-2-7B-chat**, fine-tuned with LoRA at ranks 8–256 (rank 8 is about 640M trainable parameters). Full fine-tuning and other PEFT methods appear in an appendix. **Base-model size was deliberately not varied.**
- **Summary:** LoRA still causes catastrophic forgetting. Forgetting is inversely linear in fine-tuning loss, i.e. the better the model fits the new data, the more it forgets. Forgetting follows a shifted power law in the number of *fine-tuned parameters* and in the number of update steps (OpenOrca α_f ≈ 0.035, β_f ≈ 0.147). Early stopping and changing the number of trained parameters do not avoid it. The author's reading of "larger models forget more" (citing Luo et al.) is that larger models simply reach lower fine-tuning loss. This is an inference, not something the paper tested.
- **Relation to our paper:** Relevant because the team uses LoRA. The predicted trade-off between learning and forgetting matches the team's small plasticity cost of review. "Forgetting scales with model size" should **not** be attributed to this paper.

### 1.7 Que et al. (2024): D-CPT Law **[VERIFIED]**
- **Citation:** Que, H., Liu, J., Zhang, G., Zhang, C., Qu, X., Ma, Y., Duan, F., Bai, Z., Wang, J., Zhang, Y., Tan, X., Fu, J., Su, W., Wang, J., Qu, L., & Zheng, B. (2024). D-CPT law: Domain-specific continual pre-training scaling law for large language models. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. https://openreview.net/forum?id=JzKFN5fWOk (arXiv: https://arxiv.org/abs/2406.01375)
- **Model scale:** Fitted on Qwen-1.5 **0.5B, 1.8B and 4B**. Validated on 7B (unseen size) and 14B. 0.1B–26B tokens, 9 general:domain mixture ratios, 6 domains.
- **Summary:** The law is L(N, D, r) = E + A/N^α + B·r^η/D^β + C/(r+ε)^γ. It predicts both general-corpus and domain loss for any mixture ratio, model size and data size from small runs. A cross-domain variant needs about 1% of the usual compute. Model size enters as a separate additive term with no N×r interaction. The paper makes no explicit claim about how forgetting changes with size.
- **Relation to our paper:** The fit lies entirely in the team's target range. It shows that old/new mixture trade-offs can be extrapolated from small runs, which supports a small-to-large ladder design.

### 1.8 Gu et al. (2024): CMR scaling law **[VERIFIED]**
- **Citation:** Gu, J., Yang, Z., Ding, C., Zhao, R., & Tan, F. (2024). CMR scaling law: Predicting critical mixture ratios for continual pre-training of language models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 16143–16162). https://doi.org/10.18653/v1/2024.emnlp-main.903
- **Model scale:** Llama-style models of **460M, 940M, 1.6B and 3.1B**, pretrained from scratch on 200B tokens, then continually pretrained on 20B tokens.
- **Summary:** The critical mixture ratio is the largest share of new-domain data that keeps the rise in general loss within ε = 0.05. It **increases with model size**: 29.8% (460M), 34.9% (940M), 41.4% (1.6B), 47.8% (3.1B) for Finance. Larger models tolerate more new data before general loss degrades.
- **Relation to our paper:** Direct ≤3B evidence that the old-data share needed to protect old knowledge *shrinks* as models grow. The implication is that a fixed review budget may matter relatively less for a 3B model than for a 300M model.

### 1.9 Bethune et al. (2025): Scaling laws for forgetting during finetuning with pretraining data injection **[VERIFIED]**
- **Citation:** Bethune, L., Grangier, D., Busbridge, D., Gualdoni, E., Cuturi, M., & Ablin, P. (2025). Scaling laws for forgetting during finetuning with pretraining data injection. In *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*. https://openreview.net/forum?id=vWMij23BmQ (arXiv: https://arxiv.org/abs/2502.06042)
- **Model scale:** GPT-2-style models of **41M, 109M, 334M, 665M and 1.27B**, pretrained on RedPajamaV2 at 100 tokens/param. Fine-tuned on 12 Pile domains with 300K–30M tokens and injection rates of 0, 0.1, 0.5, 1 and 5%, for about 1,500 runs.
- **Summary:** Injecting **as little as 1% pretraining data** into the fine-tuning mix prevents forgetting of the pretraining distribution, at no cost on the fine-tuning set. "Forgetting is more severe when the model is small and when the finetuning dataset is big." Small models lose up to 95% of pretraining progress versus about 20% for the largest; one figure caption says "up to 80%" instead, so the paper is internally inconsistent. Injected data acts like extra effective capacity. Injection also improved fine-tuning validation loss, more strikingly for small models.
- **Relation to our paper:** The most directly relevant scaling law, covering 41M–1.27B, a range that includes 300M. It predicts that **relative forgetting shrinks** with size and that small replay doses are effective, and the benefit is larger for small models.

### 1.10 Yıldız et al. (2024/2025): Investigating continual pretraining in LLMs **[VERIFIED; TMLR acceptance year not confirmed]**
- **Citation:** Yıldız, Ç., Ravichandran, N. K., Sharma, N., Bethge, M., & Ermis, B. (2025). Investigating continual pretraining in large language models: Insights and implications. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=aKjJoEVKgO (arXiv: https://arxiv.org/abs/2402.17400). The year is inferred from arXiv v2 (Feb 2025); OpenReview shows "Accepted by TMLR".
- **Model scale:** GPT-2 S/M/L/XL (≤1.5B), Llama-2-7B, and RoBERTa-base/large. Continual pretraining over 159 M2D2 domains (6.6B tokens).
- **Summary:** "Smaller models are particularly sensitive to continual pretraining, showing the most significant rates of both learning and forgetting." The smallest model forgets most and the largest least, though GPT-2 M and L swap ranks depending on order. The authors say the Llama-2-7B forgetting numbers should be taken "with a grain of salt". Later checkpoints forget more than earlier ones.
- **Relation to our paper:** Supports the prediction that a 300M model sits in the high-learning, high-forgetting regime, so review effects may be largest there.

### 1.11 Abbes et al. (2025): Revisiting replay and gradient alignment for continual pre-training **[VERIFIED; arXiv preprint]**
- **Citation:** Abbes, I., Subbaraj, G., Riemer, M., Islah, N., Therien, B., Tabaru, T., Kingetsu, H., Chandar, S., & Rish, I. (2025). *Revisiting replay and gradient alignment for continual pre-training of large language models* (arXiv:2508.01908). arXiv. https://arxiv.org/abs/2508.01908
- **Model scale:** Spectra (Llama-architecture) models of **99M, 560M, 1.1B and 5.7B**. Sequence English→French→German with 100B tokens per language. Replay at 0, 25 and 50%, with and without Reptile (meta-experience replay).
- **Summary:** Replay and gradient alignment stabilize continual pretraining at every size. 50% replay with Reptile gave the lowest forgetting at all four sizes. A 1B model without replay roughly matched a 560M model with 50% replay. In their compute-scaling fits, 25% replay is a better use of compute than a larger model, while 50% replay is not. The text and tables disagree in places; for example, the 560M downstream numbers in the text do not match Table 3.
- **Relation to our paper:** Shows replay still pays off at every size from 0.1B to 6B and gives a compute-equivalence framing (replay vs parameters) the team could adopt. Old/new in this paper are languages, not facts.

### 1.12 Wang et al. (2025): Learning dynamics in continual pre-training **[VERIFIED]**
- **Citation:** Wang, X., Tissue, H., Wang, L., Li, L., & Zeng, D. D. (2025). Learning dynamics in continual pre-training for large language models. In *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025, oral)*. https://openreview.net/forum?id=Vk1rNMl0J1 (arXiv: https://arxiv.org/abs/2505.07796)
- **Model scale:** LLaMA-like models of **106M, 594M and 1.72B** non-embedding parameters, plus continual pretraining of Llama-3.2-1B.
- **Summary:** A CPT scaling law that decomposes the loss curve into a distribution-shift term and a learning-rate-annealing term, predicting whole loss trajectories. Replay reduces the shift exponentially in the replay ratio. At constant LR, the distribution-shift curves for the three sizes "nearly coincide". The authors caution that their models are far below mainstream LLM scale.
- **Relation to our paper:** Supports the team's constant-LR design: once LR effects are removed, forgetting dynamics look roughly independent of size across 0.1–1.7B. It also offers a functional form for fitting old-fact loss trajectories.

### 1.13 Zheng et al. (2025): Spurious forgetting in continual learning of language models **[VERIFIED]**
- **Citation:** Zheng, J., Cai, X., Qiu, S., & Ma, Q. (2025). Spurious forgetting in continual learning of language models. In *International Conference on Learning Representations (ICLR 2025)*. https://openreview.net/forum?id=ScI7IlKGdI (arXiv: https://arxiv.org/abs/2501.13453)
- **Model scale:** Pythia-410M (instance-incremental learning on Concept-1K), LLaMA-2-7B-Chat, LLaMA-3-8B-Instruct, and a randomly initialized LM on a 200K-person synthetic biography dataset.
- **Summary:** Much apparent forgetting is lost *task alignment* rather than lost knowledge. On the synthetic biographies, task-0 accuracy fell to about 10% while a recovery probe still reached about 96%. Freezing bottom layers helps. Replay was best by far: 20% replay gave 76.93% and 50% replay 80.62% task-0 accuracy, versus 11–23% for EWC, gradient projection, LAMOL and task vectors.
- **Relation to our paper:** Answer-token loss on paraphrased questions may mix alignment loss with knowledge loss. This is consistent with the team's finding that review shifts the *level* of old-fact loss rather than its decay rate. It is worth adding a recovery probe (a few steps of re-exposure) as a metric.

### 1.14 Lee (2024): Model size and catastrophic forgetting in online continual learning **[VERIFIED; arXiv preprint]**
- **Citation:** Lee, E. (2024). *The impact of model size on catastrophic forgetting in online continual learning* (arXiv:2407.00176). arXiv. https://arxiv.org/abs/2407.00176
- **Model scale:** ResNets of varying depth and width on SplitCIFAR-10. Not LMs.
- **Summary:** In online class-incremental learning, larger models do not reliably forget less and often adapt worse.
- **Relation to our paper:** Another reminder that "bigger forgets less" depends on the setting. Non-LM.

---

## 2. Fact acquisition and memorization vs model size (≤7B emphasis)

### 2.1 Allen-Zhu & Li (2025): Physics of LMs Part 3.3, knowledge capacity scaling laws **[VERIFIED]**
- **Citation:** Allen-Zhu, Z., & Li, Y. (2025). Physics of language models: Part 3.3, knowledge capacity scaling laws. In *International Conference on Learning Representations (ICLR 2025, spotlight)*. https://openreview.net/forum?id=FxNNiUgtfa (arXiv: https://arxiv.org/abs/2404.05405)
- **Model scale:** GPT-2-style models (rotary, no dropout) from **about 1M to 0.5B parameters**, plus LLaMA/Mistral variants, trained on synthetic biographies (bioS/bioR, N = 10K–20M people).
- **Summary:** With **1000 exposures** per fact, models store about **2 bits/param** (no model exceeds 2.3). With **100 exposures**, capacity drops to about **1 bit/param**. Int8 quantization keeps capacity. At 100 exposures, gated-MLP (LLaMA/Mistral) models are about 1.3× worse. Junk data reduces capacity sharply, and prepending a domain token mitigates this.
- **Relation to our paper:** A 300M model has room for far more fictional facts than FictionalQA contains, so capacity should not limit the experiment. Exposure count matters a lot (100 vs 1000 exposures halves capacity), and a 12-review budget is in the low-exposure regime.

### 2.2 Allen-Zhu & Li (2024): Physics of LMs Part 3.1, knowledge storage and extraction **[VERIFIED]**
- **Citation:** Allen-Zhu, Z., & Li, Y. (2024). Physics of language models: Part 3.1, knowledge storage and extraction. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235, pp. 1067–1077). https://proceedings.mlr.press/v235/allen-zhu24a.html
- **Model scale:** Small GPT-2-style models on a 100k-person synthetic biography dataset (sub-1B; exact sizes not re-checked).
- **Summary:** Knowledge can be memorized yet not be extractable by questions unless the pretraining data is *augmented* (paraphrases, varied phrasings, sentence shuffling). Augmentation strongly improves question-answering extraction after fine-tuning.
- **Relation to our paper:** Reviews that use paraphrases may behave differently from verbatim reviews. FictionalQA provides paraphrased question variants, and the team evaluates on held-out paraphrases, so the choice of review format is a lever worth testing.

### 2.3 Chang et al. (2024): How do LLMs acquire factual knowledge during pretraining? **[VERIFIED]**
- **Citation:** Chang, H., Park, J., Ye, S., Yang, S., Seo, Y., Chang, D.-S., & Seo, M. (2024). How do large language models acquire factual knowledge during pretraining? In *Advances in Neural Information Processing Systems 37* (pp. 60626–60668). https://doi.org/10.52202/079017-1939 (arXiv: https://arxiv.org/abs/2406.11813)
- **Model scale:** **OLMo-1B and OLMo-7B** intermediate checkpoints (early, mid and late, at about 170B, 500B and 1.5T tokens), resumed with the original optimizer state on Dolma. Batch is about 4M tokens.
- **Summary:**
  - Fictional-knowledge passages were inserted into pretraining batches under three regimes: **once**, **duplicate** (10× at a **fixed 100-step interval**) and **paraphrase**.
  - Retention decays as a power law in steps.
  - Duplicates give bigger immediate gains but forget faster. Decay constants for OLMo-7B duplicates are about 0.20–0.26 and for paraphrases about 0.14–0.23. By step 2000, duplicate and paraphrase end at similar levels.
  - Larger batches (2048 vs 128 sequences) make knowledge more robust to forgetting.
  - The immediate per-encounter gain ("effectivity") rises from 1B to 7B, but more pretraining tokens do not raise it.
  - The authors introduce a "learnability threshold": an interval between encounters beyond which a fact is never acquired.
  - The 1B early checkpoint was much less stable.
  - **UNVERIFIED in this session: a numeric 1B-vs-7B comparison of decay constants** (it is in an appendix that could not be read).
- **Relation to our paper:** The closest prior work: fictional facts, OLMo, re-exposure at an interval, and power-law forgetting. It used uniform spacing only, so no expanding schedule was compared. It inspires a design idea: a **spacing sweep around the learnability threshold** at 1B vs 7B. It also predicts that larger models gain more per review.

### 2.4 Kim et al. (2025): Knowledge entropy decay hinders new knowledge acquisition **[VERIFIED]**
- **Citation:** Kim, J., Lee, H., Cho, H., Jang, J., Hwang, H., Won, S., Ahn, Y., Lee, D., & Seo, M. (2025). Knowledge entropy decay during language model pretraining hinders new knowledge acquisition. In *International Conference on Learning Representations (ICLR 2025, oral)*. https://openreview.net/forum?id=eHehzSDUFp (arXiv: https://arxiv.org/abs/2410.01380)
- **Model scale:** OLMo 1B (checkpoints 118k–738k) and OLMo 7B (7B shown only in figures). Pythia 1.4B was used for the entropy trend.
- **Summary:** Later pretraining checkpoints acquire new (fictional, Chang-style) knowledge worse and forget more. For OLMo-1B baseline acquisition/forgetting: 25.2/10.5% at 118k → 18.8/19.5% at 738k. "Knowledge entropy" correlates 0.94 with acquisition and −0.96 with forgetting. Mid-stage checkpoints give the best final performance. A lower LR (1e-4) sharply reduces forgetting.
- **Relation to our paper:** Which checkpoint the team picks is a confound when scaling. DataDecide models are trained at 100 tokens/param (about 5× Chinchilla), and per this paper heavily trained checkpoints learn new facts worse. Compare sizes at matched tokens/param or matched checkpoint fractions.

### 2.5 Tirumala et al. (2022): Memorization without overfitting **[VERIFIED]**
- **Citation:** Tirumala, K., Markosyan, A. H., Zettlemoyer, L., & Aghajanyan, A. (2022). Memorization without overfitting: Analyzing the training dynamics of large language models. In *Advances in Neural Information Processing Systems 35* (pp. 38274–38290). https://doi.org/10.52202/068431-2773 (arXiv: https://arxiv.org/abs/2205.10770)
- **Model scale:** **125M, 355M, 1.3B, 2.7B, 6.7B and 13B**, both causal and masked LMs (fairseq). Trained on WikiText-103 and the RoBERTa corpus.
- **Summary:** Larger models memorize training data faster in every setting and can memorize more before overfitting. In a "special batch" experiment, a batch was seen once and then normal training resumed. Exact memorization of that batch decays quickly and then levels off at a **forgetting baseline that rises with model size**, i.e. larger models forget less. The baseline is not sensitive to batch order.
- **Relation to our paper:** Direct evidence across 125M–13B that a one-time exposure is retained better by larger models. This predicts the no-review arm's delayed loss would improve with scale.

### 2.6 Carlini et al. (2023): Quantifying memorization across neural LMs **[VERIFIED]**
- **Citation:** Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramèr, F., & Zhang, C. (2023). Quantifying memorization across neural language models. In *International Conference on Learning Representations (ICLR 2023)*. https://openreview.net/forum?id=TatRHT_1cK (arXiv: https://arxiv.org/abs/2202.07646)
- **Model scale:** GPT-Neo **125M, 1.3B, 2.7B and 6B** (on the Pile), with a GPT-2 baseline. T5 and OPT families were also tested.
- **Summary:** Extractable memorization grows log-linearly with model size (a 10× increase gives about +19 percentage points, R² = 99.8%), with duplication count, and with prompt length. Within a family, larger models memorize 2–5× more. The pattern is messier across families.
- **Relation to our paper:** Verbatim memorization, not fact retention, but the duplication-count effect is the "repetition" analogue of review.

### 2.7 Biderman et al. (2023): Emergent and predictable memorization **[VERIFIED]**
- **Citation:** Biderman, S., Prashanth, U. S., Sutawika, L., Schoelkopf, H., Anthony, Q., Purohit, S., & Raff, E. (2023). Emergent and predictable memorization in large language models. In *Advances in Neural Information Processing Systems 36* (pp. 28072–28090). https://doi.org/10.52202/075280-1219 (arXiv: https://arxiv.org/abs/2304.11158)
- **Model scale:** Pythia **70M–12B**, on the same data in the same order.
- **Summary:** Small models are poor predictors of *which* sequences a 12B model memorizes. Pythia-70M has 0.956 precision but 0.197 recall; 6.9B has 0.884 precision and 0.795 recall. Partially trained checkpoints of the large model predict better.
- **Relation to our paper:** **Item-level results at 300M may not carry over to larger models**, even when aggregate trends do. Report aggregate retention, not per-fact claims.

### 2.8 Morris et al. (2025/2026): How much do language models memorize? **[VERIFIED; venue version retitled]**
- **Citation:** Morris, J. X., Sitawarin, C., Guo, C., Kokhlikyan, N., Suh, G. E., Rush, A. M., Chaudhuri, K., & Mahloujifar, S. (2025). *How much do language models memorize?* (arXiv:2505.24832). arXiv. https://arxiv.org/abs/2505.24832. OpenReview lists a version titled "How much can language models memorize?" as an **ICML 2026 spotlight** (https://openreview.net/forum?id=bA6BgSbaUi). Cite the venue version if it applies.
- **Model scale:** "Hundreds" of GPT-style models from about 500K (100K in some tables) to **1.5B** parameters. The scaling law was validated on GPT-2 small (~125M) and XL (~1.5B).
- **Summary:** Capacity is about **3.6 bits/param** (3.51 in bf16, 3.83 in fp32). Models memorize until capacity is full, after which "grokking"/double descent begins and unintended memorization falls as generalization takes over.
- **Relation to our paper:** Another capacity estimate in the same range as Allen-Zhu & Li (2–3.6 bits/param). A 300M model is far from being capacity-limited on FictionalQA.

### 2.9 Lu et al. (2024): Scaling laws for fact memorization **[VERIFIED]**
- **Citation:** Lu, X., Li, X., Cheng, Q., Ding, K., Huang, X., & Qiu, X. (2024). Scaling laws for fact memorization of large language models. In *Findings of the Association for Computational Linguistics: EMNLP 2024* (pp. 11263–11282). https://doi.org/10.18653/v1/2024.findings-emnlp.658
- **Model scale:** Qwen-1.5-architecture models from **0.6M to 308M non-embedding** (20M–0.5B total) trained from scratch on Wikidata triples. Fits mostly used models of ≤38M non-embedding parameters.
- **Summary:** Fact capacity grows linearly with model size and saturates exponentially with epochs (almost nothing below about 35 epochs, saturation near 1000). In **blocked sequential training** (30M model, no replay), the first fact type's memorization collapses to about 0. More frequent facts suppress rare ones.
- **Relation to our paper:** A clean demonstration at small scale that sequential fact training without review wipes out earlier facts. The authors recommend spreading fact types evenly, which supports interleaved review.

### 2.10 Zucchet et al. (2025): How do language models learn facts? **[VERIFIED]**
- **Citation:** Zucchet, N., Bornschein, J., Chan, S., Lampinen, A., Pascanu, R., & De, S. (2025). How do language models learn facts? Dynamics, curricula and hallucinations. In *Conference on Language Modeling (COLM 2025)*. https://openreview.net/forum?id=vBcGnragkr (arXiv: https://arxiv.org/abs/2503.21676)
- **Model scale:** Default 8-layer, **44M** non-embedding decoder. Size ablations at **163M and 400M**, plus a Hawk recurrent variant. Synthetic biographies.
- **Summary:**
  - Fact learning goes through three phases: statistics, then a plateau, then individual-specific recall. Recall circuits form during the plateau, and its length grows roughly linearly with the number of individuals.
  - Imbalanced, "celebrity" or warm-up data curricula shorten the plateau.
  - Fine-tuning on new individuals rapidly corrupts existing memories (feed-forward key-value memories).
  - Replay of pretraining data "partially mitigates the final performance drop, but not the initial decline."
  - Transition timing depends more on the data distribution than on model size.
- **Relation to our paper:** A close mechanistic analogue at ≤400M. The finding that replay helps the final level but not the initial drop parallels the team's result that review lowers the *level* without changing the forgetting *rate*.

### 2.11 Kirchenbauer et al. (2026): FictionalQA **[PARTIAL]**
- **Citation:** Kirchenbauer, J., Mongkolsupawan, J., Wen, Y., Goldstein, T., & Ippolito, D. (2026). FictionalQA: A dataset for studying memorization and knowledge acquisition. In *International Conference on Learning Representations (ICLR 2026)*. https://arxiv.org/abs/2506.05639
- **Model scale:** Base checkpoints of the Llama 3.1, Llama 3.2, Gemma 1 and Gemma 2 families. Named sizes include **Llama-3.2-1B** and **Llama-3.1-8B**; other sizes are shown only in figures. Training mixes 5% fictional documents with 95% Dolma webtext. **UNVERIFIED: whether training was full fine-tuning or PEFT** (the appendix was truncated).
- **Summary:** How fast a model fits the documents correlates strongly with model size, and larger models showed more MCQ transfer. High fiction rates (50–100%) produce pure verbatim memorization with no generalization window. Document diversity matters more than repetition.
- **Relation to our paper:** The dataset paper itself shows size dependence between 1B and 8B and recommends dilution with real webtext. The team trained on fictional data alone; consider mixing in background text, as the original authors did.

### 2.12 Kandpal et al. (2023): LLMs struggle to learn long-tail knowledge **[VERIFIED]**
- **Citation:** Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. (2023). Large language models struggle to learn long-tail knowledge. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 15696–15707). https://proceedings.mlr.press/v202/kandpal23a.html
- **Model scale:** GPT-Neo/NeoX/J **125M–20B**, BLOOM **560M–176B**, and the GPT-3 API models.
- **Summary:** QA accuracy tracks the number of relevant pretraining documents, both correlationally and causally. Larger models learn rare facts better, but the authors estimate they would need to grow by orders of magnitude to handle low-support facts.
- **Relation to our paper:** Frames review as raising the effective frequency of rare facts. The size benefit holds even below 1B.

### 2.13 Hernandez et al. (2022): Learning from repeated data **[VERIFIED; arXiv preprint]**
- **Citation:** Hernandez, D., Brown, T., Conerly, T., DasSarma, N., Drain, D., El-Showk, S., Elhage, N., Hatfield-Dodds, Z., Henighan, T., Hume, T., Johnston, S., Mann, B., Olah, C., Olsson, C., Amodei, D., Joseph, N., Kaplan, J., & McCandlish, S. (2022). *Scaling laws and interpretability of learning from repeated data* (arXiv:2205.10487). arXiv. https://arxiv.org/abs/2205.10487
- **Model scale:** Models from **1.5M to 800M** parameters.
- **Summary:** Repeating a small slice of data many times can badly hurt models. Repeating 0.1% of data 100× degrades an 800M model to the level of a 400M model. The damage peaks in a specific mid-range of model sizes for a given repetition count.
- **Relation to our paper:** Heavy review of a small fact set could harm a mid-sized model, and the harm depends on size. Watch the plasticity cost as scale changes.

### 2.14 Muennighoff et al. (2023): Scaling data-constrained language models **[VERIFIED]**
- **Citation:** Muennighoff, N., Rush, A. M., Barak, B., Le Scao, T., Tazi, N., Piktus, A., Pyysalo, S., Wolf, T., & Raffel, C. (2023). Scaling data-constrained language models. In *Advances in Neural Information Processing Systems 36* (pp. 50358–50376). https://doi.org/10.52202/075280-2191
- **Model scale:** Up to **9B** parameters and 900B tokens.
- **Summary:** Up to about 4 epochs of repetition is almost as good as unique data. Returns diminish quickly after that, and at about 40 epochs repetition is worthless.
- **Relation to our paper:** Background on repetition in pretraining. It concerns general loss, not fact retention.

---

## 3. Small-model continual pretraining, knowledge injection, and LoRA vs full fine-tuning

### 3.1 Biderman, D. et al. (2024): LoRA learns less and forgets less **[VERIFIED]**
- **Citation:** Biderman, D., Portes, J., Gonzalez Ortiz, J. J., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., & Cunningham, J. P. (2024). LoRA learns less and forgets less. *Transactions on Machine Learning Research* (Featured Certification). https://openreview.net/forum?id=aloEru2qCG
- **Model scale:** **Only Llama-2-7B.** Continued pretraining on 20B tokens of code or math, and instruction fine-tuning on about 100K examples. **Model size was not varied**; the authors leave scaling to future work.
- **Summary:**
  - LoRA substantially underperforms full fine-tuning on the target domain, especially in continued pretraining. On code CPT, HumanEval reaches 0.224 with LoRA r=256 vs 0.263 with full fine-tuning.
  - LoRA preserves more source-domain ability. On code CPT, the average of HellaSwag, ARC-C and WinoGrande is 0.617 with LoRA vs 0.545 with full fine-tuning; for math, the two forget about equally.
  - Full fine-tuning learns perturbations of 10–100× higher rank.
- **Relation to our paper:** The team's LoRA r=16 at 300M probably *dampens both* learning and forgetting compared with full fine-tuning, which could compress differences between schedules. The 7B-only evidence means a LoRA-vs-full comparison at 300M–1B would be new.

### 3.2 Shuttleworth et al. (2025): LoRA vs full fine-tuning, an illusion of equivalence **[PARTIAL: models not re-checked]**
- **Citation:** Shuttleworth, R., Andreas, J., Torralba, A., & Sharma, P. (2025). LoRA vs full fine-tuning: An illusion of equivalence. In *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*. https://openreview.net/forum?id=xp7B8rkh7L (arXiv: https://arxiv.org/abs/2410.21228)
- **Model scale:** UNVERIFIED (the abstract names no models).
- **Summary:** LoRA introduces "intruder dimensions" (new high-ranking singular vectors) that cause its forgetting and accumulate during sequential LoRA fine-tuning. This hurts continual learning.
- **Relation to our paper:** Repeated review via the same LoRA adapter could accumulate intruder dimensions, so a spectral diagnostic across arms would be informative.

### 3.3 Gupta et al. (2023): How to (re)warm your model **[VERIFIED; ICML 2023 ES-FoMO workshop / arXiv]**
- **Citation:** Gupta, K., Thérien, B., Ibrahim, A., Richter, M. L., Anthony, Q., Belilovsky, E., Rish, I., & Lesort, T. (2023). *Continual pre-training of large language models: How to (re)warm your model?* (arXiv:2308.04014). arXiv. https://arxiv.org/abs/2308.04014
- **Model scale:** Pythia **410M**, Pile (300B) → SlimPajama (297B).
- **Summary:** Re-warming the LR first increases loss on both old and new data, including forgetting upstream, but improves downstream performance over long horizons.
- **Relation to our paper:** Justifies the team's constant LR, since warm-up and decay interact with forgetting.

### 3.4 Knowledge-injection studies at 7B and above (cite as larger-scale context)
- **Ovadia, O., Brief, M., Mishaeli, M., & Elisha, O. (2024).** Fine-tuning or retrieval? Comparing knowledge injection in LLMs. In *Proceedings of EMNLP 2024* (pp. 237–250). https://doi.org/10.18653/v1/2024.emnlp-main.15. **Scale:** Llama2-7B, Mistral-7B, Orca2-7B. RAG generally beats unsupervised fine-tuning for injecting knowledge. **[VERIFIED]**
- **Jiang, Z., Sun, Z., Shi, W., Rodriguez, P., Zhou, C., Neubig, G., Lin, X. V., Yih, W., & Iyer, S. (2024).** Instruction-tuned language models are better knowledge learners. In *Proceedings of ACL 2024 (Vol. 1)* (pp. 5421–5434). https://doi.org/10.18653/v1/2024.acl-long.296. **Scale:** Llama-2 7B and 70B. Pre-instruction-tuning (QA before documents) raised QA accuracy from 30.3 to 48.1% (7B) and from 46.4 to 62.7% (70B). **[VERIFIED]**
- **Gekhman, Z., Yona, G., Aharoni, R., Eyal, M., Feder, A., Reichart, R., & Herzig, J. (2024).** Does fine-tuning LLMs on new knowledge encourage hallucinations? In *Proceedings of EMNLP 2024* (pp. 7765–7784). https://doi.org/10.18653/v1/2024.emnlp-main.444. **Scale:** PaLM 2-S, the middle of five sizes; parameter count not public. New-knowledge examples are learned more slowly than known ones and increase hallucination. **[VERIFIED]**
- **Yang, Z., Band, N., Li, S., Candès, E., & Hashimoto, T. (2025).** Synthetic continued pretraining. In *ICLR 2025 (oral)*. https://openreview.net/forum?id=07yvxWDSla. **Scale:** Llama 3 8B, continually pretrained on 455M EntiGraph tokens synthesized from 1.3M real tokens, **with RedPajama replay**. Accuracy scales log-linearly with synthetic tokens. **[VERIFIED]**
- **Relation to our paper:** These results come from 7B or larger models, or from undisclosed sizes. Cite them as motivation, not as evidence that the effects hold at 300M.

---

## 4. Spacing, repetition timing, and replay scheduling in (small) LMs

### 4.1 Atreya et al. (2026): When to review, spaced repetition for continual pre-training of LMs **[VERIFIED; arXiv preprint, Aug 2026, no venue]**
- **Citation:** Atreya, A., Batra, D., Mantri, Y. K., Bantug, G., Cowan, G. A., & Khraishi, R. (2026). *When to review: Spaced repetition for continual pre-training of language models* (arXiv:2608.17530). arXiv. https://arxiv.org/abs/2608.17530
- **Model scale:** **TinyLlama-1.1B-Chat** and **Llama-3.2-3B-Instruct**. Training appears to be full fine-tuning (AdamW, lr 1e-4); LoRA is never mentioned, but full fine-tuning is not stated explicitly. Small CNNs/MLPs for vision and tabular runs.
- **Summary:**
  - Spaced Repetition Training (SRT) schedules per-example review with SuperMemo-2, using perplexity-derived recall quality, so intervals expand adaptively.
  - Baselines: naive continual pretraining, **uniform replay at a 20/80 old/new ratio**, and perplexity-prioritized replay. **There is no fixed-interval vs expanding comparison.**
  - Old-knowledge gains over naive CPT on Wikipedia: **+37.3 points at 1.1B** (+23.8 over uniform replay) but **+8.6 at 3B** (+5.4 over uniform). The authors read this as "smaller models benefit more from prioritized review."
  - At 3B, uniform replay collapsed GSM8K/BBH (6.8/8.4 vs base 77.6/53.9), while SRT preserved them.
  - Each condition is a single run. The initial-ease ablation is non-monotonic (E0 = 1.5 / 2.5 / 3.5 → 13.2 / 22.6 / 10.5).
- **Relation to our paper:** **The closest concurrent work and the most important new citation.** It is adaptive, per-item and expanding, versus the team's fixed global uniform vs expanding schedules. Its benefit *shrinks* from 1.1B to 3B on old knowledge. It used no seeds and no matched expanding-vs-uniform control, which the team's design provides. The team's null on fixed expanding vs uniform does not conflict with it, because SRT's gains may come from *item selection* rather than timing. Note that the SRT and P4 abbreviations could be confused.

### 4.2 Feng et al. (2026): FOREVER, forgetting-curve-inspired memory replay **[PARTIAL: models and numbers not checked]**
- **Citation:** Feng, Y., Wang, H., Li, J., Chu, X., Kang, Z., Liu, Y., Wang, Y., Yu, P. S., & Wu, X.-M. (2026). FOREVER: Forgetting curve-inspired memory replay for language model continual learning. In *Proceedings of ACL 2026* (camera-ready per arXiv comment). https://arxiv.org/abs/2601.03938
- **Model scale:** "0.6B to 13B" (families not named on the abstract page).
- **Summary:** Replay is timed on a "model-centric" clock (the size of optimizer updates) rather than step counts, using an Ebbinghaus-style scheduler plus intensity-aware regularization. It reportedly beats fixed step-based replay schedules on three continual-learning benchmarks.
- **Relation to our paper:** Suggests that **step-based** spacing (as in P4) may be the wrong clock. An expanding schedule defined on cumulative update norm could behave differently.

### 4.3 Yang, Jones, Mozer & Ren (2024): Anticipatory recovery under cyclic training **[VERIFIED]**
- **Citation:** Yang, Y., Jones, M., Mozer, M. C., & Ren, M. (2024). Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training. In *Advances in Neural Information Processing Systems 37* (pp. 82438–82464). https://doi.org/10.52202/079017-2621
- **Model scale:** Pythia **160M, 410M, 1B, 1.4B and 2.8B** (full fine-tuning), GPT-2-large (812M), and vision models.
- **Summary:** When documents are revisited in a fixed cyclic order, LLMs start recovering on a document *before* seeing it again. The effect **emerges with scale**, with a sharp jump from 160M to 410M, and is stronger with more pretraining. Shuffling the order each epoch abolishes it. Online loss on epochs 2–5, cyclic vs shuffled: 1B 1.03 vs 1.51, 2.8B 1.34 vs 1.79.
- **Relation to our paper:** The strongest evidence that **the timing and structure of re-exposure matter in LMs, and in a scale-dependent way within 0.1–3B**. The team's 300M model sits just below the 410M jump, so schedule effects might appear at 0.4–3B.

### 4.4 Chang et al. (2024): learnability threshold (see 2.3)
Fixed-interval re-exposure (every 100 steps) on OLMo-1B/7B. Forgetting is a power law. Smaller batches shorten the interval beyond which a fact is never learned. This is the only verified ≤7B LM study that varies how re-exposure is presented; it compares duplicate vs paraphrase vs once but does not vary the spacing schedule.

### 4.5 Earlier and non-LM spacing and replay-scheduling work (caveat-level)
- **Amiri, H., Miller, T., & Savova, G. (2017).** Repeat before forgetting: Spaced repetition for efficient and effective training of neural networks. In *Proceedings of EMNLP 2017* (pp. 2401–2410). https://doi.org/10.18653/v1/D17-1255. A Leitner-style schedule over training instances for (pre-transformer) neural networks. **[VERIFIED citation; content not re-read]**
- **Klasson, M., Kjellström, H., & Zhang, C. (2023).** Learn the time to learn: Replay scheduling in continual learning. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2209.08660. MCTS- and RL-learned replay schedules beat fixed schedules. Vision continual-learning benchmarks (cs.CV), not LMs. **[VERIFIED]**
- **Kline, D. (2025).** *Human-like forgetting curves in deep neural networks* (arXiv:2506.12034). https://arxiv.org/abs/2506.12034. MLPs only. Prototype-similarity recall estimates trigger reviews. **[VERIFIED; preprint]**
- **Relation to our paper:** No verified study before Atreya et al. (2026) compares fixed expanding vs uniform review in a transformer LM. The team's matched-budget, multi-seed comparison appears to be **new at any LM scale**. The coverage caveat above applies.

### 4.6 BabyLM: data ordering and curricula at small scale **[VERIFIED]**
- **Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., Mosquera, R., Paranjabe, B., Williams, A., Linzen, T., & Cotterell, R. (2023).** Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *Proceedings of the BabyLM Challenge at CoNLL 2023* (pp. 1–34). https://doi.org/10.18653/v1/2023.conll-babylm.1. Curriculum learning was the most popular approach (13 teams, 41.9%) but "largely unsuccessful". Architecture changes (LTG-BERT) won.
- **Hu, M. Y., Mueller, A., Ross, C., Williams, A., Linzen, T., Zhuang, C., Cotterell, R., Choshen, L., Warstadt, A., & Wilcox, E. G. (2024).** Findings of the second BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *The 2nd BabyLM Challenge at CoNLL 2024* (pp. 1–21). https://aclanthology.org/2024.conll-babylm.1/. Again, curriculum learning "did not lead to high scores". A mixed-effects regression gave β = −3.6 (p = 0.055) for curricula.
- **Model scale:** Small LMs trained on ≤10M–100M words (parameter counts vary by submission).
- **Relation to our paper:** Mirrors the team's null on schedule. At small scale, *how data is ordered* has rarely beaten *how much* data or compute is used.

---

## 5. Model suites for a ≤few-B scaling study (open data and intermediate checkpoints)

| Suite | Sizes | Open data | Intermediate ckpts | Seeds | Citation |
|---|---|---|---|---|---|
| **DataDecide** | 14 sizes: 4M, 6M, 8M, 10M, 14M, 16M, 20M, 60M, 90M, 150M, 300M (320.0M non-emb), 530M, 750M (681M non-emb), 1B (1.18B non-emb); 100 tokens/param | Yes (25 recipes: Dolma, C4, FineWeb-Pro/Edu, Falcon, DCLM…) | >30K checkpoints | 3 (seeds 2–3 stop at 25% compute for sizes <1B) | Magnusson et al., 2025 [VERIFIED] |
| **Pythia** | 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B (+deduped) | Yes (Pile, exact order reproducible) | 154 per model | 1 (see PolyPythias) | Biderman et al., 2023 [VERIFIED] |
| **PolyPythias** | 14M, 31M, 70M, 160M, 410M (5 sizes) | Yes | ~7K new | 10 per size | van der Wal et al., 2025 [VERIFIED] (intermediate sizes per HF naming, not re-checked) |
| **OLMo (v1)** | 1B, 7B | Yes (Dolma) | Yes (used by Chang et al. 2024 and Kim et al. 2025, with optimizer state) | 1 | Groeneveld et al., 2024 [VERIFIED] |
| **OLMo 2** | 1B, 7B, 13B, 32B (1B: 4T pretrain tokens) | Yes (olmo-mix-1124, dolmino-mix-1124) | "Thousands" | 1 | Team OLMo et al., 2025 [VERIFIED] |
| **Olmo 3** | 7B, 32B | Yes ("every stage, checkpoint, data point") | Yes | 1 | Team Olmo, 2025 [VERIFIED abstract] |
| **SmolLM2** | 135M (2T tok), 360M (4T), 1.7B (11T) | Datasets released | **Not stated in paper** | 1 | Ben Allal et al., 2025 [PARTIAL] |
| **TinyLlama** | 1.1B (~1T tokens × ~3 epochs) | Public mixes (SlimPajama + StarCoder; mix composition not re-checked) | Released per authors | 1 | Zhang et al., 2024 [PARTIAL] |
| Qwen2.5 0.5/1.5/3B; Llama-3.2 1B/3B | — | **No** (open weights only) | No | — | Cite vendor reports; not suitable for data-controlled study [UNVERIFIED citations not compiled] |

Citations:
- Magnusson, I., Tai, N., Bogin, B., Heineman, D., Hwang, J. D., Soldaini, L., Bhagia, A., Liu, J., Groeneveld, D., Tafjord, O., Smith, N. A., Koh, P. W., & Dodge, J. (2025). DataDecide: How to predict best pretraining data with small experiments. In *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*. https://arxiv.org/abs/2504.11393
- Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). Pythia: A suite for analyzing large language models across training and scaling. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 2397–2430). https://proceedings.mlr.press/v202/biderman23a.html
- van der Wal, O., Lesci, P., Muller-Eberstein, M., Saphra, N., Schoelkopf, H., Zuidema, W., & Biderman, S. (2025). PolyPythias: Stability and outliers across fifty language model pre-training runs. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2503.09543
- Groeneveld, D., Beltagy, I., Walsh, E., Bhagia, A., Kinney, R., Tafjord, O., Jha, A., Ivison, H., Magnusson, I., Wang, Y., Arora, S., Atkinson, D., Authur, R., Chandu, K., Cohan, A., Dumas, J., Elazar, Y., Gu, Y., Hessel, J., . . . Hajishirzi, H. (2024). OLMo: Accelerating the science of language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Vol. 1: Long Papers)* (pp. 15789–15809). https://doi.org/10.18653/v1/2024.acl-long.841. **The P4 draft cites the arXiv version; update it to the ACL version.**
- Team OLMo, Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., Bhagia, A., Gu, Y., Huang, S., Jordan, M., Lambert, N., Schwenk, D., Tafjord, O., Anderson, T., Atkinson, D., Brahman, F., Clark, C., Dasigi, P., Dziri, N., . . . Hajishirzi, H. (2025). 2 OLMo 2 Furious. In *Conference on Language Modeling (COLM 2025)* (shorter version; arXiv:2501.00656). https://arxiv.org/abs/2501.00656
- Team Olmo. (2025). *Olmo 3* (arXiv:2512.13961). arXiv. https://arxiv.org/abs/2512.13961
- Ben Allal, L., Lozhkov, A., Bakouch, E., Martín Blázquez, G., Penedo, G., Tunstall, L., Marafioti, A., Kydlíček, H., Piqueres Lajarín, A., Srivastav, V., Lochner, J., Fahlgren, C., Nguyen, X.-S., Fourrier, C., Burtenshaw, B., Larcher, H., Zhao, H., Zakka, C., Morlon, M., . . . Wolf, T. (2025). SmolLM2: When smol goes big — Data-centric training of a (fully open) small language model. In *Conference on Language Modeling (COLM 2025)*. https://arxiv.org/abs/2502.02737
- Zhang, P., Zeng, G., Wang, T., & Lu, W. (2024). *TinyLlama: An open-source small language model* (arXiv:2401.02385). arXiv. https://arxiv.org/abs/2401.02385

---

## Summary table: size dependence reported in the literature

| Study | Sizes | Old/new | Direction with size |
|---|---|---|---|
| Ramasesh 2022 | ResNet/ViT (+ some LM) | vision tasks | Forget **less** (pretrained) |
| Mirzadeh 2022 | MLP/WRN widths | vision | Wider forgets **less**; depth no help |
| Luo 2023/25 | 1.1B–7.1B | instruction tasks | Absolute forgetting **more** |
| Ibrahim 2024 | 405M vs 10B | Pile→SlimPajama | Slightly **less** (0.27 vs 0.23 nats); same replay % works |
| Bethune 2025 | 41M–1.27B | pretrain→domain FT | **Less** relative forgetting; 1% injection suffices |
| CMR (Gu) 2024 | 460M–3.1B | general→finance | Tolerates **more** new data (CMR 29.8→47.8%) |
| Yıldız 2025 | GPT-2 S–XL, (7B) | 159 domains CPT | Small learn **and** forget most |
| Wang 2025 | 106M–1.72B | CPT | Shift curves ~**size-independent** at const LR |
| Abbes 2025 | 99M–5.7B | EN→FR→DE | Replay helps at all sizes |
| Tirumala 2022 | 125M–13B | one-shot batch | Forgetting baseline **higher** (forget less) |
| Chang 2024 | OLMo 1B/7B | fictional facts | Larger **acquires more** per exposure; power-law decay |
| Kirchenbauer 2026 | Llama/Gemma ~1B–8B | FictionalQA | Larger **fits faster**, more transfer |
| Anticipatory recovery 2024 | Pythia 160M–2.8B | cyclic docs | Structure effect **emerges ≥410M**, grows with size |
| SRT (Atreya) 2026 | 1.1B, 3B | Wiki/code snapshots | Review gain on old knowledge **shrinks** (37→9 pts) |

---

## Synthesis

### (a) Findings established at ≤few-B scale
1. **Small replay doses work from ~0.1B to ~10B.** Examples: 1% pretraining injection (Bethune, 41M–1.27B), 1% rehearsal (Scialom, about 60M–11B), 5–25% replay (Ibrahim, 405M and 10B), 20–50% replay (Zheng, Pythia-410M; Abbes, 99M–5.7B). The team's 300M finding that review helps is consistent with all of these.
2. **Forgetting of injected facts follows a power law in steps.** Duplicated exposures give larger immediate gains but decay faster, and larger batches slow forgetting (Chang, OLMo 1B/7B). Later, longer-trained checkpoints acquire less and forget more (Kim, OLMo 1B/7B).
3. **Replay or review mainly raises the level, not the decay rate.** Zucchet (≤400M) found replay mitigates the final drop but not the initial decline. Chang found duplicate and paraphrase converge by step 2000. Both match the team's finding of a lower starting point with an unchanged forgetting rate.
4. **Larger models learn facts faster per exposure** (Tirumala 125M–13B; Chang 1B→7B; Kirchenbauer 1B–8B; Carlini 125M–6B) **and retain one-shot exposures better** (Tirumala's forgetting baseline).
5. **Capacity is about 2–3.6 bits/param** (Allen-Zhu & Li ≤0.5B; Morris ≤1.5B). Capacity is not a binding constraint for FictionalQA at ≥100M. Exposure count matters (1000 vs 100 exposures doubles capacity).
6. **Data-ordering curricula rarely help at small scale** (BabyLM 2023 and 2024).

### (b) Commonly cited findings that rest on other scales: state as caveats
- **"Bigger models forget less"** rests mainly on **vision** (Ramasesh: ResNet/ViT on CIFAR; Mirzadeh: MLP/WRN) and on 10B-vs-405M loss comparisons. In the 1–7B instruction-tuning regime, **absolute forgetting grows with size** (Luo). State which metric is meant: an absolute drop vs the fraction of gains lost or loss on old data.
- **The forgetting scaling law of Kalajdzievski** comes from a single 7B model with LoRA rank varied. It says nothing directly about base-model size.
- **LoRA "learns less, forgets less"** comes from Llama-2-7B only, on code/math with 20B tokens, not on facts.
- **Knowledge-injection results** (Ovadia, Jiang, Gekhman, EntiGraph) come from **7B–70B models, or from PaLM 2-S of undisclosed size.**
- **Long-tail and memorization scaling** (Kandpal up to 176B; Carlini's cross-family results) includes models far above 7B.
- **Spacing-effect claims** come from humans and from non-LM networks (Amiri 2017, pre-transformer; Klasson 2023, vision; Kline 2025, MLPs). Physics-of-LMs and Zucchet use **synthetic biographies on ≤0.5B models**, which are smaller and more artificial than FictionalQA.
- **Mixture-law papers** that are within range (D-CPT 0.5–4B, CMR 0.46–3.1B, Wang 0.1–1.7B) model **domain loss**, not specific facts.

### (c) Will forgetting and review benefit grow or shrink from 300M to a few B?
- **Forgetting, relative (old-fact loss increase as a share of what was learned):** expected to **shrink**. Supporting evidence: Bethune (95%→20% of progress lost from 41M to 1.27B), Ibrahim (0.27→0.23 nats), Tirumala's rising forgetting baseline, Yıldız, and CMR rising with size.
- **Forgetting, absolute:** may **grow**, because larger models learn more in Stage 1 and so have more to lose (Luo; the learning–forgetting coupling in Kalajdzievski). Report both measures.
- **Review/replay benefit on old facts:** most evidence points to **a smaller relative benefit** at larger scale. SRT's gain fell from +37 to +9 points between 1.1B and 3B, CMR shows larger models tolerate more new data, and Bethune found injection matters more for small models. The benefit persists at every size, though (Abbes, Ibrahim). Larger models' higher per-exposure gain (Chang) means each of the 12 reviews may deliver more, which works the other way. **Net expectation:** a positive but smaller relative reduction than the 4.3% at 300M. Uncertainty is high, because no study measured matched-budget fact review across sizes.
- **Expanding vs uniform:** no direct evidence at any scale. Anticipatory recovery (structure-sensitive re-exposure effects) **emerges between 160M and 410M and strengthens up to 2.8B**. FOREVER suggests step-based clocks may be the wrong axis. So the 300M null **might not hold at 0.4–3B**, and this is the strongest scientific reason to scale up. The prior expectation stays a null or small effect, matching the human literature (Karpicke & Roediger, 2007).
- **LoRA caveat:** with fixed r=16, the trainable fraction falls as the base model grows. Compare at matched trainable fraction or include full fine-tuning.

### (d) Recommended scale ladder
1. **Primary ladder (data, tokenizer and recipe held constant): DataDecide 150M → 300M → 530M → 750M → 1B** on one recipe (for example the current one), with **seeds 0–2 at 1B** (only 1B has three complete seeds). This isolates size from data effects and reuses the current pipeline. Optionally add 60M/90M to fit a trend.
2. **Extension past 1B with shared data order: Pythia 410M → 1B → 1.4B → 2.8B → 6.9B.** Pythia-410M/1B overlaps rung 1 for calibration. The 160M–2.8B span is where anticipatory recovery emerged. Use **PolyPythias (10 seeds at ≤410M)** to estimate seed variance below the 410M jump.
3. **Top rung, comparable to Chang/Kim: OLMo-1B and OLMo-7B** (or OLMo 2 1B/7B). Intermediate checkpoints with optimizer state allow comparison with the published fictional-knowledge forgetting curves, and let the team pick **matched-maturity checkpoints** to avoid Kim et al.'s late-checkpoint confound.
4. **Design adjustments suggested by the literature:**
   - Run ≥5 seeds at ≤1B; the expanding-uniform CI is ±0.005 loss, so a size trend needs tight intervals.
   - Train Stage 1 to a **loss criterion** so each size starts the buffer comparably, which separates the head-start effect from the decay rate.
   - Report absolute and relative forgetting, plus a short **recovery probe** (Zheng).
   - Add a **full fine-tuning arm at ≤1B** next to LoRA, matching the LoRA trainable fraction across sizes.
   - Consider a 5% fiction / 95% webtext mix as in the FictionalQA paper.
   - Sweep **review interval around a learnability threshold** (Chang) rather than only the shape of the schedule.
   - Add an adaptive/item-selection arm (SRT-like) to separate *which items* to review from *when* to review.

## Flagged / unverified items
- Ramasesh et al. (2022): exact LM model sizes in the NLP section (PDF not accessible).
- Chang et al. (2024): numeric 1B-vs-7B decay-constant comparison (appendix not read).
- Kirchenbauer et al. (2026): full model list beyond Llama-3.2-1B and Llama-3.1-8B; whether training was full fine-tuning or PEFT.
- Shuttleworth et al. (2025): which models were used.
- FOREVER (2026): model families and numbers.
- Yıldız et al.: TMLR year (2025 inferred).
- SmolLM2: whether intermediate checkpoints are released. TinyLlama: data mix composition.
- Bethune et al.: internal inconsistency ("up to 95%" vs "up to 80%" of progress lost).
- Abbes et al.: some text vs table mismatches.
- Atreya et al. (SRT): single runs and no venue.
- Not compiled: Qwen2.5 and Llama 3.2 citations (open weights only, so unsuitable for a data-controlled study).
- Not found in this session (WebSearch quota exhausted): any peer-reviewed study comparing **fixed expanding vs uniform** review schedules in a transformer LM. The team's claim of novelty should be hedged.
