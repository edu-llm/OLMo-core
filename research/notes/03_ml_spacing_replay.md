# 03 — Prior ML work on spaced repetition, review/replay scheduling, and learning-science-inspired training

## TL;DR

- **"Replay/review reduces forgetting" is well established** (Robins 1995; Rolnick 2019; Scialom 2022; Ibrahim 2024). Our review-vs-none result is a replication; present it that way.
- **Closest prior work:**
  - **Tirumala et al. (2022, 125M):** spacing length barely mattered for long-run forgetting. This is the nearest precedent for our null.
  - **FOREVER (ACL 2026):** finds *expanding > uniform* at a matched budget (one setting, no variance reported). It contradicts us and must be addressed.
  - **SRT (2026) and LFR (CoNLL 2025):** adaptive, per-item schedulers.
- **Our niche:** no study compares *fixed* expanding vs. uniform review of facts at a matched budget, with a delayed test and seed-paired statistics. This only holds if reviews are made per-fact (see the code audit).
- **Human scheduling effects often fail to transfer to networks:** Flesch 2018 (interleaving), Xiong & Wu 2025 (spacing in context), Speckmann & Eimer 2025 (Leitner and SuperMemo in RL).
- **Framing advice:**
  - lead with the timing-only null;
  - avoid "spacing effect" without a massed arm;
  - flag LoRA as a moderator;
  - keep the constant learning rate (Luo et al. 2026).
- Definitions of the ML terms are in Section 0. Every entry lists the model sizes studied.


Prepared for: *Does Spaced Review Help a Language Model Retain Facts During Continued Training?* (Mago & Wang), P4 whitepaper. The paper uses LoRA on OLMo DataDecide 300M with FictionalQA and compares three arms: no review, uniform review, and expanding review. Each review arm gets 12 review events, and all arms have the same step budget. A 180-step no-review buffer follows. Result: review lowers delayed old-fact loss by about 4.3%, and expanding ≈ uniform.

Compiled 2026-10-02.

**Verification legend**

- **[V-pub]** The title, authors, venue and identifier were checked against the publisher, ACL Anthology, NeurIPS proceedings, OpenReview, Crossref or PubMed.
- **[V-arXiv]** Checked against the arXiv abstract page or API. The paper is a preprint with no peer-reviewed version found as of 2026-10-02.
- **[V-content]** The specific claim in the summary was also checked in the paper's full text, not only its abstract.
- **UNVERIFIED** Not checked; listed only so it can be followed up.

**Model scale field.** The team plans to stay at a few billion parameters or fewer. For each ML paper, "Model scale" lists the architectures and sizes the paper actually studied. **⚠ Scale caveat** marks a finding that comes only from much larger models (>~7B) or only from small non-LM networks (MLPs/CNNs/RL agents).

---

## 0. Definitions (with verifiable sources)

- **Catastrophic forgetting / catastrophic interference.** A network trained by gradient-based updates on new data (or a new task) often loses much of its performance on previously learned items. This is the "sequential learning problem." It is far more severe than forgetting in humans, and it comes from overlapping distributed representations being overwritten.
  - McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation, 24*, 109–165. https://doi.org/10.1016/S0079-7421(08)60536-8
  - Ratcliff, R. (1990). Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions. *Psychological Review, 97*(2), 285–308. https://doi.org/10.1037/0033-295X.97.2.285
  - French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2
  - [V-pub, all three]
- **Continual (lifelong / incremental) learning.** A model learns from a non-stationary stream of data or tasks presented in sequence. The aim is to acquire new knowledge (plasticity) while keeping old knowledge (stability), usually without retraining on all past data.
  - Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks, 113*, 54–71. https://doi.org/10.1016/j.neunet.2019.01.012
  - van de Ven, G. M., Tuytelaars, T., & Tolias, A. S. (2022). Three types of incremental learning. *Nature Machine Intelligence, 4*(12), 1185–1197. https://doi.org/10.1038/s42256-022-00568-3
  - Wang, L., Zhang, X., Su, H., & Zhu, J. (2024). A comprehensive survey of continual learning: Theory, method and application. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 46*(8), 5362–5383. https://doi.org/10.1109/TPAMI.2024.3367329
  - [V-pub]
  - Our setup is a *continued-training* (domain-incremental) form of continual learning: old facts, then new facts, from the same distribution family.
- **Rehearsal / replay.** Retraining on some previously learned information while new information is added. Robins (1995) calls this "rehearsal." *Pseudo-rehearsal* replays generated pseudo-items instead of stored ones. In neuroscience-inspired ML, "replay" means reactivating stored or generated patterns from past experience to protect memories.
  - Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318
  - Hayes, T. L., Krishnan, G. P., Bazhenov, M., Siegelmann, H. T., Sejnowski, T. J., & Kanan, C. (2021). Replay in deep learning: Current approaches and missing biological elements. *Neural Computation, 33*(11), 2908–2950. https://doi.org/10.1162/neco_a_01433
  - [V-pub]
  - Our paper's "review" is rehearsal of stored real items. The same old-fact items are re-presented.
- **Experience replay (ER).** Past experiences are kept in a buffer and mixed into later training updates. The term comes from reinforcement learning (Lin, 1992). In continual learning, "ER" usually means jointly training on current-task examples and a small episodic memory of past-task examples.
  - Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine Learning, 8*(3–4), 293–321. https://doi.org/10.1007/BF00992699
  - Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P. K., Torr, P. H. S., & Ranzato, M. (2019). *On tiny episodic memories in continual learning*. arXiv:1902.10486. https://arxiv.org/abs/1902.10486
  - Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. *Advances in Neural Information Processing Systems, 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html
  - [V-pub / V-arXiv]
- **Replay scheduling.** Deciding *when* (at which training steps or stages) and *what* (which tasks or items, in what proportion) to replay, given a fixed replay or compute budget. Klasson et al. (2023) frame it as "learning the time to learn." FOREVER (Feng et al., 2026) splits it into "when to replay" and "how (intensity) to replay." Our expanding-vs-uniform manipulation is a pure *when* manipulation: the same items, the same count, the same budget.
  - Klasson, M., Kjellström, H., & Zhang, C. (2023). Learn the time to learn: Replay scheduling in continual learning. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=Q4aAITDgdP
  - Feng, Y., et al. (2026), FOREVER, ACL 2026. https://arxiv.org/abs/2601.03938
  - [V-pub]
- **Spaced repetition; expanding vs. uniform (equal-interval) schedules.** These are human-learning terms, included so the vocabulary lines up.
  - *Spaced* practice separates repetitions in time; *massed* practice presents them back to back.
  - An *expanding* schedule increases the gap after each review. A *uniform* schedule keeps gaps equal.
  - Landauer, T. K., & Bjork, R. A. (1978). Optimum rehearsal patterns and name learning. In M. M. Gruneberg, P. E. Morris, & R. N. Sykes (Eds.), *Practical Aspects of Memory* (pp. 625–632). Academic Press. (Book chapter; no DOI. It is already cited in the paper.)
  - Karpicke, J. D., & Roediger, H. L. (2007). Expanding retrieval practice promotes short-term retention, but equally spaced retrieval enhances long-term retention. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 33*(4), 704–719. https://doi.org/10.1037/0278-7393.33.4.704
  - Note: the paper's experiment contains **no massed arm**, so it tests *schedule shape among spaced schedules*, not the spacing effect proper. See Synthesis.

---

## 1. Summary map

Column key:
- **Review helps?** Does the paper show that re-exposing old data reduces forgetting?
- **Timing isolated?** Is *when* held separate from *which items* and *how many exposures*?
- **Exp. vs uni. matched?** Is a fixed expanding schedule compared with a fixed uniform one at a matched budget?

| # | Paper | Venue | Model scale | Review helps? | Timing isolated? | Exp. vs uni. matched? |
|---|---|---|---|---|---|---|
| A1 | Amiri+ 2017 RbF | EMNLP | fastText-MLP, CNN, LSTM (small) | (efficiency focus) | No: difficulty-driven selection; reduces data per epoch | No |
| A2 | Amiri 2019 | NAACL | fastText DAN, CNN (small) | n/a (self-training sampling) | No | No |
| A3 | Prakriya+ LFR | CoNLL 2025 | Llama-2 120M/300M/500M; GPT-2 124M–1.5B | Yes (less forgetting, faster pretraining) | No: perplexity-based selection | No |
| A4 | Atreya+ SRT 2026 | arXiv | TinyLlama-1.1B, Llama-3.2-3B | Yes | No: SM-2 adaptive; exposures not matched/reported | No (no fixed-expanding baseline) |
| A5 | Feng+ FOREVER 2026 | ACL 2026 | Qwen3-0.6B/4B, LLaMA-3.1-8B, LLaMA-2-13B (LoRA) | Yes | Partly (one ablation table) | **Yes, Table 3: expanding > uniform** (one setting, no variance) |
| A6 | Lu+ MSSR 2026 | arXiv | Qwen2.5-7B, Llama-3.1-8B, Mistral-7B, Gemma2-9B (LoRA) | Yes | No (adaptive sampler + decaying volume) | Partial (fixed vs geometric vs Ebbinghaus sequences; setup unclear) |
| A7 | M'hamdi & May 2024 | NAACL | M-BERT | Yes | No (Leitner decides *what* is stored) | No |
| A8 | Kang+ VBM 2025 | CVPR | ResNet-18, ViT-B/16 | Yes | **Yes** (interval length at matched exposures) | No (constant intervals only) |
| A9 | Kline 2025 | arXiv | MLP on MNIST | Yes | No | No |
| A10 | Zuo+ 2026 | ICLR-W | 100M transformers | (data reuse) | Partly | No |
| B1 | Klasson+ 2023 | TMLR | 2-layer MLP, small ConvNet, reduced ResNet-18 | Yes | Yes (task-proportion schedules at fixed memory) | No |
| B2–B9 | Classic replay / CLS work | various | small nets / RL agents / theory | Yes | n/a | No |
| C1 | Tirumala+ 2022 | NeurIPS | 125M–13B (spacing test at 125M) | Repetition yes | Partly | No; **spacing length had minimal effect** |
| C2 | Liao+ 2025 | ACL | GPT-2 XL 1.5B, GPT-2 124M | Yes (periodic high-intensity replay) | No | No |
| C3 | Yang+ 2024 | NeurIPS | Pythia 160M–2.8B | Cyclic re-exposure: anticipatory recovery | Order vs shuffle | No |
| C4 | Chang+ 2024 | NeurIPS | OLMo-1B, OLMo-7B | Repeats help short-term; duplicates forget faster | Fixed 100-step interval only | No |
| C5 | Xiong & Wu 2025 | Acta Psychologica | GPT-4-mini-class, Llama 3.1, DeepSeek-V3 (in-context) | n/a | n/a | No; LLMs *preferred juxtaposed (massed)* in context |
| C6 | Sun+ 2026 / Spaced KD 2025 | Patterns / ICML | image ANNs; Drosophila | n/a ("spacing" = interval between teacher and student) | — | No |
| C7 | Speckmann & Eimer 2025 | RLDM | RL agents (MiniGrid) | Leitner/SuperMemo transfer poorly | — | No |
| D1–D9 | Knowledge acquisition / continual pretraining | various | 44M–10B | Replay helps (Ibrahim, Scialom, d'Autume) | — | No |

**Bottom line of the map.**
- "Review/replay reduces forgetting" is thoroughly established, including for ≤1B LMs.
- Almost every spaced-repetition-for-NN paper mixes *timing* with *item selection* (difficulty- or loss-driven) and often with *exposure count*.
- The only matched-budget comparisons of fixed expanding vs. uniform schedules found are:
  - FOREVER (ACL 2026), on task-level continual instruction tuning: expanding wins, but in one setting with no variance reported.
  - MSSR (preprint, 2026): expanding wins, but the setup is under-specified.
- **No study found tests fixed expanding vs. uniform review of *facts*, at matched count and steps, with a delayed post-buffer retention test and seed-paired statistics.**
- The closest LM result pointing toward a null is Tirumala et al. (2022): spacing length had minimal effect on the forgetting floor in a 125M LM.

---

## 2. Section A — Spaced repetition / review scheduling applied to training neural networks and LMs

### A1. Amiri, Miller & Savova (2017), "Repeat before Forgetting" [V-pub, V-content]
- **Citation:** Amiri, H., Miller, T., & Savova, G. (2017). Repeat before forgetting: Spaced repetition for efficient and effective training of neural networks. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing* (pp. 2401–2410). Association for Computational Linguistics. https://doi.org/10.18653/v1/D17-1255 (https://aclanthology.org/D17-1255/)
- **Model scale:** fastText-style MLP (IMDb sentiment), a CNN (CIFAR-10), and an LSTM seq2seq (arithmetic addition). All are small. ⚠ Scale caveat: non-LLM, small networks.
- **Summary:**
  - The paper first shows that network "retention" of an instance falls with the delay since it was last trained on. It also depends on item difficulty (loss) and network strength.
  - It then proposes RbF, a scheduler that sets each instance's next review delay from its loss and from validation performance.
  - It compares RbF with a Leitner-queue scheduler, in which queue *i* is reviewed every 2^i epochs (an expanding schedule), with curriculum learning, and with standard ("rote") training.
  - RbF uses 34–50% of the data per epoch, trains 2.9–4.8× faster, and matches or beats the baselines.
- **Relation to our paper:**
  - This is the founding citation for "spaced repetition for NN training."
  - Its goal is efficiency (skipping easy items), not retention after a delay.
  - Timing is entangled with per-item selection, and the number of exposures *differs* across conditions by design.
  - It does not compare fixed expanding vs. uniform intervals at matched exposure.

### A2. Amiri (2019), "Neural Self-Training through Spaced Repetition" [V-pub, V-content]
- **Citation:** Amiri, H. (2019). Neural self-training through spaced repetition. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1* (pp. 21–31). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1003 (https://aclanthology.org/N19-1003/)
- **Model scale:** FastText deep-averaging network (IMDb) and a basic CNN (Churn). ⚠ Scale caveat: small text classifiers.
- **Summary:** Leitner queues decide which pseudo-labelled unlabeled instances are sampled at each self-training episode. Instances are promoted or demoted by classification accuracy. This beats standard self-training baselines on IMDb and Churn.
- **Relation to our paper:** Here spaced repetition is used as a *sampling* policy, not a retention intervention. It has no forgetting test and does not isolate timing. Cite as background only.

### A3. Prakriya, Yen, Hsieh & Cong — LFR (arXiv 2409.06131, published at CoNLL 2025) [V-pub, V-content]
- **Citation:** Prakriya, N., Yen, J.-N., Hsieh, C.-J., & Cong, J. (2025). Accelerating large language model pretraining via LFR pedagogy: Learn, focus, and review. In *Proceedings of the 29th Conference on Computational Natural Language Learning* (pp. 268–290). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.conll-1.18 (https://aclanthology.org/2025.conll-1.18/; preprint https://arxiv.org/abs/2409.06131)
  - **Note:** The arXiv ID 2409.06131 *was* later published, at CoNLL 2025. Cite the CoNLL version.
- **Model scale:** Llama-2-architecture models at 120M, 300M and 500M, pretrained from scratch on SlimPajama. GPT-2 at 124M–1.5B on OpenWebText. **Directly relevant to our ≤ few-B regime; it includes a 300M Llama.**
- **Summary:**
  - Profiling pretraining shows "multiple descent": 25–78% of samples are forgotten and relearned several times.
  - LFR tracks perplexity for each data block. It "focuses" on high-perplexity blocks and periodically "reviews" all blocks.
  - It reaches lower perplexity and higher downstream accuracy using 5–19% of the training tokens. It matches Pythia models up to 2× larger.
- **Relation to our paper:**
  - Shows that review-like scheduling helps in small LMs, but the scheduler is *adaptive selection by difficulty*. It is a data-selection method, not a timing test.
  - Exposure counts differ across conditions, and there is no fixed expanding-vs-uniform contrast or delayed retention test.
  - Its "multiple descent" observation supports our premise that facts fade during continued training.

### A4. Atreya et al. (2026), "When to Review: Spaced Repetition for Continual Pre-Training of Language Models" (SRT) [V-arXiv, V-content]
- **Citation:** Atreya, A., Batra, D., Mantri, Y. K., Bantug, G., Cowan, G. A., & Khraishi, R. (2026). *When to review: Spaced repetition for continual pre-training of language models*. arXiv:2608.17530. https://arxiv.org/abs/2608.17530
- **Model scale:** TinyLlama-1.1B-Chat (main runs and all ablations) and Llama-3.2-3B-Instruct, full continual pretraining, seq len 2048. Auxiliary small CNNs/MLPs on MNIST, Fashion-MNIST, CIFAR-10 and Wine.
- **Summary:**
  - SRT schedules rehearsal for each example with SuperMemo-2 (SM-2). Per-example perplexity becomes a recall-quality score that sets the next interval: hard items return sooner, retained items later. This is an expanding schedule for each item.
  - On temporally split Wikipedia and code corpora, SRT recovers 5–37 points of the old-knowledge QA accuracy lost under naive continual pretraining (CPT). It also beats a 20/80 uniform-random replay baseline.
  - Single run per condition.
- **Relation to our paper:** **The most recent and most on-topic preprint.**
  - Timing is confounded with selection: SM-2 is driven by recall quality.
  - Only the *nominal* old/new ratio is matched. Realized exposure counts are not reported, and unused due-slots are reassigned.
  - There is **no fixed expanding or fixed uniform-interval baseline**, and no seeds or CIs.
  - Our study is complementary. We hold item set, review count and steps fixed and vary only interval shape.

### A5. Feng et al. (2026), FOREVER [V-pub (accepted, ACL 2026 camera-ready per arXiv comment and Semantic Scholar), V-content]
- **Citation:** Feng, Y., Wang, H., Li, J., Chu, X., Kang, Z., Liu, Y., Wang, Y., Yu, P. S., & Wu, X.-M. (2026). FOREVER: Forgetting curve-inspired memory replay for language model continual learning. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics*. arXiv:2601.03938. https://arxiv.org/abs/2601.03938
  - The ACL Anthology ID was not yet confirmed; use the arXiv URL until it appears.
- **Model scale:** Qwen3-0.6B (main and all ablations), Qwen3-4B, LLaMA-3.1-8B and LLaMA-2-13B. LoRA is the main setting, with full fine-tuning in an appendix. The benchmarks are task-sequence continual instruction tuning: Standard CL (5 tasks), Long Sequence (15 tasks) and SuperNI (15 tasks), with 1,000 samples per task and a 2% buffer.
- **Summary:**
  - Replay times follow Ebbinghaus intervals {1, 2, 4, 7, 15, 30…} in "model time," measured by the cumulative size of optimizer updates rather than raw steps. An intensity-aware regularizer is added.
  - FOREVER beats replay baselines, including fixed-interval replay and VBM, by roughly 1 point OP/BWT on average.
  - **Table 3 ("same replay budget"; SuperNI, Qwen3-0.6B, task order 7), OP/BWT:**
    - Ebbinghaus-style expanding {1,2,4,7,15}: 42.5/−2.8
    - Exponential {1,2,4,8,16}: 42.3/−2.6
    - Polynomial {1,4,9,16}: 41.5/−3.2
    - **Uniform: 40.9/−3.6**
    - Decreasing: 37.2/−7.8
  - The authors conclude that "increasing-spacing schedules consistently outperform uniform."
- **Relation to our paper:** **The most important prior work to discuss.** It is the only peer-reviewed matched-budget expanding-vs-uniform comparison found in LMs, and it **reports the opposite of our null** (+1.6 OP for expanding).
  - Differences reviewers will weigh:
    1. Task-level CL with accuracy metrics vs. our fact-level knowledge with held-out paraphrase loss.
    2. A single task order with no variance reported vs. our 3 paired seeds with CIs.
    3. Their schedule is in model time, and the uniform schedule spans the whole sequence. We measure *after* a 180-step no-review buffer, which is the delayed test that Karpicke & Roediger (2007) argue reverses short-term expanding advantages.
  - Our trajectory-metric result (expanding better *during* Stage 2, washed out after the buffer) offers a possible way to reconcile the two. FOREVER's measurements are effectively closer to "during-training" than to a long delay.
  - **Scale:** Its schedule comparison is at 0.6B, comparable to our setting.

### A6. Lu, He, Chen & Zha (2026), MSSR [V-arXiv, V-content (partial)]
- **Citation:** Lu, Y., He, Y., Chen, J., & Zha, H. (2026). *MSSR: Memory-aware adaptive replay for continual LLM fine-tuning*. arXiv:2603.09892. https://arxiv.org/abs/2603.09892
- **Model scale:** Qwen2.5-7B (main), Llama-3.1-8B, Mistral-7B-v0.3 and Gemma2-9B, all LoRA. ⚠ Scale caveat: 7–9B only.
- **Summary:**
  - Replay intervals expand by a saturating formula, and replay volume decays over time. A sample-level memory-strength estimate (from loss and time) weights which items are replayed.
  - It beats fixed, loss-based and accuracy-based replay by about 1 point averaged over 3 tasks, and by up to +0.108 on ARC.
  - Appendix tables compare hand-set interval sequences: Fixed (3,3,3) 76.1% vs. Geometric (1,3,7,15) 77.8% vs. MSSR (1,2,4,7,15) 78.5%. The setup for these is not specified, and the tables are internally inconsistent.
  - Only the number of optimization steps is matched. Replayed-sample counts are not said to be equal, and no seeds or variance are reported.
- **Relation to our paper:** This is a second (non-peer-reviewed) claim that expanding beats fixed. It is weaker evidence than FOREVER because exposure is not matched and timing is confounded with sampling.

### A7. M'hamdi & May (2024), Leitner-guided memory replay [V-pub, V-content]
- **Citation:** M'hamdi, M., & May, J. (2024). Leitner-guided memory replay for cross-lingual continual learning. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)* (pp. 7808–7821). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.naacl-long.432 (https://aclanthology.org/2024.naacl-long.432/)
- **Model scale:** M-BERT encoder (~180M) with intent/slot heads, on cross-lingual NLU streams.
- **Summary:** Leitner-queue skill ratings decide which items fill and stay in a fixed-size replay memory. It reduces forgetting compared with vanilla and strong replay baselines. Replaying *easy* (well-learned) items beats replaying hard ones.
- **Relation to our paper:** Leitner is used for *what* to replay, not *when*. This supports "review helps," not schedule shape. The easy-over-hard finding is an interesting contrast to RbF/SRT, which replay hard items more.

### A8. Kang, Seifer, Lee & Ryu (2025), "Do Your Best and Get Enough Rest for Continual Learning" (VBM) [V-pub (CVPR 2025 per arXiv comment), V-content]
- **Citation:** Kang, H., Seifer, G., Lee, D., & Ryu, J. (2025). Do your best and get enough rest for continual learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)*. arXiv:2503.18371. https://arxiv.org/abs/2503.18371
  - The CVF open-access page number was not retrieved.
- **Model scale:** ResNet-18 trained from scratch, and ViT-B/16 pretrained on ImageNet-21K (prompt-based CL), on Split CIFAR-10/100, Tiny-ImageNet, ImageNet-R and DomainNet. ⚠ Scale caveat: vision only, not LMs.
- **Summary:**
  - Motivated by Ebbinghaus, the method lengthens the "recall interval" between revisits of the same sample. Each batch holds V augmented views of fewer samples, and the number of epochs is divided by V so total exposures stay matched.
  - The recall-interval multiplier shows an **inverted U**: iCaRL is best at ×4 and DER++ at ×3. This adds 1–5 points across CL methods.
- **Relation to our paper:** **One of the few designs that varies interval length at matched exposure.** It tests only *constant* intervals of different lengths, not expanding vs. uniform. The inverted U (some spacing helps, too much hurts) is the analogue of the human spacing-lag effect. It suggests our 12-event schedules might differ less in shape than in average lag.

### A9. Kline (2025), "Human-like Forgetting Curves in Deep Neural Networks" [V-arXiv, V-content]
- **Citation:** Kline, D. (2025). *Human-like forgetting curves in deep neural networks*. arXiv:2506.12034. https://arxiv.org/abs/2506.12034
- **Model scale:** MLPs on MNIST. ⚠ Scale caveat: tiny non-LM networks.
- **Summary:** Recall probability is estimated from the similarity between current hidden states and stored class prototypes. The MLPs show Ebbinghaus-like forgetting curves, and scheduled reviews make knowledge more durable.
- **Relation to our paper:** FOREVER cites it as evidence of "Ebbinghaus-like forgetting in LLMs," but it studies MLPs, not LLMs. This is a citation chain worth flagging. Weak, single-author preprint; cite cautiously, if at all.

### A10. Zuo et al. (2026), "Train Smarter, Not Longer: Memorization-Guided Data Reuse" [V-arXiv]
- **Citation:** Zuo, J., Zeng, C., Chahed, I., Velikanov, M., Rhaiem, D. E., Balsebre, P., Kumar, A., Belkada, Y., & Hacid, H. (2026). *Train smarter, not longer: Memorization-guided data reuse for efficient LLM training*. 3rd DATA-FM Workshop at ICLR 2026. arXiv:2607.04969. https://arxiv.org/abs/2607.04969
- **Model scale:** 100M-parameter decoder-only transformers (FineWeb; OpenMathInstruct2).
- **Summary:** Proposes a "memorization window" from loss-retention dynamics as a guide to *when* to re-expose data in multi-epoch training. Repetition keeps helping beyond the common four-epoch limit. A full scheduler is left to future work.
- **Relation to our paper:** Conceptually adjacent: when to repeat data in small LMs. It has no expanding-vs-uniform test. Workshop paper.

### A11. Bamnodkar (2025), TFC-SR [V-arXiv] (minor)
- **Citation:** Bamnodkar, P. (2025). *Task-focused consolidation with spaced recall: Making neural networks learn like college students*. arXiv:2507.21109. https://arxiv.org/abs/2507.21109
- **Model scale:** Simple CNN (Split MNIST) and ResNet-18 (Split CIFAR-100). ⚠ Scale caveat: vision only.
- **Summary:** Adds a periodic "active recall probe" to experience replay. Final accuracy on Split CIFAR-100 is 13.17% vs. 7.40% for standard ER. The author attributes the gain to the probe, not to replay volume.
- **Relation to our paper:** Only marginally relevant. Single-author preprint.

---

## 3. Section B — Replay and rehearsal timing in continual learning (classic and scheduling work)

### B1. Klasson, Kjellström & Zhang (2023), "Learn the Time to Learn" [V-pub, V-content]
- **Citation:** Klasson, M., Kjellström, H., & Zhang, C. (2023). Learn the time to learn: Replay scheduling in continual learning. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=Q4aAITDgdP (arXiv:2209.08660)
- **Model scale:** 2-layer MLP (256 units) on Split/Permuted MNIST, FashionMNIST and notMNIST. Small 4-block ConvNet on Split CIFAR-100. Reduced ResNet-18 on Split miniImageNet. ⚠ Scale caveat: small non-LM networks.
- **Summary:**
  - With a fixed-size replay memory, the method chooses *which tasks* (task proportions) to replay at each task boundary. It uses Monte Carlo tree search, and RL (DQN/A2C) policies that generalize to new task orders.
  - Learned schedules beat Random, Equal Task Schedule (uniform over tasks) and heuristic accuracy-drop schedules, at no extra compute.
  - The authors explicitly motivate the work with spaced repetition. They state that expanding schedules have "been shown to improve memory retention better [than] uniformly spaced rehearsal" (citing Landauer & Bjork, 1977/1978; Hawley et al., 2008).
- **Relation to our paper:**
  - Establishes that *replay timing/allocation* matters at fixed budget, but at task granularity, in small vision networks.
  - It does not compare fixed expanding vs. uniform intervals.
  - Its human-literature premise (expanding > uniform) is the one our paper challenges via Karpicke & Roediger (2007). This is useful for framing.

### B2. Hayes et al. (2021), "Replay in Deep Learning" (review) [V-pub]
- **Citation:** Hayes, T. L., Krishnan, G. P., Bazhenov, M., Siegelmann, H. T., Sejnowski, T. J., & Kanan, C. (2021). Replay in deep learning: Current approaches and missing biological elements. *Neural Computation, 33*(11), 2908–2950. https://doi.org/10.1162/neco_a_01433 (arXiv:2104.04132)
- **Model scale:** Review; not applicable.
- **Summary:** Systematically compares replay in the mammalian brain with replay in ANNs. It lists biological features missing from deep-learning replay, such as sleep and offline replay, selective and prioritized replay, temporal compression, and replay timing relative to learning. It proposes them as directions for research.
- **Relation to our paper:** Cite it to motivate asking whether replay *timing* matters. Our experiment is a minimal test of one "missing element" (schedule shape).

### B3. Rolnick et al. (2019), CLEAR / experience replay for continual learning [V-pub]
- **Citation:** Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html (arXiv:1811.11682)
- **Model scale:** IMPALA-style actor-critic RL agents (DMLab/Atari multi-task). ⚠ Scale caveat: RL agents, not LMs. The architecture details are from memory of the paper and were not re-checked in full text.
- **Summary:** CLEAR mixes on-policy learning with off-policy replay plus behavioral cloning. It greatly reduces catastrophic forgetting in multi-task RL.
- **Relation to our paper:** Background evidence that replay works. It does not address timing.

### B4. Chaudhry et al. (2019), tiny episodic memories [V-arXiv, V-content]
- **Citation:** Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P. K., Torr, P. H. S., & Ranzato, M. (2019). *On tiny episodic memories in continual learning*. arXiv:1902.10486. https://arxiv.org/abs/1902.10486
  - Earlier title: "Continual learning with tiny episodic memories." No peer-reviewed venue was found.
- **Model scale:** 2-hidden-layer MLP (256 units) on MNIST. Reduced ResNet-18 on CIFAR and miniImageNet. ImageNet-pretrained ResNet-18 on CUB. ⚠ Scale caveat: small vision networks.
- **Summary:** In single-pass CL, jointly training on the current batch plus a very small episodic memory (ER) beats specialized CL methods. Even 1 example per class helps (+7–17%) without overfitting.
- **Relation to our paper:** Strong precedent that *small* replay budgets help. This is consistent with our 12-event budget producing a reliable gain.

### B5. Robins (1995), rehearsal and pseudorehearsal [V-pub]
- **Citation:** Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318
- **Model scale:** Small back-propagation networks. ⚠ Scale caveat.
- **Summary:**
  - Replicates Ratcliff (1990) "recency rehearsal" (rehearse the most recent items).
  - Shows that other rehearsal regimes, especially "sweep rehearsal" (a changing random subset each epoch), minimize forgetting better.
  - Introduces pseudorehearsal, which needs no stored data.
- **Relation to our paper:** **The earliest demonstration that *how rehearsal is scheduled/selected* changes forgetting at a similar budget.** Cite it as the origin of rehearsal-regime comparisons. Note that it varies the *selection* of rehearsed items rather than interval shape.

### B6. van de Ven, Siegelmann & Tolias (2020), brain-inspired replay [V-pub]
- **Citation:** van de Ven, G. M., Siegelmann, H. T., & Tolias, A. S. (2020). Brain-inspired replay for continual learning with artificial neural networks. *Nature Communications, 11*, Article 4069. https://doi.org/10.1038/s41467-020-17866-2
- **Model scale:** MLPs (split/permuted MNIST) and a CNN with a generative model on CIFAR-100. ⚠ Scale caveat: small vision networks. The architecture details are recalled, not re-checked.
- **Summary:** Generative "replay through feedback" of internal representations, with context gating, gives strong class-incremental performance without storing data.
- **Relation to our paper:** Background. It concerns *what* is replayed (generated vs. stored), not timing.

### B7. McClelland, McNaughton & O'Reilly (1995), complementary learning systems [V-pub]
- **Citation:** McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review, 102*(3), 419–457. https://doi.org/10.1037/0033-295X.102.3.419
- **Model scale:** Small connectionist models (theory). Not applicable.
- **Summary:** A slow-learning "neocortex" learns structure only when new items are *interleaved* with old ones. The hippocampus supports fast learning and reinstates memories to interleave them, so interleaved replay is what prevents catastrophic interference.
- **Relation to our paper:** The theoretical root of "review during new learning." Our review arms are a form of interleaving old facts into new-fact training.

### B8. Kumaran, Hassabis & McClelland (2016), CLS updated [V-pub]
- **Citation:** Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. *Trends in Cognitive Sciences, 20*(7), 512–534. https://doi.org/10.1016/j.tics.2016.05.004
- **Model scale:** Review; not applicable.
- **Summary:**
  - Extends CLS theory: replay can *reweight* experience in a goal-dependent way, not just mirror environment statistics.
  - Schema-consistent information can be integrated quickly.
  - Links the theory to experience replay in deep RL.
- **Relation to our paper:** Motivates schedule and weighting choices for replay. Also predicts that integration speed depends on consistency with prior knowledge. FictionalQA facts are novel, but they follow known schemas.

### B9. McClelland, McNaughton & Lampinen (2020), and Saxena, Shobe & McNaughton (2022), similarity-weighted interleaved learning (SWIL) [V-pub]
- **Citations:**
  - McClelland, J. L., McNaughton, B. L., & Lampinen, A. K. (2020). Integration of new information in memory: New insights from a complementary learning systems perspective. *Philosophical Transactions of the Royal Society B, 375*(1799), 20190637. https://doi.org/10.1098/rstb.2019.0637
    - **Correction to the brief:** This is *Phil. Trans. R. Soc. B*, not PNAS.
  - Saxena, R., Shobe, J. L., & McNaughton, B. L. (2022). Learning in deep neural networks and brains with similarity-weighted interleaved learning. *Proceedings of the National Academy of Sciences, 119*(27), e2115229119. https://doi.org/10.1073/pnas.2115229119
    - The PNAS paper is this follow-up.
- **Model scale:**
  - McClelland et al.: deep *linear* network with one hidden layer, on a toy hierarchy.
  - Saxena et al.: deep linear network (Fashion-MNIST), a 6-layer CNN (CIFAR-10/100), and VGG11/VGG19 (CIFAR-100).
  - ⚠ Scale caveat: vision/toy, not LMs.
- **Summary:** Interleaving only old items that are *similar* to the new item gives the same accuracy as full interleaving. McClelland et al. report ~2.5× fewer presentations, and Saxena et al. report up to ~30–160× speedups. The gain grows with the number of non-overlapping stored classes.
- **Relation to our paper:** Shows that *which* old items are reviewed can matter as much as *how much*. That is a different axis from our *when*. A natural follow-up is to review only old facts that interfere with the new ones (e.g., same fictional entity or document).

### B10. de Masson d'Autume, Ruder, Kong & Yogatama (2019), sparse experience replay in lifelong language learning [V-pub, V-content]
- **Citation:** de Masson d'Autume, C., Ruder, S., Kong, L., & Yogatama, D. (2019). Episodic memory in lifelong language learning. In *Advances in Neural Information Processing Systems 32*. https://proceedings.neurips.cc/paper/2019/hash/f8d2e80c1458ea2501f98a2cafadb397-Abstract.html (arXiv:1906.01076)
- **Model scale:** BERT-base encoder (~110M) for text classification and QA streams.
- **Summary:** Replay happens at **fixed, sparse intervals**: 100 stored examples every 10,000 new examples, a 1% ratio, one gradient step. Combined with local adaptation at test time, this greatly reduces forgetting. Memory can be cut by 50–90% with little loss.
- **Relation to our paper:** **An early LM precedent for uniform, sparse, interval-based replay.** It shows a ~1% fixed-interval budget suffices, which supports a "uniform is a good default" framing. It does not compare schedule shapes.

### B11. Huang et al. (2024), Self-Synthesized Rehearsal (SSR) [V-pub, V-content]
- **Citation:** Huang, J., Cui, L., Wang, A., Yang, C., Liao, X., Song, L., Yao, J., & Su, J. (2024). Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1416–1428). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.acl-long.77
- **Model scale:** Llama-2-7B, Llama-2-7B-chat and Alpaca-7B, on SuperNI task sequences. (At the 7B boundary.)
- **Summary:** The LLM generates its own rehearsal instances (via ICL, then refined by the latest model), which replace real stored data. This matches or beats conventional rehearsal while preserving general ability.
- **Relation to our paper:** Concerns the replay *source*, not timing. Relevant if the team later lacks access to the old facts.

---

## 4. Section C — Do neural networks / LMs show spacing or human-like memory effects?

### C1. Tirumala, Markosyan, Zettlemoyer & Aghajanyan (2022), "Memorization Without Overfitting" [V-pub, V-content]
- **Citation:** Tirumala, K., Markosyan, A., Zettlemoyer, L., & Aghajanyan, A. (2022). Memorization without overfitting: Analyzing the training dynamics of large language models. In *Advances in Neural Information Processing Systems 35* (pp. 38274–38290). https://doi.org/10.52202/068431-2773 (https://proceedings.neurips.cc/paper_files/paper/2022/hash/fa0509f4dab6807e2cb465715bf2d249-Abstract-Conference.html; arXiv:2205.10770)
- **Model scale:** Causal and masked transformer LMs at 125M, 355M, 1.3B, 2.7B, 6.7B and 13B (FairSeq, WikiText-103 and others). **The repetition and spacing experiments are at 125M.**
- **Summary:**
  - A held-out "special batch" injected once is forgotten quickly, then plateaus at a "forgetting baseline." The baseline rises with model size, so larger models forget less.
  - **Repetition and spacing (Fig. 10, 125M causal LM, WikiText-103):**
    - *Repeated (massed) injection* raises the forgetting baseline monotonically with the number of repetitions.
    - *Spaced repetition* (periodically injecting the batch) has "minimal effect on the forgetting baseline … independent of the length of spacing between the repetitions."
  - The paper explicitly cites Karpicke & Roediger (2007) and Amiri et al. (2017) as motivation.
- **Relation to our paper:** **The closest prior LM result to our null, and it should be cited prominently.**
  - It is already a peer-reviewed LM finding that *spacing length* barely matters for long-run retention. Our result extends it in several ways:
    - It compares fixed expanding vs. uniform *shapes*, where Tirumala et al. vary spacing length.
    - It holds review count and steps matched.
    - It uses LoRA, fictional QA facts, and generalization to held-out paraphrases, where Tirumala et al. measure exact memorization.
    - It reports seed-paired CIs.
  - Their massed-repetition benefit also suggests adding a massed arm to our design.

### C2. Liao, Xie, Sun, Sun & Kang (2025), "Exploring Forgetting in Large Language Model Pre-Training" [V-pub, V-content]
- **Citation:** Liao, C., Xie, R., Sun, X., Sun, H., & Kang, Z. (2025). Exploring forgetting in large language model pre-training. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 2112–2127). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.acl-long.105 (arXiv:2410.17018)
- **Model scale:** GPT-2 XL (1.5B) for the initial analysis. GPT-2 (0.1B) for most later experiments, for compute reasons.
- **Summary:**
  - Shows that forgetting happens even within pretraining on a single distribution. Perplexity hides it, so the authors introduce entity-memory metrics.
  - Memory replay reduces forgetting.
  - Forgetting curves after "high-intensity" learning resemble human curves (citing Loftus, 1985).
  - *Periodic high-intensity replay* (revisiting every 1,000 steps, 5 epochs per session) beats one-shot heavy replay at lower cost.
- **Relation to our paper:** Supports "periodic review helps" in ≤1.5B LMs. It confounds timing with replay intensity, and it has no expanding-vs-uniform test.

### C3. Yang, Jones, Mozer & Ren (2024), "Reawakening Knowledge" [V-pub, V-content]
- **Citation:** Yang, Y., Jones, M., Mozer, M. C., & Ren, M. (2024). Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training. In *Advances in Neural Information Processing Systems 37*. https://doi.org/10.52202/079017-2621 (arXiv:2403.09613; https://neurips.cc/virtual/2024/poster/94697)
  - **Correction to the brief:** The authors are Yang, Jones, Mozer & Ren, not "Yang, Mehta, Zhang."
- **Model scale:** Pythia 160M, 410M, 1B, 1.4B and 2.8B (fine-tuned), plus randomly initialized variants. Also Image GPT, ViT, VGG-19 and GPT2-large.
- **Summary:**
  - When documents are fine-tuned in a fixed cyclic order, loss on a document starts *falling before* it is seen again ("anticipatory recovery"). On Pythia-1B, more than half of the forgetting is recovered.
  - The effect emerges with scale: a sharp jump from 160M to 410M.
  - It disappears under random reshuffling.
  - Under fixed order the cyclic schedule gets better online loss than shuffling, e.g., 1B: 1.03 vs. 1.51.
- **Relation to our paper:**
  - Shows that *structured, predictable re-exposure order* changes forgetting dynamics in ≤3B LMs. Ordering is a real lever even when exposure counts are equal.
  - Important caveat for our design: if old-fact reviews recur in a fixed order, anticipatory dynamics could arise.
  - It does not compare expanding vs. uniform.
  - Mozer is also an author of classic human spacing models, so this is a natural bridge citation.

### C4. Toneva et al. (2019), example forgetting [V-pub]
- **Citation:** Toneva, M., Sordoni, A., Tachet des Combes, R., Trischler, A., Bengio, Y., & Gordon, G. J. (2019). An empirical study of example forgetting during deep neural network learning. In *International Conference on Learning Representations (ICLR 2019)*. https://arxiv.org/abs/1812.05159 (ICLR poster page: https://iclr.cc/virtual/2019/poster/753)
- **Model scale:** Small CNNs and ResNet-18-class networks on MNIST, permuted MNIST and CIFAR-10. ⚠ Scale caveat: vision.
- **Summary:** Even within i.i.d. training, individual examples undergo "forgetting events" (correct → incorrect). Some are never forgotten and others are forgotten often. Unforgettable examples can be removed with little loss in generalization.
- **Relation to our paper:** Supports the premise that items fade between exposures even without distribution shift. It motivates per-item review, but does not address timing.

### C5. Xiong & Wu (2025), interleaved and spaced practice in LLMs vs. humans [V-pub]
- **Citation:** Xiong, Y., & Wu, S. (2025). Do large language models learn like humans: Interleaved and spaced practice in morphological learning. *Acta Psychologica, 260*, 105518. https://doi.org/10.1016/j.actpsy.2025.105518
- **Model scale:** API LLMs named "GPT4mini" (sic), Llama 3.1 and DeepSeek-V3, tested **in-context** (no weight updates).
- **Summary:** In an artificial-language morphology task, humans benefited from spaced exemplars but **LLMs did better with juxtaposed (massed) input**. Interleaving effects were model-dependent. The authors conclude that human spacing and interleaving principles do not transfer straightforwardly to LLMs.
- **Relation to our paper:** This is *in-context* spacing, a different mechanism from gradient-based review. Still, it is peer-reviewed evidence that LLMs need not show human spacing effects, and it is consistent with our null. Keep the in-context vs. in-weights distinction explicit.

### C6. Sun et al. (2026), "Spacing effect improves generalization in biological and artificial systems," and Sun et al. (2025), Spaced KD [V-pub]
- **Citations:**
  - Sun, G., Huang, N., Yan, H., Zhou, J., Li, Q., Lei, B., Zhong, Y., & Wang, L. (2026). Spacing effect improves generalization in biological and artificial systems. *Patterns, 7*(6), 101564. https://doi.org/10.1016/j.patter.2026.101564
  - Sun, G., Yan, H., Wang, L., Li, Q., Lei, B., & Zhong, Y. (2025). Right time to learn: Promoting generalization via bio-inspired spacing effect in knowledge distillation. In *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*. https://openreview.net/forum?id=WuP4hwvLzo (arXiv:2502.06192)
    - Rejected at ICLR 2025, then accepted as an ICML 2025 poster, per OpenReview.
- **Model scale:** Image-classification ANNs (ResNet-family on CIFAR / Tiny-ImageNet, among others). The Patterns paper also includes *Drosophila* olfactory conditioning. ⚠ Scale caveat: vision, not LMs.
- **Summary:**
  - "Spacing" is implemented as intervals between teacher and student in knowledge distillation, in weight averaging, and in dropout-like variation.
  - Generalization follows an **inverted U** in interval length. Periodic spacing beats irregular schedules.
- **Relation to our paper:** The word "spacing" here means something different, not item-review scheduling. Cite only to show that "spacing effects in ANNs" now exist in other senses, and to distinguish our question from them. The inverted U echoes VBM.

### C7. Speckmann & Eimer (2025), Leitner/SuperMemo in multi-task RL [V-arXiv; RLDM 2025 extended abstract]
- **Citation:** Speckmann, M., & Eimer, T. (2025). *Task scheduling & forgetting in multi-task reinforcement learning*. Presented at RLDM 2025. arXiv:2503.01941. https://arxiv.org/abs/2503.01941
- **Model scale:** RL agents on MiniGrid. ⚠ Scale caveat: RL.
- **Summary:** RL agents show human-like forgetting curves in many cases. However, Leitner and SuperMemo task schedules "do not transfer as well to RL" as a value-error-based curriculum (PLR). The authors blame asymmetric learning and retention across tasks.
- **Relation to our paper:** Another negative or neutral transfer of human spaced-repetition schedulers to ANN training. Supports our framing that human scheduling heuristics are not automatically beneficial.

### C8. Non-citable or weak items (flagged)
- **SpacingLab** (GitHub: madara88645/SpacingLab). A personal laptop-scale project, not peer-reviewed. It reports spaced > massed exposures for fact retention in small-LM fine-tuning across 5 seeds.
  - **Do not cite as evidence.** Mention, if at all, only as anecdotal.
  - Only third-party summaries were seen; the source was not inspected.
- **"The Geometry of Forgetting"** (Ray Barman et al., 2026, arXiv:2604.06222) and **"The Price of Meaning"** (arXiv:2603.27116, 2026). Embedding-space and retrieval simulations of human memory phenomena, not gradient training.
  - A summary reported the argument that spacing benefits reduce to "the latest trace dominates under age-dependent corruption." That idea is relevant to interpreting our matched-final-review design (see Synthesis).
  - [V-arXiv for 2604.06222 metadata; the 2603.27116 content is UNVERIFIED]

---

## 5. Section D — When facts appear in training, knowledge acquisition and forgetting, and replay in continual pretraining

### D1. Chang et al. (2024), "How Do Large Language Models Acquire Factual Knowledge During Pretraining?" [V-pub, V-content]
- **Citation:** Chang, H., Park, J., Ye, S., Yang, S., Seo, Y., Chang, D.-S., & Seo, M. (2024). How do large language models acquire factual knowledge during pretraining? In *Advances in Neural Information Processing Systems 37* (pp. 60626–60668). https://doi.org/10.52202/079017-1939 (arXiv:2406.11813)
- **Model scale:** **OLMo-1B and OLMo-7B**, resumed from intermediate checkpoints (~170B, 500B and 1.5T tokens). Batch size swept from 2048 down to 128.
- **Summary:**
  - Fictional-knowledge passages are injected into OLMo pretraining batches. Facts are acquired through small gains at each exposure, followed by forgetting.
  - Forgetting follows a **power law** in steps.
  - *Duplicated* injections (10× identical passages, every 100 steps) give bigger immediate gains but **forget faster** than paraphrased injections.
  - Larger batches make forgetting more robust.
  - The authors posit a "learnability threshold": facts seen at intervals longer than a threshold are never acquired.
  - **Only one spacing interval (100 steps) is tested.**
- **Relation to our paper:** **The methodologically closest prior work: fictional facts, OLMo, retention after continued training.**
  - It shows exposure count and paraphrase diversity matter, and it implies intervals matter through the threshold.
  - It never varies schedule shape.
  - Our study fills exactly that gap. Our use of held-out *paraphrased* questions matches their memorization-vs-generalization distinction.
  - Also note their batch-size effect: our batch is 16, the small-batch regime that, by their findings, forgets faster.

### D2. Zucchet et al. (2025), "How do language models learn facts?" [V-pub]
- **Citation:** Zucchet, N., Bornschein, J., Chan, S., Lampinen, A., Pascanu, R., & De, S. (2025). How do language models learn facts? Dynamics, curricula and hallucinations. In *Second Conference on Language Modeling (COLM 2025)*. https://openreview.net/forum?id=vBcGnragkr (arXiv:2503.21676)
- **Model scale:** 8-layer decoder-only transformer (44M non-embedding parameters) on synthetic biographies.
- **Summary:**
  - Fact learning goes through a plateau while recall circuits form. Imbalanced data distributions shorten the plateau, which suggests data-scheduling curricula.
  - Fine-tuning on new individuals quickly corrupts old knowledge.
  - **Adding replay of pre-training data only partially mitigates this.**
- **Relation to our paper:** Shows in a small LM that replay helps only partially when new facts are injected, consistent with our modest 4.3% gain. Its data-distribution scheduling is about *frequency*, not interval shape.

### D3. Jagielski et al. (2023), "Measuring Forgetting of Memorized Training Examples" [V-pub, V-content]
- **Citation:** Jagielski, M., Thakkar, O., Tramèr, F., Ippolito, D., Lee, K., Carlini, N., Wallace, E., Song, S., Thakurta, A., Papernot, N., & Zhang, C. (2023). Measuring forgetting of memorized training examples. In *International Conference on Learning Representations (ICLR 2023)*. https://openreview.net/forum?id=7bJizxLKrR (arXiv:2207.00099)
- **Model scale:** ResNet-50 (ImageNet), Conformer-L (LibriSpeech), and a 110M decoder-only T5-architecture LM on C4.
- **Summary:**
  - Standard models empirically forget examples over training. Examples seen *early* are less exposed to privacy attacks than examples seen *later*: a recency effect.
  - The number of repetitions and example hardness most strongly control forgetting speed.
  - Nondeterminism is identified as a driver; deterministic training does not forget.
- **Relation to our paper:** Documents the recency effect, which matters because the last review position may dominate delayed retention. Supports controlling the final review time across arms.

### D4. Luo et al. (2026), "How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining" [V-pub: ICLR 2026 Oral per OpenReview]
- **Citation:** Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=T5wkZJqzkz (arXiv:2511.18903)
  - Authors verified as listed.
- **Model scale:** Main results at 1.5B parameters / 30B tokens; also mid-training settings.
- **Summary:**
  - Ascending-quality curricula beat random shuffling under a **constant LR**, but the advantage shrinks under standard LR decay, because late (high-quality) data get tiny step sizes.
  - Fixes: moderate decay, or replacing decay with model averaging (CMA/CDMA). This gives +1.64% average on benchmarks.
- **Relation to our paper:** **Directly supports our constant-LR design choice.** Under LR decay, late reviews (as in uniform) would be down-weighted relative to early ones (as in expanding), confounding schedule with LR. Cite it in Methods §2.1.

### D5. Ibrahim et al. (2024), "Simple and Scalable Strategies to Continually Pre-train LLMs" [V-pub]
- **Citation:** Ibrahim, A., Thérien, B., Gupta, K., Richter, M. L., Anthony, Q., Lesort, T., Belilovsky, E., & Rish, I. (2024). Simple and scalable strategies to continually pre-train large language models. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=DimPeeCxKO (arXiv:2403.08763)
- **Model scale:** 405M (main) and 10B, with hundreds of billions of tokens.
- **Summary:** LR re-warming, LR re-decaying, and **replay of a small fraction of previous data** together match full retraining from scratch under English→English and English→German shifts. Proposes alternatives to cosine schedules, such as infinite schedules.
- **Relation to our paper:** Establishes "replay works" for continual pretraining of small LMs. Replay there is uniform mixing at a fixed ratio; schedule shape is not studied.

### D6. Gupta et al. (2023), "Continual Pre-Training of LLMs: How to (re)warm your model?" [V-arXiv; workshop status UNVERIFIED]
- **Citation:** Gupta, K., Thérien, B., Ibrahim, A., Richter, M. L., Anthony, Q., Belilovsky, E., Rish, I., & Lesort, T. (2023). *Continual pre-training of large language models: How to (re)warm your model?* arXiv:2308.04014. https://arxiv.org/abs/2308.04014
- **Model scale:** Pythia 410M (Pile → SlimPajama).
- **Summary:** Re-warming the LR first raises upstream and downstream loss, but in the longer run improves downstream performance. It even outperforms training from scratch on a large downstream set.
- **Relation to our paper:** LR-schedule effects on forgetting. Supports reporting our 10-step warmup plus constant LR as a deliberate control.

### D7. Scialom, Chakrabarty & Muresan (2022), "Fine-tuned Language Models are Continual Learners" [V-pub]
- **Citation:** Scialom, T., Chakrabarty, T., & Muresan, S. (2022). Fine-tuned language models are continual learners. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (pp. 6107–6122). Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.emnlp-main.410
- **Model scale:** T0-3B and T0-11B (T5-based). ⚠ Scale caveat for the main 11B result.
- **Summary:** With **1% rehearsal** (a fixed fraction of earlier-task data mixed into each new task), Continual-T0 learns 8 new generation tasks while keeping ~100% of earlier performance and zero-shot ability.
- **Relation to our paper:** Strong evidence that tiny, uniformly mixed rehearsal budgets suffice in LMs. Timing is not studied.

### D8. Biderman et al. (2024), "LoRA Learns Less and Forgets Less" [V-pub]
- **Citation:** Biderman, D., Portes, J., Gonzalez Ortiz, J. J., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., & Cunningham, J. P. (2024). LoRA learns less and forgets less. *Transactions on Machine Learning Research* (Featured Certification). https://openreview.net/forum?id=aloEru2qCG (arXiv:2405.09673)
- **Model scale:** Llama-2-7B, both LoRA and full fine-tuning.
- **Summary:** LoRA underperforms full fine-tuning on the target domain (code and math) but better preserves source-domain abilities. It is a stronger "forgetting regularizer" than weight decay or dropout.
- **Relation to our paper:** Our use of LoRA (r = 16) likely *reduces* both forgetting and the room for schedule effects. This is a key scale and method caveat: full fine-tuning might reveal schedule differences that LoRA masks.

### D9. Sun et al. (2025), "How new data permeates LLM knowledge and how to dilute it" [V-arXiv] (optional)
- **Citation:** Sun, C., Aksitov, R., Zhmoginov, A., Miller, N. A., Vladymyrov, M., Rueckert, U., Kim, B., & Sandler, M. (2025). *How new data permeates LLM knowledge and how to dilute it*. arXiv:2504.09522. https://arxiv.org/abs/2504.09522
- **Model scale:** PaLM-2, Gemma and Llama, at various sizes. Exact sizes were not checked.
- **Summary:** Learning a new fact "primes" the model to misapply it elsewhere. The extent can be predicted from pre-learning token probabilities and reduced with stepping-stone augmentation or ignore-k update pruning.
- **Relation to our paper:** Cross-fact interference during knowledge injection. Peripheral.

---

## 6. Section E — Learning-science-inspired training of LMs (brief)

### E1. Warstadt et al. (2023), Findings of the (first) BabyLM Challenge [V-pub, V-content]
- **Citation:** Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., Mosquera, R., Paranjabe, B., Williams, A., Linzen, T., & Cotterell, R. (2023). Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning* (pp. 1–6). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.conll-babylm.1
- **Model scale:** Many small LMs trained on ≤10M or ≤100M words, mostly ≤ a few hundred million parameters.
- **Summary:** "Curriculum learning attempts, which accounted for a large number of submissions, were largely unsuccessful, though some showed modest improvements." The CLIMB paper (Martinez et al., 2023) found no curriculum gave widespread gains.

### E2. Hu et al. (2024), Findings of the Second BabyLM Challenge [V-arXiv; the CoNLL 2024 BabyLM proceedings entry was not directly verified]
- **Citation:** Hu, M. Y., Mueller, A., Ross, C., Williams, A., Linzen, T., Zhuang, C., Cotterell, R., Choshen, L., Warstadt, A., & Wilcox, E. G. (2024). *Findings of the second BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora*. arXiv:2412.05149. https://arxiv.org/abs/2412.05149
- **Model scale:** Small LMs on 10M/100M-word budgets.
- **Summary:** Curriculum learning was again the most popular approach but "did not lead to high scores, on average." Gains came mostly from data construction, objectives and architecture.
- **Relation to our paper (E1–E2):** Cognitively inspired *ordering* heuristics have a mostly null track record in small LMs. This is useful context for our expanding ≈ uniform null. A third-round findings paper also exists: Charpentier et al. (2025), *Proceedings of the First BabyLM Workshop*, pp. 399–420, https://doi.org/10.18653/v1/2025.babylm-main.28 [V-pub metadata only].

### E3. Flesch, Balaguer, Dekker, Nili & Summerfield (2018), blocked vs. interleaved in minds and machines [V-pub, V-content]
- **Citation:** Flesch, T., Balaguer, J., Dekker, R., Nili, H., & Summerfield, C. (2018). Comparing continual task learning in minds and machines. *Proceedings of the National Academy of Sciences, 115*(44), E10313–E10322. https://doi.org/10.1073/pnas.1800755115
- **Model scale:** Standard supervised deep networks on naturalistic tree images, plus a variant with an unsupervised generative pre-embedding. ⚠ Scale caveat: small vision networks.
- **Summary:** Humans learned two orthogonal rules *better* with blocked training. Standard deep networks suffered catastrophic forgetting under blocking and needed interleaving. Adding unsupervised pre-learning of the stimulus space reduced the networks' forgetting.
- **Relation to our paper:** The canonical peer-reviewed example of a learning-schedule effect that *reverses* between humans and networks. Use it to argue that human spacing results should not be assumed to transfer, which is our core motivation.

---

## 7. Items requested in the brief: status

| Requested item | Status |
|---|---|
| Amiri+ 2017 EMNLP | Verified (A1) |
| Amiri 2019 NAACL | Verified (A2) |
| Prakriya+ LFR 2409.06131 | Verified; **published at CoNLL 2025** (A3) |
| FOREVER 2601.03938 | Verified; **ACL 2026** (A5) |
| Klasson+ TMLR 2023 | Verified; OpenReview id Q4aAITDgdP. A withdrawn ICLR 2023 submission (forum kyJ5Mrh5Cz9) also exists, so cite the TMLR id (B1) |
| Hayes+ 2021 Neural Computation | Verified, 33(11):2908–2950 (B2) |
| Rolnick+ 2019 | Verified, NeurIPS 32 (B3) |
| Chaudhry+ 2019 | Verified; arXiv only (B4) |
| Robins 1995 | Verified (B5) |
| van de Ven+ 2020 Nat Comms | Verified (B6) |
| Kumaran+ 2016 | Verified (B8) |
| McClelland+ 1995 | Verified (B7) |
| McClelland, McNaughton & Lampinen 2020 | Verified, but it is **Phil Trans R Soc B**, not PNAS. The PNAS SWIL paper is Saxena, Shobe & McNaughton 2022 (B9) |
| Yang+ 2024 NeurIPS | Verified; authors **Yang, Jones, Mozer, Ren** (C3) |
| Toneva+ 2019 | Verified (C4) |
| "How LR decay wastes your best data" (ICLR 2026) | Verified; Luo+ (8 authors); ICLR 2026 **Oral** (D4) |
| Ibrahim+ 2024 | Verified, TMLR (D5) |
| Gupta+ 2023 | Verified, arXiv; workshop venue UNVERIFIED (D6) |
| Scialom+ 2022 | Verified, EMNLP (D7) |

**Searched for but not found:**
- A peer-reviewed study of spacing (massed vs. spaced) of *in-weights* fact learning in LLM fine-tuning with matched exposures, beyond Tirumala et al. (2022).
- Any study testing expanding vs. uniform review of facts with a delayed post-buffer test.
- Search note: the web-search budget ran out partway through. Later queries used the arXiv API, Crossref, OpenReview, ACL Anthology and Semantic Scholar directly. Some 2025–2026 work outside arXiv, ACL and OpenReview may have been missed.

---

## 8. Synthesis

### (a) What is genuinely novel about our experiment relative to this prior work

1. **It isolates timing from selection and exposure count.**
   - Nearly every "spaced repetition for NNs/LLMs" method mixes *when* with *what* and *how much*: RbF, LST, LFR, SRT, MSSR, Leitner replay, and partly FOREVER through its intensity regularizer.
   - Those methods schedule by per-item difficulty and usually change the realized number of exposures.
   - Our arms review the *same old-fact set*, with the *same 12 review events* and the *same total steps*. Only interval shape differs.
   - Among LM papers, only FOREVER's Table 3 (one setting) and MSSR's appendix (setup unclear, exposure not matched) attempt this contrast.
2. **It is a delayed-retention test after a long no-review buffer.** Prior LM replay work evaluates at the end of the task sequence or tracks curves during training. We measure retention ~194 updates after the last review, the ML analogue of the long-delay human tests (Karpicke & Roediger, 2007) where expanding advantages vanish or reverse. The dissociation in our data (expanding better *during* Stage 2, tied *after* the buffer) is a pattern no LM paper reports.
3. **It uses fact-level knowledge with no pretraining contamination, plus a generalization metric.**
   - FictionalQA facts are guaranteed novel to the model.
   - Retention is scored on *held-out paraphrased questions*, not on training strings.
   - Tirumala et al. (2022) use exact memorization. FOREVER and MSSR use task accuracy. SRT uses an unreleased QA set on real temporal corpora, where pretraining leakage is possible.
4. **Statistical design.** Seed-paired comparisons with 95% CIs, including an expanding-minus-uniform CI of [−0.0053, +0.0041] that brackets zero tightly. FOREVER, SRT and MSSR report single runs or no variance.
5. **A mechanistic decomposition: review changes the starting point, not the forgetting rate.** No prior replay-scheduling paper separates pre-buffer level from buffer-period forgetting. This links to Chang et al.'s (2024) acquisition/forgetting decomposition and is a substantive finding.
6. **Constant LR as an explicit control for schedule-by-LR confounding.** Justified by Luo et al. (ICLR 2026). Prior scheduling papers use decaying LR, which systematically down-weights late reviews.
7. **A null that directly contradicts a recent peer-reviewed positive claim.** FOREVER (ACL 2026) reports expanding > uniform at matched budget in a 0.6B LoRA setting. A carefully controlled null at a similar scale is informative, not merely negative.

### (b) What is NOT novel (state it as established and cite it)

- **"Replay/review reduces forgetting during continued training."** This is thoroughly established:
  - classic: Robins 1995; McClelland et al. 1995
  - continual learning: Chaudhry et al. 2019; Rolnick et al. 2019
  - LMs: de Masson d'Autume et al. 2019; Scialom et al. 2022; Ibrahim et al. 2024; Liao et al. 2025; Zucchet et al. 2025
  - Our 4.3% gain is a *replication* of this in a new setting, not a discovery.
- **"Small replay budgets suffice."** Shown by Chaudhry et al. 2019, de Masson d'Autume et al. 2019 (1%) and Scialom et al. 2022 (1%).
- **"Forgetting curves in NNs/LMs resemble human ones."** Shown by Amiri et al. 2017; Tirumala et al. 2022; Chang et al. 2024 (power law); Liao et al. 2025; Kline 2025; Speckmann & Eimer 2025.
- **"Applying spaced-repetition ideas to NN/LLM training."** In use since 2017: RbF, LST, LFR, Leitner replay, VBM, FOREVER, MSSR, SRT.
- **"Spacing length/schedule may not matter much for long-run retention in LMs."** Partially anticipated by Tirumala et al. (2022, 125M): spaced repetition had minimal effect on the forgetting baseline regardless of spacing length. **We must cite this.** Our null is a sharper, matched-budget, shape-specific version, not the first hint.
- **"Human learning-schedule effects need not transfer to networks."** Shown by Flesch et al. 2018 (blocking reverses); Xiong & Wu 2025 (in-context spacing reverses); Speckmann & Eimer 2025 (Leitner/SuperMemo transfer poorly in RL); BabyLM findings 2023/2024 (curricula mostly null).
- **"Replay timing/allocation can matter at fixed budget."** Shown by Robins 1995; Klasson et al. 2023; Kang et al. 2025 (inverted U in interval length); Yang et al. 2024 (order structure).

### (c) Recommended framing of the contribution

**Suggested one-sentence contribution:**
> "In a controlled knowledge-injection setting where the reviewed items, number of reviews, and total training steps are held fixed, we find that reviewing previously learned fictional facts reliably improves their retention after a long no-review delay, but that the *shape* of the review schedule (expanding vs. uniform) has no detectable effect on delayed retention, despite an apparent expanding advantage during training. This qualifies recent claims (Feng et al., 2026) that forgetting-curve-inspired expanding replay outperforms uniform replay, and parallels the human finding that expanding retrieval benefits short-term but not long-term retention (Karpicke & Roediger, 2007)."

**Concrete recommendations:**
1. **Lead with the matched-budget, timing-only null as the contribution, not "review helps."** Present review > no review as a sanity check and replication. Cite B3–B5, B10 and D5–D7.
2. **Engage FOREVER (A5) head-on** in Related Work and Discussion. Offer the measurement-time explanation: their evaluation is closer to "during training," where our trajectory metric also favours expanding. If feasible, run their schedule ({1,2,4,7,15} in steps) in our setup.
3. **Cite Tirumala et al. (2022) as the closest LM precedent** for "spacing length matters little," and position ours as a matched-budget, shape-specific, paraphrase-generalization test.
4. **Cite Chang et al. (2024) as the closest methodological precedent** (fictional knowledge, OLMo, retention), and note that they fix a single 100-step interval.
5. **Avoid "the spacing effect" in the strict sense.** Without a *massed* arm (all 12 reviews in one block, or 12 back-to-back), the study cannot speak to spaced vs. massed, which is what "spacing effect" means in Cepeda et al. (2006). Either add a massed arm (cheap, and Tirumala et al. suggest massed repetition behaves differently) or retitle and phrase as "schedule shape / expanding vs. uniform spacing."
6. **Soften "the contribution here is showing that this human phenomenon reproduces in LLM continued training"** (Conclusion). With 3 seeds, one 300M model, LoRA and one dataset, the claim should be "consistent with," not "reproduces." For an *equivalence* claim, add a pre-specified equivalence margin and a TOST test. The CI [−0.0053, +0.0041] is likely to pass a modest margin such as ±0.01 loss.
7. **Pre-empt the "final-review recency" critique.** If both review arms share the same last review time (Stage 2 step 165), delayed retention may be dominated by the most recent review. This connects to the recency effects in Jagielski et al. (2023) and to the "latest trace dominates" argument in arXiv:2604.06222. A null on shape would then be partly expected. Report each arm's last-review time and mean review lag. Consider a control in which the last-review time differs, or one matched on mean lag rather than last review.
8. **Flag LoRA as a moderator.** LoRA "forgets less" (Biderman et al., 2024), which may compress schedule differences. A small full-fine-tuning replication, which is feasible at 300M, would strengthen generality.
9. **Keep the constant-LR justification and cite Luo et al. (ICLR 2026).**
10. **Scale positioning for a ≤ few-B follow-up.** The most comparable evidence is at ≤3B:
    - Tirumala 125M
    - LFR 120M–1.5B
    - Chang OLMo-1B/7B
    - Yang Pythia 160M–2.8B
    - FOREVER Qwen3-0.6B
    - SRT 1.1B/3B
    - Liao GPT-2 0.1B/1.5B
    - Ibrahim 405M
    - Gupta 410M
    - Zucchet 44M
    - Jagielski 110M

    Evidence only from ≥7B (MSSR, SSR, Biderman, Scialom-11B) or only from vision/RL networks (VBM, Klasson, Kline, Robins, Chaudhry, van de Ven, Saxena, Flesch, Sun et al., Speckmann) should be cited as background, with an explicit scale or domain caveat. Yang et al. (2024) show some re-exposure phenomena *emerge* between 160M and 410M. A model-size sweep (for example 150M / 300M / 1B DataDecide or OLMo checkpoints) would directly address whether schedule effects emerge with scale.
