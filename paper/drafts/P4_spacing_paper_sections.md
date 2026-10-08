# Review schedules and fact retention in language-model continued training: section collage

*Working draft, 2026-10-04. This is not a paper. It is a set of interchangeable sections for assembling the spacing paper later. The original whitepaper ([P4_whitepaper_writeup.md](../source/P4_whitepaper_writeup.md)) is unchanged.*

## TL;DR

- **Protocol status, 2026-10-07.** The revised [spacing PRD](../../plans/PRD_spacing_interleaving_reruns.md), Part 1, governs implementation after scientific and cost review. The [RERUN] passages below reflect the October 4 design and must be reconciled with the revised protocol before manuscript assembly; they do not describe completed experiments.
- **How to use this file.** Every section carries a status tag. To build a paper, pick one version of each section and paste them together.
  - **[PILOT]** describes the experiment as it was actually run, according to the code on branch `anshulm/fictionalqa-review` (see [methods_verification.md](../../research/methods_verification.md)). Use it for a pilot or workshop paper now.
  - **[BOTH]** is true whatever the reruns show: introduction framing, related work, terminology.
  - **[RERUN]** is written as if the experiments in [PRD_spacing_interleaving_reruns.md](../../plans/PRD_spacing_interleaving_reruns.md), Part 1, had been run. Design values and results are `[TODO]` slots.
- **Number provenance.**
  - Unmarked numbers trace to the code, the pinned dataset, or the team's committed `REVIEW_LAB.md`.
  - **†** marks numbers taken from the original whitepaper that have not yet been recomputed from the run logs. These logs are the `runs/` folder, which still has to be requested from the team.
  - **‡** marks numbers we reconstructed by replaying the code's sampling procedure. Confirm them against logged example IDs.
- **What changed from the whitepaper.** Each item is fixed in the [PILOT] sections:
  - review events are described as what they were (one 16-statement batch from one document style);
  - the evaluation set is described accurately;
  - "trajectory change" is redefined to match the code;
  - the model size is corrected;
  - three citations are corrected;
  - the human comparison is scoped;
  - no em-dashes.
- **Open placeholders** are listed at the end of the file.

---

## 0. Title options

- **[PILOT]** Review of earlier facts during LoRA continued training: a pilot comparison of expanding and uniform review schedules
- **[RERUN]** Review schedule shape and delayed fact retention in language-model continued training

---

## Abstract

### Abstract [PILOT]

Continued training on new data degrades a language model's knowledge of facts it learned earlier. Human memory research suggests two remedies: reviewing old material, and spacing those reviews on an expanding schedule.

We ran a pilot test of both remedies.
- **Setup.** We adapted a 371.5M-parameter OLMo-architecture model (DataDecide, Dolma 1.7 recipe) with LoRA. Stage 1 trained it on 60 updates of statements of fictional facts from FictionalQA. Stage 2 then trained on 180 updates of new fictional facts, followed by a 180-update buffer with no exposure to the old facts.
- **Arms.** Two review arms replaced 12 Stage-2 updates with review events, placed at uniform or expanding intervals with the same first and last review update. Each review event was one batch of 16 old-fact statements from a single document style. A third arm had no review.
- **Results.** Across three paired seeds, both review schedules lowered answer-token cross-entropy on old-fact questions after the buffer by 0.17 nats relative to no review: from 3.981 to 3.809 (uniform) and 3.808 (expanding). This corresponds to a 19% higher geometric-mean probability of the answer tokens. Expanding and uniform review differed by −0.0006 nats (95% paired CI [−0.0053, +0.0041]†).
- **The limit of the design.** Review events sampled facts at random within a style, so roughly half of the evaluated facts were never reviewed‡. The two schedules therefore differed in the timing of style-level batches rather than of reviews of individual facts.

We interpret the result as evidence that a small budget of batch-level review reduces forgetting in this setting. Testing schedule shape properly requires per-fact review schedules.

### Abstract [RERUN] (template)

Continued training on new data degrades a language model's knowledge of earlier facts. In human learning, reviewing material counteracts forgetting, but whether reviews should be spaced on an expanding or a uniform schedule remains unsettled: meta-analyses find no reliable long-delay difference between the two.

We tested review schedules in LM continued training.
- **Design.** Each previously learned fictional fact received exactly [TODO: k] reviews. The schedules were no review, uniform, expanding or massed. First and last review were matched per fact, and the total number of training updates and tokens was matched across arms.
- **Controls.** We evaluated against a never-trained control set to separate fact-specific retention from domain adaptation. We used [TODO: n] paired seeds with resampled fact splits, LoRA and full fine-tuning, and models of 371.5M and [TODO: 1B] parameters.
- **Results.** Review [TODO: effect, e.g. "lowered drift-corrected old-fact loss by X nats (95% CI […])"]. Expanding and uniform review [TODO: "were equivalent within ±Δ nats (TOST p = …)" or the observed difference]. Uniform review [TODO: outperformed/did not differ from] massed review.

[TODO: implication, one sentence, scoped to the tested setting.]

---

## 1. Introduction

### 1.1 Problem [BOTH]

A language model that keeps training after deployment must acquire new information without losing what it already knows. Gradient updates on new data overwrite weights that also encode earlier knowledge, so performance on earlier material degrades: catastrophic interference (McCloskey & Cohen, 1989; French, 1999). The standard remedy in continual learning is replay, re-training on a small amount of earlier data while new data are learned (Robins, 1995; Rolnick et al., 2019). Replaying a small fraction of earlier data reduces forgetting in language models: 1% rehearsal in continual instruction tuning (Scialom et al., 2022), and 5–25% replay in continual pretraining (Ibrahim et al., 2024).

Replay leaves open *when* earlier material should be revisited. Human learning research has studied this question for over a century. Distributing study of the same material over time, rather than massing it, improves retention. Only 12 of 271 comparisons in a meta-analysis of verbal recall showed no benefit or a reversal (Cepeda et al., 2006).

Whether the reviews should come at *expanding* intervals, starting close together and spreading out, or at *uniform* intervals is less settled:
- Expanding schedules were proposed for test-type rehearsal and gave a modest same-session advantage (Landauer & Bjork, 1978).
- They underlie common flashcard schedulers.
- Controlled comparisons at long delays find no reliable advantage over equal spacing (Cepeda et al., 2006; Karpicke & Roediger, 2007; Kang et al., 2014), and a meta-analysis of retrieval practice estimates the difference at *g* = 0.034 (Latimier et al., 2021).

### 1.2 Gap [BOTH]

Several training methods borrow spaced-repetition ideas, and two recent studies report advantages for forgetting-curve-inspired replay schedules:
- **Earlier methods:** scheduling training examples by difficulty (Amiri et al., 2017), and revisiting data blocks that a model is likely to forget during pretraining (Prakriya et al., 2025).
- **Feng et al. (2026):** expanding replay intervals outperformed uniform intervals at a matched replay budget in one continual instruction-tuning setting.
- **Atreya et al. (2026):** per-example SM-2 scheduling outperformed uniform replay in continual pretraining.

These methods change *which* items are revisited and *how often*, as well as *when*. That makes the effect of timing hard to isolate.

The closest test of timing alone found that periodically re-injecting a held-out batch had little effect on how much of it a 125M language model retained in the long run, "independent of the length of spacing between the repetitions" (Tirumala et al., 2022). No study we found compares fixed expanding and uniform review of facts at a matched review budget with a delayed retention test.

### 1.3 This study and contributions

#### 1.3 [PILOT] version

We asked two questions in a controlled continued-training setting:
1. Does reviewing previously learned facts improve their retention after a long stretch of training without review?
2. At a fixed review budget, does an expanding schedule retain more than a uniform one?

We used fictional facts (FictionalQA; Kirchenbauer et al., 2026), so that all fact knowledge had to be acquired during our training, and a constant learning rate, so that later updates were not down-weighted (Luo et al., 2026).

Contributions:
1. **Review.** A small review budget (12 of 180 Stage-2 updates) lowered old-fact answer loss after a 180-update buffer by 0.17 nats in a 371.5M model across three paired seeds.
2. **Schedule.** Expanding and uniform schedules of style-level review batches gave final old-fact losses within 0.001 nats of each other.
3. **Limits of the pilot design.** We report the limits that prevent a fact-level conclusion about schedule shape. Most evaluated facts received zero or one review, and the evaluation metric partly reflects domain adaptation. The design that resolves both is given in [PRD reference / follow-up].

#### 1.3 [RERUN] version

We tested review timing at the level of individual facts.
- **Schedules.** Each previously learned fictional fact received exactly [TODO: k] review exposures on a no-review, uniform, expanding or massed schedule. The first and last review of each fact occurred at the same update in all spaced arms, so the arms differed only in the placement of the intermediate reviews.
- **Control set.** We measured retention against a set of never-trained fictional facts. This separates fact-specific memory from adaptation to the domain and answer format, which also lowers loss on unseen facts (Kirchenbauer et al., 2026).

Contributions:
1. [TODO: effect of review vs. no review on drift-corrected retention, with CI.]
2. [TODO: expanding vs. uniform result with the TOST outcome at the preregistered margin.]
3. [TODO: spaced vs. massed result.]
4. [TODO: whether review changed the level or the rate of forgetting, on both the loss and accuracy scales.]
5. [TODO: whether the results held under full fine-tuning and at 1B parameters.]

---

## 2. Background and related work [BOTH]

### 2.1 Spacing and schedule shape in human memory

**Definitions.**
- **Spacing effect:** better retention after spaced than after massed study of the same item (Cepeda et al., 2006).
- **Inter-study interval:** the gap between study episodes.
- **Retention interval:** the gap between the last episode and the test.
- **Schedule shape:** with three or more episodes, the intervals can be equal (uniform), progressively longer (expanding) or progressively shorter (contracting) (Cepeda et al., 2006).

**The spacing effect is large and reliable.** For retrieval practice, spaced practice outperforms massed practice with *g* = 0.74 (Latimier et al., 2021). The optimal gap grows with the retention interval (Cepeda et al., 2008).

**Schedule shape matters much less.**
- In Cepeda et al.'s (2006) synthesis, expanding and fixed schedules did not differ reliably: 62.0% vs. 58.6% correct, *t*(42) = 0.5, *p* = .61.
- Karpicke and Roediger (2007) found expanding retrieval better at 10 minutes but equal spacing better at 2 days. They traced the long-delay advantage to the later *first* retrieval in the equal schedule. When the first retrieval was matched, the schedule of later retrievals had no reliable effect (their Experiment 3).
- Total spacing improved retention while schedule shape did not (Karpicke & Bauernschmidt, 2011).
- Meta-analytically, expanding and uniform retrieval practice differ by *g* = 0.034 [−0.10, 0.17] (Latimier et al., 2021).

**Expanding schedules can still help in specific conditions:**
- when forgetting between tests is fast (Storm et al., 2010);
- for restudy spread over days after weak initial learning (Toppino et al., 2018);
- for performance *during* training. Kang et al. (2014) found that expanding practice gave higher average recall over the training period than equal-interval practice (.49 vs. .41), with no reliable difference at the final test.

**Two distinctions matter when mapping this literature onto language-model training:**
1. **Restudy vs. retrieval.** Most schedule-shape studies use retrieval practice, in which testing itself improves retention (Roediger & Karpicke, 2006). A language-model review event is closer to *restudy*: a gradient update on the statement itself. For re-presentation of information, Landauer and Bjork (1978) predicted and observed that uniform spacing does at least as well as expanding (.62 vs. .58, not significant). Their expanding advantage was confined to test-type practice.
2. **Comparing forgetting rates.** Whether a manipulation changes the *level* or the *rate* of forgetting depends on the scale of measurement (Loftus, 1985). In human verbal memory, degree of learning shifts the forgetting curve without changing its slope on common scales (Slamecka & McElree, 1983).

### 2.2 Replay and spaced repetition in neural network training

**Replay.**
- Rehearsal of earlier items reduces catastrophic forgetting in connectionist networks (Robins, 1995).
- Experience replay does the same in deep reinforcement learning (Rolnick et al., 2019).
- In language models, small replay fractions preserve earlier skills during continual instruction tuning (Scialom et al., 2022) and continual pretraining (Ibrahim et al., 2024).
- Learned schedules for *which* tasks to replay and *when* can outperform fixed schedules at a fixed memory size in small vision networks (Klasson et al., 2023).

**Spaced repetition.**
- Methods inspired by spaced repetition schedule training examples by estimated difficulty (Amiri et al., 2017), or revisit data blocks with high perplexity during language-model pretraining (Prakriya et al., 2025).
- FOREVER triggers replay on an Ebbinghaus-style schedule measured in "model time" (accumulated parameter-update magnitude). In one ablation at an identical replay budget, an increasing-interval schedule reached 42.5 overall performance against 40.9 for uniform intervals (Qwen3-0.6B, one task order; Feng et al., 2026).
- Spaced Repetition Training keeps per-example SM-2 review states in continual pretraining and improves the stability–plasticity trade-off over uniform replay at 1.1B and 3B parameters (Atreya et al., 2026; preprint).
- Both methods select *which* items or tasks to replay as well as when. Neither isolates fixed schedule shape for individual facts.

**Timing alone.**
- In a 125M causal language model, re-injecting a batch several times in a row raised its long-run retention, while periodic, spaced re-injection had minimal effect regardless of the spacing length (Tirumala et al., 2022).
- Fine-tuning on documents in a fixed cyclic order produces "anticipatory recovery": loss on a document falls before it reappears. The effect emerges between 160M and 410M parameters (Yang et al., 2024). Re-exposure order can therefore change forgetting dynamics at the model sizes studied here.

### 2.3 Acquisition and forgetting of facts in language models

- **Acquisition and forgetting.** Language models acquire a fact through small increases in its probability at each exposure, and subsequent training erodes these gains with a power-law dependence on training steps (Chang et al., 2024, OLMo 1B and 7B).
- **Extraction.** Storing a fact is not the same as being able to retrieve it from a question. Models trained on biographies extract facts reliably only when the training data present each fact in varied forms (Allen-Zhu & Li, 2024).
- **FictionalQA.** FictionalQA was built to study these processes with facts that cannot occur in pretraining data (Kirchenbauer et al., 2026). Its authors report that training on short declarative statements leads to fast memorization with little transfer. They also report that training on some fictional documents improves performance on questions about others, a "leaky" transfer that reflects learning of the fictional domain as well as of individual facts.
- **Drift control.** A never-trained control set separates these components (O'Neill, 2026; preprint).
- **Adapter choice.** Fine-tuning through low-rank adapters learns less and forgets less than full fine-tuning (Biderman et al., 2024), so the choice of adapter can change the size of forgetting effects.

### 2.4 Terminology used in this paper [BOTH]

- **Old facts / new facts:** facts trained in Stage 1 / Stage 2.
- **Review event:** one optimizer update that trains on old-fact data. Its content is specified in Methods.
- **Arms:** no review, uniform review, expanding review (and, in the reruns, massed review and generic insertion).
- **Buffer:** the final period of training on new facts only, after the last review event.
- **Delayed retention:** old-fact performance at the end of the buffer.

We use **"schedule shape"** for the expanding-vs-uniform contrast and reserve **"spacing effect"** for spaced vs. massed comparisons.

---

## 3. Methods

### 3 [PILOT] Methods as run

#### 3.1 Model and adaptation

- **Base model:** the DataDecide 300M model pretrained on the Dolma 1.7 recipe (`allenai/DataDecide-dolma1_7-300M`, final default-seed checkpoint `step45787-seed-default`, commit `4b1b42ff`; Magnusson et al., 2025). It has the OLMo architecture (Groeneveld et al., 2024) with 16 layers, model width 1024, SwiGLU feed-forward layers, untied input and output embeddings, and 371.5M parameters.
- **Adaptation:** LoRA (Hu et al., 2022) with rank 16, α = 32 and dropout 0.05 on all linear layers of the transformer blocks, giving 5.24M trainable parameters.
- **Optimizer:** AdamW (β₁ = 0.9, β₂ = 0.95, weight decay 0.1) with gradient clipping at 1.0, a learning rate of 1 × 10⁻⁴ with a 10-step linear warmup and no decay, and fp16 mixed precision.
- **Optimizer reset:** the optimizer, including its warmup, was re-initialized at the start of Stage 2.
- **Batches:** each update used 16 sequences of at most 64 tokens. No training or evaluation sequence exceeded 37 tokens, so none was truncated.

#### 3.2 Data

- **Source:** the `fict_qa` configuration of FictionalQA (Kirchenbauer et al., 2026; revision `131cb74`). It contains 7,500 question–answer rows generated from 100 fictional events, each described in documents of five styles (blog, corporate, encyclopedia, news, social).
- **Deduplication:** we kept one canonical question per duplicate cluster.
- **Split:** with a fixed data seed, we assigned 20 events to old facts and 40 events to new facts, giving 386 old and 689 new facts. Because the split is by event, no fictional event contributed to both sets. The remaining 40 events were not used.
- **Training text:** each fact's declarative statement, formatted as `Fictional fact: <statement>`, with loss on all tokens.
- **Evaluation text:** the dataset's question for the same fact, formatted as `Question: <question>\nAnswer:`. Loss was computed on the tokens of the short answer and the end-of-sequence token.
- **Train/test overlap:** the question wording never appeared in training. The answer string occurred verbatim in the corresponding training statement for 1,003 of the 1,075 facts (93%).

#### 3.3 Training stages

Every arm trained for 420 updates. In each update, the 16 sequences were sampled with replacement from a single document style, and styles were visited in a fixed rotation.

1. **Stage 1 (60 updates):** old-fact statements, 12 updates per style. All arms of a seed started from the same Stage-1 checkpoint. With 28–180 old facts per style, the expected number of Stage-1 exposures per fact ranged from 1.1 (news) to 6.9 (blog), and 4–8 of the evaluated old facts were never sampled in Stage 1‡.
2. **Stage 2 (180 updates):** new-fact statements, except at the review events of the review arms.
3. **Buffer (180 updates):** new-fact statements only.

#### 3.4 Review conditions

- **Review event:** one update on 16 old-fact statements sampled with replacement from a single style, with styles taken in rotation.
- **Budget:** both review arms had 12 review events, each replacing one Stage-2 new-fact update.
- **Schedules** (Stage-2 update indices, counting from 0):
  - **Uniform:** 8, 22, 37, 51, 65, 79, 94, 108, 122, 136, 151, 165.
  - **Expanding** (gaps growing by a factor of 1.35): 8, 10, 13, 17, 22, 29, 38, 51, 68, 91, 123, 165.
- **What was matched:** the first and last review event occurred at the same update in both arms, and the old facts sampled at each review event were identical across arms within a seed. Only the timing differed.
- **Per-style recency was not matched.** Because styles rotated across review events, the last review of each style differed between arms (Table M1). The final review event occurred 194 updates before the final evaluation.

**Table M1.** Update index of the last review event for each document style.

| Style | Old facts | Uniform | Expanding |
|---|---:|---:|---:|
| Blog | 28 | 151 | 123 |
| Corporate | 54 | 165 | 165 |
| Encyclopedia | 41 | 108 | 51 |
| News | 180 | 122 | 68 |
| Social | 83 | 136 | 91 |

*Stage-2 update indices (0–179). Blog and corporate styles received three review events, the others two.*

#### 3.5 Evaluation

- **Items:** 16 old-fact and 16 new-fact questions per style (80 each), identical across seeds and arms.
- **Loss:** for each style, the token-weighted mean answer-token cross-entropy (in nats, including the end-of-sequence token). We report the unweighted mean across the five styles.
- **Checkpoints:** Stage-2 updates 29, 59, 89, 119, 149 and 179, and after 15, 60 and 180 buffer updates.
- **Measures:**
  - **final old-fact loss** (primary; end of the buffer);
  - **pre-buffer old-fact loss** (Stage-2 update 179);
  - **buffer increase** (final minus pre-buffer);
  - **change from Stage 1** (final minus the end-of-Stage-1 loss);
  - **trajectory change**, the mean change from the end-of-Stage-1 loss over all nine checkpoints. Six of these lie in Stage 2 and three in the buffer.
  - **final new-fact loss**;
  - **joint loss**, the unweighted mean of final old- and new-fact loss.
- **Exact match:** greedy exact match was computed for every item but is not analysed here [TODO: report from logs].

#### 3.6 Statistics

- **Seeds:** we ran three seeds (17, 23, 42). A seed set the LoRA initialization and the order of sampled training examples. It did not change the event split or the evaluated items.
- **Reporting:** means ± SD across seeds, and differences between arms paired by seed with 95% confidence intervals [CHECK: confirm CI method; the reported intervals are consistent with paired t-intervals, df = 2].
- **No preregistration:** no analysis was preregistered.

### 3 [RERUN] Methods for the planned experiments

*Written in the past tense for later use. Values marked [TODO] are set in calibration (PRD Part 1, phase S1).*

#### 3.1R Model and adaptation

- **Base model:** as in the pilot (§3.1).
- **Adaptation:** we trained each arm both with LoRA (configured as in the pilot) and with full fine-tuning (learning rate [TODO]).
- **Optimizer:** AdamW as in the pilot, with a constant learning rate after a single 10-step warmup at the start of Stage 1. The optimizer state was carried across all stages.
- **Scale check:** a confirmation set of arms used the DataDecide 1B model on the same data recipe [TODO: repository and revision].

#### 3.2R Data

- **Source:** FictionalQA as in the pilot.
- **Split per seed:** for each seed we drew a fresh event-level split into
  - 20 old-fact events;
  - 40 new-fact events;
  - 20 never-trained control events;
  - 10 format-teaching events.
- **Guessability filter:** we removed questions that a model without access to the source documents answered correctly, using the dataset's blind-answer grades (threshold [TODO]).
- **Format-teaching events:** question–answer pairs from these events were mixed into every stage of every arm at [TODO]% of each batch. They were never evaluated. This taught the answer format without exposing any evaluated fact in question form.
- **Training text:** old and new facts were trained as declarative statements, as in the pilot.
- **Evaluation:** every retained old, new and control fact, in two forms: the dataset's open question, and its four-choice multiple-choice version.

#### 3.3R Training stages and review schedules

- **Stage 1.** Every old fact appeared exactly [TODO: E] times in shuffled order. E was chosen in calibration so that mean old-fact exact match after Stage 1 lay between [TODO] and [TODO].
- **Per-fact review schedules.** In Stage 2, each old fact *f* was assigned a start update *s_f*, staggered evenly across facts, and received exactly *k* = [TODO] review exposures:

  | Arm | Review exposures for fact *f* |
  |---|---|
  | Uniform | at *s_f*, *s_f* + *L*/(*k*−1), …, *s_f* + *L* |
  | Expanding | from *s_f* to *s_f* + *L*, with gaps growing by a factor *r* = [TODO: 1.35; sensitivity 2.0] |
  | Massed | on the *k* consecutive updates ending at *s_f* + *L* |
  | Generic insertion | at the uniform arm's positions, but with generic web text in place of the old-fact statements |
  | No review | none |

  So first and last review were matched for every fact in the uniform and expanding arms, and last review for every fact in all three review arms.
- **Insertion.** At each update, scheduled review statements replaced an equal number of new-fact statements in the 16-sequence batch, so every arm processed the same number of training sequences and tokens.
- **Buffer.** After Stage 2, all arms trained for [TODO] updates on new facts only.
- **Seeds.** We ran [TODO: 10] seeds. A seed set the event split, LoRA initialization and data order, and was shared by all arms.

#### 3.4R Measures

- **Per-item measures:**
  - answer-token cross-entropy;
  - greedy exact match;
  - four-choice accuracy;
  - the likelihood margin, log *p*(correct answer) minus log-sum-exp over the three distractors.
- **Checkpoints:** every [TODO] Stage-2 updates, and after 1, 15, 60, 180 and [TODO] buffer updates.
- **Primary outcome:** drift-corrected delayed retention, the mean old-fact answer loss minus the mean control-fact answer loss at the final checkpoint.
- **Exposure log:** we logged every exposure of every fact, so the realised schedules could be verified.

#### 3.5R Statistical analysis

- **Preregistration:** hypotheses, primary outcome, the equivalence margin (Δ = [TODO] nats) and the analysis plan were preregistered before the main runs [TODO: link].
- **Models:** linear mixed-effects models with crossed random effects for fact (nested in event) and seed (Baayen et al., 2008), with by-seed random slopes for arm where estimable (Barr et al., 2013). This treats facts as a sample from a population of items rather than as fixed (Clark, 1973).
- **Equivalence:** expanding and uniform review were compared with two one-sided tests at ±Δ (Lakens, 2017).
- **Forgetting:** buffer forgetting was modelled over all buffer checkpoints, on both the loss and accuracy scales.
- **Multiplicity:** secondary contrasts were Holm-corrected.

---

## 4. Results

### 4 [PILOT] Results

**Table R1.** Old- and new-fact answer-token loss (nats) by arm. Means ± SD across three paired seeds; lower is better.

| Arm | Review events | Pre-buffer old loss† | Buffer increase | Final old loss | Change from Stage 1† | Trajectory change† | Final new loss | Joint loss† |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No review | 0 | 3.830 | +0.151 ± 0.030† | 3.981 ± 0.031† | −0.223 ± 0.021 | −0.305 ± 0.014 | 2.993 ± 0.038† | 3.487 ± 0.031 |
| Uniform | 12 | 3.648 | +0.161 ± 0.009† | 3.809 ± 0.015† | −0.396 ± 0.015 | −0.423 ± 0.021 | 3.018 ± 0.024† | 3.413 ± 0.019 |
| Expanding | 12 | 3.650 | +0.158 ± 0.008† | 3.808 ± 0.016† | −0.396 ± 0.014 | −0.436 ± 0.023 | 3.015 ± 0.019† | 3.411 ± 0.017 |

*Means for buffer increase, final old loss and final new loss match the team's committed results; † marks values not yet recomputed from the run logs. Trajectory change averages six Stage-2 and three buffer checkpoints (§3.5).*

**Review lowered delayed old-fact loss.**
- After the 180-update buffer, final old-fact loss was 0.172 nats lower with uniform review and 0.173 nats lower with expanding review than with no review. The paired 95% CIs were [−0.214, −0.131]† and [−0.210, −0.137]†.
- On the probability scale, the geometric-mean probability of the answer tokens was 0.0222 with uniform review against 0.0187 with no review, a ratio of 1.19.
- Both review arms ended the buffer with lower old-fact loss (3.809 and 3.808) than the no-review arm had at the start of the buffer (3.830†).

**Expanding and uniform review gave nearly identical retention.**
- **At the end of the buffer:** the difference in final old-fact loss (expanding minus uniform) was −0.0006 nats (95% paired CI [−0.0053, +0.0041]†).
- **During training:** expanding review had a more negative trajectory change than uniform review (−0.436 vs. −0.423†) [TODO: CI].
- **At intermediate delays:** the two review arms differed by at most 0.002 nats at buffer delays of 15, 60 and 180 updates (Table R2).

**Table R2.** Old-fact loss (nats) at three delays into the buffer. Means across three paired seeds†.

| Buffer updates | No review | Uniform | Expanding |
|---:|---:|---:|---:|
| 15 | 3.846 | 3.664 | 3.666 |
| 60 | 3.880 | 3.698 | 3.699 |
| 180 | 3.981 | 3.809 | 3.808 |

**Old-fact loss rose by similar amounts during the buffer in all arms.**
- **Loss scale:** the increase was 0.151 (no review), 0.161 (uniform) and 0.158 (expanding) nats. The paired differences from no review had 95% CIs of [−0.072, +0.091]† (uniform) and [−0.066, +0.079]† (expanding); each interval is about half the size of the increase itself.
- **Probability scale:** the geometric-mean answer probability fell by 0.0030 in the no-review arm (0.0217 to 0.0187) and by 0.0039 in the uniform arm (0.0260 to 0.0222).

**Old-fact loss also fell during new-fact training.** In every arm, old-fact loss at the end of the buffer was lower than at the end of Stage 1 (change from Stage 1: −0.223 to −0.396†). This includes the no-review arm, which received no old-fact training after Stage 1.

**Review had a small, uncertain cost on new facts.** Final new-fact loss was 0.025 (uniform) and 0.022 (expanding) nats higher than with no review. The paired 95% CIs, [−0.026, +0.076]† and [−0.033, +0.076]†, include zero.

**Most evaluated facts received little or no review.** Replaying the sampling procedure with the run seeds showed:
- 31–40 of the 80 evaluated old facts appeared in at least one review event;
- 11–17 appeared in two or more‡;
- the facts drawn at each review event were the same in both review arms, so per-fact review counts were identical across schedules within a seed.

**Figure R1.** [Use `paper/figures/whitepaper_fig1_old_fact_loss.png`.] Old-fact answer-token loss (nats) against updates since the end of Stage 1, for no review (grey), uniform review (blue) and expanding review (orange). Lines are means across three paired seeds; shaded bands show ±1 SD across seeds [CHECK: confirm band definition in the plotting code]. The dashed line marks the start of the 180-update buffer. Loss falls in all arms during Stage 2, falls further in the review arms, and rises in all arms during the buffer.

### 4 [RERUN] Results skeleton

*Tables and figures to be filled from the rerun. Report every contrast with CIs and per-seed values.*

- **Table R1R.** Drift-corrected old-fact loss, exact match and four-choice accuracy at the final checkpoint, by arm and adaptation method (mean, 95% CI). [TODO]
- **Table R2R.** Planned contrasts:
  - review vs. no review;
  - expanding vs. uniform (TOST at ±Δ);
  - uniform vs. massed;
  - uniform vs. generic insertion.

  Report each as an estimate with its 95% CI, *p*, and the Holm-adjusted *p*. [TODO]
- **Figure R1R.** Old-fact and control-fact loss over training, by arm, with 95% CI bands. [TODO]
- **Figure R2R.** Final old-fact accuracy against the number of updates since each fact's last review, by arm. [TODO]
- **Table R3R.** Buffer forgetting slopes on the loss and accuracy scales, with the arm × time interaction. [TODO]
- **Table R4R.** The 1B confirmation. [TODO]

---

## 5. Discussion

### 5 [PILOT] Discussion

**Review.**
- Twelve review events, 6.7% of Stage-2 updates, lowered old-fact loss after a long no-review buffer, consistent with the replay literature (Scialom et al., 2022; Ibrahim et al., 2024). The review arms still had lower loss after 180 buffer updates than the no-review arm had when the buffer began.
- The design cannot say how much of this benefit is specific to the reviewed facts, for two reasons:
  1. About half of the evaluated facts were never reviewed‡.
  2. Old-fact loss fell during new-fact training even without review. This is the leaky transfer FictionalQA's authors describe (Kirchenbauer et al., 2026), and it implies that part of what our metric measures is adaptation to the fictional domain and answer format.
- Comparing reviewed with never-reviewed facts within the review arms, using the existing logs, would separate the two components [TODO: run this analysis].

**Schedule shape.**
- Expanding and uniform review produced final old-fact losses within 0.001 nats of each other.
- This tie should not be read as a fact-level null. Review events were assigned to document styles in rotation, so the schedules governed when each *style* was revisited, and each evaluated fact received zero to three reviews.
- Per-style recency also differed: under the expanding schedule, three of five styles were last reviewed 45–57 updates earlier than under the uniform schedule. A recency advantage for the uniform arm would predict lower uniform loss; we observed none.
- The result is consistent with the human literature in a restricted sense:
  - when first- and last-review timing are matched, schedule shape has little effect on long-delay retention (Karpicke & Roediger, 2007, Experiment 3; Latimier et al., 2021);
  - for re-presentation rather than retrieval, uniform spacing does at least as well as expanding (Landauer & Bjork, 1978).
- The lower trajectory change for expanding review parallels Kang et al.'s (2014) finding of higher average recall during expanding practice with no final-test difference. In our data, this measure averages Stage-2 and buffer checkpoints and has no interval estimate yet.

**Relation to FOREVER.**
- Feng et al. (2026) report an advantage for increasing replay intervals at a matched budget. Their setting differs from ours in three ways:
  1. it replays tasks in continual instruction tuning, not facts;
  2. it measures intervals in accumulated parameter change rather than in updates;
  3. it reports performance over a task sequence rather than after a dedicated no-review delay.
- The per-fact design in our follow-up is needed to test whether the difference reflects the setting or the batch-level manipulation used here.

**Level and rate of forgetting.**
- On the loss scale, the three arms lost similar amounts during the buffer, which suggests that review lowered the starting point rather than the rate of forgetting.
- We do not claim that the rates were equal:
  - the confidence intervals on the differences are about half as wide as the effect;
  - on the probability scale the review arm lost slightly more;
  - comparisons of forgetting rate depend on the measurement scale (Loftus, 1985).

**Strength of learning.**
- After Stage 1, old facts were only weakly learned. An answer-token loss near 3.8 nats corresponds to a geometric-mean answer-token probability of about 2%.
- Facts received few training exposures, and training used short declarative statements, which FictionalQA's authors associate with memorization and weak transfer (Kirchenbauer et al., 2026).
- Our conclusions therefore concern the retention of partially learned facts.

### 5 [BOTH] Reusable paragraph: why human schedule effects need not transfer

Human spacing effects are usually explained by retrieval processes during study: study-phase retrieval, encoding variability, or the extra benefit of retrieving an item that is partly forgotten (Cepeda et al., 2006; Karpicke & Roediger, 2007). A teacher-forced gradient update on a training statement involves no retrieval attempt. Its effect depends on the current loss of the reviewed item and on interference from the surrounding updates. Expanding schedules exploit the human pattern of fast early forgetting followed by slower forgetting (Storm et al., 2010). They would be expected to help a network only if its forgetting followed a similar time course, which the power-law forgetting reported for language models (Chang et al., 2024) makes plausible but does not guarantee.

---

## 6. Limitations

### 6 [PILOT] Limitations

1. **Review was not scheduled per fact.**
   - Review events sampled facts at random within a style, so most evaluated facts received zero or one review‡.
   - Per-style last-review times differed between arms.
   - The schedule comparison is therefore between style-level batch schedules.
2. **The metric mixes fact-specific memory with domain and format adaptation.** No never-trained control was evaluated. Exact match was logged but not analysed.
3. **Facts were weakly learned and unevenly exposed.** Expected Stage-1 exposures ranged from 1.1 to 6.9 per fact across styles, and a few evaluated facts were never trained‡.
4. **The format of training and testing differed.** Training used statements and testing used questions. The answers appeared verbatim in the training statements.
5. **Inference is limited.** There were three seeds, one event split, 80 evaluated old facts treated as fixed, no preregistered equivalence margin, and confidence intervals computed outside the committed code†.
6. **The scope is narrow:** one 371.5M model, LoRA only (which learns and forgets less than full fine-tuning; Biderman et al., 2024), one expansion ratio, one review budget, one buffer length, and a constant learning rate.
7. **There was no massed arm**, so the study does not test the spacing effect in the spaced-vs-massed sense.

### 6 [RERUN] Remaining limitations (edit after the reruns)

- **Synthetic facts.** Fictional facts with short declarative statements may be learned and forgotten differently from facts embedded in natural documents.
- **Scale.** We tested models up to [TODO: 1B] parameters. Effects of re-exposure order can change with scale (Yang et al., 2024).
- **Restudy only.** Review was restudy of the training statement. Retrieval-like review, for example training on the question form, was [TODO: tested as a secondary arm / not tested].
- **Learning-rate schedule.** We used a constant learning rate, so the results may not hold under schedules that decay the learning rate during the review period (Luo et al., 2026).

---

## 7. Conclusion [PILOT]

In a 371.5M model adapted with LoRA, a small budget of review events lowered old-fact loss after a 180-update period without review. Expanding and uniform schedules of these review events gave nearly identical delayed retention. Because review events were assigned to document styles rather than to individual facts, the pilot does not test schedule shape at the level of facts. The follow-up experiment assigns every fact its own matched review schedule and measures retention against never-trained control facts.

---

## References

- Allen-Zhu, Z., & Li, Y. (2024). Physics of language models: Part 3.1, knowledge storage and extraction. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR Vol. 235, pp. 1067–1077). https://proceedings.mlr.press/v235/allen-zhu24a.html
- Amiri, H., Miller, T., & Savova, G. (2017). Repeat before forgetting: Spaced repetition for efficient and effective training of neural networks. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing* (pp. 2401–2410). https://doi.org/10.18653/v1/D17-1255
- Atreya, A., Batra, D., Mantri, Y. K., Bantug, G., Cowan, G. A., & Khraishi, R. (2026). *When to review: Spaced repetition for continual pre-training of language models* [Preprint]. arXiv. https://arxiv.org/abs/2608.17530
- Baayen, R. H., Davidson, D. J., & Bates, D. M. (2008). Mixed-effects modeling with crossed random effects for subjects and items. *Journal of Memory and Language, 59*(4), 390–412. https://doi.org/10.1016/j.jml.2007.12.005
- Barr, D. J., Levy, R., Scheepers, C., & Tily, H. J. (2013). Random effects structure for confirmatory hypothesis testing: Keep it maximal. *Journal of Memory and Language, 68*(3), 255–278. https://doi.org/10.1016/j.jml.2012.11.001
- Biderman, D., Portes, J., Gonzalez Ortiz, J. J., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., & Cunningham, J. P. (2024). LoRA learns less and forgets less. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=aloEru2qCG
- Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin, 132*(3), 354–380. https://doi.org/10.1037/0033-2909.132.3.354
- Cepeda, N. J., Vul, E., Rohrer, D., Wixted, J. T., & Pashler, H. (2008). Spacing effects in learning: A temporal ridgeline of optimal retention. *Psychological Science, 19*(11), 1095–1102. https://doi.org/10.1111/j.1467-9280.2008.02209.x
- Chang, H., Park, J., Ye, S., Yang, S., Seo, Y., Chang, D.-S., & Seo, M. (2024). How do large language models acquire factual knowledge during pretraining? In *Advances in Neural Information Processing Systems 37*. https://arxiv.org/abs/2406.11813
- Clark, H. H. (1973). The language-as-fixed-effect fallacy: A critique of language statistics in psychological research. *Journal of Verbal Learning and Verbal Behavior, 12*(4), 335–359. https://doi.org/10.1016/S0022-5371(73)80014-3
- Feng, Y., Wang, H., Li, J., Chu, X., Kang, Z., Liu, Y., Wang, Y., Yu, P. S., & Wu, X.-M. (2026). FOREVER: Forgetting curve-inspired memory replay for language model continual learning. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics*. https://arxiv.org/abs/2601.03938
- French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2
- Groeneveld, D., Beltagy, I., Walsh, E., Bhagia, A., Kinney, R., Tafjord, O., Jha, A. H., Ivison, H., Magnusson, I., Wang, Y., Arora, S., Atkinson, D., Authur, R., Chandu, K. R., Cohan, A., Dumas, J., Elazar, Y., Gu, Y., Hessel, J., . . . Hajishirzi, H. (2024). OLMo: Accelerating the science of language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 15789–15809). https://doi.org/10.18653/v1/2024.acl-long.841
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR 2022)*. https://openreview.net/forum?id=nZeVKeeFYf9
- Ibrahim, A., Thérien, B., Gupta, K., Richter, M. L., Anthony, Q., Lesort, T., Belilovsky, E., & Rish, I. (2024). Simple and scalable strategies to continually pre-train large language models. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=DimPeeCxKO
- Kang, S. H. K., Lindsey, R. V., Mozer, M. C., & Pashler, H. (2014). Retrieval practice over the long term: Should spacing be expanding or equal-interval? *Psychonomic Bulletin & Review, 21*(6), 1544–1550. https://doi.org/10.3758/s13423-014-0636-z
- Karpicke, J. D., & Bauernschmidt, A. (2011). Spaced retrieval: Absolute spacing enhances learning regardless of relative spacing. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 37*(5), 1250–1257. https://doi.org/10.1037/a0023436
- Karpicke, J. D., & Roediger, H. L., III. (2007). Expanding retrieval practice promotes short-term retention, but equally spaced retrieval enhances long-term retention. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 33*(4), 704–719. https://doi.org/10.1037/0278-7393.33.4.704
- Kirchenbauer, J., Mongkolsupawan, N., Wen, Y., Goldstein, T., & Ippolito, D. (2026). FictionalQA: A dataset for studying memorization and knowledge acquisition. In *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=SUNC1eJGMr
- Klasson, M., Kjellström, H., & Zhang, C. (2023). Learn the time to learn: Replay scheduling in continual learning. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=Q4aAITDgdP
- Lakens, D. (2017). Equivalence tests: A practical primer for *t* tests, correlations, and meta-analyses. *Social Psychological and Personality Science, 8*(4), 355–362. https://doi.org/10.1177/1948550617697177
- Landauer, T. K., & Bjork, R. A. (1978). Optimum rehearsal patterns and name learning. In M. M. Gruneberg, P. E. Morris, & R. N. Sykes (Eds.), *Practical aspects of memory* (pp. 625–632). Academic Press. https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/Landauer.Bjork_.1978.pdf
- Latimier, A., Peyre, H., & Ramus, F. (2021). A meta-analytic review of the benefit of spacing out retrieval practice episodes on retention. *Educational Psychology Review, 33*(3), 959–987. https://doi.org/10.1007/s10648-020-09572-8
- Loftus, G. R. (1985). Evaluating forgetting curves. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 11*(2), 397–406. https://doi.org/10.1037/0278-7393.11.2.397
- Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *The Fourteenth International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=T5wkZJqzkz
- Magnusson, I., Tai, N., Bogin, B., Heineman, D., Hwang, J. D., Soldaini, L., Bhagia, A., Liu, J., Groeneveld, D., Tafjord, O., Smith, N. A., Koh, P. W., & Dodge, J. (2025). DataDecide: How to predict best pretraining data with small experiments. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR Vol. 267, pp. 42487–42502). https://proceedings.mlr.press/v267/magnusson25a.html
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), *Psychology of learning and motivation* (Vol. 24, pp. 109–165). Academic Press. https://doi.org/10.1016/S0079-7421(08)60536-8
- O'Neill, C. (2026). *Can a language model learn facts continually in its weights?* [Preprint]. arXiv. https://arxiv.org/abs/2607.11020
- Prakriya, N., Yen, J.-N., Hsieh, C.-J., & Cong, J. (2025). Accelerating large language model pretraining via LFR pedagogy: Learn, focus, and review. In *Proceedings of the 29th Conference on Computational Natural Language Learning* (pp. 268–290). https://doi.org/10.18653/v1/2025.conll-1.18
- Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318
- Roediger, H. L., III, & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science, 17*(3), 249–255. https://doi.org/10.1111/j.1467-9280.2006.01693.x
- Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html
- Scialom, T., Chakrabarty, T., & Muresan, S. (2022). Fine-tuned language models are continual learners. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (pp. 6107–6122). https://doi.org/10.18653/v1/2022.emnlp-main.410
- Slamecka, N. J., & McElree, B. (1983). Normal forgetting of verbal lists as a function of their degree of learning. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 9*(3), 384–397. https://doi.org/10.1037/0278-7393.9.3.384
- Storm, B. C., Bjork, R. A., & Storm, J. C. (2010). Optimizing retrieval as a learning event: When and why expanding retrieval practice enhances long-term retention. *Memory & Cognition, 38*(2), 244–253. https://doi.org/10.3758/MC.38.2.244
- Tirumala, K., Markosyan, A., Zettlemoyer, L., & Aghajanyan, A. (2022). Memorization without overfitting: Analyzing the training dynamics of large language models. In *Advances in Neural Information Processing Systems 35* (pp. 38274–38290). https://proceedings.neurips.cc/paper_files/paper/2022/hash/fa0509f4dab6807e2cb465715bf2d249-Abstract-Conference.html
- Toppino, T. C., Phelan, H.-A., & Gerbier, E. (2018). Level of initial training moderates the effects of distributing practice over multiple days with expanding, contracting, and uniform schedules: Evidence for study-phase retrieval. *Memory & Cognition, 46*(6), 969–978. https://doi.org/10.3758/s13421-018-0815-7
- Yang, Y., Jones, M., Mozer, M. C., & Ren, M. (2024). Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training. In *Advances in Neural Information Processing Systems 37*. https://openreview.net/forum?id=YSs1z5udBY

---

## Open placeholders and checks

| Tag | Where | What is needed |
|---|---|---|
| † | Abstract, §4 Tables R1–R2, CIs throughout | Recompute from the `runs/` logs (request them from the team), together with the CI script |
| ‡ | Abstract, §3.3, §4, §5, §6 | Confirm the per-fact exposure counts against logged example IDs |
| [CHECK] | §3.6 | Confirm the CI method (likely a paired t-interval, df = 2) |
| [CHECK] | Figure R1 caption | The definition of the shaded band in the plotting code |
| [TODO] | §3.5, §5 | Exact-match results; the reviewed-vs-unreviewed fact analysis |
| [TODO] | §4 trajectory change | A CI for the expanding-minus-uniform trajectory difference |
| [TODO] | §1.3 [PILOT] contribution 3 | Replace "[PRD reference / follow-up]" with a pointer suited to the venue |
| [TODO] | All [RERUN] sections | Calibrated design values and results (PRD Part 1) |
