# Does Spaced Review Help a Language Model Retain Facts During Continued Training?

**An expanding- versus uniform-interval study on OLMo with FictionalQA**

Anshul Mago and William · OLMo FictionalQA Review Experiment
Code: https://github.com/anshulmago1/olmo-fictionalqa-review

---

## Abstract

When a language model keeps training on new material, facts it learned earlier tend to fade. A natural fix borrowed from human learning is *review*: periodically re-expose the model to old facts while it takes on new ones. We ask two questions. First, does reviewing previously learned facts actually improve their long-delay retention under a fixed review budget? Second, does an *expanding* review schedule—reviews that start close together and space out over time, mirroring the spacing effect in human memory—beat a plain *uniform* schedule? Using LoRA fine-tuning on OLMo DataDecide 300M and the FictionalQA dataset, we find that review helps: both review schedules reduced delayed old-fact loss by roughly 4.3% relative to no review, and the advantage survived a long no-review buffer across all three seeds. But expanding and uniform review were statistically and practically indistinguishable. The experiment supports review as a technique; it does not support any particular spacing schedule—mirroring a human literature in which expanding retrieval, too, shows no reliable long-delay advantage over uniform practice.

## 1. Introduction and Background

Continual learning in neural networks runs into catastrophic forgetting: as a model updates on new data, the representations that encoded earlier knowledge drift, and performance on the old task degrades (McCloskey & Cohen, 1989; French, 1999). This is a well-documented obstacle for any system expected to keep learning after deployment rather than being retrained from scratch.

Human memory research offers a candidate remedy. The *spacing effect*—the finding that learning is more durable when study sessions are distributed over time rather than massed together—is one of the most robust results in cognitive psychology (Cepeda et al., 2006), and *expanding-interval* schedules (reviewing an item after progressively longer gaps) are a common practical implementation, familiar from flashcard systems such as Anki, and are shown to greatly improve memory in humans (Landauer & Bjork, 1978; Cepeda et al., 2006). Even among humans, though, whether an *expanding* schedule specifically beats a plain *uniform* one is contested: equally spaced retrieval can match or exceed expanding practice when memory is tested after a long delay (Karpicke & Roediger, 2007). It is tempting to assume the same principle should govern how we rehearse old data into a model that is still training.

Whether that intuition transfers to gradient-based learning in a transformer is not obvious. The mechanisms behind human spacing—consolidation, retrieval difficulty, encoding variability (Cepeda et al., 2006)—do not have clean analogues in a network trained by stochastic gradient descent. So the question is empirical: given a fixed budget of review opportunities, does *when* we schedule them matter, or is it enough simply *that* we review at all?

We isolate this question with a controlled setup. To make sure we are measuring retention of newly taught knowledge rather than facts the model already absorbed during pretraining, we use FictionalQA, a dataset of invented facts that cannot appear in the pretraining corpus. We teach a set of old facts, then continue training on new facts while interspersing a fixed number of reviews of the old ones, and finally push the model through a long stretch of new-fact training with no review at all to measure how well the old knowledge holds up after a realistic delay.

## 2. Methods

### 2.1 Model and training

We fine-tuned OLMo DataDecide 300M (approximately 377M total parameters; Magnusson et al., 2025; Groeneveld et al., 2024) using LoRA (rank 16, α = 32, applied to all linear layers; Hu et al., 2021), which left 5.24M trainable parameters. Working through a low-rank adapter keeps the intervention lightweight and confines the changes to a small, well-defined set of weights, which is appropriate for a study about incremental knowledge injection rather than full retraining.

Optimization used a batch size of 16, a sequence length of 64 tokens, and a learning rate of 1 × 10⁻⁴ with a 10-step linear warmup held constant thereafter. The constant (non-decaying) schedule is deliberate: it ensures that exposures occurring later in training are not systematically down-weighted relative to earlier ones, which would otherwise confound the review schedule with learning-rate decay.

All facts came from FictionalQA (Kirchenbauer et al., 2025; https://huggingface.co/datasets/jwkirchenbauer/fictionalqa), a corpus of fictional facts. Because the facts are invented, any measured knowledge of them must have been acquired during our fine-tuning rather than carried over from pretraining—this is what makes the retention signal clean.

We ran every condition under three seeds (17, 23, and 42) and report means with standard deviations across those seeds. Comparisons between conditions are paired by seed.

### 2.2 Experimental phases

Each run proceeded through three phases. Note that each arm—no review, uniform, and expanding—has the same number of training steps.

- **Stage 1 (60 updates):** teach the old fictional facts.
- **Stage 2 (180 updates):** introduce new fictional facts. In the two review arms, reviews of the Stage 1 facts are interspersed among these updates.
- **Buffer (180 updates):** continue training on new facts with no review of old facts at all.

The buffer is the heart of the design. It simulates the realistic situation in which a model keeps learning long after its last chance to rehearse a given fact, and it lets us measure retention at a genuine delay rather than immediately after review, when any method looks good. The final review occurred at Stage 2 step 165, so by the time we take the final measurement—after the full buffer—the model has gone roughly 194 updates since it last saw an old fact.

### 2.3 Review conditions

All review arms were held to the same budget of 12 review events, so any difference between them reflects scheduling rather than a difference in total review effort.

- **No review:** old facts are never repeated after Stage 1.
- **Uniform:** the 12 reviews are spread at roughly equal intervals across Stage 2.
- **Expanding:** reviews begin close together early in Stage 2 and are spaced progressively farther apart.

### 2.4 Metrics

All losses are answer-token cross-entropy measured on a held-out evaluation set—paraphrased questions about the facts that never appear in training—so the scores reflect retained knowledge rather than memorized training text (lower is better). We track:

- **Pre-buffer old loss** — old-fact knowledge measured immediately before the buffer begins.
- **Final old loss** — old-fact knowledge after the full 180-update buffer. This is our primary delayed-retention metric.
- **Buffer forgetting** — final old loss minus pre-buffer old loss; positive means the model lost ground during the buffer.
- **Old-loss change** — final old loss minus the loss at the end of Stage 1; negative means the old facts ended up better learned than they were right after initial teaching.
- **Trajectory change** — the average old-loss change across Stage 2, capturing how well knowledge was held *during* training rather than only at the end.
- **New loss** — final loss on the newly introduced facts, a measure of plasticity.
- **Joint loss** — an equal-weight average of final old-fact loss and final new-fact loss.

## 3. Results and discussion

### 3.1 Review improves delayed retention

As expected, review of facts improved retention. Both review arms ended the buffer with substantially lower old-fact loss than the no-review control, and the effect was consistent across all three seeds.

| Condition | Reviews | Pre-buffer old loss | Buffer forgetting | Final old loss | Old-loss change | Trajectory change | New loss | Joint loss |
|-----------|:-------:|:-------------------:|:-----------------:|:--------------:|:---------------:|:-----------------:|:--------:|:----------:|
| No review | 0  | 3.830 | +0.151 ± 0.030 | 3.981 ± 0.031 | −0.223 ± 0.021 | −0.305 ± 0.014 | 2.993 ± 0.038 | 3.487 ± 0.031 |
| Uniform   | 12 | 3.648 | +0.161 ± 0.009 | 3.809 ± 0.015 | −0.396 ± 0.015 | −0.423 ± 0.021 | 3.018 ± 0.024 | 3.413 ± 0.019 |
| Expanding | 12 | 3.650 | +0.158 ± 0.008 | 3.808 ± 0.016 | −0.396 ± 0.014 | −0.436 ± 0.023 | 3.015 ± 0.019 | 3.411 ± 0.017 |

*Values are means across three paired seeds; ± values are standard deviations across seeds.*

Paired against the control, the final old-loss reductions are clear and their confidence intervals sit well away from zero:

| Comparison | Final old-loss difference | 95% paired CI |
|------------|:-------------------------:|:-------------:|
| Uniform − no review    | −0.1727 | [−0.2140, −0.1314] |
| Expanding − no review  | −0.1733 | [−0.2100, −0.1366] |
| Expanding − uniform    | −0.0006 | [−0.0053, +0.0041] |

In relative terms, uniform review cut delayed old-fact loss by about 4.34% and expanding by about 4.35%.

### 3.2 Retention holds throughout the buffer

Tracking old-fact loss at several points into the buffer shows that the review advantage is not a fragile artifact of one measurement point—it is present early and persists as the delay grows.

| Buffer length | No review | Uniform | Expanding |
|---------------|:---------:|:-------:|:---------:|
| 15 updates  | 3.846 | 3.664 | 3.666 |
| 60 updates  | 3.880 | 3.698 | 3.699 |
| 180 updates | 3.981 | 3.809 | 3.808 |

Both review arms kept a clear margin over no review at every checkpoint, and the two review schedules stayed nearly on top of each other the whole way.

The old-fact loss trajectory plot (old-fact answer-token loss vs. updates after old-fact acquisition, with the "180-step no-review buffer starts" line marked, and the three curves for no review, uniform, and expanding). This is the natural place for it: it visually anchors the retention-over-the-buffer discussion in this subsection, showing all three conditions dropping together during Stage 2, the review arms pulling below no-review, and all three rising again once the buffer begins.

### 3.3 Review changes the starting point, not the forgetting rate

A useful way to read the mechanism: review helped mainly by leaving the old facts *better learned before* the buffer, not by slowing the rate at which they were subsequently forgotten. Every condition lost roughly 0.15–0.16 loss units over the buffer, and the differences in that forgetting rate are indistinguishable from zero.

| Comparison | Difference in buffer forgetting | 95% paired CI |
|------------|:-------------------------------:|:-------------:|
| Uniform − no review   | +0.0097 | [−0.0718, +0.0911] |
| Expanding − no review | +0.0066 | [−0.0663, +0.0794] |
| Expanding − uniform   | −0.0031 | [−0.0118, +0.0057] |

So review does not appear to install a more durable memory that decays more slowly; it simply starts the buffer from a lower loss, and that head start carries through.

### 3.4 A small plasticity cost

Rehearsing old facts is not free. Both review arms paid a small numerical penalty on new-fact learning during stage 2 and the buffer, though in each case the confidence interval crosses zero, so the cost is suggestive rather than established:

- Uniform vs. no review: +0.0252 new loss, 95% CI [−0.0258, +0.0762]
- Expanding vs. no review: +0.0216 new loss, 95% CI [−0.0329, +0.0760]

On the equal-weight joint metric, which balances old- and new-fact performance, review still comes out ahead: uniform improved joint loss by 0.0737 and expanding by 0.0758. The gap between the two schedules was 0.0021, with a confidence interval spanning zero.

### 3.5 Expanding versus uniform

The one place expanding review looked better was the trajectory metric—the average old-loss change measured *during* Stage 2 (−0.436 for expanding vs. −0.423 for uniform). This is most plausibly an artifact of expanding's front-loaded schedule: piling reviews early in Stage 2 keeps old-fact loss lower through the middle of training, which flatters an average taken over that window. Once training ended and both arms passed through the same buffer, that edge disappeared. On the metric that actually matters for deployment—retention after a long delay—the two schedules were functionally tied, with an expanding-minus-uniform interval of [−0.0053, +0.0041] hugging zero.

## 4. Conclusion

Under a matched review budget, periodically reviewing old facts improved their retention after a prolonged stretch of continued training. The effect was about 4.3% on delayed old-fact loss, it survived a substantial 180-update no-review buffer, and it appeared in all three paired seeds. Review earns its keep.

The spacing schedule, however, did not matter. Uniform and expanding reviews were statistically and practically indistinguishable on final retention; expanding's apparent advantage during training was a byproduct of front-loading its reviews and washed out once the buffer intervened. The experiment supports review itself, not a specific spacing strategy. Notably, that null mirrors the human literature, where equally spaced practice matches or beats expanding retrieval once memory is tested after a long delay (Karpicke & Roediger, 2007); the contribution here is showing that this human phenomenon reproduces in LLM continued training—review transfers, but the fine-grained spacing schedule confers no long-delay advantage in either setting.

The practical implication follows directly. For a production system, uniform review is the sensible default: it matches expanding review's retention without the added complexity of a schedule that has to decide how fast to stretch its intervals. If a more elaborate scheduler is going to earn its place, this experiment did not find the evidence for it.

A few caveats bound these conclusions. The study used a single small model (300M), a single fact domain (fictional QA), LoRA rather than full fine-tuning, and three seeds. It is entirely possible that spacing effects emerge at larger scales, over longer horizons, with more review events, or under different kinds of knowledge—these are the natural directions for follow-up. What we can say from this experiment is narrow and, we think, well supported: review helps, and how you space it does not.

## References

Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin, 132*(3), 354–380. https://doi.org/10.1037/0033-2909.132.3.354

French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2

Groeneveld, D., Beltagy, I., Walsh, P., Bhagia, A., Kinney, R., Tafjord, O., et al. (2024). OLMo: Accelerating the Science of Language Models. *arXiv:2402.00838.* https://arxiv.org/abs/2402.00838

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685.* https://arxiv.org/abs/2106.09685

Karpicke, J. D., & Roediger, H. L. (2007). Expanding retrieval practice promotes short-term retention, but equally spaced retrieval enhances long-term retention. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 33*(4), 704–719. https://doi.org/10.1037/0278-7393.33.4.704

Kirchenbauer, J., Mongkolsupawan, J., Wen, Y., Goldstein, T., & Ippolito, D. (2025). FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition. *arXiv:2506.05639.* https://arxiv.org/abs/2506.05639

Landauer, T. K., & Bjork, R. A. (1978). Optimum rehearsal patterns and name learning. In M. M. Gruneberg, P. E. Morris, & R. N. Sykes (Eds.), *Practical Aspects of Memory* (pp. 625–632). Academic Press.

Magnusson, I., Tai, N., Bogin, B., Heineman, D., Hwang, J., Soldaini, L., Bhagia, A., Liu, J., Groeneveld, D., Tafjord, O., Smith, N. A., Koh, P. W., & Dodge, J. (2025). DataDecide: How to Predict Best Pretraining Data with Small Experiments. *arXiv:2504.11393.* https://arxiv.org/abs/2504.11393

McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation, 24*, 109–165.
