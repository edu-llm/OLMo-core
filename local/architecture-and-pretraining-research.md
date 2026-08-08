# Making a Better Small Model: Architecture and Pretraining Levers for OLMo-core

**A brainstorm, literature scan, and experiment plan for improving model quality and training efficiency, validated at 370M and scaled toward 7B**

> **Status:** This is a research-planning document — a survey of what to change, why we think it will help, and how we would test it honestly. It is not a results writeup. Every proposed experiment is designed to be run at matched budget so that a measured difference can be attributed to the intervention rather than to spending more compute. Where a technique is already available in OLMo-core, the relevant config is named; where it would require new code, that is flagged explicitly.

---

## 1. Purpose and scope

We are building toward our own 7-billion-parameter model. For now we develop and screen ideas on the 370M OLMo model (`olmo2_370M`: `d_model=1024`, 16 layers, 16 heads), which is cheap enough to iterate on and — as this document argues — close enough in kind to the 7B target that most decisions transfer. The near-term goal is a *better base model*: the same or better quality for less training compute, or higher quality at a fixed parameter budget, or cheaper to serve at inference. Tailoring the model to education is a later objective and is deliberately out of scope here, with one exception: the timing-of-data question in Section 5.6 turns out to be both a general pretraining lever *and* the natural hook for later educational specialization, so we develop it fully.

Two constraints shape everything below.

First, **OLMo-core is unusually modular.** Attention, the sequence mixer itself, normalization, positional encoding, the feed-forward block, the optimizer, the learning-rate schedule, the data mixture, and the MoE machinery are all selected through config (`Registrable` registries or enum dispatch), not hard-coded. In practice this means many of the interventions we would want to try are *configuration changes, not engineering projects*, and even the ambitious ones (a Mamba-style layer, a reasoning objective) slot into abstractions that already exist. That inverts the usual cost model: the expensive part is often not building a technique, it is running a clean enough experiment to know whether it helped.

Second, we inherit a **methodological discipline** from the project's earlier learning-science work (see `experiment-history.md`): match budgets before comparing; expect the headline effect to shrink to something simpler; use contamination-free evaluation; treat a null result as a real deliverable; and — the confound that has bitten this project before — hold the learning-rate schedule identical across arms, because data or changes introduced late under a decaying LR barely move the model and can masquerade as a curriculum effect. Those lessons are baked into the designs in Section 6, and they are exactly what let us take swings at riskier ideas without fooling ourselves.

---

## 2. What today's small models actually struggle with

It helps to be concrete about the failure modes a "better" model would fix, because they point at different levers.

**Training instability at scale.** Loss spikes, attention-logit blow-ups, and divergence become more likely as models and learning rates grow. This is a direct tax on training efficiency, because the usual defense is a conservatively low learning rate that wastes compute. Much of the recent architecture work on OLMo 2 and its peers is, at bottom, about buying back that lost learning rate through stability (Section 4.2).

**Compute- and data-inefficiency.** The Chinchilla result says that for a *fixed training budget* the optimal model trains on roughly 20 tokens per parameter. But that is the wrong objective for us: a deployable model is trained once and served many times, so it pays to train a *smaller* model far past its compute-optimal point on high-quality data — the regime the strong small models (Llama, Qwen, OLMo 2) live in. There the binding constraint becomes "how good is each token of data" and "how much does each token teach," which is what data-quality filtering, rephrasing augmentation, and token-selective losses all attack (Sections 5.3 and 5.6).

**The KV cache and long context.** At inference, memory and latency are dominated by the key/value cache, which grows linearly with context length and with the number of key/value heads. A 7B model that is expensive to serve at long context is a worse product than a slightly less accurate one that is cheap. Grouped-query attention, sliding-window attention, latent attention (MLA), and hybrid SSM designs are all attacking this (Sections 4.1 and 4.6).

**Knowledge and reasoning per parameter.** Small models have less room to store facts and to run multi-step reasoning. The structural responses are mixture-of-experts (add knowledge capacity without per-token compute), richer objectives like multi-token prediction, and — most ambitiously — folding chain-of-thought *into* pretraining rather than eliciting it afterward (Sections 4.4, 5.4, 5.6).

**Forgetting under continued training.** Relevant because our roadmap includes later domain adaptation: models forget earlier knowledge as they train on new data. The project's own review experiment already characterized this (`expanding-interval-review-whitepaper.md`); any late-stage specialization phase must be evaluated for retention, not just for gains on the new domain.

---

## 3. How to read the brainstorm: the two cost tiers

Every lever below is tagged with how much it costs *us* to try, which given OLMo-core's modularity is mostly a question of whether the code already exists:

- **[Config]** — already implemented; trying it is a config change and a training run. These are the cheap screens and should come first.
- **[Code]** — not currently in OLMo-core; would require implementing a module behind an existing abstraction before it can be tested. Higher-effort, and should be justified by a config-level signal first or by a large expected payoff.

Some levers are **[Config→Code]**: a first, informative version is config-only, and a stronger version needs code.

---

## 4. Architecture levers

### 4.1 Attention: the KV-cache and long-context frontier

This is the richest classical area, because it trades directly against serving cost.

**Grouped-query attention (GQA) head-count sweep. [Config]** `AttentionConfig.n_kv_heads` below `n_heads` gives GQA; `=1` gives multi-query. Fewer KV heads shrink the cache proportionally at usually-small quality cost until the ratio gets aggressive. The cheapest inference win available.

**Sliding-window attention (SWA). [Config]** `SlidingWindowAttentionConfig` supports a per-layer pattern (e.g. `[4096,4096,4096,-1]`: three local layers then one global, repeating — the OLMo 3 recipe). Local layers cost less and cap KV growth; periodic global layers preserve long-range reach. The design question is window size and local:global ratio.

**QK-normalization. [Config]** Normalizing queries and keys before the dot-product (`qk_norm`, with a per-head option) is a main OLMo 2 stability fix — it prevents the attention-logit growth that triggers spikes. Already on in `olmo2_*`; the experiment is to confirm it lets us raise the learning rate at our scale.

**Multi-head latent attention (MLA). [Code]** DeepSeek-V2/V3's KV-cache idea: project keys/values into a low-rank latent and cache only the compressed latent, shrinking the cache dramatically at near-full quality. **Not implemented** (no MLA/KV-LoRA path in `attention/`). Highest-ceiling classical attention change; a strong candidate for a real engineering investment once the cheaper GQA/SWA screens quantify how much KV pressure costs us.

### 4.2 Normalization and residual placement: buying back learning rate

**Reordered-norm block. [Config]** OLMo 2 normalizes the *outputs* of the attention and feed-forward sublayers (`reordered_norm` block type) — part of its stability recipe, and already the `olmo2_*` default. Alternatives to compare: `peri_norm` (Peri-LN) and `default_scaled` (`1/sqrt(layer_id)` scaling). A cheap study comparing pre-norm / reordered / peri at fixed compute, measuring the *maximum stable learning rate* (not just final loss), tells us which placement trains fastest at our scale.

**RMSNorm variant and precision. [Config]** `rms`, `qwen_rms`, `fused_rms`, `cute_rms` differ in speed/rounding, not really quality. A throughput knob, not a headline.

### 4.3 Positional encoding

**RoPE theta and long-context scaling. [Config]** `RoPEConfig.theta` (500k default) and the `RoPEScalingConfig` family (YaRN, position-interpolation, Llama-3 stepwise) are context-length levers, relevant when we extend the 7B window in a late phase rather than to from-scratch quality.

### 4.4 Mixture-of-experts: knowledge capacity without per-token cost

**MoE with shared experts and aux-loss-free balancing. [Config]** OLMo-core has the full stack: `MoEConfig` with fine-grained experts, an optional dense `shared_mlp` (DeepSeek/OLMoE shared-expert pattern), and DeepSeek-V3's **auxiliary-loss-free load balancing** via a per-expert router bias (`bias_gamma` on `MoERouterConfig`), which avoids the quality tax of a heavy load-balancing loss. Ready factories: `smallmoe` (32 experts, top-4, shared MLP), `olmoe_1B_7B` (dropless, 64 experts, top-8). The most promising *structural* route to more capability per unit of inference compute; heavy to compare fairly (Section 6, Experiment 6).

### 4.5 Embeddings and the LM head

**Tied embeddings. [Config]** `tie_word_embeddings` shares input embedding and output projection — usually favorable at 370M (the embedding matrix is a big parameter fraction), less obviously at 7B.

**z-loss on logits. [Config]** A small penalty on the softmax log-partition (`z_loss_multiplier`) keeps logits from drifting; an OLMo 2 stability ingredient. Keep on as a control.

**Logit soft-capping. [Code]** Gemma-style logit bounding is **not implemented**; low priority unless z-loss proves insufficient.

### 4.6 Hybrid SSM / attention architectures — the deepest architectural bet

This is the frontier direction, and OLMo-core is unusually well-positioned for it because it already registers a recurrent, linear-attention-style mixer (`gated_delta_net`) behind the `Registrable` `SequenceMixerConfig` — the exact abstraction a state-space layer would plug into — and sliding-window attention already exists.

The premise: attention is quadratic with an unbounded KV cache but recalls any past token exactly. A selective **state-space model (SSM)** — Mamba / Mamba-2 — is linear-time and carries a *fixed-size* recurrent state, so it compresses history cheaply but blurs precise recall. Mamba-2's **state-space duality** shows an SSM is a structured-matrix sequence transform computable either as a recurrence or as an attention-like quadratic form; that is why the families are roughly matched in expressivity at small–medium scale, and Mamba-2's core layer runs 2–8× faster than Mamba-1. The winning recipes do not pick one — they combine them:

- **Layer-wise interleaving (Samba). [Config→Code]** Alternate SSM layers (cheap long-range compression) with SWA layers (sharp local recall). Samba (3.8B, 3.2T tokens), trained at 4K context, extrapolated to 1M-token perplexity zero-shot and 256K with perfect passkey recall, at **3.73× the throughput** of a GQA transformer on 128K-token prompts. In our codebase, interleaving the registered `gated_delta_net` mixer with SWA layers is a **config-level** first cut; swapping in a true Mamba-2 layer is the **[Code]** upgrade.
- **Parallel hybrid heads (Hymba). [Code]** Run attention heads and SSM heads *in parallel within one layer*, fusing high-resolution recall with context summarization, plus learnable "meta tokens" and cross-layer KV sharing. **Hymba-1.5B beat Llama-3.2-3B** by +1.3% average accuracy with an **11.7× smaller KV cache** and **3.5× throughput** — a striking result at exactly our target scale. A parallel-head block is new code, but it subclasses the same block abstraction OLMo-core already uses.

Why this matters for us specifically: the strongest published *small* models increasingly are hybrids, and the efficiency wins (cache, throughput, length extrapolation) are largest precisely in the 1–2B range where we are prototyping. This is the architecture experiment most likely to produce a genuinely differentiated 7B model rather than a well-tuned clone.

---

## 5. Pretraining levers

### 5.1 The optimizer: the cheapest large win

**Muon (and NorMuon / Dion). [Config]** OLMo-core registers `muon`, `nor_muon`, `dion` alongside `adamw` and `lion`. Muon replaces AdamW's per-coordinate rule with a matrix-orthogonalized update on 2-D weights; scaled up (weight decay + a per-parameter update-scale correction) it reports ~**2× the compute efficiency of AdamW** at compute-optimal training, demonstrated on a 3B/16B-active MoE trained on 5.7T tokens. Already implemented, so an AdamW-vs-Muon screen is the highest expected-value experiment we can run nearly for free (Experiment 1).

**Skip-step wrappers. [Config]** `skip_step_adamw` / `skip_step_lion` skip an update when loss or grad-norm spikes beyond a rolling threshold — stability insurance that composes with anything.

### 5.2 The learning-rate schedule and mid-training

**Warmup-Stable-Decay (WSD). [Config]** `wsd` (plus cosine, inverse-sqrt, composable/sequential) holds LR flat then decays late. Two virtues: we can branch continued-training runs off the stable-phase checkpoint without committing to a horizon, and it concentrates "annealing" into a defined final phase — the standard place a high-quality data mix pays off. But *where* the good data belongs is itself a research question (Section 5.6).

### 5.3 Data quality and mixture: still the biggest classical lever

**Model-based filtering (DCLM). [Config to run; external to build the filter]** DataComp-LM found model-based data filtering was the dominant factor in pretraining quality — a 7B hitting 64% MMLU with ~6.6× less compute than a comparably-scored baseline. OLMo-core's composable loader (`ComposableDataLoaderConfig` with document/token/instance-level `mixing` sources and ratios) is built for mixture experiments; the cost is producing the filtered corpus, not the run.

### 5.4 The training objective

**Multi-token prediction (MTP). [Code]** DeepSeek-V3 predicts several future tokens via small extra heads, densifying the per-token signal and doubling as a speculative-decoding aid. **Not implemented** (today's objective is next-token CE plus z-loss / MoE aux losses). The most credible pure-objective change; a mid-tier engineering investment.

### 5.5 Precision and throughput

**FP8 / low precision. [Config, partial]** The train module exposes `float8_config`. A throughput ("free tokens") lever, not a quality one; validate that loss tracks the higher-precision baseline before trusting it on a long run.

### 5.6 The *timing* of augmented, instruction-formatted, and reasoning data

This is the most novel pretraining direction, and it is where a teammate's proposal points. The standard recipe puts the "nice" data — cleaned, instruction-shaped, high-quality — at the *end* (the anneal). The proposal is to invert that: what if post-training-*style* data (augmented, rephrased, instruction/QA-formatted, reasoning-annotated) goes at the *beginning*, or is spread throughout? That is a question about the **curriculum of data *format***, and it is genuinely underexplored at scale. Three recent techniques make it concrete and buildable, and a fourth pushes it as far as it goes:

- **Rephrasing augmentation (WRAP). [Config to run]** Use an instruction-tuned model to paraphrase raw web docs into cleaner styles (Wikipedia-like, Q&A, simplified), and pretrain on real + synthetic *jointly*. WRAP reported ~**3× pretraining speedup**, **>10% perplexity** improvement, and **>2% zero-shot QA** gains, crediting style diversity that matches downstream evaluation and higher per-token quality.
- **Instruction pretraining. [Config to run]** Convert raw corpora into ~200M instruction-response pairs (40+ task categories) with an open-source synthesizer and mix them in *from the start*. In continual pretraining this let **Llama-3-8B rival Llama-3-70B** — instruction-shaped data during pretraining changes the base model itself, not just the fine-tune.
- **Selective language modeling (Rho-1). [Code]** Score every token with a reference model and apply the loss *only to high-value tokens*. On math this matched DeepSeekMath with **~3% of the tokens (~33× efficiency)** and up to **+30% few-shot** accuracy — a token-level lever that composes with any mixture. New code (a scored-token loss mask), but small.
- **Reasoning injection (Quiet-STaR). [Code]** The radical version: teach the model to emit a short internal rationale (bracketed by learnable "thought" start/end tokens) at many positions during ordinary text, sampled in parallel, rewarded by how much it improves prediction of the *actual* next text (REINFORCE-style), blended via a mixing head. On continued pretraining alone — no task fine-tuning — this lifted zero-shot **GSM8K 5.9→10.9%** and **CommonsenseQA 36.3→47.2%**. This is chain-of-thought moved *into* pretraining.

The teammate's intuition — "we usually feed slightly-altered / post-training-style data at the *end*; what if we do it at the *beginning*?" — is exactly the kind of learning-science question this project is built to answer (do "worked examples and clean explanations first" help a model the way they help people?), and it can be tested with the same matched-budget discipline. Experiment 3 is built around it; the reasoning-injection version is Experiment 4.

---

## 6. Prioritized experiment plan

Ordering principle: run the cheap, already-implemented screens with the largest expected payoff first; use them to decide which expensive **[Code]** investments are justified; and validate hyperparameter transfer continuously so the 7B run is chosen from evidence. All experiments are at 370M unless stated, use contamination-free evaluation where retention is at issue, and hold the LR schedule fixed across arms within a comparison. Each design names the specific numbers from the literature it is trying to reproduce, so a run either clears that bar or gives us an informative null.

| # | Experiment | Cost tier | Payoff | Core question |
|---|---|---|---|---|
| 1 | Muon (and NorMuon) vs AdamW | [Config] | High, cheap | Fewer tokens/FLOPs to target loss? (bar: ~2×) |
| 2 | **Hybrid SSM/attention (Samba interleave → Hymba parallel heads)** | [Config→Code] | Very high | Better quality-per-cache and length extrapolation than pure attention? |
| 3 | **Timing of augmented / instruction data: front-load vs uniform vs anneal** | [Config to run] | Very high | Does post-training-*style* data help *most* at the start? |
| 4 | **Reasoning-in-pretraining (Quiet-STaR at 370M)** | [Code] | High ceiling, risky | Do latent rationales during pretraining raise zero-shot reasoning? |
| 5 | KV-cache frontier: GQA/SWA sweep → MLA | [Config→Code] | Medium–high (serving) | How much cache can we cut at iso-quality? |
| 6 | MoE done right (fine-grained + shared + aux-loss-free) | [Config] | High, heavy | More quality per active param / per train FLOP? |
| 7 | Multi-token prediction (MTP) | [Code] | Medium | Does a denser objective speed learning + enable spec-decoding? |
| 8 | Hyperparameter-transfer ladder 370M→1B→7B | [Config] | De-risks 7B | Do LR/init tuned small transfer to large? |

### Experiment 1 — Optimizer efficiency: Muon vs AdamW

**Hypothesis.** At matched compute, Muon reaches a given validation loss in meaningfully fewer tokens than a well-tuned AdamW, toward the ~2× compute-efficiency reports at larger scale.

**Design.** `olmo2_370M`, identical data order, identical WSD schedule shape. Arms: `adamw`, `muon`, and `nor_muon` as a third point. The essential fairness requirement is a *fair LR sweep for every arm* — the classic way to flatter a new optimizer is to under-tune the baseline. Handle the embedding/norm parameter groups identically across arms (Muon's matrix update applies only to 2-D weights; the optim group overrides make this explicit).

**Metric.** Tokens and FLOPs to reach matched validation-loss thresholds; a small downstream suite for sanity. Report loss-vs-tokens *curves*, since the claim is efficiency, not just endpoint quality.

**Confounds.** Under-tuned baseline; mismatched effective batch size; inconsistent parameter-group handling. A vanished effect after proper AdamW tuning is a publishable null.

### Experiment 2 — Hybrid SSM/attention architecture

**Hypothesis.** A hybrid that interleaves a linear-recurrent (SSM-family) mixer with sliding-window attention matches or beats a pure-attention model of equal parameters and training compute on quality, while cutting KV cache and improving throughput and length extrapolation — reproducing the *direction* of Samba/Hymba at 370M.

**Design.** Three arms at matched parameters and matched training FLOPs, all trained at a 4K context:
1. **Baseline:** `olmo2_370M`, full attention.
2. **Samba-style interleave [Config→Code]:** alternate the registered `gated_delta_net` mixer with SWA layers (e.g. recurrent : SWA in a fixed repeating pattern), varying the ratio as the key hyperparameter. This is the config-level first cut; if it signals, swap the recurrent mixer for a true Mamba-2 layer as the [Code] upgrade.
3. **Hymba-style parallel heads [Code]:** a block running attention and SSM heads in parallel, with cross-layer KV sharing.

Evaluate quality in-distribution *and* run explicit **length-extrapolation probes** — passkey retrieval and phonebook-style lookup at 16K/32K/64K despite 4K training — because that is where hybrids are supposed to shine and where a pure-attention baseline should struggle.

**Metric.** Quality (loss + downstream) at equal params/FLOPs; KV-cache size and tokens/sec at long context; extrapolation accuracy vs context length. The deliverable is a quality-vs-serving-cost frontier plus an extrapolation curve. Bars to reference: Samba's 3.73× throughput and clean 256K passkey recall; Hymba-1.5B's +1.3% accuracy at 11.7× smaller cache.

**Confounds.** Parameter-count matching across heterogeneous layer types (count carefully); the recurrent mixer's own hyperparameters (state size) need their own small sweep or the hybrid is handicapped; SWA window must be exercised by a long-enough eval context.

**Why start config-only.** The interleave arm is buildable today; it tells us whether the hybrid *shape* helps before we invest in a bespoke Mamba-2/Hymba layer.

### Experiment 3 — The timing of augmented / instruction-formatted data (the teammate's idea)

**Hypothesis.** Post-training-*style* data (WRAP rephrasings + instruction/QA pairs synthesized from the raw corpus) helps *most* when it is present early or throughout pretraining, not only in a late anneal — and the standard "save it for the end" recipe leaves value on the table.

**Design.** This is a placement study, so content and budget are held fixed and only *when* the augmented data appears varies. Build one augmented pool once (rephrase a slice of the base corpus WRAP-style into Wikipedia/Q&A/simplified styles, and synthesize instruction-response pairs from it), then train matched-token, matched-LR-schedule arms that differ only in schedule of that pool:
1. **Anneal (standard baseline):** augmented data concentrated in the final phase.
2. **Front-loaded (the proposal):** augmented data concentrated early, raw web text later.
3. **Uniform:** augmented data spread evenly throughout.
4. **Decay-control:** the *base* mix (no augmentation) in the final phase, matched schedule.

The comparison that isolates the real effect is **front-loaded / uniform vs anneal**, with the decay-control arm present to catch the learning-rate-decay artifact that has fooled this project before (`experiment-history.md`, lesson 6): if "anneal wins" only because *anything* in the decay window looks good, arm 4 exposes it. Optionally add a **Rho-1 selective-loss arm [Code]** — mask the CE loss to high-value tokens scored by a reference model — as a token-level version of "spend the budget where it teaches most."

**Metric.** Downstream benchmark average (held-out, contamination-checked — the synthetic pool must be screened against eval sets); *when* capabilities emerge (track downstream vs tokens, not just the endpoint — the interesting claim is that front-loading changes the learning trajectory); and a base-domain retention probe. Bars: WRAP's ~3× speedup and >2% QA; Instruction-Pretraining's base-model gains.

**Confounds.** The LR-decay confound (arm 4 is the guard); loss-not-comparable-across-distributions (use downstream metrics, since augmented and raw data have different loss scales); synthetic-data contamination of the eval; and rephrasing-model bias (the augmenter's style leaking into evaluation-shaped gains — report OOD tasks too).

**Why this is the on-brand experiment.** It is a learning-science question about *format curriculum*, testable with the team's existing matched-budget machinery, and it doubles as the exact design we would reuse to fold educational data in early rather than late.

### Experiment 4 — Reasoning-in-pretraining (Quiet-STaR at 370M) [Code]

**Hypothesis.** Training the 370M model to generate short latent rationales during ordinary text — rewarded only when the rationale improves prediction of the real following tokens — raises zero-shot reasoning without any task fine-tuning, reproducing the direction of Quiet-STaR's GSM8K 5.9→10.9% and CommonsenseQA 36.3→47.2%.

**Design.** A continued-pretraining study from a base `olmo2_370M` checkpoint (cheaper and cleaner than from-scratch). Implement the Quiet-STaR machinery [Code]: learnable thought start/end tokens, tokenwise-parallel rationale sampling, a mixing head that blends with-thought and without-thought predictions, and a REINFORCE-style reward tied to next-text likelihood. Two arms at matched tokens: base continued-pretraining vs Quiet-STaR continued-pretraining. Ablate rationale length and thought frequency.

**Metric.** Zero-shot GSM8K, CommonsenseQA, and a general perplexity check (reasoning should disproportionately help hard-to-predict tokens, not harm easy ones — the mixing head is what protects the latter). Report compute overhead honestly: rationale sampling is expensive, so the fair question is gain *per unit of extra compute*.

**Confounds.** Reward variance (REINFORCE is noisy — expect seeds to matter); the compute overhead making "gains" unfair unless normalized; and 370M possibly being below the scale where latent reasoning emerges — a null here is informative about scale, not a dead end.

**Risk posture.** Highest-risk, highest-ceiling item. Justified as a research bet because it targets the reasoning-per-parameter problem head-on and is exactly the kind of "cool new thing" worth a contained trial at small scale before any 7B commitment.

### Experiment 5 — KV-cache frontier: GQA/SWA sweep, then MLA

**Hypothesis.** We can cut the KV cache substantially with negligible quality loss at 370M, and the ordering of options (GQA vs SWA vs, eventually, MLA) gives a clear serving recommendation for 7B.

**Design.** Matched-compute sweeps from the `olmo2_370M` baseline:
- **GQA [Config]:** `n_kv_heads` in {16, 8, 4, 2, 1}.
- **SWA [Config]:** global vs OLMo-3-style patterns (`[4096,4096,4096,-1]` and variants), varying window and local:global ratio.
- **MLA [Code]:** if GQA/SWA show cache pressure is the binding serving cost, implement latent attention and place it on the same Pareto plot.

**Metric.** Quality (loss + downstream) vs KV-cache size / inference memory / throughput — a Pareto curve whose "knee" is the recommendation.

**Confounds.** Too-short eval context to exercise SWA; conflating the sweeps (run separately, combine winners).

### Experiment 6 — MoE done right

**Hypothesis.** At matched *active* parameters and matched *total training FLOPs*, a fine-grained MoE with a shared expert and aux-loss-free balancing beats the dense baseline.

**Design.** Dense `olmo2_370M`-class baseline vs a `smallmoe`-style MoE (fine-grained experts + dense `shared_mlp` + `bias_gamma` aux-loss-free balancing). The fairness conditions *are* the experiment: pre-declare the matched axis (active-param-matched vs FLOP-matched; ideally report both), match data, and monitor routing.

**Metric.** Quality per active parameter and per training FLOP; serving-cost note; router load-balance and dead-expert count (a collapsed router invalidates the run).

**Confounds.** The matched-axis ambiguity; router collapse; parallelism complexity (why it is heavy despite being config-level).

### Experiment 7 — Multi-token prediction [Code]

**Hypothesis.** Adding MTP heads (predict the next *k* tokens) speeds convergence per token and yields a spec-decoding-friendly 7B, at modest cost.

**Design.** Implement extra prediction heads + an auxiliary MTP loss in the train module; screen at 370M against the next-token-only baseline at matched compute; ablate *k*.

**Metric.** Tokens-to-target-loss; downstream; and measured speculative-decoding acceptance rate as a bonus deliverable.

**Confounds.** Getting the loss weighting between the primary and auxiliary heads right; separating "denser signal helps" from "more parameters in the heads help" (match head parameters).

### Experiment 8 — Hyperparameter-transfer ladder

**Hypothesis.** LR and initialization tuned at 370M transfer, with a known correction, to 1B and 7B, so we need not tune blind at 7B.

**Design.** The methodological backbone. Using OLMo-core's `InitMethod` options (including depth-aware `llama_depth`) and a principled parameterization, fit loss-vs-compute scaling curves at 370M and ~760M/1B (`olmo2_760M`, `olmo2_1B`), confirm the optimal LR moves predictably with width/depth, and *predict* the 7B loss before committing the run. Run continuously as Experiments 1–7 produce runs at each scale.

**Metric.** Predicted vs actual loss at each rung; stability of the optimal LR under the chosen parameterization. A large predicted-vs-actual gap is itself the finding — it names the knob that failed to transfer.

**Confounds.** Skipping the 1B rung (it is the cheap checkpoint that catches a broken extrapolation); changing more than one thing between rungs.

---

## 7. A staged roadmap

1. **Cheap screens first (weeks).** Experiment 1 (optimizer) and Experiment 5's GQA/SWA phase are nearly free and settle *how we train* and *how we serve*. Run them immediately.
2. **The two flagship bets, in parallel.** Experiment 2 (hybrid SSM/attention) and Experiment 3 (data-timing) are the highest-upside, most differentiated directions. Experiment 3 is gated only by building the augmented pool once; Experiment 2's interleave arm is buildable today.
3. **The research swing.** Experiment 4 (Quiet-STaR) as a contained, well-instrumented trial — high ceiling, and a clean null at 370M is still a scale finding worth writing up.
4. **The heavy structural work, once earned.** Experiments 6 (MoE), and the [Code] upgrades in 2 (Mamba-2/Hymba), 5 (MLA), and 7 (MTP), justified by the screens.
5. **Transfer ladder throughout.** Experiment 8 runs continuously so the 7B configuration is chosen from evidence, not hope.

Throughout, the discipline from the earlier learning-science work applies unchanged: match the budget, control the learning-rate schedule, prefer contamination-free evaluation, isolate the simplest mechanism that explains the effect, and write up the nulls. The reason we can afford to chase the ambitious ideas — hybrids, reasoning-in-pretraining, format curricula — is that OLMo-core has already built most of the plumbing, and the team already knows how to run an experiment that will not lie to us.

---

## 8. A wider menu of ideas to screen

The eight experiments above are the ones we would design in full first, but the point of this section is breadth: a catalog of further novel-ish directions, each with a one-line rationale, the concrete evidence or number that makes it interesting, its cost tier, and where it would attach in OLMo-core. These are candidates to promote into full designs, not finished protocols. A few — nGPT, batch-size scheduling, Dion — are notable because the code already exists and nobody has screened them at our scale.

### 8.1 More architecture / efficiency ideas

- **nGPT (normalized transformer). [Config — already in codebase]** Represent every embedding and hidden vector on a unit hypersphere and make the residual stream a normalized "walk" toward the answer; reported multiplicatively faster convergence. OLMo-core already registers the `normalized` transformer type, block, attention, feed-forward, and LM head — this is a nearly-free screen that is sitting unused. Caveat: incompatible with tied embeddings.

- **Tokenizer-free / byte-latent modeling (BLT). [Code, large]** Drop the BPE tokenizer and operate on raw bytes grouped into *entropy-based dynamic patches* — more compute where the text is unpredictable, less where it is not. BLT matched tokenizer-based LLMs up to 8B params / 4T bytes with better inference scaling and robustness, and gains on reasoning and long-tail generalization. Especially interesting for an *educational* model: robustness to spelling, typos, arithmetic digits, and morphology that tokenizers mangle.

- **Latent recurrent depth / looped transformer. [Code]** Reuse a recurrent block N times to "think longer" at inference without adding parameters. The Huginn model (3.5B, 800B tokens) improved reasoning *dramatically* by iterating more at test time — up to a compute-equivalent of a ~50B model — with no special training data and small context windows. A parameter-efficient route to reasoning that is orthogonal to Quiet-STaR.

- **Vocabulary / tokenizer scaling. [Config — tokenizer swap]** Larger vocabularies favor larger models and change the compute/quality trade; an under-examined knob when we move from 370M to 7B. Cheap to include as a variable in the transfer ladder (Experiment 8).

- **NoPE (no positional encoding). [Config]** Decoder-only models can length-generalize *better* without any explicit positional signal, relying on the causal mask alone. A one-line ablation worth running alongside the RoPE-theta and SWA work.

- **Attention sinks / register tokens. [Config→Code, small]** A few always-attended "sink"/register tokens stabilize attention and enable streaming at long context (and echo Hymba's learnable meta tokens). Cheap insurance for long-context serving.

- **LayerSkip / early-exit + self-speculative decoding. [Code]** Train with layer dropout and early-exit heads so easy tokens exit shallow; doubles as self-speculative decoding for faster inference. A serving-efficiency play for the 7B.

### 8.2 More objective / reasoning ideas

- **Continuous latent chain-of-thought (Coconut). [Code]** Feed the last hidden state straight back as the next input embedding instead of decoding a token, so reasoning happens in continuous space and can hold several branches at once (BFS-like), beating explicit CoT on search-heavy planning with a better accuracy/efficiency trade. The natural sibling to Experiment 4 — pick one latent-reasoning bet to run well.

- **Fill-in-the-middle (FIM). [Code, small]** Reorder a fraction of documents into prefix–suffix–middle so the model learns infilling, not just left-to-right. Near-free extra capability (standard for code models) and useful for any editing/completion product.

- **Sparse upcycling. [Code]** Initialize an MoE from the *dense* 370M checkpoint rather than from scratch, recycling pretraining compute. Directly de-risks and cheapens the MoE study (Experiment 6).

- **In-pretraining distillation. [Code]** Blend a stronger teacher's logits into the CE loss during pretraining (not only after). Connects to the team's peer-distillation thread (`four-model-peer-distillation-protocol.md`) but applied at the base-model stage.

### 8.3 More data ideas

- **In-context pretraining (document ordering). [Config-ish — needs a corpus sort]** Order documents so *related* ones are adjacent in each context window instead of random concatenation, giving real cross-document signal. Reported +8% in-context learning, +15% reading comprehension, +16% context faithfulness — with only the ordering changed and the pipeline reused. OLMo-core already threads intra-document masking, so the missing piece is a nearest-neighbor sort of the corpus.

- **Synthetic "textbook-quality" data (phi-style). [External build]** Generate high-quality synthetic explanatory text and train (partly) on it. On the direct path to the eventual educational model, and a strong complement to the augmentation-timing study (Experiment 3).

- **Best-fit packing + document-boundary masking. [Config / partly present]** Pack sequences to avoid truncating documents and mask attention across document boundaries, which measurably reduces hallucination and spurious cross-doc dependence. OLMo-core already supports intra-doc masking (`cu_doc_lens`); the packing policy is the addition.

- **Data-constrained repetition analysis. [Config + analysis]** If high-quality tokens are scarce (likely, once we filter hard), how many epochs can we repeat them before returns diminish? The scaling-law result is that up to ~4 epochs is nearly as good as fresh data — a direct input to how aggressively to reuse our best corpus.

### 8.4 More optimization / training ideas

- **Weight averaging (LAWA / EMA / model soup). [Code, small callback]** Average checkpoints along the training trajectory (or across runs) for a near-free quality and stability bump. One of the cheapest wins available and a natural OLMo-core training callback.

- **Batch-size scheduling / critical batch size. [Config — already in codebase]** Grow the batch size over training to sit near the critical batch size as it rises; OLMo-core already has a `batch_size_scheduler`. An efficiency knob to fold into Experiment 1.

- **SOAP / Shampoo (second-order). [Config for Dion; SOAP is Code]** Preconditioned/second-order optimizers as a second front alongside Muon. Dion is already registered; SOAP would be new. Screen against the Muon winner from Experiment 1.

- **μP (maximal update parameterization). [Code-ish]** Parameterize widths so the optimal learning rate is *invariant* to model size — the principled backbone that makes Experiment 8's transfer exact rather than empirical. May need init/scaling additions beyond the existing `InitMethod` options.

---

## Appendix A — Where each lever lives in OLMo-core

| Lever | Status | Config / entry point |
|---|---|---|
| GQA / MQA | Config | `AttentionConfig.n_kv_heads` |
| Sliding-window attention | Config | `SlidingWindowAttentionConfig` (per-layer pattern) |
| QK-norm | Config | `AttentionConfig.qk_norm`, `use_head_qk_norm` |
| Reordered / peri / scaled norm blocks | Config | `TransformerBlockType` (`reordered_norm`, `peri_norm`, `default_scaled`) |
| RMSNorm variants | Config | `LayerNormType` (`rms`, `qwen_rms`, `fused_rms`, `cute_rms`) |
| RoPE theta / YaRN scaling | Config | `RoPEConfig.theta`, `RoPEScalingConfig` (YaRN, PI, stepwise) |
| Tied embeddings | Config | `TransformerConfig.tie_word_embeddings` |
| z-loss | Config | `LMHeadConfig` / `z_loss_multiplier` (train module) |
| MoE (+ shared expert, aux-loss-free LB) | Config | `MoEConfig`, `MoERouterConfig.bias_gamma`; `smallmoe`, `olmoe_1B_7B` |
| **Recurrent / linear-attention mixer (SSM-family)** | Config | `SequenceMixerConfig` registry (`gated_delta_net`) — basis for the Samba-style interleave |
| Muon / NorMuon / Dion / Lion | Config | `OptimConfig` registry (`muon`, `nor_muon`, `dion`, `lion`) |
| Skip-step (spike) optimizer | Config | `skip_step_adamw`, `skip_step_lion` |
| WSD / cosine / composable schedules | Config | `Scheduler` registry (`wsd`, `cos_with_warmup`, `composable`) |
| Data mixture / ratios (for augmentation-timing) | Config | `ComposableDataLoaderConfig` + mixing sources |
| Sequence-length curriculum | Config | `VSLCurriculum*`, `SequenceLengthSchedulerCallback` |
| Init methods (incl. depth-aware) | Config | `InitMethod` (`normal`, `llama`, `llama_depth`, `fan_in`, `normalized`) |
| FP8 / low precision | Config | `TransformerTrainModuleConfig.float8_config` |
| **nGPT (normalized transformer)** | Config | `TransformerType.normalized` + `normalized` block/attn/FF/LM-head (unscreened) |
| **Batch-size scheduling** | Config | `batch_size_scheduler` callback |
| **True Mamba-2 layer** | **Code** | new `SequenceMixerConfig` implementation |
| **Hymba-style parallel hybrid-head block** | **Code** | new `TransformerBlockType` |
| **Multi-head latent attention (MLA)** | **Code** | new `AttentionType` |
| **Native sparse attention (NSA)** | **Code** | not implemented |
| **Multi-token prediction (MTP)** | **Code** | extra heads + aux loss in train module |
| **Selective LM / Rho-1 token loss mask** | **Code** | reference-model token scoring + masked CE |
| **Quiet-STaR reasoning objective** | **Code** | thought tokens + parallel rationale sampling + mixing head + RL reward |
| **Logit soft-capping** | **Code** | not implemented |

## Appendix B — Sources

Grounded in the following, plus the project's own `experiment-history.md`. (External literature current to early 2026; retrieved during drafting where possible — note that keyword web search was unavailable in this environment due to an AWS Bedrock model-access error, so a few figures are cited from prior knowledge rather than re-fetched.)

**Architecture**
- **OLMo 2** (reordered-norm, QK-norm, z-loss, Dolmino late-stage annealing) — arXiv:2501.00656.
- **DeepSeek-V3** (MLA, fine-grained + shared experts, aux-loss-free load balancing, MTP, FP8) — arXiv:2412.19437.
- **Mamba-2 / State Space Duality** (SSM↔attention duality; 2–8× faster core layer) — arXiv:2405.21060.
- **Samba** (Mamba + sliding-window-attention layer-wise hybrid; 3.73× throughput; 256K passkey recall from 4K training) — arXiv:2406.07522.
- **Hymba** (parallel attention + SSM heads; 1.5B > Llama-3.2-3B, 11.7× smaller cache, 3.5× throughput) — arXiv:2411.13676.
- **Native Sparse Attention (NSA)** (natively-trainable hierarchical sparse attention) — arXiv:2502.11089.

**Pretraining & data**
- **Muon at scale / Moonlight** (~2× compute efficiency vs AdamW; 3B/16B-active MoE on 5.7T tokens) — arXiv:2502.16982.
- **DataComp-LM (DCLM)** (model-based filtering as the dominant quality lever; 64% MMLU at 7B, ~6.6× less compute) — arXiv:2406.11794.
- **WRAP — Web Rephrase Augmented Pre-training** (~3× speedup, >10% perplexity, >2% zero-shot QA) — arXiv:2401.16380.
- **Instruction Pre-Training** (200M synthesized instruction-response pairs; Llama-3-8B rivals 70B in continual pretraining) — arXiv:2406.14491.
- **Rho-1 / Selective Language Modeling** (reference-model token scoring; DeepSeekMath parity with ~3% of tokens, up to +30% few-shot math) — arXiv:2404.07965.
- **Chinchilla** (compute-optimal tokens-per-parameter) — arXiv:2203.15556.

**Reasoning in pretraining**
- **Quiet-STaR** (latent rationales during ordinary text; zero-shot GSM8K 5.9→10.9%, CommonsenseQA 36.3→47.2%, no task fine-tuning) — arXiv:2403.09629.

**Further-menu ideas (Section 8)**
- **Byte Latent Transformer (BLT)** (tokenizer-free entropy-based byte patching; matches BPE to 8B/4T bytes, better inference scaling and robustness) — arXiv:2412.09871.
- **Coconut — Chain of Continuous Thought** (reasoning in latent space by feeding hidden state back as input; BFS-like, beats CoT on search-heavy planning) — arXiv:2412.06769.
- **Latent recurrent-depth reasoning (Huginn)** (looped block, test-time reasoning up to ~50B-equivalent compute at 3.5B params) — arXiv:2502.05171.
- **In-Context Pretraining** (order related documents adjacently; +8% ICL, +15% reading comprehension, +16% context faithfulness) — arXiv:2310.10638.
- Additional directions cited from prior knowledge (not re-fetched here): nGPT normalized transformer; fill-in-the-middle; sparse upcycling; phi-style synthetic textbooks; best-fit packing; data-constrained-scaling repetition (~4-epoch rule); weight averaging (LAWA / model soup); SOAP/Shampoo; and μP parameterization.
</content>
