# Latent-Space Reasoning: Research Directions and Experiments for OLMo-core

**A survey of the latent-reasoning frontier, with concrete experiments to build and run in OLMo-core**

> **Status:** A research-planning and literature-scan document, not a results writeup. It maps the current landscape of latent-space reasoning, proposes experiments ordered from cheapest to most ambitious, and points to where each would attach in OLMo-core. It stands on its own — the directions here are chosen on their research merits and do not depend on, or need to connect to, any of the project's prior experiments. Techniques already implementable via config are marked **[Config]**; those needing new code are **[Code]**, with **[Code, small/medium/heavy]** indicating effort.

---

## 1. Purpose and scope

**Latent-space reasoning** is the idea that a model should do its multi-step "thinking" inside its own continuous hidden states rather than by emitting natural-language chain-of-thought (CoT) tokens. The motivation is threefold: language is a narrow channel for computation ("most tokens ensure textual coherence, not reasoning"), emitting long CoT is slow and expensive at inference, and forcing reasoning into words may actually constrain what the model can represent. Reasoning in latent space promises richer intermediate representations, lower latency, and — in the depth-recurrent form — the ability to spend *more* compute on *harder* inputs without generating more text.

This document surveys the frontier, then proposes a ladder of experiments we can build in OLMo-core, from a data-only curriculum that needs no new code to an architecture that reasons by iterating a recurrent block at inference. The goal is to establish, at the 370M scale we prototype on, which latent-reasoning approach is worth carrying toward a larger model.

---

## 2. The landscape: two axes

The cleanest way to organize the field — and the framing used by the 2025 survey — is by **where the extra computation goes**.

### 2.1 Vertical (depth-wise): think *harder* per token

Spend more internal compute per token, emitting no reasoning text. The model's forward pass does more work before it commits to an output.

- **Recurrent depth / looped transformer.** Apply a recurrent block repeatedly, unrolling to arbitrary depth at test time. Reasoning happens purely in latent space, needs no special training data, and works with small context windows. The **Huginn** model (3.5B params, 800B tokens) improved reasoning *dramatically* by iterating more at inference — up to a compute-equivalent of a **~50B-parameter model** — without changing its parameter count.
- **Pause / filler tokens.** Append learnable "pause" tokens and delay reading the output until the last one, giving the model extra hidden vectors to compute on. Goyal et al. reported gains (~18% EM on SQuAD, smaller on GSM8K) at 1B scale, but *only when the tokens were used in both pretraining and finetuning*. A cautionary companion result ("Let's Think Dot by Dot") shows filler tokens help only on specific parallelizable problems and require dense supervision — so this is a "does free latent compute help *here*" probe, not a guaranteed win.

### 2.2 Horizontal (sequence-wise): think *longer*, but silently

Replace explicit CoT *text* with *latent states* that carry reasoning from step to step across the sequence.

- **Continuous thoughts (Coconut).** Instead of decoding a hidden state into a word and re-embedding it, feed the last hidden state back directly as the next input embedding — a "continuous thought." Because it is not collapsed to one token, a single thought can keep several reasoning branches alive at once, letting the model do a breadth-first search rather than committing to one path. Coconut beats explicit CoT on **search-heavy planning** tasks and gives a better accuracy/efficiency trade-off.
- **Compressed contemplation tokens (CCoT).** Generate a *variable-length* sequence of continuous "contemplation tokens" that are compressed representations of a full reasoning chain, distilled from explicit CoT. The number of tokens is a tunable dial trading accuracy for latency, and it applies to off-the-shelf decoder models.
- **Discrete latent tokens (Token Assorted).** Abstract the *early* reasoning steps into discrete latent tokens (produced by a VQ-VAE) while keeping later steps as text, mixing latent and text tokens during training. Shortens reasoning traces while preserving accuracy.
- **Internalized CoT (iCoT / stepwise internalization).** Start from a model trained on explicit CoT, then *gradually remove* the intermediate steps and keep finetuning until the model answers directly — forcing the reasoning into its internal computation. This got **GPT-2-small to 99% on 9×9 multiplication** (standard training fails past 4×4) and **Mistral-7B above 50% on GSM8K with no emitted steps**. Notably, this is a pure data/curriculum method — no architecture change.

### 2.3 The training-signal sub-axis

Cutting across both is *how* the latent behavior is learned:

- **Curriculum / distillation from explicit CoT** — Coconut's staged replacement, iCoT's step removal, CCoT's compression. Cheapest and most stable; needs CoT data to distill from.
- **Reinforcement-style reward** — **Quiet-STaR** teaches the model to emit a short internal rationale (between learnable "thought" start/end tokens) at many token positions during ordinary text, rewarded by how much the rationale improves prediction of the *actual* following text. On continued pretraining alone, no task finetuning, it lifted zero-shot **GSM8K 5.9→10.9%** and **CommonsenseQA 36.3→47.2%**. Most general and most ambitious; higher variance.
- **From-scratch mechanism in pretraining** — Huginn's recurrent pretraining, pause tokens in pretraining. Bakes the capability in but is the most expensive to iterate on.

### 2.4 Frontier developments (2025–2026)

The area has moved quickly, and several developments postdate the core papers above and reshape the priorities. (Grounded in a live literature pass; see the updated sources appendix.)

- **Superposition theory — *why* continuous thoughts help.** A NeurIPS 2025 analysis proves each continuous thought can act as a *superposition* of many discrete reasoning paths, effectively running a breadth-first search instead of committing to one branch. Formally, a **two-layer transformer using D continuous-CoT steps solves directed-graph reachability** (D = graph diameter), whereas the best known discrete-CoT constant-depth construction needs **O(n²) steps** — and the superposition behavior **emerges during training without explicit supervision**. This is the strongest theoretical case yet for latent reasoning, and it tells us *which* tasks (search/planning with high branching) should show the largest gains. Follow-on work (CoT2, "Emergence of Superposition") studies how to induce and exploit it.

- **Single-stage self-distillation (CODI).** Rather than Coconut's finicky multi-stage curriculum, CODI jointly trains an explicit-CoT "teacher" and an implicit-CoT "student" in **one stage**, aligning the hidden states of a designated distillation token across all layers. It was the first implicit-CoT method to **match explicit CoT on GSM8K at GPT-2 scale**, at ~**3.1× compression** and **2.7–5.9× speedup**, and it avoids the forgetting seen in staged methods. Public code exists. For us this is likely a *better first continuous-thought build than Coconut*.

- **RL-elicited latent reasoning (no CoT traces needed).** HRPO (NeurIPS 2025) and HyRea (ICLR 2026) use reinforcement learning to draw latent reasoning out of a model's own abilities instead of distilling it from CoT data. HRPO adds a learnable **gate** that mixes prior hidden states into sampled token embeddings and uses token sampling to supply the stochasticity RL needs — so it optimizes without CoT trajectories and stays interpretable (it even shows cross-lingual patterns and shorter completions). Related: Soft Thinking, Soft-GRPO, LePO. This removes the main data dependency of the distillation methods.

- **Latent *diffusion* reasoning (backtracking + parallelism).** LaDiR encodes reasoning steps into latent "thought-token blocks" with a VAE, then runs a **latent diffusion denoiser with blockwise bidirectional attention**, so the model can *revise earlier thoughts* (backtracking) and explore diverse trajectories in parallel — capabilities autoregressive latent reasoning structurally lacks (once an AR latent is emitted it cannot be changed). "Reasoning with Latent Tokens in Diffusion LMs" (2026) finds latent tokens especially help tasks needing global coherence/lookahead. A heavier build for genuinely new reasoning behavior.

- **Adaptive latent computation (ponder/halting).** "Learning to Ponder" and adaptive-anchor-refinement work (2025–2026) add a *learned halting* signal so the model spends more latent steps on hard inputs — validating the adaptive-depth idea and giving us a baseline (now promoted to Experiment L10).

- **The central open problem: distributional shift & interpretability.** A recurring 2026 finding is that raw, unconstrained latent states drift from the model's vocabulary distribution, which both **hurts performance** (methods "frequently suffer severe degradation vs explicit reasoning") and **removes the visible CoT** that was our main monitoring artifact. Active responses — logit-lens decoding of latents, linear/causal probes, and regularizing latents toward the vocabulary simplex — are simultaneously a *quality* lever and a *safety* lever, and the space is wide open at small scale. A 2026 position paper ("LLM Reasoning Is Latent, Not the Chain of Thought") reframes reasoning as latent-trajectory formation; a companion critique ("Observable Patterns Are Not Explanations") warns against over-reading probe results without causal tests.

---

## 3. The one rule that governs every comparison

Latent methods **trade emitted tokens for hidden compute.** A latent model that "matches CoT with fewer tokens" may simply be doing the same work in a different place. So every comparison in this document is specified at **matched inference compute (FLOPs / wall-clock latency)**, not matched token count. Reporting accuracy-vs-compute curves — not single points — is mandatory, because the entire value proposition is a better point on that curve.

Two further caveats worth stating up front:

- **Interpretability / faithfulness.** Latent reasoning is, by construction, not human-readable. We lose the (partial) transparency of CoT. Where it matters, we should *probe* the latent states — train small linear decoders to check what a continuous thought actually encodes — rather than assume the model is "reasoning" in there.
- **Scale sensitivity.** Several of these effects may only emerge above some scale. A null result at 370M is therefore informative about *scale*, not necessarily about the method — and should be written up as such.

---

## 4. Where each approach attaches in OLMo-core

OLMo-core's modularity makes most of these tractable; the table maps each to its natural insertion point.

| Approach | Tier | Where it plugs in |
|---|---|---|
| iCoT (step-removal curriculum) | [Config / data] | Composable data loader (`ComposableDataLoaderConfig`) + a scheduler callback to fade CoT tokens over training |
| Pause / filler tokens | [Code, small] | Extra learnable rows in the embedding table + sequence construction in the data collator |
| Coconut (continuous thought) | [Code, medium] | Train-module forward loop: feed last hidden state back as next input embedding for K steps before decoding |
| Quiet-STaR (RL rationales) | [Code, medium] | Thought start/end tokens + parallel rationale sampling + mixing head + REINFORCE-style loss in the train module |
| Recurrent depth (Huginn) | [Code, heavy] | New model/block variant that loops a shared recurrent block a variable number of times; adaptive halting optional |
| CCoT (compressed tokens) | [Code, medium] | Module to emit variable-length continuous tokens, distilled from explicit CoT traces |
| Token Assorted (discrete VQ) | [Code, heavy] | VQ-VAE over reasoning states + extended vocabulary of latent tokens |
| CODI (single-stage self-distillation) | [Code, medium] | Joint teacher/student loss in train module + feature-distillation on a designated token across layers |
| HRPO (RL-elicited hybrid) | [Code, heavy] | Learnable hidden-state→embedding gate + a GRPO-style policy-optimization loop (new to OLMo-core) |
| LaDiR (latent diffusion) | [Code, heavy] | VAE over thought-blocks + a latent diffusion denoiser with bidirectional block attention |
| Adaptive latent depth (ponder) | [Code, medium] | Learned halting head over recurrent depth / continuous-thought count |

The recurring theme: the *forward/loss orchestration* lives in the transformer train module and the LM head, and the *block* is a clean abstraction to subclass — so continuous-thought loops, recurrent depth, and reasoning objectives all have a natural home without touching unrelated code.

---

## 5. Experiment ladder (cheapest → most ambitious)

All experiments are at 370M unless noted, use contamination-checked reasoning benchmarks, and — per Section 3 — compare at matched inference compute with accuracy-vs-compute curves. Each names the literature result it targets so a run either clears that bar or yields an informative null.

### L1 — Internalized CoT via step-removal curriculum [Config / data — cheapest, run first]

- **Topic / hypothesis.** Reasoning can be pushed "into the model's head" by a curriculum that starts from explicit chain-of-thought and gradually deletes the written steps, so the 370M model ends up solving multi-step problems with few or no emitted steps at much lower inference cost than full CoT.
- **What we're testing.** Whether a 370M model can internalize step-by-step reasoning at all, and whether *how fast* the steps are removed matters — i.e., can it hold accuracy while emitting near-zero reasoning tokens.
- **How we're testing.** From a base (or CoT-finetuned) checkpoint, run continued training in which the fraction of retained CoT tokens decays on a schedule from 100% to ~0% — a data-loader curriculum plus a scheduler callback, no architecture change. Arms: explicit-CoT baseline, abrupt removal, gradual removal. Measure accuracy with zero/near-zero emitted steps vs. explicit CoT at matched inference compute, plus tokens emitted per solved problem. **Bars:** GPT-2-small → 99% on 9×9 multiplication; Mistral-7B → >50% GSM8K with no steps (larger/different models, so treat as directional at 370M). **Run first:** needs no new code, isolates the core question, and produces a reusable CoT-curriculum harness the later experiments reuse.
- **Confounds.** The removal *schedule* is itself a variable (see Novel Angle 6.2); the model may need explicit-CoT competence first; verify no leakage of answers into the "no-step" format.

### L2 — Pause / filler tokens in pretraining [Code, small]

- **Topic / hypothesis.** Giving the model blank "pause" tokens to compute on — placeholders that carry no words but add forward-pass steps before answering — improves reasoning/QA, and the gain requires their presence in *pretraining*, not just finetuning.
- **What we're testing.** Whether "free" per-token latent compute helps at our scale, and on which kinds of task — including the known failure mode where filler tokens do nothing.
- **How we're testing.** Add N learnable pause tokens to the embedding table, insert them before answer positions, and delay output extraction until the last one. Because we control pretraining, run the full pretrain+finetune protocol the literature found necessary, and sweep N. Baseline: an identical model with no pause tokens at matched compute. Report reasoning/QA accuracy vs. pause-token count, segmenting by whether a task is the parallelizable kind filler tokens should help (the "Dot by Dot" caveat). **Bar:** Goyal et al.'s ~18% SQuAD EM and small GSM8K gain at 1B.
- **Confounds.** Pause tokens can do nothing without the right supervision (a documented failure mode); ensure the compute they add is counted in the "matched compute" baseline.

### L3 — Coconut: continuous thought in the loop [Code, medium — flagship horizontal]

- **Topic / hypothesis.** Reasoning in the model's continuous hidden space — feeding the last hidden state back as the next input embedding for K latent steps instead of decoding it to a word — matches or beats explicit CoT, especially on search/planning, at lower inference compute.
- **What we're testing.** Whether continuous thoughts trained by Coconut's staged curriculum (progressively replacing written reasoning steps with continuous ones) reason better per unit of compute, and whether their non-committal nature yields a breadth-first-search advantage on branching problems.
- **How we're testing.** Implement the continuous-thought loop in the train module plus Coconut's staged curriculum. Arms: no-CoT baseline, explicit-CoT, Coconut. Evaluate on GSM8K **and** a genuinely search-heavy planning task (ProntoQA / ProsQA-style), reporting accuracy-vs-inference-compute curves, and run a probing analysis (linear decoders on the continuous thoughts) to check what they encode. **Bar:** Coconut > explicit CoT on search-heavy planning with a better accuracy/efficiency trade.
- **Confounds.** Curriculum stability (staged replacement is finicky — exactly what L6/CODI aims to fix); matched-compute accounting for the K latent steps; planning tasks must be genuinely search-heavy or the advantage won't show.

### L4 — Recurrent-depth latent reasoning [Code, heavy — flagship vertical]

- **Topic / hypothesis.** A model that loops a shared recurrent block a *variable* number of times can "think longer" at inference — improving reasoning as iterations increase — without adding parameters, reproducing the direction of Huginn's ~50B-equivalent gains at a few-B scale.
- **What we're testing.** Whether spending more latent compute (more iterations) at test time raises reasoning accuracy, where that saturates, and whether a small looped model can punch above its parameter count.
- **How we're testing.** Build a recurrent-depth variant (a core block iterated R times, with a stable recurrence and input injection each step); pretrain a small version; at inference, sweep R and plot the reasoning-vs-R curve. Compare against a same-parameter, fixed-depth baseline at matched *inference* compute (equal FLOPs, not equal layers), and report accuracy-per-parameter vs. a deeper fixed model. Huginn's open code + recipe make this the most buildable of the ambitious options. **Bar:** monotone reasoning gains with R, approaching a much larger fixed model's performance.
- **Confounds.** Recurrence stability at depth; fair matched-compute baseline (equal FLOPs, not equal layers); ensuring gains come from iteration, not from the extra parameters in the recurrence machinery.

### L5 — Discrete latent reasoning tokens (Token Assorted / VQ) [Code, heavy — stretch]

- **Topic / hypothesis.** Abstracting the *early* reasoning steps into discrete VQ latent tokens (keeping later steps as text) shortens reasoning traces at iso-accuracy, and discrete latents may be more stable and interpretable than continuous ones.
- **What we're testing.** Whether discrete "symbol" latents compress the reasoning trace without losing accuracy, and how they compare to the continuous approach (L3/L6) on stability and interpretability.
- **How we're testing.** Train a VQ-VAE over reasoning states, extend the vocabulary with the resulting latent tokens, and train on mixed latent/text traces. Compare trace length and accuracy against explicit CoT and against the continuous variant, reporting trace-length reduction at matched accuracy and accuracy-vs-compute. Run last, informed by whether continuous latents worked at our scale.
- **Confounds.** VQ codebook collapse; the added complexity of a two-stage (VQ → LM) pipeline; matching the comparison to L3/L6 fairly.

> **Experiments L6–L10 below are additions from the 2025–2026 frontier pass. They do not replace L1–L5; several are stronger or cheaper variants that should be run alongside them (e.g. L6 is likely a better first continuous-thought build than L3).**

### L6 — CODI: single-stage self-distillation of CoT into continuous space [Code, medium — likely the best first continuous build]

- **Topic / hypothesis.** Compressing explicit CoT into continuous thoughts via *single-stage* self-distillation is more stable than Coconut's multi-stage curriculum and matches explicit CoT at our scale.
- **What we're testing.** Whether a jointly-trained explicit "teacher" + implicit "student," aligned on a distillation token's hidden states, gives Coconut-level (or better) latent reasoning without the staged-curriculum forgetting.
- **How we're testing.** Implement CODI's joint objective — teacher CE on explicit CoT + student CE on continuous-thought reasoning + a feature-level distillation loss aligning the designated token's activations across all layers — and run it head-to-head against L3 (Coconut) and an explicit-CoT baseline at 370M, at matched inference compute. Public reference code de-risks the build. **Bars:** GSM8K parity with explicit CoT at GPT-2 scale, ~3.1× compression, 2.7–5.9× speedup, and less forgetting than L3.
- **Confounds.** Distillation-token placement and loss-weight tuning; fair compression accounting (same effective compute); making the L3 comparison apples-to-apples.

### L7 — RL-elicited hybrid latent reasoning (HRPO-style) [Code, heavy — no CoT data needed]

- **Topic / hypothesis.** Reinforcement learning can *elicit* latent reasoning from the model's own abilities — no CoT distillation corpus — while keeping it interpretable.
- **What we're testing.** Whether a learnable gate that blends previous hidden states into sampled token embeddings, optimized by a policy-gradient objective on a verifiable reward, matches or beats distillation-based latent reasoning (L3/L6).
- **How we're testing.** Add the gate (initialized to mostly token-embeddings, annealed toward more hidden-state content) and train with a GRPO-style objective on verifiable-reward tasks (math/logic with checkable answers). Formulate the hybrid discrete/continuous action space by defining the policy density only at the text-emitting positions (latent positions are deterministic given prior text and just replayed on the gradient pass) — a recent result that makes the gradients tractable. **Note:** OLMo-core has no first-class RL loop today, so this needs a new policy-optimization train module — the heaviest build here, but it is where the frontier is heading. **Bars:** HRPO's gains over prior latent methods with shorter completions.
- **Confounds.** RL instability; reward design; ensuring the interpretability claim holds (probe the gated states).

### L8 — Superposition: measure it, then encourage it [Code, small–medium — cheap analysis, high insight]

- **Topic / hypothesis.** Continuous thoughts win by holding a *superposition* of reasoning paths; we can verify this and improve results by explicitly encouraging it while fighting distributional shift.
- **What we're testing.** (a) Do our continuous thoughts (from L3/L6) actually encode multiple reasoning frontiers at once? (b) Does encouraging superposition and constraining latents toward the vocabulary distribution improve both accuracy and interpretability?
- **How we're testing.** Use **directed-graph reachability** — the exact setting of the superposition theory, where a two-layer transformer with D continuous steps should beat discrete CoT — as the probe task. Then: (1) apply logit-lens + linear *and causal* probes to test multi-frontier encoding; (2) try a CoT2-style variant that composes K discrete tokens per step to dial parallelism; (3) add a regularizer pulling latents toward the vocab simplex and measure the effect on accuracy and on logit-lens readability. **Bars:** solve-rate advantage growing with graph diameter/branching factor, as theory predicts.
- **Confounds.** Probes can find non-causal structure (hence causal interventions, per the 2026 "Observable Patterns Are Not Explanations" critique); regularizer-strength trade-off.

### L9 — Latent diffusion reasoning (LaDiR-style) [Code, heavy — stretch, genuinely new behavior]

- **Topic / hypothesis.** Reasoning by *denoising* latent thought-blocks (instead of emitting them left-to-right) lets the model backtrack/revise earlier thoughts and explore diverse trajectories in parallel — things autoregressive latent reasoning cannot do.
- **What we're testing.** Whether self-correcting, parallel latent-diffusion reasoning beats AR CoT and AR latent baselines on tasks needing global coherence and lookahead (math, code, puzzle-planning).
- **How we're testing.** VAE-encode reasoning steps into latent thought-token blocks; train a latent diffusion denoiser with blockwise bidirectional attention; at inference use diversity guidance for multiple trajectories and adaptive test-time compute for iterative refinement. Run last, informed by L3/L6/L8.
- **Confounds.** Two-stage (VAE + diffusion) complexity; fair compute-matching to AR baselines; diffusion training stability at small scale.

### L10 — Adaptive latent depth (learned halting / ponder) [Code, medium — promotes Novel Angle 6.1]

- **Topic / hypothesis.** Letting the model decide *how long to think* in latent space — more steps on hard inputs, fewer on easy — beats a fixed latent budget on accuracy-per-compute.
- **What we're testing.** Whether a learned halting policy over recurrent-depth iterations (L4) or continuous-thought count (L3/L6) dominates the best fixed-budget setting at matched *average* compute.
- **How we're testing.** Add a PonderNet/ACT-style halting head driven by a difficulty signal (next-token entropy or recurrence convergence); compare adaptive vs. best fixed budget, plotting accuracy vs. *average* compute. Recent "Learning to Ponder" work supplies motivation and a baseline. **Bars:** the adaptive curve should dominate the fixed-budget curve.
- **Confounds.** Halting-policy training stability; matched-*average*-compute accounting (not peak); difficulty-signal calibration.

---

## 6. Novel angles worth claiming

These are less-explored combinations that would be genuinely new contributions rather than reproductions.

### 6.1 Adaptive latent depth (learned halting) — *now promoted to Experiment L10*

Combine recurrent depth (L4) with a **learned halting** signal (in the spirit of PonderNet / adaptive computation time) so the model spends more latent iterations on *hard* tokens and fewer on easy ones, allocating compute by difficulty. A clean efficiency story — "compute where it's needed." The 2025–2026 literature ("Learning to Ponder," adaptive-anchor-refinement) has begun validating this direction, so it graduates from a speculative angle to full Experiment L10; the open contribution is doing it *at small scale* with a principled difficulty signal (next-token entropy or recurrence convergence) and honest matched-*average*-compute accounting.

### 6.2 The *schedule* of internalization

In L1, the rate at which CoT tokens are removed is a free design choice nobody has studied carefully. Does gradual removal beat abrupt? Is there an optimal fade curve? Does re-exposing removed steps intermittently (rather than monotone removal) help the model consolidate? This turns "internalize the reasoning" into a controlled study of *how* to fade a reasoning scaffold — a small, self-contained, and publishable question.

### 6.3 Latent-thought-length curriculum

For continuous thoughts (L3) or contemplation tokens (CCoT), grow the *number* of latent steps over training as a difficulty curriculum — start with 1–2 thoughts, expand as the model stabilizes — and test whether this beats a fixed budget. Analogous to a from-easy-to-hard schedule but over the model's *internal* reasoning length.

### 6.4 Latent reasoning × capacity (MoE)

A speculative but interesting pairing: route *latent reasoning steps* through a mixture-of-experts so different "thoughts" can invoke different experts. Whether latent reasoning benefits from conditional capacity per step is, as far as I found, open.

---

## 7. Recommended reading (ranked)

1. **Coconut — "Training Large Language Models to Reason in a Continuous Latent Space"** (Hao et al., arXiv:2412.06769). The canonical continuous-latent-reasoning paper; the place to start.
2. **"Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"** (Geiping et al., arXiv:2502.05171). The strongest vertical/recurrent-depth work, with open code and data recipe — read this if you want to build (L4).
3. **Quiet-STaR — "Language Models Can Teach Themselves to Think Before Speaking"** (Zelikman et al., arXiv:2403.09629). Reasoning-in-pretraining via an RL-style reward; foundational and the most general training signal.
4. **"From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step"** (Deng et al., arXiv:2405.14838). The cheapest to reproduce (L1) and the clearest curriculum-based method.
5. **"A Survey on Latent Reasoning"** (arXiv:2507.06203) and its living GitHub bibliography **LatentCoT-Horizon**. The field map plus a curated, continuously-updated reading feed — the best way to track what's new.

Secondary, by theme:
- **Compression:** "Compressed Chain of Thought" (CCoT, arXiv:2412.13171).
- **Discrete latents:** "Token Assorted: Mixing Latent and Text Tokens" (arXiv:2502.03275).
- **Simplest entry point / cautionary pair:** "Think Before You Speak: Pause Tokens" (Goyal et al., arXiv:2310.02226) and "Let's Think Dot by Dot" (filler-token limits).

Frontier additions (2025–2026), highly recommended:
6. **CODI — "Compressing Chain-of-Thought into Continuous Space via Self-Distillation"** (Shen et al., EMNLP 2025, arXiv:2502.21074). The single-stage self-distillation method behind Experiment L6; read right after Coconut. Code: github.com/zhenyi4/codi.
7. **"Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought"** (Zhu et al., NeurIPS 2025, arXiv:2505.12514). The theory that explains *why* continuous CoT works and *which* tasks it helps; grounds Experiment L8.
8. **HRPO — "Hybrid Latent Reasoning via Reinforcement Learning"** (NeurIPS 2025, arXiv:2505.18454). RL-elicited latent reasoning with no CoT data; behind Experiment L7. Code: github.com/Yueeeeeeee/HRPO.
9. **LaDiR — "Latent Diffusion Enhances LLMs for Text Reasoning"** (Kang et al., arXiv:2510.04573). Latent-diffusion reasoning with backtracking/parallelism; behind Experiment L9.
10. **"LLM Reasoning Is Latent, Not the Chain of Thought"** (2026 position paper, arXiv:2604.15726) and **"Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention"** (arXiv:2606.01243). The interpretability / distributional-shift frontier that Experiment L8 engages with.

---

## 8. Suggested order of attack

1. **L1 (internalized CoT)** first — no new code, builds the CoT-curriculum harness, answers "can a 370M internalize steps at all?"
2. **L2 (pause tokens)** in parallel — small code, tests whether free latent compute helps here, with the known failure mode as an explicit check.
3. **L6 (CODI)** as the *preferred* first continuous-thought build — single-stage, stable, public code — with **L3 (Coconut)** run as the comparison rather than the lead. This reflects the 2025 finding that CODI's self-distillation avoids Coconut's staged-curriculum fragility.
4. **L8 (superposition analysis)** early and cheap — it is mostly probing on top of L3/L6, and it tells us whether the mechanism is even present at 370M before we invest further.
5. **L4 (recurrent depth)** as the flagship vertical build — the biggest architectural payoff, using Huginn's open recipe — with **L10 (adaptive halting)** layered on once it works.
6. **Heavy frontier bets, once earned:** **L7 (RL-elicited, HRPO)** if we want to drop the CoT-data dependency (needs a new RL loop in OLMo-core), and **L9 (latent diffusion, LaDiR)** for backtracking/parallel reasoning. **L5 (discrete VQ)** as a stretch informed by L3/L6.

The discipline throughout: compare at matched inference compute, report accuracy-vs-compute curves not points, probe the latent states (with *causal* interventions, not just linear probes) rather than assuming they reason, watch for distributional shift of the latents away from the vocabulary space, and treat a null at 370M as a scale finding worth writing down.

---

## Appendix — Sources

Grounded in the following papers. The L1–L5 core (below) was assembled during an initial targeted pass; the L6–L10 frontier additions came from a later **live keyword-search pass (2025–2026)** once web search was available. Figures are from abstracts/landing pages plus prior knowledge; the LatentCoT-Horizon bibliography is the recommended way to catch anything missed.

**Continuous / horizontal**
- Coconut — arXiv:2412.06769 — https://arxiv.org/abs/2412.06769
- Compressed Chain of Thought (CCoT) — arXiv:2412.13171 — https://arxiv.org/abs/2412.13171
- Token Assorted (discrete VQ latent tokens) — arXiv:2502.03275 — https://arxiv.org/abs/2502.03275

**Vertical / depth**
- Recurrent-depth latent reasoning (Huginn) — arXiv:2502.05171 — https://arxiv.org/abs/2502.05171
- Pause / filler tokens — arXiv:2310.02226 — https://arxiv.org/abs/2310.02226
- "Let's Think Dot by Dot" (filler-token limits; cited from prior knowledge, not fetched this pass — verify on open) — arXiv:2404.15758 — https://arxiv.org/abs/2404.15758

**Training signal / curriculum**
- Quiet-STaR — arXiv:2403.09629 — https://arxiv.org/abs/2403.09629
- Stepwise internalization of CoT (iCoT) — arXiv:2405.14838 — https://arxiv.org/abs/2405.14838

**Field map**
- A Survey on Latent Reasoning — arXiv:2507.06203 — https://arxiv.org/abs/2507.06203
- Living bibliography: **LatentCoT-Horizon** (GitHub) — reach it via the link on the survey's abstract page above rather than a guessed URL.

**Frontier additions (2025–2026 live pass)**
- CODI (single-stage self-distillation) — arXiv:2502.21074 — https://arxiv.org/abs/2502.21074 — code: https://github.com/zhenyi4/codi
- Reasoning by Superposition (theory) — arXiv:2505.12514 — https://arxiv.org/abs/2505.12514
- CoT2 (compose K tokens; superposition in practice) — see the "Emergence of Superposition" line — https://arxiv.org/abs/2509.23365
- HRPO — Hybrid Latent Reasoning via RL — arXiv:2505.18454 — https://arxiv.org/abs/2505.18454 — code: https://github.com/Yueeeeeeee/HRPO
- HyRea — Learning to Reason over Continuous Tokens with RL (ICLR 2026) — https://openreview.net/forum?id=lebJ6wz1vj
- LaDiR — Latent Diffusion Enhances LLMs for Text Reasoning — arXiv:2510.04573 — https://arxiv.org/abs/2510.04573
- Reasoning with Latent Tokens in Diffusion LMs — arXiv:2602.03769 — https://arxiv.org/abs/2602.03769
- Learning to Ponder — Adaptive Reasoning in Latent Space — arXiv:2509.24238 — https://arxiv.org/abs/2509.24238
- Unlocking the Black Box of Latent Reasoning (interpretability-guided intervention) — arXiv:2606.01243 — https://arxiv.org/abs/2606.01243
- Observable Patterns Are Not Explanations (causal-geometric critique) — arXiv:2606.12689 — https://arxiv.org/abs/2606.12689
- LLM Reasoning Is Latent, Not the Chain of Thought (2026 position paper) — arXiv:2604.15726 — https://arxiv.org/abs/2604.15726

> Note: several 2026 arXiv IDs above were returned by live search results; open via the URL to confirm the exact version, as very recent identifiers occasionally shift.
</content>
