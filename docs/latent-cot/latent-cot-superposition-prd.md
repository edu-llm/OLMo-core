# Continuous-Thought Reasoning on OLMo-370M: A Superposition & Distributional-Shift PRD

**Does a continuous chain-of-thought, built stably by self-distillation, reason by superposition where theory says it must — and does constraining latents toward vocabulary space fix the distributional-shift that degrades latent reasoning?**

P4 · Latent Reasoning · Execution PRD v1 (OLMo-370M-locked)

> **Status:** Pre-registered experimental design and execution protocol. It fixes the model, the tasks, the arms, the matched budgets, the metrics, the analysis plan, and the confound controls *before* any outcome is examined. These design docs live in `docs/latent-cot/` (tracked); generated data and run/eval outputs are gitignored under `data/` and `runs/` (no weights/datasets/outputs committed); the **code** is committed and pushed to the feature branch `latent-cot-superposition-amy` only (never `main`). See "Status & changes" below for what's actually built.

---

## Status & changes since this design was written (updated 2026-08-05)

**Build status.** Phases 1–8 are implemented, unit-tested (**129 tests**, CPU), style-clean, and pushed to branch **`latent-cot-superposition-amy`**. The only step not done is the **370M training on GPU** (this environment is CPU-only) — the driver, eval, benchmark, and runbook are ready. The per-phase `✅ DONE` notes in §8 are the source of truth for what's built; see also `progress.md` (changelog), `handoff.md` (agent brief), and `phase8-runbook.md` (the GPU procedure).

**Notable changes from this design as originally written** (annotated in place; the design text below is preserved):
- **Platform dataset shape (new; not in the original design).** The dataset is published to the eduLLM platform as **`sft/graph-reachability-depth`** via **`sft-conversations/v1`** — the only fitting *registered* profile (the natural `eval-items/v1` is unregistered in `edullm-data` v0.2.0). Each row carries a `messages[]` array (user = query+edges, assistant = BFS reasoning + yes/no) plus the full `Example` as metadata. Added per the `edullm-dataset-design` / `edullm-datasets` skills. See §3 and `src/scripts/latentcot/publish_dataset.py`.
- **Optimize `.loss`, not `.ce_loss`.** `LMOutputWithLoss.ce_loss` is detached (logging only); `codi_loss` optimizes `.loss`. A grad-flow regression test guards this.
- **Four control tokens, not three.** Added `<thought>` as the student latent-slot placeholder (design named `<bot>/<eot>/<distill>`); all four live at unused padded ids 100348–100351.
- **Per-example student (batch dim 1).** Sidesteps the variable-length-prefix problem without left-padding/attention masks; batched/bucketed is a deferred optimization.
- **`arm_mode` added to the confound whitelist** — A0/A1 are structurally different objectives, so the whitelist is `(arm_mode, num_continuous_thoughts, vocab_reg, vocab_reg_weight)`. *(Later extended with `vocab_reg_entropy_floor` — see the 2026-08-04 bullet below.)*
- **Direct training loop, not the framework `Trainer`** (Phase 8): the per-example student doesn't fit the token-array `DataLoader`; the loop reuses the same `arm_loss` the `CodiTransformerTrainModule` uses.
- **Secondary tasks (ProsQA / GSM8K) not implemented.** Only the synthetic directed-graph task was built; the compute-parity reproduction on ProsQA/GSM8K is deferred (§2 secondary objective).
- **Matched starts via shared init**, not necessarily a checkpoint file: all arms build from the same `--init-seed` (identical weights) or the same `--init-checkpoint`.
- **Best-model S3 init (design pivot; updated 2026-08-04).** The confirmatory sweep now forks a *real* general-pretrained base — the "best model" `s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/` (W&B run `f08ey8cm`) — via `train_codi.py --rung olmo3_370M --init-checkpoint s3://…`, rather than a from-scratch seeded init. This *refines* the §3/§5/§7 "all arms fork the same base checkpoint" requirement (still satisfied — the base is shared and identical across arms, so the confound control holds) but shifts **A0's role** from "matched-random-init upper anchor" to **the best model fine-tuned the normal way = the fair baseline**; the latent arms are measured against that same fine-tuned start, isolating the *training method*. A from-scratch `olmo2_370M` run (drop `--init-checkpoint`) remains available as a no-creds sanity check.
- **Rung is `olmo3_370M`, not `olmo2_370M` (updated 2026-08-04).** The best model was pretrained as olmo3 (olmo2 + sliding-window + flash_2). The two factories' state dicts are **identical** (same keys/shapes, 474M params) and numerically equivalent for our sequence lengths (≪ the 4096 sliding window), so the §3 model spec and the §3.1 escalation ladder (written as `olmo2_*`) carry over unchanged — the escalation rungs would likewise use their `olmo3_*` counterparts.
- **Entropy floor: implemented, default OFF, switchable (updated 2026-08-04).** R1's optional anti-collapse entropy floor (§4.2, §7 "superposition = collapse", §10 open question) is wired through but defaults to `0.0` so the **first sweep is the clean one-variable A3-vs-A4 comparison**. It is now a per-arm field switchable at runtime (`train_codi.py --vocab-reg-entropy-floor <nats>`); the empirical protocol (see `phase8-runbook.md` §4) is to inspect A3 decodability/probes for collapse first and only then re-run A3 with the floor on (noting the added-term caveat vs the L2 control). This resolves the §10 "R1's exact form needs a pilot" open question in favor of *empirical-first*, keeping both forms pre-registered.
- **Confound whitelist extended (pre-registration edit; updated 2026-08-04).** `vocab_reg_entropy_floor` was added to `ARM_WHITELIST` (now `(arm_mode, num_continuous_thoughts, vocab_reg, vocab_reg_weight, vocab_reg_entropy_floor)`), so A3 may differ from A4 in the floor without tripping `assert_arms_differ_only_in`. With the floor off (the default), this is a no-op and A3/A4 still differ only in `vocab_reg`.
- **Benchmark + pre-flight scripts (new; updated 2026-08-04).** `compare_models.py` tabulates our arms' solve-rate-by-depth against the best-model baseline (A0) with the per-depth advantage + slope (the §6.2 head-to-head); `verify_checkpoint.py` strict-loads the S3 base into `olmo3_370M` and exercises one plain + one continuous-thought forward/backward (run first on the GPU box). Eval/compare/train all auto-detect the GPU (`--device auto`).
- **All arms fine-tune the best model; WSD warmup now in the driver (updated 2026-08-05).** Following the S3-init pivot, *every* arm (A0–A4) is a fine-tune of the shared pretrained best model — none train from scratch. The Phase-8 direct loop (`train_arm`) previously used a **constant LR** (a from-scratch-era default), which contradicted the §7/§7.1 pre-registered **WSD** schedule. It now follows that same WSD (linear warmup `--warmup-steps 200` default + 10% linear decay tail), applied by hand in the shared loop so it is byte-identical across arms and stays confound-clean. **Why it helps:** on a *good* pretrained base, a full-LR first step can spike the loss and destroy the pretraining we forked for; warmup eases the optimizer in, and this reconciles the code with the pre-registered design (it is a fidelity fix, not a design change). `preflight.py` now checks this same `WSD(warmup=200, decay_fraction=0.1)`.
- **LR screen surfaced into the runbook (updated 2026-08-05).** §3.1 rule 2 / §7.1 already pre-register that only the LR-schedule *shape* is fixed and the peak LR is screened per rung; the runbook now makes this explicit — screen peak LR on 1 seed of A0/A2 over `{1e-5, 2e-5, 5e-5, 3e-4}` and fix the winner for all arms. **Why it helps:** fine-tuning wants a smaller peak LR than a from-scratch run; screening avoids silently over-writing the base at 3e-4. `--lr`/`--warmup-steps` are recorded in each `metrics.json`.
- **K bumped 8 → 10 for headroom (pre-registration edit; updated 2026-08-05).** `DEFAULT_K` (and all script/runbook defaults) went from 8 to 10; the data is unchanged (deepest graph still depth **8**; K is applied at load via `encode_example(ex, K)`, so no regen). **Why it helps:** superposition theory needs K ≥ D, and K=8 gave the deepest graph *zero* optimization slack — a depth-8 failure would be ambiguous between "can't superpose" and "one step short after imperfect training." K=10 leaves ~2 steps of headroom so a depth-8 failure is attributable to capability, while the depth-vs-advantage *slope* (the actual superposition signature) is unaffected. K remains a frozen, whitelisted, arm-invariant value (A0/A1 ignore it).
- **Best-model comparison is vs A0, never the raw checkpoint zero-shot (clarification; updated 2026-08-05).** The raw best model never saw the graph format, so evaluating it cold (≈ chance) would inflate our arms unfairly. `compare_models.py` therefore compares the latent arms against **A0 = the best model given identical fine-tuning the conventional way** (same weights/data/steps/schedule/seeds, explicit CoT). The only thing that varies is the *reasoning method*, so a positive, depth-increasing advantage is attributable to latent superposition — not to pretraining, data, or compute. (An optional cold-start floor line can be added if a reference point is wanted; it is not the comparator.)

---

## 0. Why this PRD (which experiment this is, and why it was chosen)

From the ten latent-reasoning directions in `latent-reasoning-directions.md`, this PRD builds **L6 (CODI) as the substrate** and **L8 (superposition + distributional-shift regularization) as the science**. The selection is deliberate and defended on three axes:

- **Novelty.** L8 is the only direction that attacks the field's *stated central open problem* — that raw, unconstrained latent states drift from the model's vocabulary distribution, which both hurts accuracy and destroys interpretability ([survey, arXiv:2507.06203](https://arxiv.org/abs/2507.06203)) — with a concrete, testable intervention (pull the continuous thoughts toward the vocabulary manifold). Everything else on the list is a faithful reproduction of a published method.
- **Promise.** CODI is the most likely substrate to actually work at our scale: single-stage self-distillation that already matched explicit CoT on GSM8K at GPT-2 scale, with public code ([Shen et al., EMNLP 2025, arXiv:2502.21074](https://arxiv.org/abs/2502.21074)). OLMo-370M is squarely in that scale band.
- **Rigor / low confounds.** The superposition theory gives a *provable* separation on directed-graph reachability — a two-layer transformer needs only D continuous steps (D = graph diameter) versus O(n²) discrete-CoT steps ([Zhu et al., NeurIPS 2025, arXiv:2505.12514](https://arxiv.org/abs/2505.12514)). That hands us a task where the predicted effect has a known *shape* (advantage grows with diameter), which is far harder to fake with a confound than a single benchmark number.

The coolest single concept on the list is arguably L4 (recurrent-depth test-time scaling), but it requires recurrent pretraining from scratch and is much harder to evaluate confound-free at 370M; it is deferred. This PRD is re-targetable to L4 if priorities change.

---

## 1. Literature review

**Continuous chain-of-thought.** Standard chain-of-thought (CoT) reasons in language, emitting one discrete token per step ([Wei et al., 2022, arXiv:2201.11903](https://arxiv.org/abs/2201.11903)). Coconut replaces the emitted token with a *continuous thought* — the last hidden state fed back as the next input embedding, never decoded to a word — and trains it with a multi-stage curriculum that progressively swaps language steps for continuous ones. It outperforms discrete CoT on planning/search with high branching, at lower inference cost ([Hao et al., 2024, arXiv:2412.06769](https://arxiv.org/abs/2412.06769)). The information passed between steps is a dense vector rather than a one-token bottleneck.

**Why continuous CoT helps: superposition.** A continuous thought can be a weighted blend of many candidate next-states — a *superposition* — rather than a single sampled token, so it can carry several search frontiers at once and expand them in parallel (a breadth-first search) instead of committing to one path. The formal result: a **two-layer transformer using D continuous-CoT steps solves directed-graph reachability**, where D is the graph diameter, whereas the best-known discrete-CoT constant-depth construction needs **O(n²) decoding steps**; and this superposition behavior **emerges during training without explicit supervision** ([Zhu et al., NeurIPS 2025, arXiv:2505.12514](https://arxiv.org/abs/2505.12514)). Follow-up work studies how to induce/exploit it ([CoT2 / "Emergence of Superposition," arXiv:2509.23365](https://arxiv.org/abs/2509.23365)).

**Stable training by self-distillation (why CODI, not Coconut's curriculum).** Coconut's multi-stage curriculum is fragile and prone to forgetting. CODI trains an explicit-CoT *teacher* and an implicit continuous-thought *student* **jointly in a single stage**, aligning the hidden states of a designated "distillation token" across all layers; it was the first implicit-CoT method to **match explicit CoT on GSM8K at GPT-2 scale**, at ~3.1× compression and 2.7–5.9× speedup, avoiding the staged-curriculum forgetting ([Shen et al., EMNLP 2025, arXiv:2502.21074](https://arxiv.org/abs/2502.21074); code: [github.com/zhenyi4/codi](https://github.com/zhenyi4/codi)). At our exact scale band, CODI is the appropriate, validated choice of substrate.

**The central open problem: distributional shift.** Recent work repeatedly finds that unconstrained continuous thoughts drift away from the model's vocabulary-embedding distribution, which both degrades performance ("methods frequently suffer severe degradation vs explicit reasoning") and removes the readable CoT trace we would use for oversight ([survey, arXiv:2507.06203](https://arxiv.org/abs/2507.06203); interpretability-guided intervention, [arXiv:2606.01243](https://arxiv.org/abs/2606.01243)). This motivates the novel intervention here: **regularize continuous thoughts toward the vocabulary manifold** and test whether it recovers both accuracy and decodability.

**Interpretability caveats.** Latent reasoning is not human-readable, so its claims must be probed. The standard toolkit is logit-lens (read a hidden state through the LM head to a vocab distribution), linear probing, and *causal* activation interventions. A 2026 critique warns that observable probe structure is not proof of use without causal tests ([Observable Patterns Are Not Explanations, arXiv:2606.12689](https://arxiv.org/abs/2606.12689)); a position paper reframes reasoning as latent-trajectory formation ([LLM Reasoning Is Latent, Not the Chain of Thought, arXiv:2604.15726](https://arxiv.org/abs/2604.15726)). We therefore require causal, not merely correlational, probe evidence.

**Task substrate.** Coconut's own evaluation used **ProsQA**, a graph-reachability-style reasoning dataset built to require planning/search ([Hao et al., 2024, arXiv:2412.06769](https://arxiv.org/abs/2412.06769)); ProntoQA is a related ontology-reasoning set. Directed-graph reachability is exactly the setting of the superposition theorem, and its difficulty is controllable (vary the number of nodes, branching factor, and diameter), which is what lets us measure the *shape* of the predicted effect rather than one point.

**Synthesis for this experiment.** (i) Build the continuous-thought reasoner with **CODI** (stable, scale-appropriate) rather than Coconut's curriculum, on the OLMo-370M we already have. (ii) Evaluate primarily on **synthetic directed-graph reachability** with controllable diameter, because the superposition theory predicts a *specific dependence on diameter* that a confound would not reproduce. (iii) Test the **novel distributional-shift fix** (vocab-manifold regularization) against unconstrained CODI, with a matched-strength non-vocab regularizer as the control that isolates the *direction* of the constraint from the mere fact of regularizing. (iv) Require causal probe evidence for any superposition claim.

---

## 2. Objectives

**Primary claim A (superposition, as predicted by theory).** On directed-graph reachability, the accuracy advantage of continuous-thought reasoning over explicit discrete CoT **grows with graph diameter D**, and causal probes show the continuous thoughts encode and *use* multiple reachable-frontier candidates simultaneously.

**Primary claim B (distributional-shift fix, the novel contribution).** Regularizing continuous thoughts toward the vocabulary manifold **improves reasoning accuracy and latent decodability** over unconstrained CODI, and the gain is attributable to the *vocabulary-space direction* of the constraint (it beats a matched-strength generic L2 regularizer), **without erasing** the inference-compute advantage of latent reasoning.

**Secondary objectives.** (i) Reproduce, at OLMo-370M scale, CODI's core result that continuous thoughts **match explicit CoT at lower inference compute** (a compute-parity curve, not a single point). (ii) Produce a reusable OLMo-core continuous-thought training/eval harness and a labeled diagnostic set of graph problems by diameter.

**Explicit non-goals.** Not a claim about large-scale models (370M only; scale sensitivity is expected and reported). Not a claim that latent reasoning beats explicit CoT in general — only where theory predicts (branching search). Not a general reasoning-benchmark SOTA attempt. No RL (that is L7). No architecture change to attention/MoE. No deployment.

---

## 3. Model and inputs

- **Model:** `olmo2_370M` from OLMo-core (`d_model=1024`, `n_layers=16`, `n_heads=16`, `hidden_size_multiplier=1.5`, RMSNorm eps 1e-6, RoPE θ=500k, reordered-norm + QK-norm). This is the model we already have; all arms start from **matched initial weights** — *(as built)* the same `--init-seed` (identical deterministic init = the shared "base") or the same `--init-checkpoint`.
- **Primary task — synthetic directed-graph reachability.** Programmatically generated graphs with controllable node count *n*, branching factor *b*, and diameter *D*; question = "is node T reachable from node S?" plus the BFS-frontier trace for CoT supervision. Held-out test graphs use disjoint seeds and unseen (n, b, D) combinations (OOD depths 5, 8) to test generalization, not memorization. Contamination-free by construction. *(As built: reachable/unreachable are balanced and expand to matched frontier depth, so the label can't be read off frontier depth — a confound fix added in Phase 1; see `data/graph_gen.py`.)*
- **Published shape (eduLLM platform), as built.** Emitted by `src/scripts/latentcot/gen_graph_data.py` as the **`sft-conversations/v1`** layout — one group dir `conversations/` with `train-00000.jsonl` + `heldout-00000.jsonl`; each row = the full `Example` (edges/source/target/reachable/depth/frontiers/path/seed) **plus** a `messages[]` array (user = query + edges, assistant = BFS reasoning + yes/no). Dataset id **`sft/graph-reachability-depth`**; publish with `publish_dataset.py`. Our loader reads these shards directly (`Example.from_dict` ignores `messages`).
- **CoT supervision source.** For every training problem we generate the explicit reasoning trace (the BFS frontier expansion) so CODI's teacher branch has ground-truth CoT to self-distill from.
- **Secondary tasks — ProsQA / GSM8K: DEFERRED (not implemented).** The compute-parity reproduction (§2 secondary objective) on ProsQA (Coconut's graph-reasoning set) and GSM8K was planned but not built; only the synthetic directed-graph task exists.
- **Compute:** single-node GPU (1–8 accelerators). This is continued-training / finetuning scale, not pretraining scale.

### 3.1 Parameter-scaling ladder and escalation protocol

370M is the *primary* rung, but the PRD is explicitly designed so the model size can be raised if 370M proves below the scale at which superposition emerges (a real risk — see §10). Model size is changed by **one line** — the `TransformerConfig` factory call in the training script (`MODEL_CONFIG = TransformerConfig.olmo2_760M(vocab_size=...)`) or a CLI override — with everything else held fixed. All factories are in `src/olmo_core/nn/transformer/config.py`; verified dims:

| Rung | Factory (`config.py`) | d_model | n_layers | n_heads | Role |
|---|---|---|---|---|---|
| 190M | `olmo2_190M` (:601) | 768 | 12 | 12 | optional — not used by default (we start at 370M) |
| **370M** | `olmo2_370M` (:616) | 1024 | 16 | 16 | **PRIMARY — start here; also used for the pipeline smoke test** |
| 600M | `olmo2_600M` (:631) | 1344 | 16 | 16 | escalation 1 |
| 760M | `olmo2_760M` (:646) | 1536 | 16 | 16 | escalation 2 |
| 1B | `olmo2_1B` (:661 → `llama2_1B` :1143) | 2048 | 18 | 16 | escalation 3 |
| 3B | `olmo2_3B` (:696) | 3328 | 16 | 16 | later, separate compute decision |
| 7B | `olmo2_7B` (:714 → `llama2_7B` :1159) | 4096 | 32 | 32 | eventual target |

All rungs already use reordered-norm + QK-norm + RoPE θ=500k + RMSNorm eps 1e-6, so the architecture is held constant across the ladder — only width/depth change. `vocab_size` is always `TokenizerConfig.dolma2().padded_vocab_size()`.

**Escalation rules (pre-registered, to keep the scaling clean of confounds).**
1. **One axis at a time.** Only the model factory changes between rungs. Data, tokenizer, number of continuous thoughts K, the arms A0–A4, the LR-schedule *shape*, and seeds are all held fixed.
2. **Re-sweep the learning rate per rung.** The optimal LR does not transfer for free across scale; run the small pre-registered LR grid at each new rung (this is the one hyperparameter allowed to move, and it moves for all arms identically). A μP-style parameterization can replace the sweep later but is out of scope here.
3. **Comparisons stay within-rung.** Arms are always compared at the *same* size and at matched inference compute; never compare an arm at one size to an arm at another. Escalation turns the primary claims into a **scaling question** — plot the gate-A slope and the gate-B gain *as a function of model size* — rather than a cross-size comparison.
4. **K is held fixed across rungs** so the continuous-thought budget is comparable; cost scales roughly as (params × (1 + K sequential passes)), so 760M ≈ ~2× and 1B ≈ ~3–4× the 370M cost.

**Trigger to escalate (decision rule).** After the 370M primary run with ≥5 seeds:
- If **gate A is null but the probes show a weak, present, and diameter-increasing** superposition signal → escalate one rung (likely below emergence scale).
- If **gate A signal is entirely absent even in-distribution** *and* the secondary CODI compute-parity reproduction (§6) also fails → suspect an implementation/method problem; debug on a small / CPU-friendly test config before spending on escalation.
- Escalate **at most one rung per iteration**; stop when a gate passes or at 1B (beyond 1B is a separate, explicit compute decision, not an automatic step).

---

## 4. Method

### 4.1 The continuous-thought substrate (CODI) — [Code, medium]

Following [CODI](https://arxiv.org/abs/2502.21074), a single training step runs the shared-weight model twice and combines three losses:

1. **Teacher branch (explicit CoT):** standard next-token cross-entropy over `question + explicit-CoT + answer`.
2. **Student branch (continuous thought):** the model consumes `question`, then a `<bot>` marker, then generates **K continuous thoughts** (each step's last-layer hidden state fed back as the next input embedding, no decode), then `<eot>`, then is trained with cross-entropy on the `answer` tokens only.
3. **Feature-distillation loss:** align the hidden-state activations of a designated **distillation token** between the teacher and student branches, across all layers (an L2 / smooth-L1 on the per-layer activations). This is the mechanism that transfers the teacher's reasoning into the student's continuous thoughts in one stage.

Total loss = CE_teacher + CE_student + λ·distill. K (number of continuous thoughts) is a fixed hyperparameter per run; the answer supervision plus the distillation loss are the only signals the continuous thoughts receive (they get no direct token-level target — gradients reach them through the answer and distillation losses).

### 4.2 The novel intervention: vocabulary-manifold regularization — [Code, small]

The distributional-shift hypothesis says unconstrained continuous thoughts `h_t` drift off the manifold of real token embeddings, hurting accuracy and readability. We add a regularizer that pulls each continuous thought toward that manifold. Two concrete forms, pre-registered, one chosen after a pilot:

- **(R1) Soft-decode entropy / commitment.** Compute the logit-lens distribution `p_t = softmax(W_U · h_t)` over the vocabulary and penalize its distance from the convex hull of embeddings, e.g. minimize `‖h_t − E·p_t‖²` where `E` is the embedding matrix (pull `h_t` toward a *mixture* of real token embeddings — a superposition that still lives in vocab space). Optionally add a mild entropy floor so it is a *mixture*, not a collapse to one token. *(As built: the entropy floor is implemented but **off by default** — a whitelisted per-arm `vocab_reg_entropy_floor`, switchable via `train_codi.py --vocab-reg-entropy-floor`; the first sweep runs it off for a clean A3-vs-A4 and it is switched on only if collapse is observed. See "Status & changes".)*
- **(R2) Nearest-embedding distance.** Penalize distance to the nearest token embedding. *(As built: top-1 nearest by logit-lens — the argmax token's embedding; see `loss.vocab_manifold_reg`.)*

Strength γ is swept. R1 is the primary form because it is exactly a "superposition constrained to vocabulary space," aligning the intervention with the mechanism.

### 4.3 Superposition measurement — [Code, small–medium]

For the trained continuous-thought models, on graph-reachability problems with known solution frontiers:

- **Logit-lens:** decode each continuous thought's `p_t` and check whether the top-mass tokens correspond to *multiple* currently-reachable frontier nodes at once (superposition signature).
- **Linear probes:** train linear classifiers on frozen `h_t` to predict the set of reachable nodes at step t; report accuracy vs. a shuffled-label control probe.
- **Causal interventions (required):** patch/ablate the frontier directions in `h_t` and measure the causal effect on the final answer. A superposition claim is only accepted if the probe-identified directions are *causally* used (per the 2026 critique).

---

## 5. Experimental design

All arms start from the same OLMo-370M base checkpoint, use identical data, data order, seeds, optimizer, and **identical LR schedule** (WSD; held fixed so no arm benefits from a decay artifact). Comparisons are at **matched inference compute** and reported as accuracy-vs-compute *curves*, never single points.

| Arm | Description | Role |
|---|---|---|
| **A0** | Explicit discrete CoT (teacher-style) | Readable upper anchor; the "discrete" side of the superposition comparison |
| **A1** | No-CoT direct answer | Lower anchor |
| **A2** | CODI continuous thoughts, unconstrained | The substrate / reproduction |
| **A3** | CODI + vocabulary-manifold regularization (R1) | **The novel intervention** |
| **A4** | CODI + matched-strength generic L2 on `h_t` | **Confound control:** isolates that it's the *vocab-space direction*, not regularization per se |
| **A5** *(optional/secondary)* | Coconut multi-stage curriculum | Stability comparison vs CODI |

**Primary comparison A (superposition).** Across a grid of graph diameters D, compute `acc_continuous(D) − acc_discrete(D)` (A2/A3 vs A0). The theory predicts this difference is **positive and increasing in D**. Fit the slope with paired seeds.

**Primary comparison B (distributional-shift fix).** `A3 − A2` (does vocab regularization help?), gated by `A3 − A4 > 0` (is it the vocab *direction*, not just regularization?), plus logit-lens decodability of A3 vs A2, all at matched inference compute.

**Seeds.** ≥3 paired seeds for screening; ≥5 for any confirmatory claim. All comparisons paired by seed.

---

## 6. Metrics and success criteria

- **Primary gate A (superposition):** the slope of `acc_continuous(D) − acc_discrete(D)` vs D is **positive with a 95% paired CI excluding 0**, AND causal interventions confirm frontier directions are used (causal effect CI excludes 0). Passing means the theory-predicted mechanism is present at 370M.
- **Primary gate B (distributional-shift fix):** `A3 − A2` accuracy **> 0, 95% CI excludes 0**, AND `A3 − A4 > 0` (vocab direction matters), AND logit-lens decodability(A3) > decodability(A2), AND A3's inference-compute advantage over A0 is preserved (within a pre-set tolerance). Passing means vocab-manifold regularization is a real fix, not generic regularization.
- **Secondary:** A2 reaches A0's accuracy at strictly lower inference compute on ≥1 task (compute-parity curve); generalization to unseen (n, b, D) holds (no memorization collapse).
- **Reporting rule:** every headline number is an accuracy-vs-compute curve with paired-seed CIs; a positive mean with a CI crossing 0 is reported as *underpowered*, not as success or failure.

---

## 7. Confounds and mitigations

This is the core of the PRD; each row is a way the result could lie, and the control that prevents it.

| Confound | How it would fake a result | Mitigation |
|---|---|---|
| **Compute confound** | Latent arm "wins" just by doing more hidden compute | Match inference FLOPs/latency; report curves; count the K continuous-thought passes in the budget |
| **Training-budget confound** | An arm sees more tokens/steps | Identical tokens, steps, data order, seeds across arms |
| **Optimizer/LR-schedule confound** | Late data under decaying LR flatters an arm | Identical WSD schedule and optimizer for all arms (held fixed) |
| **Base-model confound** | Arms start from different checkpoints | All arms fork the *same* OLMo-370M checkpoint |
| **Task-triviality confound** | Graphs so easy discrete CoT already saturates → no headroom to show superposition | Calibrate graph sizes so discrete CoT is well below ceiling; sweep D across a range with real difficulty |
| **Memorization masquerading as reasoning** | Model memorizes train graphs | Test on unseen (n, b, D); report generalization gap |
| **Probe over-interpretation** | Linear probe finds frontier structure that isn't used | Require *causal* interventions + shuffled-label control probes |
| **"Regularization, not direction"** | A3 gains come from *any* regularizer | A4 = matched-strength generic L2 control; claim requires A3 > A4 |
| **K (thought count) confound** | A3 vs A2 differ in effective compute via K | Fix K identical between A2 and A3; sweep K separately |
| **Distillation-token placement** | Result hinges on an arbitrary token position | Ablate distillation-token position; report sensitivity |
| **Contamination** | Eval leakage inflates accuracy | Synthetic generator with disjoint train/test seeds; no public-graph reuse in the primary task |
| **Superposition = collapse** | R1 collapses `h_t` to a single token, killing the mechanism it's meant to preserve | Entropy floor in R1; verify logit-lens mass stays multi-modal where multiple frontiers exist. *(As built: floor default-off, switched on empirically if the `decodability` + `superposition_mass` probes show collapse.)* |

---

## 8. Build checklist (executable, no-questions)

This section is written so an agent can implement the PRD end to end without further clarification. All new code lives under a new package `src/olmo_core/latentcot/` and new scripts under `src/scripts/latentcot/`; experiment data lives under `data/` (gitignored) and run/eval outputs under `runs/` (gitignored) — never committed. **Line references were verified against `main` @ `d663bae` (2026-08-01)**, after the upstream refactor that removed `src/edullm/` and `src/scripts/orcd/` — the `olmo_core` core files this PRD depends on were untouched by that refactor and all anchors below resolved exactly (only the two `llama2_*` factory lines and the optim import lines were nudged). Re-read before editing in case later commits shift them.

**Integration-point map (verified against the tree).**

| What | Symbol | File:line |
|---|---|---|
| Model-size factories | `TransformerConfig.olmo2_*` | `nn/transformer/config.py:601–725` |
| Model forward (accepts pre-computed embeddings) | `Transformer.forward(..., input_embeddings=None, ...)` | `nn/transformer/model.py:523–536` |
| Embedding lookup / pass-through | `h = self.embeddings(input_ids)` vs `h = input_embeddings` | `nn/transformer/model.py:585–592` |
| Return hidden states when no head | `if self.lm_head is not None … else return h` | `nn/transformer/model.py:604–615` |
| Embedding table | `self.embeddings = nn.Embedding(vocab_size, d_model, …)` | `nn/transformer/model.py:132` |
| Unembedding matrix (logit-lens / reg) | `self.lm_head.w_out.weight` (shape `[vocab, d_model]`) | `nn/lm_head.py:174` |
| LM-head loss output | `LMOutputWithLoss(logits, loss, ce_loss, z_loss)` | `nn/lm_head.py:143–151` |
| Train module (subclass this) | `class TransformerTrainModule(TrainModule)` | `train/train_module/transformer/train_module.py:65` |
| Microbatch loop + loss call | `self.model_forward(...) → (_, loss, ce_loss, z_loss)`; `loss.backward()` | `train/train_module/transformer/train_module.py:397–422` |
| Forward wrapper | `def model_forward(...)` | `train/train_module/transformer/train_module.py:541` |
| Metric logging | `self.record_metric(name, val, ReduceType, namespace="train")` | `train/train_module/transformer/train_module.py:446–457` |
| Train-module config `.build(model)` | `TransformerTrainModuleConfig.build` | `train/train_module/transformer/config.py:341` |
| Optimizer / scheduler | `AdamWConfig`, `WSD` (registered `"wsd"`) | `optim/__init__.py:2` (AdamWConfig), `:15` (WSD); `optim/scheduler.py:158` (`@Scheduler.register("wsd")`), `:160` (class) |
| Data + entry pattern to copy | `NumpyFSLDatasetConfig`, `NumpyDataLoaderConfig`, `trainer.fit()`, `trainer.load_checkpoint()` | `src/scripts/train/template.py:103–217` |

Reference implementation to port: [github.com/zhenyi4/codi](https://github.com/zhenyi4/codi).

---

### Phase 0 — Environment & baseline sanity
- [ ] **0.1** Install: `pip install -e '.[all]'`. Confirm `python -c "import olmo_core"` works.
- [ ] **0.2** Copy `src/scripts/train/template.py` → `src/scripts/latentcot/train_codi.py`. Set `MODEL_CONFIG = TransformerConfig.olmo2_370M(vocab_size=TOKENIZER_CONFIG.padded_vocab_size())` (the primary rung — we start here; there is no separate 190M rung), `SEQUENCE_LENGTH=1024` for the smoke run.
- [ ] **0.3** Run `python src/scripts/latentcot/train_codi.py dry_run smoke` and confirm the config prints, then run a ~50-step tiny train to confirm the pipeline is wired.
- [ ] **Done when:** dry_run prints a valid config at 370M and the 50-step smoke train runs with no import/build errors.

### Phase 1 — Synthetic graph-reachability data — ✅ DONE (branch `latent-cot-superposition-amy`, commit 16bf3e4)
- [x] **1.1** Create `src/olmo_core/latentcot/data/graph_gen.py` with `generate(*, num_nodes, branching, depth, seed, reachable) -> Example`. `Example` holds the directed graph, source `0`, target, `reachable`, `depth`/`distance`, per-step BFS `frontiers` (the teacher reasoning trace **and** the probing labels), and a shortest `path`. *(Answer token(s) rendered in Phase 2.)*
- [x] **1.2** Serialize balanced train/test to `data/latentcot/` via `src/scripts/latentcot/gen_graph_data.py` — disjoint seed ranges and **OOD depths (5, 8) in test only**. Generated 2400 train / 960 test.
- [x] **1.3** Independent (second-implementation) BFS verification of every instance; balanced depth histogram printed; train∩test graph hashes empty. Formal suite: `src/test/latentcot/test_graph_gen.py` (65 tests passing).
- [x] **Done when:** train/test files exist, BFS-verified, with a printed diameter histogram; train∩test graph hashes are empty. ✅

### Phase 2 — Tokenization & special tokens — ✅ DONE (commit 6bd6704)
- [x] **2.1** `src/olmo_core/latentcot/tokens.py`: dolma2 vocab (100278; padded 100352) + control tokens at unused padded ids **100348–100351** (no resize). Deviation: **four** control tokens, not three — added `<thought>` as the student latent-slot placeholder (its embedding is overwritten by the continuous thoughts in Phase 3). Cached dolma2 tokenizer loader (needs `tokenizers`, installed via `uv`).
- [x] **2.2** `src/olmo_core/latentcot/data/encode.py`: renders query + BFS-frontier CoT + yes/no answer, and builds two **structurally parallel** views — teacher `question <bot> cot <eot> <distill> answer` and student `question <bot> THOUGHT*K <eot> <distill> answer` (shared `<distill>`+answer suffix for clean feature alignment). Uses OLMo-core's native boolean `label_mask` (student supervises answer only; teacher supervises CoT+answer) rather than pre-shifted labels, plus `<bot>`/`<distill>` positions. `dataset.py`: `LatentCotDataset` over the Phase-1 JSONL + `collate` (right-pad; `label_mask` padded `False`).
- [x] **Done when:** round-trip decode reproduces the rendered problem; control ids never collide with real tokens (all ≥ 100278); label masks isolate exactly the intended spans. `src/test/latentcot/test_encode.py` (7 tests). ✅

### Phase 3 — Continuous-thought forward path — ✅ DONE (commit b86a061)
- [x] **3.1** `nn/transformer/model.py`: added an additive `return_hidden_states: bool = False` kwarg to `Transformer.forward` that returns the post-block hidden states (mirrors the `lm_head is None` branch). Default `False` → existing behavior unchanged (verified: existing transformer tests pass, 59 passed / 42 GPU-skipped). Only +9 lines; the newer local `black` wants to reformat a *pre-existing* line (301) but the repo's pinned black leaves it — so `model.py` kept minimal.
- [x] **3.2** `src/olmo_core/latentcot/cot.py`: `embed_tokens` (replicates the model's own `embed_scale`/`embedding_norm`), `run_continuous_thoughts` (feeds the last-position hidden state back as the next input embedding for K steps via `input_embeddings`), and `student_forward` (embed prefix → K thoughts → splice → suffix → final forward with optional labels). Note: step 4 (answer logits/loss) lives in `student_forward`; the CODI dual-branch + distillation loss is Phase 4.
- [x] **3.3** `src/test/latentcot/test_cot.py`: K∈{1,2,4} produce correctly-shaped thoughts and `loss.backward()` puts nonzero gradient at the continuous-thought positions (verified via `embeds.grad`, since the returned `thoughts` is a parallel `cat`) and on the embedding table.
- [x] **Done when:** test passes for K∈{1,2,4} on a reduced-size 2-layer CPU model (not the full 370M). ✅

### Phase 4 — CODI train module (dual branch + distillation + regularizer) — ✅ DONE (commit 97221ae)
> Implemented as a testable core `latentcot/loss.py::codi_loss` + a thin `latentcot/train_module.py` wrapper. Deviations from the sketch below: the loss lives in `loss.py` (not inline in the microbatch); it optimizes `LMOutputWithLoss.loss` (the `.ce_loss` field is detached — logging only); students are processed per-example (batch dim 1) to avoid the variable-length-prefix problem (batched/bucketed = Phase-5 optimization); `distill_token_id` dropped (positions come from the data). Mechanism validated on a tiny CPU model; the literal 370M run is Phase 8 (GPU).
- [x] **4.1** `codi_loss` per microbatch: teacher branch (explicit CoT); student branch (`run_continuous_thoughts`, answer-only CE); per-layer `<distill>` activation capture via forward hooks; distillation `smooth_l1(student, teacher.detach())`; `vocab_manifold_reg` (R1/R2/L2/none); `loss = teacher.loss + student.loss + λ_d·distill + γ·reg`; metrics recorded. Per microbatch:
      1. **Teacher branch:** `model_forward` on `teacher_input_ids/labels` → `ce_teacher`. Register forward hooks on `model.blocks` to capture per-layer activations at `distill_pos` (teacher, `.detach()`).
      2. **Student branch:** `run_continuous_thoughts(...)` (Phase 3) → `ce_student` on the answer span only; capture per-layer `distill_pos` activations (student).
      3. **Distillation loss:** `distill = smooth_l1(student_acts, teacher_acts)` summed/averaged over layers.
      4. **Vocab regularizer** (on each continuous thought `c_t`): compute `logits_t = c_t @ model.lm_head.w_out.weight.T`; `p_t = softmax(logits_t)`; **R1** `reg = ‖c_t − E·p_t‖²` where `E = model.embeddings.weight`, with an entropy floor on `p_t` to prevent collapse. (Provide `R2` = distance to k-NN embeddings, and `L2` = `‖c_t‖²` as the **A4 control**, behind a flag.)
      5. `loss = ce_teacher + ce_student + λ_d·distill + γ·reg`; `loss.backward()`.
      6. `self.record_metric(...)` for `ce_teacher`, `ce_student`, `distill`, `reg` (`:446`).
- [x] **4.2** `CodiTransformerTrainModuleConfig(TransformerTrainModuleConfig)` with `num_continuous_thoughts`, `distill_weight`, `vocab_reg` (str: none/R1/R2/L2), `vocab_reg_weight`, `vocab_reg_entropy_floor`; `.build()` overridden to return `CodiTransformerTrainModule` (verified on CPU). *(`distill_token_id` dropped — distill position comes from the data.)*
- [x] **Done when (mechanism):** on a tiny CPU model, `ce_student` drops 11.7→<2 in 150 steps, `distill` is computed, and all four metrics are produced; grad-flow regression test guards the `.loss` vs detached-`.ce_loss` pitfall. The literal 370M 100-step run is deferred to Phase 8 (GPU). ✅ `src/test/latentcot/test_codi.py` (4 tests; suite total 90).

### Phase 5 — Arms as configs (differ only by pre-registered flags) — ✅ DONE (commit 7fcf039)
- [x] **5.1** `src/olmo_core/latentcot/arms.py`: the 5 arms as overrides on one shared base `CodiTransformerTrainModuleConfig` (same model/data/optim/WSD/seed/base-checkpoint). **A0** `explicit_cot` (teacher CE), **A1** `no_cot` (direct `question <distill> answer` CE), **A2** `codi` vocab_reg=none, **A3** `codi`+R1, **A4** `codi`+L2 (control at matched γ). Multi-mode training via `arm_loss` (loss.py) + `arm_mode` dispatch in the train module; `codi_collate` yields the `{"examples": [...]}` batch.
- [x] **5.2** `assert_arms_differ_only_in(configs, whitelist)` compares `as_dict()` outside the whitelist and raises naming any offending field (catches accidental LR/seed/data confounds). **Whitelist = `(arm_mode, num_continuous_thoughts, vocab_reg, vocab_reg_weight)`** — `arm_mode` added to the PRD's list because A0/A1 are structurally different objectives. *(Later extended with `vocab_reg_entropy_floor` (default 0.0 = off) so A3 can carry R1's anti-collapse floor without a confound; see "Status & changes".)*
- [x] **Done when:** the confound assertion passes on the shared base (and fails on a tampered LR), and each of A0–A4 reduces its primary CE over a short run. `src/test/latentcot/test_arms.py` (10 tests; suite total 100). *(Full Trainer-driven 100-step runs at 370M are Phase 8/GPU.)* ✅

### Phase 6 — Eval & probing harness — ✅ DONE (commit 1736251)
- [x] **6.1** `evaluate.py`: per-arm answer prediction (codi decodes at `<distill>`; no_cot from `question <distill>`; explicit_cot greedy-generates to `<distill>`), `solve_rate_by_depth`, and `gate_a_curve` + `linear_slope` (`acc_continuous(D) − acc_discrete(D)` vs depth). `scripts/latentcot/eval.py` loads per-arm checkpoints and emits it.
- [x] **6.2** `inference_token_cost` as a forward-compute proxy that counts the K sequential continuous-thought passes (for the accuracy-vs-compute curve). *(Token-count proxy, not FLOP-exact.)*
- [x] **6.3** `probes.py`: **logit-lens** + `decodability`, **superposition_mass** (per-thought mass on frontier-node tokens), **linear_probe_accuracy** with a **shuffled-label control** (validated: 1.00 vs ~0.4), and **causal_ablation_margin_change** (project a direction out of the thoughts, measure the answer-margin change).
- [x] **Done when:** `eval.py` emits the gate-A curve+slope, the gate-B table (A2/A3/A4 acc + decodability), and the probe utilities are validated (`test_probes.py`, `test_evaluate.py`; suite total 112). Real gate plots + paired CIs come from Phase 8 on trained 370M arms. ✅

### Phase 7 — Matched-budget dry run & confound checks — ✅ DONE (commit 445f030)
- [x] **7.1** `preflight.py::per_arm_compute` reports per-arm forward-token cost **with the K continuous-thought passes counted**; `scripts/latentcot/preflight.py` prints it. *(Interpretation: we assert equality of the confound-relevant budget — config outside the arm whitelist, base checkpoint, problem seeds — and **report** per-arm FLOPs rather than equalizing them, since the arms use different token views and CODI's extra passes are intrinsic; arm fairness is enforced at matched **inference** compute in the eval.)*
- [x] **7.2** `assert_same_base_checkpoint` (fingerprint identical across arms) + `assert_disjoint_seeds` (train/test problem seeds don't overlap). Plus `assert_arms_differ_only_in` for the matched config.
- [x] **Done when:** `preflight()` runs all checks and returns a report, raising on the first failure; validated by `test_preflight.py` (6 tests) — passes on matched arms, fails on a tampered LR / mismatched checkpoint / overlapping seeds. Suite total: 118. ✅

### Phase 8 — Run, analyze, escalate — ⏳ CODE READY (commit c363732); runs pending GPU
> Driver + eval + runbook are implemented and unit-tested on tiny CPU models. The boxes below stay unchecked until the actual **370M runs on GPU** are done — that's the one step this CPU-only environment can't execute. Turnkey procedure: `phase8-runbook.md`.
- [ ] **8.1** Run ≥3 paired seeds (screen), then ≥5 (confirm), at 370M. Evaluate gates A and B (§6) with paired CIs. *(Driver: `src/scripts/latentcot/train_codi.py` / core `olmo_core.latentcot.train_driver`; eval: `src/scripts/latentcot/eval.py`.)*
- [ ] **8.2** Apply the §3.1 escalation decision rule. If triggered, pass `--rung olmo2_760M` (then `1B`), re-sweep `--lr` only, and repeat at the next rung; record the gate metric vs. model size.
- [ ] **Done when:** gates are decided with CIs at the current rung, and the scale decision (stop / escalate) is recorded in the results writeup.

**Commands** (see the runbook for the full loop): `verify_checkpoint.py --init-checkpoint s3://… --model olmo3_370M` (pre-flight: strict-load + forward-path smoke, run first) → `preflight.py` (gate) → `train_codi.py --arm <A0..A4> --rung olmo3_370M --init-checkpoint s3://… --init-seed 0 --seed <s> ...` (per arm × seeds, all forking the best model; GPU auto-detected; writes `model.pt` + `metrics.json`) → `eval.py --model olmo3_370M --arm A2=<...model.pt> ...` (gates A/B + report.json) → `compare_models.py --model olmo3_370M --baseline A0=<...> --ours A2=<...> ...` (§6.2 head-to-head vs the best-model baseline: solve-rate-by-depth + per-depth advantage slope). Dataset under `data/` and run/eval outputs under `runs/` (both gitignored); publishing the dataset is separate (`publish_dataset.py`, AWS creds).

---

## 9. Compute budget (to be calibrated in a dry run)

370M at finetuning/continued-training scale on synthetic graphs + a slice of GSM8K/ProsQA. The dominant extra cost is CODI's dual branch and the K sequential continuous-thought passes (training cost scales roughly with K). Rough order: single-digit to low-tens of GPU-hours per arm on one modern accelerator; the full arm × seed × diameter grid is a small-cluster-day, not a pretraining run. A `dry_run` mode must print per-arm token/step/FLOP budgets and confirm they match before any confirmatory seed launches.

---

## 10. Risks and open questions

- **370M may be below the emergence scale** for clean superposition; a null on gate A is then a *scale* finding (report it, per non-goals), not a refutation of the method.
- **CODI transfer at 370M on graph tasks** is unproven (its result is GSM8K at GPT-2 scale); the secondary compute-parity reproduction de-risks this before the primary claims are trusted.
- **R1's exact form** (mixture pull vs entropy floor weighting) needs a pilot; both forms are pre-registered so the choice is not a garden-of-forking-paths. *(As built: resolved empirical-first — the floor is default-off for the clean A3-vs-A4 sweep and switched on only if the pilot shows collapse; both forms remain pre-registered. See "Status & changes".)*
- **Causal-probe design** for "multiple frontiers" must be specified crisply (which directions, how patched) before looking at outcomes.
- **Open question:** does vocab-manifold regularization trade off against the superposition advantage (constraining to vocab space might reduce the very mixing that helps)? Gate B's "preserve the compute advantage" condition is designed to catch exactly this tension.

---

## 11. Acceptance criteria (pre-registration integrity)

Before any outcome is examined: the arms, the diameter grid, the seeds, the K value(s) *(frozen at **K=10**; deepest graph is depth 8, so K gives ~2 steps of headroom — see "Status & changes")*, the regularizer forms (R1 primary, R2 secondary), the metrics, and the confound controls above are frozen in this document. A run counts as valid only if: all arms share the base checkpoint, tokens, data order, and LR schedule; the `dry_run` confirms matched budgets; the graph generator's train/test seeds are disjoint; causal-probe and control-probe protocols were specified pre-hoc; and no arm's hyperparameters were tuned on the test set. Underpowered results (CI crossing 0) are reported as such. *(As built: the pre-registration checks are `preflight.py` / `preflight()`, with the confound-whitelist gate `assert_arms_differ_only_in`. The **code** is committed and pushed to the feature branch `latent-cot-superposition-amy` only, never `main`; generated data/outputs are gitignored under `data/`/`runs/`, and these design docs live in `docs/latent-cot/`.)*

---

## 12. Deliverables

- Trained arms A0–A4 (+ optional A5), paired seeds, from the shared OLMo-370M base.
- Accuracy-vs-compute curves per task; the `acc_continuous(D) − acc_discrete(D)` vs D plot with paired CIs (gate A); the A2/A3/A4 comparison with decodability (gate B).
- The superposition probing report (logit-lens visualizations, linear-probe accuracy vs control, causal-intervention effects).
- The reusable OLMo-core continuous-thought harness and the graph-reachability diagnostic set.
- A short results writeup (separate from this PRD), including any nulls.

---

## References

- Chain-of-thought prompting — Wei et al., 2022 — https://arxiv.org/abs/2201.11903
- Coconut (continuous latent reasoning; ProsQA) — Hao et al., 2024 — https://arxiv.org/abs/2412.06769
- CODI (single-stage self-distillation) — Shen et al., EMNLP 2025 — https://arxiv.org/abs/2502.21074 — code: https://github.com/zhenyi4/codi
- Reasoning by Superposition (theory; graph-reachability separation) — Zhu et al., NeurIPS 2025 — https://arxiv.org/abs/2505.12514
- CoT2 / Emergence of Superposition — https://arxiv.org/abs/2509.23365
- A Survey on Latent Reasoning (distributional-shift problem) — https://arxiv.org/abs/2507.06203
- Unlocking the Black Box of Latent Reasoning (interpretability-guided intervention) — https://arxiv.org/abs/2606.01243
- Observable Patterns Are Not Explanations (causal-probe critique) — https://arxiv.org/abs/2606.12689
- LLM Reasoning Is Latent, Not the Chain of Thought (position paper) — https://arxiv.org/abs/2604.15726

> Sourcing note: literature above was gathered via a live search + fetch pass. A few very recent (2026) arXiv identifiers should be confirmed on open, as recent IDs occasionally shift.
</content>
