# Continuous-Thought Reasoning on OLMo-370M: A Superposition & Distributional-Shift PRD

**Does a continuous chain-of-thought, built stably by self-distillation, reason by superposition where theory says it must — and does constraining latents toward vocabulary space fix the distributional-shift that degrades latent reasoning?**

P4 · Latent Reasoning · Execution PRD v1 (OLMo-370M-locked)

---

## TL;DR

**Question.** Does a continuous (latent) chain-of-thought reason by *superposition* — carrying several graph search-frontiers at once, rather than committing to one path — where theory says it must? And does pulling the latent thoughts toward vocabulary space fix the distributional shift that otherwise degrades latent reasoning?

**Setup.** One task: synthetic directed-graph reachability with controllable depth D ("is the target reachable from the source?"). One model: OLMo-370M (`olmo3_370M`). **All five arms are fine-tunes of the *same* pretrained "best model" checkpoint, on the same data, with the same schedule and seeds — the only thing that varies between them is the reasoning/training method.** That single-variable design *is* the confound control.

**The five arms.**

| Arm | Method | Role |
|---|---|---|
| **A0** | Explicit written-out CoT (standard fine-tune) | **Fair baseline** — the yardstick every latent arm is measured against |
| **A1** | No CoT (question → answer directly) | Floor — task difficulty with no intermediate reasoning |
| **A2** | CODI continuous thoughts, unconstrained | The latent substrate (plain latent reasoning) |
| **A3** | CODI + **R1** vocabulary-manifold regularizer | **The hypothesis / the fix** — keeps thoughts near the real token-embedding manifold |
| **A4** | CODI + matched-strength generic **L2** on the thoughts | **Control for A3** — isolates the *vocab-space direction* from regularization-in-general |

Each latent arm (A2/A3/A4) computes **K = 10** continuous thought vectors before answering — one BFS-style expansion per step. K ≥ the deepest graph (depth 8), with ~2 steps of headroom. (An optional A5 = Coconut curriculum is a secondary stability comparison; see §5.)

**The two pre-registered gates (pass/fail).**
- **Gate A — superposition.** Plot the advantage `acc(latent) − acc(A0)` as graph depth D grows. Theory predicts it is **positive and increasing in D** (superposition compounds over BFS steps). Pass = the slope's 95% paired-seed CI excludes 0 **and** causal probes confirm the frontier directions are actually *used*. This is the headline signal `compare_models.py` computes.
- **Gate B — is R1 the reason.** `A3 > A2` (the regularizer helps) **and** `A3 > A4` (it is the vocab-space *direction*, not merely regularizing) **and** A3's latents are more decodable — all **without** losing the latent inference-compute advantage over A0.

Everything below fixes the model, data, arms, budgets, metrics, and every confound control *before* any outcome is examined.

---

> **Status:** Pre-registered experimental design and execution protocol. It fixes the model, the tasks, the arms, the matched budgets, the metrics, the analysis plan, and the confound controls *before* any outcome is examined. These design docs live in `docs/latent-cot/` (tracked); generated data and run/eval outputs are gitignored under `data/` and `runs/` (no weights/datasets/outputs committed); the **code** is committed and pushed to the feature branch `latent-cot-superposition-amy` only (never `main`). See "Status & changes" below for what's actually built.

---

## Status & as-built notes (updated 2026-08-05)

**Build status.** Phases 1–8 are implemented, unit-tested (**146 tests**, CPU), style-clean, and pushed to branch **`latent-cot-superposition-amy`** (never `main`). The only remaining step is the **370M training on GPU** (this environment is CPU-only); the driver, eval, benchmark, and runbook are ready. Source of truth for what's built: the per-phase `✅ DONE` notes in §8, plus `progress.md` (changelog), `handoff.md` (agent brief), and `phase8-runbook.md` (the GPU procedure).

**As-built deltas from the original design** (the design text below is preserved; these are the ways the implementation refines it):

- **Base = the real "best model"; all arms fork it.** The confirmatory sweep forks a general-pretrained base — `s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/` (W&B run `f08ey8cm`) — via `train_codi.py --rung olmo3_370M --init-checkpoint s3://…`. The base is shared and identical across arms, so the §7 base-model confound control holds; this fixes **A0's role** as *the best model fine-tuned the normal way = the fair baseline* (see TL;DR). Dropping `--init-checkpoint` gives a from-scratch `olmo2_370M` run as a no-creds sanity check.
- **Rung is `olmo3_370M`, not `olmo2_370M`.** The best model was pretrained as olmo3 (olmo2 + sliding-window + flash_2). The two factories' state dicts are identical (same keys/shapes, 474M params) and numerically equivalent at our sequence lengths (≪ the 4096 window), so the §3 spec and the §3.1 ladder (written `olmo2_*`) carry over — escalation rungs use their `olmo3_*` counterparts.
- **All arms fine-tune → WSD warmup in the driver.** Because every arm is a fine-tune of the shared base, the Phase-8 direct loop (`train_arm`) applies the §7 pre-registered **WSD** schedule (linear warmup, `--warmup-steps 200` default, + 10% linear-decay tail), by hand so it is byte-identical across arms. Warmup stops a full-LR first step from spiking the loss and destroying the pretraining we forked (a fidelity fix reconciling the code with the design, not a design change). `preflight.py` checks the same `WSD(warmup=200, decay_fraction=0.1)`.
- **Peak LR is screened, not assumed.** Per §3.1 rule 2 / §7.1 only the schedule *shape* is fixed; screen peak LR on 1 seed of A0/A2 over `{1e-5, 2e-5, 5e-5, 3e-4}` and fix the winner for all arms (fine-tuning wants a smaller peak than a from-scratch 3e-4). `--lr`/`--warmup-steps` are recorded in each `metrics.json`.
- **K = 10** (was 8; `DEFAULT_K` + all script/runbook defaults). K ≥ the deepest depth (8) with ~2 steps of headroom, so a depth-8 failure means "can't superpose," not "one step short." Data is unchanged — K is applied at load via `encode_example(ex, K)`, no regen — and the depth-vs-advantage *slope* is unaffected. K is frozen, whitelisted, arm-invariant (A0/A1 ignore it).
- **R1 entropy floor: implemented, default OFF, switchable.** R1's optional anti-collapse floor (§4.2, §7 "superposition = collapse", §10) is a whitelisted per-arm field defaulting to `0.0`, switchable at runtime (`train_codi.py --vocab-reg-entropy-floor <nats>`). The first sweep runs it off for a clean one-variable A3-vs-A4; per `phase8-runbook.md` §4 it is switched on only if A3's decodability/superposition-mass probes show collapse (noting the added-term caveat vs the L2 control). Resolves the §10 "R1's exact form" question empirical-first; both forms stay pre-registered.
- **Confound whitelist** (the fields arms may differ in) = `(arm_mode, num_continuous_thoughts, vocab_reg, vocab_reg_weight, vocab_reg_entropy_floor)`, enforced by `assert_arms_differ_only_in`. `arm_mode` is included because A0/A1 are structurally different objectives; `vocab_reg_entropy_floor` so A3 can carry the floor without a confound (a no-op while off, so A3/A4 still differ only in `vocab_reg`).
- **Comparison is vs A0, never the raw checkpoint zero-shot.** The raw best model never saw the graph format (≈ chance cold), so scoring it zero-shot would inflate our arms unfairly. `compare_models.py` therefore compares the latent arms against **A0** (the best model given identical conventional fine-tuning), isolating the *reasoning method*. An optional cold-start floor line can be added for reference but is not the comparator.
- **Dataset published as `sft/graph-reachability-depth`** via the `sft-conversations/v1` profile — the only fitting *registered* profile (`eval-items/v1` is unregistered in `edullm-data` v0.2.0). Each row = the full `Example` as metadata **plus** a `messages[]` array (user = query+edges, assistant = BFS reasoning + yes/no). Publish with `src/scripts/latentcot/publish_dataset.py`; our loader reads the shards directly (`Example.from_dict` ignores `messages`). See §3. Added per the `edullm-dataset-design` / `edullm-datasets` skills.
- **Implementation shape.** Optimize `LMOutputWithLoss.loss` (`.ce_loss` is detached/logging-only; a grad-flow test guards this); **four** control tokens (added `<thought>` as the student latent slot) at unused padded ids 100348–100351; the CODI student is processed **per example (batch dim 1)** to sidestep variable-length prefixes (batched/bucketed is a deferred optimization); Phase 8 uses a **direct training loop**, not the framework `Trainer`, reusing the same `arm_loss` as `CodiTransformerTrainModule`.
- **Checkpointing + a train-carved validation split.** `train_arm` writes rolling checkpoints every `--save-every` steps (default 500 ≈ ~10 saves/run), keeping the last `--keep-last` (default 2) plus a `best.pt` selected by accuracy on a **validation split carved off the training set** (`--val-fraction` default 0.1, seeded independently of `--seed` so it is identical across arms/seeds). The gate **test** set is never used for selection — that would be model selection on the eval data, which §11 pre-registers against. The final last-step weights remain the canonical `model.pt`. The policy is byte-identical across arms, so it stays confound-clean.
- **Thoughts pass through the final norm before feedback (scale fix, pre-run).** The last block's output is the **pre**-final-norm residual stream (this model keeps the final norm inside `LMHead`), so feeding it straight back compounded its magnitude every step: measured on the `olmo3_370M` rung at K=10, thought RMS ran **5.8 → 52** against a real-token embedding RMS of **1.0**, and 60 training steps on an 8-layer proxy amplified the endpoint a further **3.2×** (5.1 → 16.5). `cot.run_continuous_thoughts` now applies `cot.final_norm` (the LM head's own norm, identity-safe when a head/norm is absent) before each feedback step, matching the `hidden_states[-1]` convention Coconut/CODI feed — thought RMS is then flat **1.000** across all K, equal to the embedding scale. Two reasons this matters beyond tidiness: under `reordered_norm` blocks the SwiGLU sees the *unnormalized* residual, so the forked pretrained weights were being pushed off their operating point precisely at the thought positions; and **R1/L2 (A3/A4) incidentally suppress the drift while unregularized A2 does not**, an arm-dependent artifact in a controlled comparison — Gate B was protected by L2 as the matched control, but Gate A's A2 term was carrying it. Learning was *not* measurably hurt at 60 steps (final loss 9.427 vs 9.421, within noise), so this is a validity/robustness fix for a 5,000-step run, not a bug repair. **Decided and applied before the first GPU hour**, so no pre-registered outcome was observed first. K is uniform (10) for every example regardless of depth, so the old drift was constant across depths and did not bias the §6 depth *slope* directly — it shifted the level of the latent arms. Diagnostics `grad_norm` (pre-clip) and `thought_rms` are now logged every `log_every` steps in `metrics.json.train_history` as tripwires. Probes kept out-of-tree in `local/latentcot_norm_probe.py` / `local/latentcot_norm_ab.py`.
- **Throughput: bf16 autocast + TF32 (done); packing and a KV cache (not done).** The driver ran fp32 with no autocast, no TF32 and no `torch.compile`, and every forward at **batch dim 1** — 16 teacher + 160 thought-loop + 16 final = **192 sequential batch-1 forwards per optimizer step**, i.e. ~960k per arm per seed over 5,000 steps, at ~9.6e13 FLOPs/step (A100 bf16 peak would be 309 ms/step at 100% MFU; strict fp32-no-TF32 peak, 4.9 s). `--precision bf16` (new default) now wraps the training forward, in-loop val scoring, and the final gate scoring in bf16 autocast and enables TF32; `--precision fp32` is bit-identical to before. The distill smooth-L1 and R1's 100k-way softmax are pinned to fp32 (`.float()`, a no-op on the fp32 path). Precision lives in the shared driver, so it is arm-invariant, and is recorded in `metrics.json` alongside `batch_size`. **Still on the table:** sequence packing via `cu_doc_lens`/`max_doc_len` (already public on `Attention.forward`, flash-varlen underneath — no core change needed) would collapse the teacher and final forwards and fully batch A0/A1; prefix-length bucketing is *not* a substitute (23 distinct prefix lengths over 84 examples). The 9.7× recompute in the thought loop (1,916 token-positions/example vs 197 cached) **cannot** be fixed with core's `KVCacheManager`: it writes in-place into registered buffers, so it is an inference cache, and CODI training needs gradients through all K steps. It is also unusable for *eval* validation here — there is no `Transformer`-level API (only `Attention.init_kv_cache_manager`) and `TorchAttentionBackend` raises "doesn't support KV caching", so the CPU/Mac path cannot exercise it; deferred rather than shipped untested on the code path that produces the gates.
- **`--batch-size` is a gradient-noise knob, not a throughput one.** Because examples are processed one at a time, raising it adds sequential forwards instead of widening a tensor. Measured (CPU, tiny model, A2, K=10): per-example step time 155.6 ms at batch 1 → 71.1 / 71.7 / 64.0 ms at batch 2 / 4 / 8, i.e. **flat from 2 onward** — the only gain is amortizing the fixed per-step AdamW + grad-clip cost, and it saturates immediately, while total step time rises linearly (142 → 287 → 512 ms). At the runbook's `--batch-size 16` that amortization is already fully captured. Raising it further buys effective batch size (less gradient noise) at linear wall-clock cost, and would require re-screening peak LR; it does **not** improve MFU until the per-example loop is packed. Batch size is arm-invariant by construction (a `train_arm` argument, not an `Arm` field) and is now recorded in `metrics.json`. Audit scripts kept out-of-tree: `local/latentcot_mfu_audit.py`, `local/latentcot_batch_scaling.py`.
- **First campaign is a 1-seed pilot (decided 2026-08-07), not the confirmatory sweep.** §5/§11 pre-register the gates on **paired-seed 95% CIs**, and §5 called for 3 screening seeds then 5 confirmatory. The first GPU campaign is being run at **1 seed × 5 arms** to bound cost and shake the harness out at 370M. Consequence, stated plainly: **neither gate can be passed or failed from it** — a single seed provides no seed-level variance, so it yields point estimates only. What it does settle: that the harness runs clean at 370M (loss decreasing, `thought_rms` ≈1 and flat, `grad_norm` stable, no OOM), the calibrated per-arm cost, whether A2 comes anywhere near A0, and the sign/shape of the depth curve. A bootstrap CI over the 960 held-out **items** is legitimate to report but measures test-item noise, not init/data-order variance, and is **not** a substitute for the paired-seed criterion. The confirmatory sweep (≥3, then 5 seeds) is unchanged and still required; the pilot's seed-1 runs are reusable as one of its seeds. Anything written from the pilot must label itself a pilot — presenting it as a gate outcome would violate §11. Procedure and cost scaling: `phase8-runbook.md` §3 and §6.
- **Compute requirements are estimated, not measured.** `phase8-runbook.md` §0b carries the sizing for a GPU-availability request: 5 independent single-GPU jobs (no multi-node, no interconnect), ≈2,420 PFLOPs for the campaign and ≈740 for the slowest arm, ⇒ **~13 h on 5×A100 in parallel / ~43 h serialized at an assumed 5% MFU**, and **≈25 GB peak memory** per CODI arm (dominated by ~17 GB of retained activations, because `codi_loss` keeps all 16 examples' K-chains alive before a single `backward()`; scales linearly with `--batch-size`). Derived from FLOPs × assumed MFU in `local/latentcot_gpu_estimate.py` using real token counts — the runbook includes a 10-minute calibration command to replace the estimate with a measurement before booking. Two operational notes: A0's *eval* nearly equals its training cost (`greedy_generate` has no KV cache, so every generated CoT token is a full forward), and because there is no `--resume` flag a preempted job restarts from the base checkpoint — prefer non-preemptible instances.
- **Peak LR was NOT screened for the first submitted pilot — a live deviation from §3.1/§7.1.** The as-built note above pre-registers screening peak LR on 1 seed of A0/A2 over `{1e-5, 2e-5, 5e-5, 3e-4}`. The calibration job that was to do it never got past loading the base checkpoint (below), so `.edullm/run.yaml` submits `--lr 2e-5`: a defensible fine-tune peak, chosen because 3e-4 is a from-scratch value and fine-tuning a pretrained base wants roughly an order of magnitude less, but **not measured**. Any numbers this pilot produces carry that caveat, and it compounds the 1-seed caveat: the run is a screen twice over. The screen is still owed before any confirmatory sweep. Had it run, the plan was 3 points × ~150 steps of A2 at batch 8, ~20 minutes on one A10G.
- **Phase 8 status: four platform failures diagnosed and fixed, blocked on the base checkpoint.** In order, each caught by a run that died in seconds rather than by review: (1) the research image had no `flash-attn`, so every `olmo3_*` config raised at construction — fixed by installing the prebuilt wheel in `.edullm/Dockerfile` (verified in-image: `has_flash_attn_2(): True`, and `olmo3_370M` builds at 474,022,912 params on an A100); (2) `tokenizers` was absent from the image and is reached by the first dataset access — installed, with the dolma2 tokenizer baked into the image cache and `load_tokenizer()` falling back to `local_files_only` so no Hub egress is needed; (3) `$EDULLM_CHECKPOINT_DIR` is an `s3://` URI that `pathlib.Path` silently rewrites to a relative local path, which would have written every artifact into a directory named `s3:` and lost it at container exit — `--out` now stages locally and mirrors, and `train_arm` refuses a URI outright; (4) `load_checkpoint` probed only `<dir>/.metadata` while `Checkpointer` writes `<dir>/model_and_optim/` — both are probed now. **Still open:** the base checkpoint at `s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/` cannot be read under either layout, and that bucket is referenced nowhere outside this repo's own docs; see `.edullm/research-image-findings.md` §4b. Everything upstream of that load is green on real hardware: dataset generation, the §11 pre-registration gate, the image, and model construction.
- **Secondary tasks (ProsQA / GSM8K) not implemented** — only the synthetic directed-graph task was built; the compute-parity reproduction (§2 secondary objective) is deferred.
- **Benchmark + pre-flight scripts.** `compare_models.py` (arms vs A0: solve-rate-by-depth + per-depth advantage slope, the §6.2 head-to-head) and `verify_checkpoint.py` (strict-load the S3 base into `olmo3_370M`, exercise one plain + one continuous-thought forward/backward — run first on the GPU box). Train/eval/compare auto-detect the GPU (`--device auto`).

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
- [x] **3.1** Read the post-block hidden states out of the model. Originally an additive `return_hidden_states: bool = False` kwarg on `Transformer.forward` (+9 lines in `nn/transformer/model.py`). **Superseded on `latent-superposition-module`:** the same tensor is now captured by a forward hook on the last block (`cot._capture_last_block`), so **no shared file is touched at all**. `Transformer.forward` assigns `h = block(h, ...)` in a loop with the LM head as the next statement, so the last block's output *is* what the kwarg returned — numerically identical, and checkpoints from runs made under the kwarg stay valid. The LM head that the hook path cannot skip is reduced to one position by `logits_to_keep=1` (already on `main`; it slices before the vocab projection). Motive: `Transformer.forward` is the most contended function in the repo and a dozen other workstreams are editing it concurrently. See `module-prd.md`.
- [x] **3.2** `src/olmo_core/latentcot/cot.py`: `embed_tokens` (replicates the model's own `embed_scale`/`embedding_norm`), `run_continuous_thoughts` (feeds the last-position hidden state back as the next input embedding for K steps via `input_embeddings`), and `student_forward` (embed prefix → K thoughts → splice → suffix → final forward with optional labels). Note: step 4 (answer logits/loss) lives in `student_forward`; the CODI dual-branch + distillation loss is Phase 4.
- [x] **3.2b** `cot.final_norm(model, hidden)` applies the LM head's final norm (identity when there is no head/norm, e.g. pipeline parallelism or `NormalizedLMHead`), and `run_continuous_thoughts` applies it to every thought before feeding it back — the scale fix in "Status & as-built notes". One call site covers training, eval, probes, and `verify_checkpoint.py`; no core change was needed, which is the boundary working as intended.
- [x] **3.3** `src/test/latentcot/test_cot.py`: K∈{1,2,4} produce correctly-shaped thoughts and `loss.backward()` puts nonzero gradient at the continuous-thought positions (verified via `embeds.grad`, since the returned `thoughts` is a parallel `cat`) and on the embedding table. Plus `test_thought_scale_does_not_compound_over_k` — an 8-layer/K=10 regression guard on thought magnitude (verified to fail at ratio 5.56 vs its 1.5 threshold if `final_norm` is neutralized; the 2-layer/K=2 fixtures cannot see the drift at all) and `final_norm` unit tests for the pass-through cases.
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
- [ ] **8.1a** *(first)* Run the **1-seed pilot** (A0–A4, seed 1) at 370M — a harness/cost screen, **not** a gate outcome; see "Status & as-built notes" and `phase8-runbook.md` §3/§6.
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
