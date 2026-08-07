# Handoff: Latent-CoT Superposition experiment (branch `latent-cot-superposition-amy`)

Context brief for an AI agent taking over this work. Read this + `latent-cot-superposition-prd.md`
(the full spec) + `progress.md` (per-phase changelog) + `phase8-runbook.md` (the GPU procedure).
Everything below is implemented, unit-tested (**140 tests, CPU**), style-clean, and pushed to the
branch. The only thing not done is the **370M GPU run**.

---

## 0. GPU request — the five things a scheduling agent needs

**1. What we're running, and which module it's testing.**
5 independent single-GPU training processes — `src/scripts/latentcot/train_codi.py`, one per
experiment arm (A0, A1, A2, A3, A4) — 1 seed each, 5,000 steps each, `--rung olmo3_370M`
(474M params). No multi-node, no multi-GPU per job, no interconnect requirement.
Module under test: **`olmo_core.latentcot`** — specifically the continuous-thought forward path
(`cot.py`), the CODI loss with the novel vocabulary-manifold regularizer (`loss.py`), and the
Phase-8 driver (`train_driver.py`). The one line of shared code involved outside that package is an
additive `return_hidden_states` kwarg on `Transformer.forward`
(`src/olmo_core/nn/transformer/model.py`, default off, no behavior change for anyone else).

**2. Peak GPU memory: ≈ 25 GB per CODI arm; request ≥ 40 GB.**
Estimated, not yet measured (this dev box is CPU-only). Breakdown: params fp32 1.9 GB + AdamW
states 3.8 GB + grads 1.9 GB + **retained activations ≈ 17 GB**. That last term dominates because
`codi_loss` accumulates all 16 examples' graphs and calls `backward()` once, so every example's
whole K=10 chain is alive at the same time; it is computed as
`student_len × d_model × n_layers × ~10 saved tensors/layer × 2 bytes (bf16) × (K+2) forwards ×
batch 16` and scales **linearly with `--batch-size`**. A0/A1 need far less. 40 GB is comfortable,
24 GB is marginal and would need a smaller batch. Source: `local/latentcot_gpu_estimate.py`.

**3. How many hours: ~13 h wall-clock on 5×A100 in parallel, ~43 h if serialized on one.**
Derived, not measured: FLOPs ÷ (dense bf16 peak × assumed MFU), using real token counts from the
difficulty grid. Campaign ≈ 2,420 PFLOPs total; slowest single arm ≈ 740 PFLOPs (a CODI arm — 192
batch-1 forwards per step). **Plan against the 5% MFU column**, because the loop is
launch-latency-bound at batch 1, not compute-bound:

| GPU | 5 arms serial on 1 GPU | 5 arms parallel on 5 GPUs |
|---|---|---|
| A100-40/80GB | 43 h | **13 h** |
| H100-80GB | 14 h | 4 h |
| L40S-48GB | 74 h | 23 h |
| RTX A6000-48GB | 87 h | 27 h |

Add ~6 h for the LR screen (short runs, see runbook §3). **Do not pay for H100s expecting 3.2×** —
at batch 1 we are nowhere near saturating either card, so the FLOPs ratio does not transfer.
There is **no** multi-GPU parallelism within an arm (the direct loop has no DDP/FSDP), so more than
5 GPUs buys nothing. A 10-minute calibration command in runbook §0b replaces these estimates with a
measurement — run it before booking a long reservation.

**4. When we need it by: we need it now.**

**5. Resuming from a checkpoint: yes for init, no for run-state.**
Every arm **forks a shared pretrained base** — `--init-checkpoint
s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/` (W&B run `f08ey8cm`), needs AWS
creds — so this is a fine-tune, not from scratch, and *all five arms load the identical file*
(that shared start is the experiment's central confound control). It is **not** a resume of a
previous latent-CoT run; there is no prior run state to continue, and there is currently **no
`--resume` flag**. Within a run, rolling checkpoints are written every `--save-every` steps
(default 500, last 2 kept, plus a `best.pt`) purely for crash recovery and best-selection, at
~1.9 GB each / ~7.5 GB per run dir. If a job is preempted it restarts from the base checkpoint, so
**preemptible/spot instances cost up to a full run**, not one interval — prefer non-preemptible, or
budget for restarts.

---

## 1. What we're building (one paragraph)
A research harness in the OLMo-core repo to test **latent chain-of-thought** on a small model.
The model reasons in continuous "thoughts" (Coconut/CODI style) instead of emitting text, and we
test two claims on a synthetic **directed-graph reachability** task: **(gate A) superposition** —
the continuous-reasoning accuracy advantage over explicit CoT grows with graph depth (theory:
arXiv:2505.12514); **(gate B) the novel fix** — regularizing the continuous thoughts toward the
vocabulary manifold ("R1") improves accuracy + interpretability over unconstrained CODI, and does
so because of the *vocabulary-space direction* (must beat an "L2" control). Substrate is CODI
(single-stage self-distillation, arXiv:2502.21074). Primary model rung = **`olmo3_370M`**.

## 2. Repo / branch / environment
- Repo: `github.com/edu-llm/OLMo-core`. Branch **`latent-cot-superposition-amy`** off `main` @ `d663bae`.
- **Workflow rules (important):** all new code is namespaced under `latentcot`; design docs live in
  `docs/latent-cot/` (tracked); generated data + run/eval outputs are gitignored under
  `data/`/`runs/`; scratch analysis scripts live untracked in `local/`; push ONLY to our branch,
  never `main`; confirm before pushing.
- **Dev env is CPU-only (no GPU).** Python via **`.venv/bin/python`** (uv-managed, 3.12). Dev deps
  `pytest` + `tokenizers` installed via `uv pip install --python .venv/bin/python ...` (`.venv` untracked).
  `mypy` is **not** installed here, so `make type-check` has never run on this code.
- Style: line-length 100; `isort --profile black` + `black --target-version py312`. The venv's
  `ruff` is NEWER than the repo's pin and flags pre-existing `typing.Optional`/`Dict` usage
  repo-wide (an untouched core file reports 59) — don't chase those UP/C4 warnings. Repo
  convention: `typing.List/Optional`.

## 3. Files (all under `src/`)
Package `src/olmo_core/latentcot/`:
- `data/graph_gen.py` — layered directed-graph reachability generator; `Example`, `generate(...)`, BFS helpers.
- `data/encode.py` — `encode_example(ex, K)` → teacher/student/direct token views + boolean `label_mask` + positions; also `render_messages` / `to_sft_record` (platform dataset rows).
- `data/dataset.py` — `LatentCotDataset`, `collate`, `codi_collate` (→ `{"examples": [...]}`).
- `tokens.py` — dolma2 vocab; control tokens `<bot>/<eot>/<distill>/<thought>` at padded ids 100348–100351; `load_tokenizer`/`encode`/`decode`.
- `cot.py` — `embed_tokens`, `final_norm`, `run_continuous_thoughts`, `student_forward`.
- `loss.py` — `codi_loss` (teacher+student CE + distillation + vocab reg), `vocab_manifold_reg` (R1/R2/L2/none), `explicit_cot_loss`, `no_cot_loss`, `arm_loss` dispatcher.
- `train_module.py` — `CodiTransformerTrainModule(TransformerTrainModule)` + config. **Unused by Phase 8** (the direct loop is what runs); kept as the framework-native path.
- `arms.py` — 5 arms (A0–A4), `DEFAULT_K = 10`, `build_arm_config`, `assert_arms_differ_only_in`, `ARM_WHITELIST`.
- `evaluate.py` — `predict_reachable`, `greedy_generate`, `solve_rate_by_depth`, `gate_a_curve`+`linear_slope`, `overall_accuracy`, `mean_decodability`, `inference_token_cost`, `run_eval`.
- `probes.py` — `logit_lens`, `decodability`, `superposition_mass`, `linear_probe_accuracy` (+ shuffled control), `causal_ablation_margin_change`.
- `preflight.py` — `checkpoint_fingerprint`, `assert_same_base_checkpoint`, `assert_disjoint_seeds`, `per_arm_compute`, `preflight`.
- `train_driver.py` — `resolve_device`, `build_model`, `load_checkpoint`, `iter_batches`, `train_arm`, `configure_precision`, `autocast_ctx`, `PRECISIONS`.

Scripts `src/scripts/latentcot/`: `gen_graph_data.py`, `train_codi.py`, `eval.py`, `compare_models.py`,
`verify_checkpoint.py`, `preflight.py`, `publish_dataset.py` (+ `README.md`).
Tests `src/test/latentcot/`: `test_graph_gen`, `test_encode`, `test_cot`, `test_codi`, `test_arms`,
`test_evaluate`, `test_probes`, `test_preflight`, `test_train_driver`, `test_publish_shape` —
**140 tests, all pass** (`.venv/bin/python -m pytest -q src/test/latentcot/`, ~75 s).

**One core-file edit** (the only change outside `latentcot`): `src/olmo_core/nn/transformer/model.py`
added an additive `return_hidden_states: bool = False` kwarg to `Transformer.forward` (+6 lines;
default off = no behavior change; semantically identical to the pre-existing `lm_head is None`
branch). It returns post-block hidden states, used by the continuous-thought loop. Core never
imports `latentcot`; the dependency runs one way only.

## 4. Key design decisions / how it fits together
- Tokenization uses the real **dolma2** vocab so the 370M embeddings apply; only 4 control tokens
  are new (unused padded ids, no embedding resize).
- Two structurally parallel views share the `<distill>`+answer suffix: **teacher** `q <bot> cot <eot> <distill> ans`,
  **student** `q <bot> THOUGHT*K <eot> <distill> ans`. The `<distill>` token's hidden state is what CODI aligns
  (teacher detached → student) across all layers, captured via forward hooks.
- Continuous thoughts are produced by forwarding with `input_embeddings=` (bypasses the lookup) +
  `return_hidden_states=True`, then **passed through `cot.final_norm`** (the LM head's own norm)
  before being fed back — see gotchas.
- **All five arms are fine-tunes of the same S3 base checkpoint**, so the LR follows WSD
  (`--warmup-steps 200` + 10% decay tail), identical across arms, living in the shared loop.
- **K = 10** for every latent arm (≥ the deepest graph, depth 8, with ~2 steps of headroom).
- Arms share one base config and differ ONLY in the whitelist
  `(arm_mode, num_continuous_thoughts, vocab_reg, vocab_reg_weight, vocab_reg_entropy_floor)`.
- **Precision:** `--precision bf16` (default) = bf16 autocast + TF32 on CUDA; `fp32` is
  bit-identical to the pre-flag driver. Distill smooth-L1 and R1's softmax are pinned to fp32.
- Peak LR is **screened, not assumed** (runbook §3) — fine-tuning wants less than 3e-4.

## 5. Gotchas already discovered (don't re-hit these)
- **`LMOutputWithLoss.ce_loss` is detached** ("logging only"). Optimize `.loss`; log `.ce_loss`. There's a
  grad-flow regression test guarding this.
- The deep K-step continuous-thought graph needs **gradient clipping** (`max_grad_norm`) or it destabilizes.
- To verify grad flows through thoughts, check `embeds.grad` at the thought positions — the returned
  `thoughts` tensor is a *parallel* `cat`, so its `.grad` reads zero even when grad is flowing.
- **`return_hidden_states=True` gives the PRE-final-norm residual stream** (this model's final norm
  lives inside `LMHead`). Fed back raw, thought magnitude compounds every step: 5.8 → **52** by
  K=10 on this rung, vs an embedding RMS of 1.0, and training amplified it further. `final_norm`
  fixes it (flat 1.000). This also removed a real confound — R1/L2 incidentally suppress that drift
  while unregularized A2 does not. Watch `thought_rms` in `train_history`; it should stay ≈1 and flat.
- CODI students are processed **per example (batch dim 1)**, so `--batch-size` is a gradient-noise
  knob, **not** a throughput knob: per-example step time is flat from batch 2 up while total step
  time grows linearly. Throughput would come from packing (`cu_doc_lens` is already public on
  `Attention.forward` — no core change needed), not from the batch flag.
- Core's `KVCacheManager` **cannot** speed up the training thought loop (it writes in-place into
  registered buffers = an inference cache; CODI needs grads through all K steps). For *eval* it is
  blocked differently: no `Transformer`-level API, and `TorchAttentionBackend` raises "doesn't
  support KV caching", so the CPU dev box can't exercise it. Both deferred, documented in the PRD.
- `--rung` on `train_codi.py` still defaults to `olmo2_370M` while `eval.py`/`compare_models.py`/
  `verify_checkpoint.py` default to `olmo3_370M`. **Always pass `--rung olmo3_370M` explicitly**
  (the runbook does); a forgotten flag trains one architecture and evaluates as another.
- The unit fixtures use 2-layer models and K=2 — that regime is too shallow to reveal
  magnitude/scale bugs in the latent path. `test_cot.py` has an 8-layer/K=10 guard for that reason.

## 6. What's left — the Phase 8 GPU runs (code done, execution pending)
Phases 1–8 are implemented + unit-tested on tiny CPU models. Remaining work:
1. **Run the 1-seed pilot** (runbook §3): A0–A4, one seed, 5,000 steps, forking the S3 base.
2. **Gates + probes** (runbook §4): `eval.py` for Gate A (slope of A2−A0 vs depth) and Gate B
   (A3 > A2 and A3 > A4 on accuracy + decodability), then `compare_models.py` for the head-to-head.
3. **Escalate** per PRD §3.1 (`--rung olmo3_600M/760M/1B`) only if 370M is under-powered.

**Honest status of the 1-seed decision.** The pre-registered Gate A/B criteria are **paired-seed
95% CIs**; a single seed provides no seed-level variance, so this pilot yields *point estimates
only* and cannot pass or fail either gate as written. Treat it as a screen answering "is there any
depth-increasing signal, and does the harness run clean at 370M?" Within-run item bootstrap CIs
over the 960 held-out examples are still legitimate and worth reporting, but they capture test-item
noise, not init/data-order variance. The confirmatory sweep still needs ≥3 seeds (PRD §5/§11), so
budget for a 3–5× larger campaign once the pilot looks promising. Any claim published off 1 seed
would be a pre-registration violation — say "pilot" in anything written from it.

## 7. Commit trail (branch, oldest → newest)
`40b0ede` scaffold · `16bf3e4` Phase1 (+`e6fc31a` confound fix) · `6bd6704` Phase2 · `b86a061` Phase3 ·
`97221ae` Phase4 · `7fcf039` Phase5 · `1736251` Phase6 · `445f030` Phase7 · `04b22ed`+`0282221` platform
dataset shape · `9e776b0` Phase8 driver · `7e53aa9` docs/data reorg · `2b64944` compare_models ·
`3ce170e` GPU/device + smoke test · `c363732` R1 entropy floor · `3998f9a` PRD status ·
`30fcf20` WSD warmup + K=10 · `0df02ee` style · `4356569` PRD TL;DR · `13b0175` checkpointing +
val split · `1714591` thought normalization · `0812cd9` bf16/TF32 precision.
