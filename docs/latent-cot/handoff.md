# Handoff: Latent-CoT Superposition experiment (branch `latent-cot-superposition-amy`)

Context brief for an AI agent taking over this work. Read this + `latent-cot-superposition-prd.md`
(the full spec) + `progress.md` (per-phase changelog). Everything below is implemented,
unit-tested, and pushed to the branch.

## 1. What we're building (one paragraph)
A research harness in the OLMo-core repo to test **latent chain-of-thought** on a small model.
The model reasons in continuous "thoughts" (Coconut/CODI style) instead of emitting text, and we
test two claims on a synthetic **directed-graph reachability** task: **(gate A) superposition** —
the continuous-reasoning accuracy advantage over explicit CoT grows with graph depth (theory:
arXiv:2505.12514); **(gate B) the novel fix** — regularizing the continuous thoughts toward the
vocabulary manifold ("R1") improves accuracy + interpretability over unconstrained CODI, and does
so because of the *vocabulary-space direction* (must beat an "L2" control). Substrate is CODI
(single-stage self-distillation, arXiv:2502.21074). Primary model = `olmo2_370M`.

## 2. Repo / branch / environment
- Repo: `github.com/edu-llm/OLMo-core`. Branch **`latent-cot-superposition-amy`** off `main` @ `d663bae`.
- **Workflow rules (important):** all new code is namespaced under `latentcot`; generated data +
  design docs live in `docs/latent-cot/` (tracked); generated data + run/eval outputs are gitignored under `data/`/`runs/`; push ONLY to our branch, never `main`; confirm before pushing.
- **Env is CPU-only (no GPU).** Python via **`.venv/bin/python`** (uv-managed, 3.12). Dev deps
  `pytest` + `tokenizers` were installed via `uv pip install --python .venv/bin/python ...` (`.venv` untracked).
- Style: line-length 100; run `uvx black --line-length 100 <dirs>`, `uvx isort <dirs>`,
  `uvx ruff check --select E4,E7,E9,F --ignore F403,F405,E501 <dirs>` (the repo's effective ruleset;
  the bare `uvx` ruff/black are NEWER/stricter than the repo's pinned versions — don't chase their
  extra UP/C4 warnings, and keep the one core file minimal). Repo convention: `typing.List/Optional`.

## 3. Files (all under `src/`)
Package `src/olmo_core/latentcot/`:
- `data/graph_gen.py` — layered directed-graph reachability generator; `Example`, `generate(...)`, BFS helpers.
- `data/encode.py` — `encode_example(ex, K)` → teacher/student/direct token views + boolean `label_mask` + positions; also `render_messages` / `to_sft_record` (platform dataset rows).
- `data/dataset.py` — `LatentCotDataset`, `collate`, `codi_collate` (→ `{"examples": [...]}`).
- `tokens.py` — dolma2 vocab; control tokens `<bot>/<eot>/<distill>/<thought>` at padded ids 100348–100351; `load_tokenizer`/`encode`/`decode`.
- `cot.py` — `embed_tokens`, `run_continuous_thoughts` (feed last hidden state back as next input embedding for K steps), `student_forward`.
- `loss.py` — `codi_loss` (teacher+student CE + distillation + vocab reg), `vocab_manifold_reg` (R1/R2/L2/none), `explicit_cot_loss`, `no_cot_loss`, `arm_loss` dispatcher.
- `train_module.py` — `CodiTransformerTrainModule(TransformerTrainModule)` + `CodiTransformerTrainModuleConfig`; `arm_mode` dispatch; per-batch loss = `arm_loss`.
- `arms.py` — 5 arms (A0–A4), `build_arm_config`, `assert_arms_differ_only_in`, `ARM_WHITELIST`.
- `evaluate.py` — `predict_reachable`, `solve_rate_by_depth`, `gate_a_curve`+`linear_slope`, `overall_accuracy`, `mean_decodability`, `inference_token_cost`, `run_eval`.
- `probes.py` — `logit_lens`, `decodability`, `superposition_mass`, `linear_probe_accuracy` (+ shuffled control), `causal_ablation_margin_change`.
- `preflight.py` — `checkpoint_fingerprint`, `assert_same_base_checkpoint`, `assert_disjoint_seeds`, `per_arm_compute`, `preflight`.

- `train_driver.py` — `build_model`, `iter_batches`, `train_arm` (Phase 8 driver core, tested).

Scripts `src/scripts/latentcot/`: `gen_graph_data.py`, `eval.py`, `preflight.py`, `publish_dataset.py`, `train_codi.py`.
Tests `src/test/latentcot/`: `test_graph_gen`, `test_encode`, `test_cot`, `test_codi`, `test_arms`, `test_evaluate`, `test_probes`, `test_preflight` — **118 tests, all pass**.

**One core-file edit** (the only change outside `latentcot`): `src/olmo_core/nn/transformer/model.py`
added an additive `return_hidden_states: bool = False` kwarg to `Transformer.forward` (+9 lines;
default off = no behavior change). It returns post-block hidden states, used by the continuous-thought loop.

## 4. Key design decisions / how it fits together
- Tokenization uses the real **dolma2** vocab so `olmo2_370M` embeddings apply; only 4 control tokens
  are new (unused padded ids, no embedding resize).
- Two structurally parallel views share the `<distill>`+answer suffix: **teacher** `q <bot> cot <eot> <distill> ans`,
  **student** `q <bot> THOUGHT*K <eot> <distill> ans`. The `<distill>` token's hidden state is what CODI aligns
  (teacher detached → student) across all layers, captured via forward hooks.
- Continuous thoughts are produced by forwarding with `input_embeddings=` (bypasses the lookup) +
  `return_hidden_states=True`, feeding the last-position hidden state back K times.
- Arms (A0 explicit-CoT anchor, A1 no-CoT anchor, A2 CODI, A3 CODI+R1, A4 CODI+L2 control) share one
  base config and differ ONLY in the whitelist `(arm_mode, num_continuous_thoughts, vocab_reg, vocab_reg_weight)`.

## 5. Gotchas already discovered (don't re-hit these)
- **`LMOutputWithLoss.ce_loss` is detached** ("logging only"). Optimize `.loss`; log `.ce_loss`. There's a
  grad-flow regression test guarding this.
- The deep K-step continuous-thought graph needs **gradient clipping** (`max_grad_norm`) or it destabilizes.
- To verify grad flows through thoughts, check `embeds.grad` at the thought positions — the returned
  `thoughts` tensor is a *parallel* `cat`, so its `.grad` reads zero even when grad is flowing.
- CODI students are processed **per example (batch dim 1)** to avoid the variable-length-prefix problem
  (no left-padding/attention mask). Batched/bucketed processing is a deferred optimization.

## 6. How to run (CPU)
```bash
.venv/bin/python -m pytest -q src/test/latentcot/            # 122 tests
# generate the platform-compliant dataset (sft-conversations/v1 layout):
.venv/bin/python src/scripts/latentcot/gen_graph_data.py     # -> data/latentcot/graph-reachability-depth/conversations/{train,heldout}-00000.jsonl
.venv/bin/python src/scripts/latentcot/preflight.py \
    --train-data data/latentcot/graph-reachability-depth/conversations/train-00000.jsonl \
    --test-data  data/latentcot/graph-reachability-depth/conversations/heldout-00000.jsonl   # pre-registration gate
# publish (needs edullm-data + AWS creds; run where creds exist):
.venv/bin/python src/scripts/latentcot/publish_dataset.py --source data/latentcot/graph-reachability-depth
```

## 6b. Platform dataset shape (eduLLM)
Datasets must publish through the `edullm-data` validator (skills: `edullm-dataset-design`,
`edullm-datasets`). We use **`sft/graph-reachability-depth`** via **`sft-conversations/v1`**
(the only registered JSONL-records profile — `eval-items/v1` is unregistered in v0.2.0). Rows
carry `messages[]` (validated) + all `Example` fields (metadata, consumed by our loader).
`edullm-data` is installed in `.venv` only (not in `pyproject`); only `publish_dataset.py` needs it.
Publishing writes to `s3://edullm-landing` → validator promotes to `s3://edullm-data`.

## 7. What's left — the Phase 8 GPU runs (code done, execution pending)
All phases 1–8 are **implemented + unit-tested on tiny CPU models** (126 tests). The only thing not
done is the actual **370M training on GPU** (this env is CPU-only). The driver + eval + runbook are ready:
1. `src/scripts/latentcot/train_codi.py` (core `olmo_core.latentcot.train_driver`) trains one arm at a rung
   via a direct `arm_loss` loop (matched init via `--init-seed`, per-run `--seed`), saves `model.pt` +
   `metrics.json`. Run A0–A4 × ≥3 (screen) then ≥5 (confirm) seeds.
2. `src/scripts/latentcot/eval.py` loads the per-arm `model.pt` → gate A (superposition slope vs depth),
   gate B (R1 vs L2 on acc + decodability), causal probes; aggregate paired CIs across seeds.
3. Apply the PRD §3.1 escalation ladder (`--rung olmo2_600M/760M/1B`) if 370M is under-powered.

**Turnkey commands + seed loop + escalation triggers: `phase8-runbook.md`.**

## 8. Commit trail (branch)
`40b0ede` scaffold · `16bf3e4` Phase1 (+`e6fc31a` confound fix) · `6bd6704` Phase2 · `b86a061` Phase3 ·
`97221ae` Phase4 · `7fcf039` Phase5 · `1736251` Phase6 · `445f030` Phase7.
