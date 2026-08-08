# Latent-CoT Superposition — Build Progress

**Log every change here.** Newest entries at the bottom. Keep each one short.

Full spec: `latent-cot-superposition-prd.md`. Isolation contract + port plan: `module-prd.md`.
Current branch: **`latent-superposition-module`** (off `main` @ `08df5aa0`). Push to our branch
only; `main` untouched. Generated data lives untracked in `data/latentcot/`.
Primary model rung: `olmo3_370M` (`olmo2_370M` for the earlier phases below).

Phases 1–8 were built on the predecessor branch `latent-cot-superposition-amy` (off `main` @
`d663bae`); see the 2026-08-08 entry at the bottom for how they arrived here and what changed.

## Setup
- Pulled `main` (fast-forward to `d663bae`). Upstream refactor removed `src/edullm/`; the local `ssh.py` timeout tweak was saved to `local/ssh-timeout.patch`.
- Installed dev deps into `.venv` via `uv`: `pytest`, `tokenizers` (not tracked).
- All new code namespaced under `src/olmo_core/latentcot/`, `src/scripts/latentcot/`, `src/test/latentcot/`.

## Phase 1 — Synthetic graph-reachability data ✅ (`16bf3e4`, fix `e6fc31a`)
- `data/graph_gen.py`: layered directed-graph generator; every edge advances one level so reachable distance == `depth` (no shortcuts). BFS `frontiers` double as the teacher reasoning trace and the superposition probing labels. Non-trivial negatives (target isolated).
- `scripts/latentcot/gen_graph_data.py`: builds balanced train/test JSONL to `data/latentcot/` (gitignored), independent-BFS verification, depth histogram, train/test disjointness (OOD depths 5,8 in test).
- **Confound fix (`e6fc31a`):** negatives now expand to the *same* frontier depth as positives, so the label can't be read off frontier depth.
- Tests: `test_graph_gen.py` (71).

## Phase 2 — Tokenization & token views ✅ (`6bd6704`)
- `tokens.py`: dolma2 vocab + 4 control tokens (`<bot>/<eot>/<distill>/<thought>`) at unused padded ids 100348–351 (no resize). *(Deviation: 4 not 3 — `<thought>` is the student latent-slot placeholder.)*
- `data/encode.py`: renders query + BFS-CoT + yes/no; parallel teacher (`q <bot> cot <eot> <distill> ans`) and student (`q <bot> THOUGHT*K <eot> <distill> ans`) views sharing the `<distill>`+answer suffix. Framework-native boolean `label_mask` (student→answer only; teacher→cot+answer).
- `data/dataset.py`: `LatentCotDataset` + `collate` (right-pad; mask padded False).
- Tests: `test_encode.py` (7).

## Phase 3 — Continuous-thought forward path ✅ (`b86a061`)
- `nn/transformer/model.py` (core, +9 lines, additive): `return_hidden_states: bool=False` on `Transformer.forward` returns post-block hidden states. Default off → no behavior change (existing transformer tests still pass).
- `latentcot/cot.py`: `embed_tokens`, `run_continuous_thoughts` (feed last hidden state back as next input embedding for K steps via `input_embeddings`), `student_forward`.
- Tests: `test_cot.py` (8) — K∈{1,2,4} shapes + gradient flows through the thought chain on a tiny CPU model.

## Phase 4 — CODI train module ✅ (`97221ae`)
- `latentcot/loss.py`: `codi_loss` (per example): teacher branch (explicit CoT) + student branch (continuous thoughts) + smooth-L1 distillation of the `<distill>` hidden state across all layers (teacher detached→student, via forward hooks) + `vocab_manifold_reg` (R1 = pull toward `E·softmax(logit-lens)` + entropy floor; R2 nearest-embedding; L2 control; none).
- `latentcot/train_module.py`: `CodiTransformerTrainModule` + config subclass; per-batch loss = `codi_loss`; inherits optimizer/scheduler/grad-clip/checkpointing/metrics.
- **Key fix:** `LMOutputWithLoss.ce_loss` is *detached* (logging only) → must optimize `.loss`. (Regression test guards it.)
- **Simplification:** students processed per-example (batch dim 1) to avoid the variable-length-prefix problem; batched/bucketed is a Phase-5 optimization. Literal 370M run is a GPU task (Phase 8); mechanism validated on a tiny CPU model.
- Tests: `test_codi.py` (4) — grad-flow guard, `ce_student` 11.7→<2 in 150 steps, reg variants, config build. Suite total: **90 tests**.

## Phase 5 — Arms + confound assertion ✅ (`7fcf039`)
- `arms.py`: 5 arms (A0 explicit-CoT, A1 no-CoT, A2 CODI, A3 CODI+R1, A4 CODI+L2 control) sharing one base config, differing only in the whitelist `(arm_mode, num_continuous_thoughts, vocab_reg, vocab_reg_weight)`. `assert_arms_differ_only_in()` fails on any out-of-whitelist confound (e.g. LR/seed).
- `loss.py`: `arm_loss` dispatcher → `explicit_cot_loss` / `no_cot_loss` / `codi_loss`.
- `encode.py`: added the direct (no-CoT) view `question <distill> answer` (A1).
- `train_module.py`: `arm_mode` field + `train_batch` dispatches via `arm_loss`.
- `dataset.py`: `codi_collate` → `{"examples": [...]}`.
- **Deviation:** `arm_mode` added to the whitelist (A0/A1 are structurally different objectives). Full 370M Trainer runs are Phase 8 (GPU).
- Tests: `test_arms.py` (10) — all arms present, confound assertion pass/fail, collate, unknown-mode error, each of A0–A4 reduces its primary CE. Suite total: **100 tests**.

## Phase 6 — Eval + probing harness ✅ (`1736251`)
- `probes.py`: `logit_lens`, `decodability`, `superposition_mass`, `linear_probe_accuracy` (+ shuffled control), `causal_ablation_margin_change`.
- `evaluate.py`: per-arm answer prediction (codi decodes at `<distill>`; no_cot from `question <distill>`; explicit_cot greedy-gens to `<distill>`), `solve_rate_by_depth`, `gate_a_curve`+`linear_slope`, `overall_accuracy`, `mean_decodability`, `inference_token_cost`, and `run_eval` (report: per-arm acc/solve-rate/decodability, gate A = A2−A0 curve+slope, gate B = A2/A3/A4).
- `scripts/latentcot/eval.py`: loads per-arm checkpoints, runs held-out set, writes `report.json` + tables + optional gate-A plot.
- Tests: `test_probes.py` (6) + `test_evaluate.py` (6). Linear probe beats shuffled control (1.00 vs ~0.4). Suite total: **112 tests**. (Real gate plots/CIs come from Phase 8 on trained 370M arms.)

## Phase 7 — Matched-budget dry-run + integrity checks ✅ (`445f030`)
- `preflight.py`: `checkpoint_fingerprint`, `assert_same_base_checkpoint`, `assert_disjoint_seeds`, `per_arm_compute` (forward-token cost with the K passes **counted**), and `preflight()` (matched-config via `assert_arms_differ_only_in` + checkpoint + seed checks → report; raises on first failure).
- `scripts/latentcot/preflight.py`: builds arm configs from one base, loads problems, runs preflight, prints report.
- **Interpretation:** match the confound-relevant budget (config outside whitelist, base checkpoint, problem seeds) and *report* per-arm FLOPs with K counted — don't equalize raw training FLOPs (arms use different token views); fairness is at matched *inference* compute in eval.
- Tests: `test_preflight.py` (6). Suite total: **118 tests**.

## Dataset platform-compliance (eduLLM) ✅ (`04b22ed`)
- Per the `edullm-dataset-design` + `edullm-datasets` skills, a custom dataset must publish through the platform validator. Only 5 profiles are registered in `edullm-data` v0.2.0 (`pretrain-tokens`, `sft-conversations`, `eval-results`, `token-order`, `tokenizer`) — `eval-items/v1` is NOT one — so we reshape into **`sft-conversations/v1`**. Dataset id **`sft/graph-reachability-depth`**.
- `encode.py`: `render_messages` (user=query+edges, assistant=BFS reasoning+yes/no) + `to_sft_record` (Example fields + `messages[]`).
- `gen_graph_data.py`: emits the compliant layout `conversations/{train,heldout}-00000.jsonl` under `data/latentcot/graph-reachability-depth/` (publish source = only the group dir; meta.json written outside it); asserts 0 train/heldout leakage via the validator's dedup key.
- `scripts/latentcot/publish_dataset.py`: the `publish()` call (group_meta: record_schema, partitions, dedup, leakage). Needs `edullm-data` + AWS creds (sb-aws) — run where creds exist.
- `preflight.py`/`eval.py` default paths updated to the new layout. Tests: `test_publish_shape.py` (4). Suite total: **122**.
- `edullm-data` installed into `.venv` only (not in pyproject); no module/test imports it — only `publish_dataset.py` at publish time.

## Phase 8 — Training driver ✅ code (`9e776b0`); the 370M runs are pending GPU
- `train_driver.py` (library, tested): `build_model` (deterministic init shared across arms via `init_seed` = the base), `iter_batches`, `train_arm` (direct loop reusing `arm_loss` + AdamW + grad clip).
- `scripts/latentcot/train_codi.py`: per-arm driver → build at a rung, optional shared init ckpt, train, save state_dict + end-of-run heldout eval. GPU auto-detected.
- `eval.py`: `load_checkpoint` reads a plain `.pt` (from the driver) or a framework dir.
- Tests: `test_train_driver.py` (3) — iter_batches shapes, `train_arm` reduces loss, drives all arm modes. Suite total: **126**.
- **Turnkey procedure:** `phase8-runbook.md` (dataset → preflight → train A0–A4 × seeds → eval gates A/B + probes → escalation).
- **NOT run here (CPU-only):** the actual seeded 370M training + gate CIs. Direct loop (not the framework Trainer) because the CODI student is per-example (variable-length prefix); same `arm_loss`, so behavior matches.

---

## 2026-08-08 — Ported to `latent-superposition-module`; zero shared-code edits

Squash-merged all of phases 1–8 from `latent-cot-superposition-amy` onto a branch cut from
`main` @ `08df5aa0`, then hardened the isolation. Squashed rather than merged so the 1.6 MB of
`local/` files pushed in `e0dff82a` never becomes an ancestor here. Authorship of the
platform-image and lint fixes is philote-dev's; credited in the commit message.

**The one core edit is gone.** `Transformer.forward`'s `return_hidden_states` kwarg is replaced by
`cot._capture_last_block`, a forward hook on the last block, plus `logits_to_keep=1` to shrink the
LM head we can no longer skip to a single position. The captured tensor is the same one the kwarg
returned (`forward` assigns `h = block(h, ...)` and the head is the next statement), so this is
numerically identical and **checkpoints from the live pilot stay valid**. `git diff origin/main --
src/olmo_core/nn src/olmo_core/train src/olmo_core/data src/olmo_core/optim
src/olmo_core/distributed` is now empty. Motive: a dozen workstreams are editing
`nn/transformer/` concurrently and `forward` is the worst place to hold a line.

**Two CI failures found and fixed** — both would have gone red, and the second class of thing has
blocked the image build before:
- `make style-check`: `train_driver.py`'s `Protocol` stubs were formatted by black ≥24 (compact
  `def f() -> int: ...`), which the pinned `black>=23.1,<24.0` rejects. Reformatted to the pinned
  style. Note the local `.venv` has black 26.5.1 and ruff 0.16.1, both far newer than the pins —
  run the pinned versions (`uvx black@23.12.1`, `uvx ruff@0.15.22`) or the answers are noise.
- `make type-check`: 12 mypy errors, all in latentcot. Fixed at the root: `Arm.vocab_reg` and
  `CodiTransformerTrainModuleConfig.vocab_reg` are now `loss.VocabReg` (the `Literal` `arm_loss`
  accepts, so an arm-table typo is a type error); `run_eval` takes `Dict[str, Any]`; `train_arm`'s
  `save_dir` is `Optional[Union[str, Path]]` because *rejecting* an `s3://` string is its job, with
  the Path-narrowed form now a separate `save_path` local; `Example.from_dict` builds real
  `(int, int)` edges; two tests assert their loop ran before indexing `Optional` metrics.
  mypy was not installed in `.venv` — added (pyproject already declared it).

**Not carried over:** `.mcp.json` (personal `sb-aws-creds` tooling; stays untracked).
**Carried, and shared:** `.edullm/Dockerfile` (+70 lines, append-only: flash-attn wheel,
`tokenizers`, baked dolma2 cache) — required, since every `olmo3_*` config asserts flash-2 at
construction and `main` still has neither. Worth its own PR to `main`.

Verified: 146 latentcot tests pass; 61 core transformer tests pass / 42 GPU-skipped (confirming
`model.py` untouched); pinned isort, black, ruff and mypy all clean.

Still open (unchanged by this port): the pilot `run_019fdf83`'s outcome, the unscreened peak LR,
n=1 seeds, and the `local/` blobs still reachable from `e0dff82a` on the two old pushed branches.

## 2026-08-08 — W&B tracking (new: `tracking.py`)

Nothing was reaching W&B before this. The Phase-8 driver is a direct loop, not the framework
`Trainer`, so it never builds a callback list and `WandBCallback` — how everything else in this
repo reaches W&B — was simply not in play. Metrics existed only as stdout and an end-of-run
`metrics.json`.

- **`latentcot/tracking.py`** (new): `resolve_project()` + `ArmTracker`. Follows the platform's
  convention from `.edullm/train_on_corpus.py` rather than a new one — enable iff
  `EDULLM_WANDB_PROJECT` (or `--wandb-project`) names a project, let the wandb client read
  `WANDB_RUN_GROUP` itself (so `group=` is *not* passed), default `WANDB_INIT_TIMEOUT=60`.
- **`train_arm(on_log=...)`**: a plain callable invoked with each `train_history` entry. The
  training core stays free of any metrics dependency and unit-testable; `tracking.py` is the only
  file that imports `wandb`.
- **`train_codi.py`**: starts the tracker *before* training (so a run that dies mid-way keeps its
  curve), streams per-step metrics, writes the gate numbers to the run summary — including
  `solve_rate/depth_N` flattened to scalars, since gate A is a slope over depth and a nested dict
  can't be compared across arms — and on an exception marks the run failed with the reason on it.
  Run name is `<arm>-seed<n>`; config carries every arm-defining field.
- **Fail-open throughout, and this is the point:** missing `wandb`, unset `WANDB_API_KEY`, failed
  `init`, failed `log`, even an `on_log` that raises — each degrades to one stderr line and an
  untracked run. A metrics sidecar must never cost a day of A100 time. 14 new tests in
  `test_tracking.py` assert exactly these paths; suite total **160**.
- **`metrics.json` now records `{"wandb": {active, url, reason}}`**, because `run.yaml` redirects
  each arm's stdout to a `train.log` that does not survive (see below), so otherwise there'd be no
  durable answer to "was this tracked?".
- **`.edullm/run.yaml`**: `--wandb-project latent-cot-superposition` written into the command, so
  tracking does not depend on remembering a submit-time flag. `WANDB_API_KEY` still arrives with
  the submission and cannot live in this file — a key committed here would be a secret in an image
  built from the commit — so `edullm submit --wandb-project latent-cot-superposition` remains the
  right way to launch.

**Found while doing this, pre-existing, NOT fixed (W&B section):** `run.yaml` does
`mkdir -p "$EDULLM_CHECKPOINT_DIR/A$i"` and `> "$EDULLM_CHECKPOINT_DIR/A$i/train.log"` where that
variable is an `s3://` URI. Those are *shell* operations, so they hit a local relative path
(`s3:/bucket/...`) that dies with the container — the same `Path()`-mangling trap the file documents
for Python, one layer out where no Python guard can catch it. The per-arm logs are therefore not
durable. Verified end-to-end against a fake `wandb`: config, per-step logs, group-from-env, summary,
`finish(0)`, and both untracked paths completing normally.

## 2026-08-08 — MoE base support (new: `moe.py`)

The pretrained checkpoint the arms fork will be a **Mixture-of-Experts** model, not a dense one.
The continuous-thought path itself needed no change — `MoETransformerBlock.forward` returns a plain
tensor, so `cot._capture_last_block` reads it exactly as it reads a dense block. What did need
changing is everything the framework `Trainer` does for an MoE run and the Phase-8 direct loop
never did.

**The real bug: the MoE auxiliary losses were an arm-dependent confound.** Each router computes a
load-balancing loss per forward and welds it to the activation with `attach_auxiliary_loss`, whose
backward hands the aux loss gradient `1.0` *regardless of how the main loss was scaled*. Each
forward also normalizes by its own token count, so the aux pressure per step scales with the
**number of forwards** — and `codi_loss`'s `/n` over examples does not touch it. Forwards per
example: **1** for A0/A1, **K+2 = 12** for A2/A3/A4. So the router would be pushed ~12× harder in
the latent arms than in the baselines — on exactly the `acc(A2) − acc(A0)` comparison Gate A is
defined on. Same species as the pre-run thought-norm drift, and it would have been just as
invisible.

- **`latentcot/moe.py`** (new): `is_moe_model`, `count_forwards`, `normalized_aux_losses`,
  `reset_router_state`, `finish_step`, `collect_router_metrics`, `describe_moe`. Every one is a
  no-op on a dense model, so nothing changes for the dense rungs.
- **`normalized_aux_losses(model, n)`** divides each router's `lb_loss_weight`/`z_loss_weight` by
  the step's forward count for the duration of the step, restoring them in a `finally` so a failed
  step cannot leave the model detuned. `None` weights stay `None` — off must stay off.
- **`post_batch()` is now called** each step (the `bias_gamma` score-bias update, i.e. aux-loss-free
  balancing). Nothing in the loop called it, so that mechanism was silently inert.
- **Router metrics reset per step and land in `train_history`** under `moe/…`, so they stream to
  W&B. A fine-tune can quietly collapse the routing and these are the series that show it. Per-block
  series are dropped; only the totals are logged.
- **`CodiTransformerTrainModule.train_batch` had the same gap** — it overrides the parent wholesale
  (the CODI loss is per-example, not a token-array microbatch), so it inherited none of the parent's
  MoE bookkeeping. Now repeated there too.
- `metrics.json` and the W&B config record `describe_moe(model)` — expert count, top-k, aux weights
  as actually built. The arms fork a checkpoint whose MoE shape comes from *its* config, so "what
  did we load" should be on the record rather than reconstructed later.

**A wrong design, caught by a test.** The first attempt threaded `loss_div_factor` through every
forward, which is the obvious-looking lever. It is the wrong one: `LMHead` passes it to
`_finalize_loss`, so it divides the **cross-entropy** too — measured on a dense 2-layer model,
`loss_div_factor=1234` moved the loss from 11.67 to 0.00945, exactly 1/1234, silently rescaling the
LM objective and the effective LR. That threading was reverted; scaling the router weights is the
only knob that moves the aux term alone. A GPU test now pins that CE is unchanged by the
correction.

**MoE needs CUDA, so MoE tests are GPU-only.** Every MoE path here routes through
`olmo_core.kernels.moe`, which is Triton — `import triton` fails outright on macOS, and both the
dropless and capacity-factor paths assert `kernels is not None`. This repo's own MoE tests are all
`@requires_gpu`; ours follow. 21 new tests in `test_moe.py`: 12 run on CPU (the forward-count
arithmetic, the weight scale/restore including the raising case, the dense no-ops), 9 are
`@requires_gpu` and cover a real MoE — thoughts on MoE blocks, all five arms training, router
metrics reaching the history, the aux loss actually shrinking, and CE not moving. Suite total
**174 passed, 9 skipped** on CPU.

**Not verified, and cannot be here:** the GPU-marked tests have never executed — this machine has
no CUDA and cannot install triton. They need one GPU run before the MoE pilot is trusted. Also
still open: `--rung` must name an MoE factory that takes `(vocab_size, **kwargs)` (e.g.
`olmoe_1B_7B`); `llama_like_moe` needs explicit expert args and cannot be reached through that flag
as written. Exact MoE hyperparameters are pending from the user.

## 2026-08-08 — Reframed as post-training techniques (new: `techniques.py`)

The experiment is no longer the deliverable; the **final model** is. The five arms are now a catalog
of **selectable post-training techniques**, so that when the results say which one wins, naming it
is a flag rather than a code change. The experiment path (`--arm A0..A4`) is kept intact so the
study's runs stay reproducible.

**Where this sits:** post-training, SFT stage. Every technique starts from a pretrained checkpoint.

- **`latentcot/techniques.py`** (new): `TECHNIQUES` keyed by readable name, `get_technique(name,
  **overrides)`, `as_arm()` (so selection changes nothing downstream), `describe_techniques()` for
  `--list-techniques`. Seven entries — the five arms plus the two combinations that were implemented
  and wired to no arm: `codi-r2` (nearest-embedding reg) and `codi-r1-entropy` (R1's anti-collapse
  entropy floor). A test pins that the five arm-derived techniques still mean exactly what those
  arms meant, so the study's results keep applying.
- **`train_codi.py`**: `--technique` (mutually exclusive with `--arm`), `--list-techniques`, and the
  run label / `metrics.json` / W&B config now record which technique was used.

**`Technique.needs_cot_data` is not decoration.** CODI trains a teacher branch on the written-out
chain of thought and aligns the latent thoughts to its `<distill>` hidden state, so every `codi-*`
technique needs examples carrying *both* views. On post-training data with no reasoning traces the
distillation term has nothing to distill from and the latent techniques degrade to a slower
`no-cot`. Only `no-cot` runs on ordinary SFT data. **This is a data requirement for the final
model**, and the synthetic BFS traces the study used will not be there.

### Loading an arbitrary pretrained model (the gap that blocked "yes")

`build_model` can only name a registered `TransformerConfig` factory and hardcodes this project's
vocab size. Loading is `strict=True`, so the built architecture has to match the weights exactly —
fine for the study's own rungs, useless for an arbitrary pretrained MoE whose expert shape lives in
its own config, and `--rung` cannot reach `llama_like_moe` at all.

- **`read_model_config(path)`** reads the `model` key out of the `config.json` that
  `ConfigSaverCallback` writes beside every checkpoint. Probes the step dir, one level up, and the
  parent of a `.pt` file.
- **`build_model_from_config(cfg)`** rebuilds from it. Verified: `TransformerConfig.from_dict` round-
  trips exactly, including MoE configs, which rebuild as `MoETransformer` with their expert
  parameters and strict-load cleanly.
- `train_codi.py` prefers the checkpoint's config over `--rung` automatically, and says so.
- **`tokens.assert_control_tokens_fit(model)`** runs after the load. The four control tokens sit at
  `padded_vocab_size - 1..4`, safe only because dolma2 pads 100278 real tokens to 100352 and leaves
  those rows unused. A checkpoint with a smaller vocab would index off the end of the embedding
  mid-training; now it fails at load, naming the assumption.

Verified end-to-end on CPU: a fixture checkpoint (weights + `config.json`) with `--rung` deliberately
naming a factory that does not exist trains under `--technique codi-r1`, building from the config,
strict-loading, and recording `built_from_checkpoint_config: true`.

**Still not confirmable from here:** whether a real pretrained MoE loads. Two unknowns remain — an
expert-parallel-sharded checkpoint relies on DCP resharding into a single-GPU non-EP model, and the
tokenizer must actually be dolma2 (confirmed as the intent; the guard now enforces the vocab size).
`verify_checkpoint.py` settles both on real hardware in about a minute and should be the first thing
run against the real checkpoint. Suite: **207 passed, 9 skipped** (the 9 are the GPU MoE tests,
still never executed).
