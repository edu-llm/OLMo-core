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
