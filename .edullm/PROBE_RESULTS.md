# HPO probe results

Authoritative source: W&B project [`eduLLM/hpo-probe`](https://wandb.ai/eduLLM/hpo-probe).  
Local winner vectors (cross-checked): [`.edullm/final-validation-vectors.json`](final-validation-vectors.json).  
Probe contract background: [`.edullm/THREE_ARM_HPO.md`](THREE_ARM_HPO.md).  
Next step after these winners: [`.edullm/FINAL_VALIDATION.md`](FINAL_VALIDATION.md) (370M / ~10B).

RegMix pretraining probes and the **curriculum learning** probe are documented in separate sections below.

## Study setup (RegMix contract)

| Item | Value |
|------|--------|
| Dataset | `pretrain/regmix-10b` v1 (`regmix-10b`) |
| Search model | OLMo2 ~190M, exact fidelity |
| Aggregate search budget | ~2B tokens (`controller.budget_tokens`) |
| Target lineage horizon | ~500M tokens (`controller.target_tokens` = 500,039,680) |
| Quantum / first rung | 50,003,968 tokens |
| Scheduler | WSD (warmup / stable / decay) |
| Optimized fields | `lr`, `weight_decay`, `beta2_gap`, `eps`, `warmup_fraction`, `decay_fraction`, `terminal_lr_ratio`, `global_batch_mult`, `max_grad_norm` |
| Controller stack | Brainlift: FT-PFN + ifBO + IPBT/BTT |
| Held-out search metric | `eval/lm/regmix-10b-val/CE loss` (search-validation callback) |

Two finished **RegMix** arms are reported here:

1. **no-proxy** — stock 12-layer `olmo2_190M`, fully trainable, Centaur enabled (30% multi-action).
2. **no-centaur** — post-hoc exact arm (`hpo-no-centaur-exact.json`: `proxy_removed_after_failed_admission`). Same stock `olmo2_190M` / exact fidelity; Centaur disabled. W&B run name: `hpo-no_centaur-exact-v1-…`.

The u-μP / frozen-layer proxy arm (`full_acronym_soup`) is not a winner source for final validation; the proxy cohort recorded `reporting_only` after failed admission.

The **curriculum learning** winner (`curriculum_quadratic_mtld`) is documented in [Curriculum learning probe](#curriculum-learning-probe) below.

## Per-arm outcomes (RegMix)

| | **no-proxy** | **no-centaur** |
|--|--------------|----------------|
| W&B run ID | `904ea39d368dfe412048a6063c1600df` | `06e12699f744b8d2e562e78afa003b7f` |
| Display name | `hpo-no_proxy-aligned-v4-runpod-20260809-113617-no_proxy` | `hpo-no_centaur-exact-v1-runpod-20260809-132223-no_centaur` |
| State | finished | finished |
| Winner trial | `t9_0` | `t8_0` |
| Search-validation CE | **2.786524534225464** | **2.7904467582702637** |
| Trusted for selection | true | true |
| Trials spawned | 24 | 22 |
| Controller steps logged | 112 | 149 |
| Search tokens charged | 2,350,186,496 | 2,000,158,720 |
| Budget tokens (run config) | 2,350,186,496 | 2,000,158,720 |
| Exact retrain tokens | 0 | 0 |
| Total tokens charged | 2,350,186,496 | 2,000,158,720 |
| Accelerator seconds (search) | 49,058.11 | 57,840.46 |
| Total A100-hours (logged) | 13.627 | 16.067 |
| Winner checkpoint (RunPod path) | `…/no_proxy/aligned-v4/checkpoints/trials/t9_0/step30520` | `…/no_centaur/exact-v1/checkpoints/trials/t8_0/step30520` |
| Model parameterization | standard `olmo2_190M`, depth 12 | standard `olmo2_190M`, depth 12 |
| Fidelity | exact | exact |
| Centaur | on (arm policy) | off |
| Full-fidelity candidates (`top_five`) | **1** (`t9_0` only) | **1** (`t8_0` only) |

Notes:

- Committed arm JSON `hpo-no-proxy.json` lists `budget_tokens: 2000158720`; the finished **no-proxy** run used **2,350,186,496** (aligned-v4). Report the W&B value as what was charged.
- **no-centaur** matched the exact-arm budget of 2,000,158,720 tokens.
- `controller.top_candidates(5)` / `_persist_study_result` only keep trials that reached `target_tokens` (500,039,680). Both arms promoted a single lineage to that horizon (full-fidelity rescue), so `top_five_full_fidelity` is length 1 — not a truncated top-five dump.

## Winning hyperparameter vectors (RegMix)

Values below match both W&B `winner/hyperparameters/*` and `.edullm/final-validation-vectors.json`.

### no-proxy — trial `t9_0`

| Hyperparameter | Value |
|----------------|-------|
| `lr` | 0.0004125460019173203 |
| `weight_decay` | 0.01473432082609167 |
| `beta2_gap` | 0.0014689794923786166 |
| `eps` | 1.4798352708540092e-12 |
| `warmup_fraction` | 0.01131976840436488 |
| `decay_fraction` | 0.05 |
| `terminal_lr_ratio` | 0.021488995515927797 |
| `global_batch_mult` | 0.5 |
| `max_grad_norm` | 0.3 |

### no-centaur — trial `t8_0`

| Hyperparameter | Value |
|----------------|-------|
| `lr` | 0.00030060095254686933 |
| `weight_decay` | 0.01 |
| `beta2_gap` | 0.001 |
| `eps` | 1e-12 |
| `warmup_fraction` | 0.007007066567546487 |
| `decay_fraction` | 0.05 |
| `terminal_lr_ratio` | 0.0 |
| `global_batch_mult` | 0.5 |
| `max_grad_norm` | 0.34844106730841967 |

## Curriculum learning probe

**Training mode:** curriculum learning — Arm 9 quadratic warmup pacing (`arm9_warmup_quadratic_n10_token_fraction_v1`) over MTLD difficulty order on `curriculum/opt-with-synthetic-10b` v1, parent corpus `pretrain/opt-with-synthetic-10b` v1. Probe contract: [`.edullm/hpo-curriculum-quadratic-mtld.json`](hpo-curriculum-quadratic-mtld.json).

### Study setup (curriculum contract)

| Item | Value |
|------|--------|
| Arm | `curriculum_quadratic_mtld` |
| Training | **Curriculum learning** (quadratic warmup, MTLD token order) |
| Parent dataset | `pretrain/opt-with-synthetic-10b` v1 |
| Curriculum order | `curriculum/opt-with-synthetic-10b` v1 (`mtld` group) |
| Pacing | `arm9_warmup_quadratic_n10_token_fraction_v1` |
| Search model | OLMo2 ~190M, exact fidelity |
| Aggregate search budget | ~2.01B tokens (`controller.budget_tokens`) |
| Target lineage horizon | ~503M tokens (`controller.target_tokens` = 503,316,480) |
| Quantum / first rung | 50,331,648 tokens |
| Base global batch (search) | 524,288 tokens (512 Ki) |
| Scheduler | WSD (warmup / stable / decay) |
| Optimized fields | `lr`, `weight_decay`, `beta2_gap`, `eps`, `warmup_fraction`, `decay_fraction`, `terminal_lr_ratio`, `global_batch_mult`, `max_grad_norm` |
| Controller stack | Brainlift: FT-PFN + ifBO + IPBT/BTT |
| Held-out search metric | `eval/lm/opt-with-synthetic-10b-val/CE loss` |
| Centaur | on (30% multi-action) |

### Per-arm outcome (curriculum)

| | **curriculum_quadratic_mtld** |
|--|-------------------------------|
| W&B run ID | `7f74348409e054561b12348d0f5a815b` |
| State | finished |
| Winner trial | `t4_0` |
| Search-validation CE | **2.7663214206695557** |
| Trusted for selection | true |
| Search tokens charged | 2,013,265,920 |
| Total A100-hours (logged) | 10.47 |
| Study result (RunPod path) | `/workspace/olmo-artifacts/curriculum-quadratic-mtld-hpo/study-result.json` |
| Model parameterization | standard `olmo2_190M`, depth 12 |
| Fidelity | exact |
| Centaur | on (arm policy) |
| Full-fidelity candidates (`top_five`) | **1** (`t4_0` only) |

### Winning hyperparameter vector — curriculum learning, trial `t4_0`

Values match W&B `winner/hyperparameters/*` and `/workspace/olmo-artifacts/curriculum-quadratic-mtld-hpo/study-result.json`.

| Hyperparameter | Value |
|----------------|-------|
| `lr` | 0.0006550313203869464 |
| `weight_decay` | 0.011568982457596117 |
| `beta2_gap` | 0.021090019944083084 |
| `eps` | 1.4198780561886085e-07 |
| `warmup_fraction` | 0.07599507819381664 |
| `decay_fraction` | 0.1756549080475558 |
| `terminal_lr_ratio` | 0.027642727977629657 |
| `global_batch_mult` | 1.4515939717052118 |
| `max_grad_norm` | 1.2429510134280908 |

**370M validation (curriculum learning):** finished in W&B project [`eduLLM/hpo-validation`](https://wandb.ai/eduLLM/hpo-validation) as [`3576162a7edea5d8bebeafdf50053740`](https://wandb.ai/eduLLM/hpo-validation/runs/3576162a7edea5d8bebeafdf50053740) (`quadratic-mtld-370m-mb32k-v2`). Used the winner vector above on OLMo2-370M with the same curriculum identity; global batch overridden to **262,144** tokens/step (256 Ki) to align with other HPO validation runs. Budget: ~10B tokens (38,146 steps).

## Additional full-fidelity winners

**Both arms have only the single selected winner at full fidelity.** No second (or third…) full-fidelity candidate was recovered after searching W&B for either arm.

### Ranked full-fidelity set (complete)

| Rank | Arm | Trial | Search-val CE | Role |
|------|-----|-------|---------------|------|
| 1 | no-proxy | `t9_0` | 2.786524534225464 | selected winner |
| 1 | no-centaur | `t8_0` | 2.7904467582702637 | selected winner |
| 1 | curriculum_quadratic_mtld | `t4_0` | 2.7663214206695557 | selected winner (curriculum learning) |

Hyperparameters for each row are the primary winner vectors above (identical to `winner` and the sole `top_five_full_fidelity[]` entry).

### Sources checked (W&B `eduLLM/hpo-probe`)

| Source | no-proxy (`904ea39d…`) | no-centaur (`06e12699…`) |
|--------|------------------------|---------------------------|
| Run summary `top_five_full_fidelity` | length 1 → `t9_0` | length 1 → `t8_0` |
| Artifact `study-result` (`hpo-study-result`) | `study-result:v0` JSON, `top_five_full_fidelity` length 1 | `study-result:v1` JSON, length 1 |
| Artifact `controller-state` observations | Only `t9_0` has `tokens == 500039680` | Only `t8_0` has `tokens == 500039680` |
| Controller `final_evaluation` event | `t9_0` only | `t8_0` only |
| Run history / tables | No separate top-N payload (only `hpo/best_search_validation_ce`) | Same |
| RunPod `run.log` artifact | Mentions study-result / winner; no extra full-fidelity list | Same |
| Local RunPod paths | Not accessible from this laptop | Not accessible |

Code path that wrote the length-1 lists: `_persist_study_result` → `controller.top_candidates(5)` in `.edullm/hpo_on_corpus.py` / `src/olmo_core/hpo/controller.py`, mirrored by `HpoProbeSession.record_study_result`.

### Not full-fidelity (context only)

Highest **partial** fidelities from controller-state observations (search-val CE at each trial’s max tokens). These are **not** additional full-fidelity winners and were never eligible for `top_five_full_fidelity`.

**no-proxy** — next after `t9_0` (500,039,680): five trials stopped at 150,011,904 tokens — `t16_0` (CE 3.14810), `t8_0` (3.16539), `t3_0` (3.17184), `t17_0` (3.20260), `t7_0` (3.22089).

**no-centaur** — next after `t8_0` (500,039,680): no 150M runners; best partials at 100,007,936 — `t9_0` (CE 3.34995), `t16_0` (3.39177), `t3_0` (3.40825), `t10_0` (3.43760), `t17_0` (3.47886).

## Winner comparison

| | no-proxy `t9_0` | no-centaur `t8_0` | Diff / note |
|--|-----------------|------------------|-------------|
| Search-val CE | 2.78652 | 2.79045 | no-proxy better by ≈0.00392 |
| `lr` | 4.13e-4 | 3.01e-4 | no-proxy higher |
| `weight_decay` | 0.0147 | 0.01 (floor) | no-proxy higher |
| `beta2_gap` | 0.00147 | 0.001 (floor) | no-proxy higher |
| `eps` | 1.48e-12 | 1e-12 (floor) | similar |
| `warmup_fraction` | 0.0113 | 0.0070 | no-proxy longer warmup |
| `decay_fraction` | 0.05 | 0.05 | same (floor) |
| `terminal_lr_ratio` | 0.0215 | 0.0 | no-proxy anneals to non-zero terminal LR |
| `global_batch_mult` | 0.5 | 0.5 | same (floor) |
| `max_grad_norm` | 0.3 (floor) | 0.348 | no-centaur slightly higher |

Both winners chose the lower edge of `global_batch_mult` and `decay_fraction`. Several no-centaur fields sit on search-space floors (`weight_decay`, `beta2_gap`, `eps`, `terminal_lr_ratio`).

## MixLaw mix01 control vs probe winners

Reference control for final 370M / ~10B validation: the finished **mix01** run with **cosine LR decay** on the **RegMix validation mixture** (not `olmo-mix-1124`, not MixLaw/LightGBM-optimized mixes).

| | Control (mix01) | no-proxy `t9_0` | no-centaur `t8_0` |
|--|-----------------|-----------------|-------------------|
| Role | baseline / ladder recipe | HPO winner | HPO winner |
| W&B source | [`eduLLM/mixlaw/uc48hhcz`](https://wandb.ai/eduLLM/mixlaw/runs/uc48hhcz) | [`eduLLM/hpo-probe/904ea39d…`](https://wandb.ai/eduLLM/hpo-probe/runs/904ea39d368dfe412048a6063c1600df) | [`eduLLM/hpo-probe/06e12699…`](https://wandb.ai/eduLLM/hpo-probe/runs/06e12699f744b8d2e562e78afa003b7f) |
| Cloned to `hpo-validation` | [`9xy0yl6v`](https://wandb.ai/eduLLM/hpo-validation/runs/9xy0yl6v) (`mixlaw-mix01-control-cosine-regmix`; control steps × **16**) | [`e65cd29c…`](https://wandb.ai/eduLLM/hpo-validation/runs/e65cd29c7dd34fcd6b8a34eb81f8e716) (256 Ki batch override) | not yet launched |
| Model | OLMo2-370M | OLMo2-370M (validation) | OLMo2-370M (validation) |
| Search model | — | OLMo2 ~190M | OLMo2 ~190M |
| Training data | mix01 domain mixture (`validation_mixtures_10b.json` weights on `olmo-127b` shards) | `regmix-10b` sealed corpus | `regmix-10b` sealed corpus |
| Scheduler | **Cosine + warmup** (`CosWithWarmup`) | **WSD** (warmup / stable / decay) | **WSD** |
| `lr` (peak) | **0.0004** | 0.0004125 | 0.0003006 |
| `weight_decay` | **0.1**† | 0.01473 | 0.01 (floor) |
| `beta2_gap` (`β₂ = 1 − gap`) | **0.05** (β₂ = 0.95)† | 0.00147 (β₂ ≈ 0.9985) | 0.001 (floor, β₂ = 0.999) |
| `eps` | **1e-8**† | 1.48e-12 | 1e-12 (floor) |
| Warmup | **24 steps** (~100.7M tokens, ≈ **1.01%** of 10B) | `warmup_fraction` **0.0113** | `warmup_fraction` **0.0070** |
| Decay schedule | cosine over all post-warmup steps (`lr_alpha_f` **0.1** → terminal LR **4e-5**) | `decay_fraction` **0.05** tail | `decay_fraction` **0.05** tail |
| `terminal_lr_ratio` | **0.1** (`lr_alpha_f`; cosine floor) | **0.0215** (WSD terminal) | **0.0** (floor) |
| Global batch (tokens) | **4,194,304** (4 Mi) | **16,384** (`mult` 0.5)‡ | **16,384** (`mult` 0.5)‡ |
| `max_grad_norm` | **1.0**† | 0.3 (floor) | 0.348 |

† **Not logged on the mix01 W&B run** (`uc48hhcz` / `run-meta:v1`). Values match the shared 370M validation ladder (`SkipStepAdamW` + `CosWithWarmup`, `max_grad_norm=1.0`, `weight_decay=0.1`, `betas=(0.9, 0.95)`, stock `eps`) used by the MixLaw / curriculum 370M recipes (`train_stack` on the control run).

‡ Probe winners at `global_batch_mult=0.5` yield **16,384** tokens/step on the 370M validation recipe (`BASE_PROBE_GLOBAL_BATCH_TOKENS=32_768`). The control uses **256×** that batch. The live no-proxy validation run overrides to **262,144** tokens/step (256 Ki).

### What aligns vs what differs

**Roughly aligned**

- Peak LR: control **4e-4** sits between the two winners (3.0e-4 – 4.1e-4).
- Warmup length: control **24 steps / ~1.0%** of 10B is close to no-proxy `warmup_fraction` **0.0113**; no-centaur warmup is shorter.
- `decay_fraction` **0.05** on both winners matches the search-space floor but is **not comparable** to cosine decay shape on the control.

**Material differences**

- **Scheduler family:** cosine (control) vs WSD (probe winners and final validation). `terminal_lr_ratio` / `lr_alpha_f` are not interchangeable.
- **Global batch:** control **4 Mi** vs probe-default **16 Ki** (and **256 Ki** on the current RunPod validation). This dominates step count and learning-rate noise scale.
- **Optimizer tail:** control uses much higher `weight_decay` (**0.1**) and `beta2_gap` (**0.05**) than either winner; both winners sit at or near search floors for `eps`, `max_grad_norm`, and (for no-centaur) `weight_decay` / `beta2_gap` / `terminal_lr_ratio`.
- **Data path:** mix01 **domain-weighted** stream over multi-source `olmo-127b` shards vs **single-corpus** `regmix-10b` for probe and validation.
- **Model scale during search:** probe tuned on **~190M**; control and validation are **370M**.

Use the cloned control in `hpo-validation` for task-loss curves on the **optimizer-step x-axis** (control steps multiplied by **16** so 4 Mi-token batches align with 256 Ki validation steps); interpret HP gaps above when comparing training dynamics, not just downstream BPB.

## Provenance

| Arm | Entity/project | Run ID | Trial | Local vector name |
|-----|----------------|--------|-------|-------------------|
| no-proxy | `eduLLM/hpo-probe` | [`904ea39d368dfe412048a6063c1600df`](https://wandb.ai/eduLLM/hpo-probe/runs/904ea39d368dfe412048a6063c1600df) | `t9_0` | `no-proxy-winner` |
| no-centaur | `eduLLM/hpo-probe` | [`06e12699f744b8d2e562e78afa003b7f`](https://wandb.ai/eduLLM/hpo-probe/runs/06e12699f744b8d2e562e78afa003b7f) | `t8_0` | `no-centaur-winner` |
| curriculum_quadratic_mtld | `eduLLM/hpo-probe` | [`7f74348409e054561b12348d0f5a815b`](https://wandb.ai/eduLLM/hpo-probe/runs/7f74348409e054561b12348d0f5a815b) | `t4_0` | curriculum-learning winner → `hpo-validation` [`3576162a…`](https://wandb.ai/eduLLM/hpo-validation/runs/3576162a7edea5d8bebeafdf50053740) |
| mix01 control | `eduLLM/mixlaw` | [`uc48hhcz`](https://wandb.ai/eduLLM/mixlaw/runs/uc48hhcz) | — | cloned → `hpo-validation` [`9xy0yl6v`](https://wandb.ai/eduLLM/hpo-validation/runs/9xy0yl6v) (steps × 16) |

Cross-check: W&B winner HPs and trial IDs match `.edullm/final-validation-vectors.json` exactly (within float representation).

## Metrics not recovered

- Additional full-fidelity candidates beyond the single winner per arm — **confirmed absent** in study-result artifacts and controller-state observations (not merely missing from summary flattening).
- Separate held-out / untouched evaluator CE beyond search-validation CE (`exact_retrain_tokens` = 0 on both runs).
- Platform dollar cost / approval class (not logged on these runs; use `edullm check --json` if needed for a future submission).
- Original RunPod job directories from this laptop (W&B artifacts are the durable mirror used above).

## Next step

- **Curriculum learning:** 370M / ~10B validation **finished** ([`3576162a…`](https://wandb.ai/eduLLM/hpo-validation/runs/3576162a7edea5d8bebeafdf50053740)); winner vector documented above.
- **RegMix arms:** final 370M / ~10B validation on the sealed RegMix corpus. Vectors are wired in `.edullm/final-validation-vectors.json`; see `.edullm/FINAL_VALIDATION.md` for the launch contract.
