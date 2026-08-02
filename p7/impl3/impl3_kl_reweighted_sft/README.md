# Implementation 3 — Low-KL / forgetting-aware SFT (KL-reweighted loss)

Keep Impl 2's pedagogy gains while cutting the math/logic forgetting it causes, by
biasing SFT toward the **KL-minimal** solution (RL's Razor) — via a **training-objective
change only** (per-token loss reweighting). No data rewriting, no generation. Drops
straight into the Impl-2 loss.

## Idea (PRD §3.1)
Much of SFT's KL is incidental/stylistic: within a correct tutor turn most tokens are
already high-probability under the base, and only a few carry the behavioral pivot. We
put **higher density on low-gap tokens, lower on high-gap ones** — a reweighting of
*where* the loss is spent, which selects among task-solving solutions rather than
trading task loss for KL (this is NOT a KL penalty). Guardrail: only operate in the
temperature regime that still passes the pedagogy rubric (§3.1).

## Weight signal (§3.2) — record BOTH, compare
- **a — base-surprise**: `s_t = -log π₀(y_t | context)` (one frozen-base pass; single-stage).
- **b — forward-KL**: `s_t = KL(π₀(·|ctx) ‖ π_SFT(·|ctx))` per token (two-stage: needs a
  vanilla Impl-2 SFT via `--sft_model_id`).

## Normalization (§3.3)
Global **mean-1** multiplier over pedagogy tokens (`m_t = N_ped · softmax_ped(−z(s_t)/T)`),
general tokens = 1. Preserves the pedagogy:general ratio and overall LR automatically.
`s_t` is standardized once with a robust z-score. Implemented in `common/weighting.py`
and applied by `common/sft_train.py`'s `WeightedTrainer`.

## Files
| File | What it does |
|---|---|
| `precompute_weights.py` | Compute + cache `s_t` once per variant (optional; the first train run also caches). |
| `train_kl_sft.py`       | One `(variant, T)` weighted SFT run; keeps ≥10 checkpoints. |
| `config.yaml`           | Inherits the Impl-2 recipe + the sweep grid. |

## Sweep (§3.4)
Grid: `variant ∈ {a, b} × T ∈ {2, 4, 8, 16, 32}` (+ the vanilla Impl-2 baseline line). The ladder
was shifted up from `{0.5,1,2,4,8}` on 2026-07-29: low `T` over-concentrates the reweighting and
`T≤1` behaved badly in the variant-a runs; higher `T` is a gentler reweighting (→ vanilla as `T→∞`).
```bash
# variant a (base-surprise): no vanilla SFT needed
for T in 2 4 8 16 32; do
  python train_kl_sft.py --variant a --temperature $T --config config.yaml
done
# variant b (forward-KL): point at your vanilla Impl-2 SFT
for T in 2 4 8 16 32; do
  python train_kl_sft.py --variant b --temperature $T \
      --sft_model_id ../impl1_2_prompting_sft/out/impl2-sft --config config.yaml
done
```

## Definition of done (§3.6) — three graphs (RL's Razor Fig 3), both conditions
Vs vanilla Impl 2 at matched pedagogy (CIs): reduced forgetting, lower new-task KL,
a Pareto improvement over the LR/steps curve, SI-gating preserved. Each of ~10 lines
(a/b × 5 T + baseline) is a checkpoint trajectory (≥10 ckpts). Reuse `common/kl.py`
for the KL axis and the eval suite for pedagogy/math. Report `kl_new_SI` and
`kl_ped_noSI`.
