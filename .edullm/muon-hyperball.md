# MuonH (Hyperball) vs MuonW, dense and MoE, at 370M

Four arms: `{olmo2_370M, olmo2_370M_moe} × {muon_w, muon_h}`.

## What is being tested

Hyperball ([arXiv 2606.16899](https://arxiv.org/abs/2606.16899)) fixes each constrained weight
matrix's Frobenius norm at `R = ||W_0||_F` and normalizes the update to unit Frobenius norm, so
one step moves exactly `η_t · R` before a radial projection back onto the sphere. Constrained
matrices take **no weight decay** — the constraint replaces it. MuonH is that wrapper with
Muon's `msign(M_t)` as the base update.

`src/olmo_core/optim/hyperball.py` implements both arms as one optimizer with a `constraint`
switch, so they share the identical momentum and Newton–Schulz path and the comparison isolates
the wrapper. The dion-backed `MuonConfig` is untouched.

## The three things that would invalidate this comparison

1. **The run must finish its learning-rate decay.** The paper's own finding is that Hyperball
   "starts slightly worse but overtakes WD as the learning rate decays". A run truncated before
   the cosine schedule completes is not a shorter version of this experiment — it is an
   experiment biased toward MuonW. Every arm gets a schedule that completes inside its step
   budget; no arm is stopped early and compared to one that was not.

2. **`--learning-rate` is not the same quantity across arms.** For `muon_w` it is scaled per
   matrix by `adjust_lr` (Moonlight's `0.2·√max(d_in,d_out)`). For `muon_h` it is a *relative*
   step size — dimensionless, `||ΔW||/||W|| ≈ η`. A single LR shared between the arms compares
   nothing, so each arm needs its own value. See "LR sweep" below.

3. **`--init-method fan_in` on every arm.** `R` is measured from `W_0`, so the initializer sets
   the absolute step length. The paper uses `std = 1/√d_in`, which is `fan_in` here;
   `llama_like` and so every `olmo2_*` factory default to `normal` (std 0.02). Both arms must
   use the same one, and `fan_in` is the one the method was designed around.

## Why the MoE arm needed library work

OLMo-core stores expert weights with the expert dimension folded into rows — `w1` is
`(num_experts · d_model, hidden_size)`. That is 32 independent matrices in one 2D tensor, so
orthogonalizing it whole mixes experts and computes the radius and `adjust_lr` from the stacked
shape. The `block_rows` param-group option makes `msign`, both Frobenius norms, the radius and
`adjust_lr` run per expert; `default_group_overrides` derives it from the owning module's
`num_experts`, for **both** arms. This is an extension — the paper says nothing about MoE.

Convenient consequence: FSDP shards those tensors on the expert-major dimension, so at any
world size dividing the expert count every block is already rank-local and the step needs no
communication. Dense attention/MLP matrices are all-gathered instead.

## Parameter matching

`olmo2_370M_moe` is matched on **active** parameters, not total:

| | total | non-embedding | active/token |
|---|---|---|---|
| `olmo2_370M` | 474.0M | 371.3M | 474.0M |
| `olmo2_370M_moe` | 1,078.5M | 975.8M | 373.9M |

32 experts × hidden 512, `top_k=4`, no shared MLP. Everything else — `d_model`, depth, heads,
QK-norm, RoPE theta, reordered-norm blocks — is held equal to the dense config.

## What lands in W&B

The platform supplies the project (`EDULLM_WANDB_PROJECT`) and puts the experiment slug in
`WANDB_RUN_GROUP`, so every arm of one `--experiment` groups on its own. The run *name* is the
platform run id, which is what `edullm status` and the lineage record use and which says nothing
about the arm — so the arm goes on as **tags**: `muon_h` / `muon_w`, the model factory,
`init-fan_in`, and `lr-<value>`. Filter or group on those.

Alongside the usual loss and throughput, `MuonMetricsCallback` logs every step:

| metric | reads as |
|---|---|
| `optim/radius_relative_drift_max` | `max\|‖W_b‖_F / R_b − 1\|`. **MuonH only.** Should sit at the fp32 accumulation floor for the whole run. |
| `optim/matrix_norm_{mean,min,max}` | Frobenius norms over constrained blocks. Pinned on MuonH by construction; free to move on MuonW, and where it settles is what Hyperball pins. |

**Check the drift metric before reading any loss curve.** If it climbs, the constraint stopped
holding — a shard boundary splitting an expert, a resume that recovered the wrong radius — and
the arm is no longer testing Hyperball. The run still trains and still reports a loss, and it
will probably lose; that reads as a result about the method and is a result about a bug. Both
are invisible in the loss alone, which is the entire reason this metric is logged.

These are rank-local: a sharded matrix contributes its own slice, so the drift is reduced with
`max` (the worst rank is the one worth seeing) and the norms with `mean`.

## Running it

Stage 1, smoke: four short single-GPU runs, one per arm, to prove each path trains on real data
under FSDP and to **measure tokens/s** so the full runs can be sized against the 24 h attempt
bound rather than guessed at.

```bash
for arm in dense-muonw dense-muonh moe-muonw moe-muonh; do
  edullm check --json --spec .edullm/run-smoke-$arm.yaml \
    --experiment muonh-smoke --dataset olmo-150b-dolma2-v1 \
    --team scratch --compute gpu-1xl40s --hours 1 --attempts 1
done
```

Always pass `--hours` and `--attempts`. The workload profile's defaults are the maximums, which
price high enough to park the run waiting for an approver instead of starting it. Read the real
number out of `check --json` (`cost`, `approval_class`) — never from this file.

Stage 2, the comparison itself: sized from stage 1, one spec per arm, `--data-seed` held equal
across arms and varied only to add seeds.
