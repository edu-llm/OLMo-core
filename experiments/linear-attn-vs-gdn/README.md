# Linear attention vs Gated DeltaNet (370M)

A controlled comparison of **plain (ungated) linear attention** against **Gated
DeltaNet (GDN)** at the 370M OLMo-3 scale, holding everything else fixed: same
data mix, same recipe, and the **same `fla` chunked-scan Triton kernel family**.

## What varies (and only this)

`linear` is a faithful *ablation* of `GatedDeltaNet` — identical `w_q/w_k/w_v/w_out`
projections, identical short causal convs (silu), identical QK-L2-norm, identical
head layout and output RMSNorm placement. The **only** difference is the
recurrence:

| | recurrence | kernel |
|---|---|---|
| `--mixer gdn` | gated delta rule `S_t = (diag(a_t) − β_t k_tk_tᵀ)S_{t-1} + β_t k_tv_tᵀ` | `fla.ops.gated_delta_rule.chunk_gated_delta_rule` |
| `--mixer linear` | ungated sum `S_t = S_{t-1} + k_tv_tᵀ`, `o_t = q_tS_t` | `fla.ops.linear_attn.chunk_linear_attn` |

GDN additionally carries the gate projections `w_a`/`w_b`/`w_g` and `A_log`/`dt_bias`
(≈34M params over 16 layers, dominated by `w_g`). That parameter delta is intrinsic
to the mechanism and is printed by `--dry-run` so it stays transparent.

Neither mixer uses document-boundary resets (`chunk_linear_attn` has no `cu_seqlens`
path), so the comparison is symmetric.

## Files

- `olmo_linear_attn.py` — `LinearAttention` module + `LinearAttentionConfig`
  (registered as `"linear_attention"`). Additive library code; **no OLMo-core edits**.
- `train_mixer.py` — training entry with `--mixer {attention,gdn,linear}`; reuses the
  proven 370M dolma2 ladder recipe (LR/global-batch/schedule) and only swaps the
  block sequence mixer. Logs to W&B (`entity=eduLLM`, `project=pretraining`).

## Run

```bash
# CPU config check
python experiments/linear-attn-vs-gdn/train_mixer.py t --mixer linear --dry-run

# 10B-token runs, one B200 each (default data = 10B water-fill mix, seq 4096)
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc-per-node=1 \
  experiments/linear-attn-vs-gdn/train_mixer.py linear-attn-370m-10b --mixer linear \
  --save-folder s3://edullm-olmo-370m-ckpts/linear-attn-vs-gdn/linear --work-dir /mnt/nvme/olmo-work

CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc-per-node=1 \
  experiments/linear-attn-vs-gdn/train_mixer.py gdn-370m-10b --mixer gdn \
  --save-folder s3://edullm-olmo-370m-ckpts/linear-attn-vs-gdn/gdn --work-dir /mnt/nvme/olmo-work
```

Requires `flash-linear-attention==0.4.1` in the training venv.
