# Checkpoint transfer smoke (v2, MoE)

Ben’s idea: post-train earlier base $M_s$, apply $\Delta=\mathrm{FT}-M_s$ onto later $M_t$.

Paper: https://arxiv.org/abs/2503.20110

## v2 (accurate-enough smoke)

- **MoE** `top_k=4` / `num_experts=40` at d_model=512 (same routing pattern as team 4/40; not full 7B)
- **Real tokens** from `math-frontload-100m` (dolma2), not random ids
- Held-out CE on a disjoint shard region
- Requires SFT gain ≥ 0.05 CE before calling transfer decisive

## Launch (not main)

```bash
# on edullm/ckpt-transfer-smoke
edullm check --experiment ckpt-transfer-smoke-moe --dataset math-frontload-100m-v1 --hours 4 --attempts 1
# ceiling should be ~$3.22; only submit if under $50
edullm submit --experiment ckpt-transfer-smoke-moe --dataset math-frontload-100m-v1 --hours 4 --attempts 1
```
