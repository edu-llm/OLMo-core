# Checkpoint transfer smoke (v4)

Ben’s idea: post-train an **early** checkpoint \(M_s\) in parallel, then **fit** that onto a later base \(M_t\) via \(\Delta = \mathrm{FT}-M_s\).

Paper: https://arxiv.org/abs/2503.20110

## v4 = MoE + actual SFT

| Piece | Choice |
|-------|--------|
| Architecture | **MoE 4-of-40** twin (`d_model=512`, 8 layers) |
| Bases | Synthetic close-gap \(M_s \to M_t\) on real `math-frontload` tokens |
| Post-train | Full-weight **assistant-masked SFT** on `math-sft-60m` |
| Fit | Zero-shot \(M_t+\Delta\) only |
| Metric | Held-out assistant CE retention |

**Not in this run:** DPO, RLVR, real team MoE intermediates, generation evals.

## Launch (only when approved — never main)

```bash
# on edullm/ckpt-transfer-smoke
edullm check --experiment ckpt-transfer-moe-sft-v4 --dataset math-frontload-100m-v1 --hours 8 --attempts 1
# abort if ceiling > $20; submit only after explicit OK
```
