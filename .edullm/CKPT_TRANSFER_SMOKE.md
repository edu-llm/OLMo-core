# Checkpoint transfer smoke

Investigates Ben’s idea: post-train an **earlier** pretrain checkpoint, then move
that change onto a **later** checkpoint via $\Delta = \mathrm{FT}(M_s) - M_s$,
$M_t + \Delta$.

Paper: [Efficient Model Development through Fine-tuning Transfer](https://arxiv.org/abs/2503.20110).

## Checkpoints (team)

| Role | URI |
|---|---|
| $M_s$ | `s3://edullm-checkpoints/olmo2-370m-cpt/edullm-370M-30B/step15000-unsharded/model.pt` |
| $M_t$ | `s3://edullm-checkpoints/olmo2-370m-cpt/edullm-370M-30B/step20000-unsharded/model.pt` |

First submit uses `--mode synthetic_twin` (same experiment, keys guaranteed).
`--mode team_s3` loads the URIs above when the old-OLMo dump loads into olmo_core.

## Branch / push

Use `edullm/ckpt-transfer-smoke` only. **Do not push to `main`.**

## Launch

```bash
edullm check --experiment ckpt-transfer-smoke --dataset none --hours 2
# only after check is clean and you approve:
edullm submit --experiment ckpt-transfer-smoke --dataset none --hours 2
```
