# Checkpoint transfer smoke (v3)

Ben’s idea: post-train an **early** pretrain checkpoint \(M_s\) in parallel, then **fit** that post-train onto a later base \(M_t\) via \(\Delta = \mathrm{FT}-M_s\), \(M_t+\Delta\).

Paper: https://arxiv.org/abs/2503.20110

## v3 (what this run answers)

| Piece | Choice |
|-------|--------|
| Bases | Real `edullm-370M-30B` **step15000 → step20000** |
| Post-train | Full-weight **assistant-masked SFT** on `math-sft-60m` |
| Fit | Zero-shot \(M_t+\Delta\) only (no continue-finetune) |
| Metric | Held-out assistant CE retention |
| Not in this run | DPO, RLVR, MoE final, generation evals |

**Go claim:** early SFT on our close-gap pair can be fitted onto the later base and keep >25% of the SFT gain.

**Not claimed:** full SFT→DPO→RLVR on Joe’s MoE will transfer.

## Launch (not main)

Use branch `edullm/ckpt-transfer-smoke` only.

```bash
# platform still needs a runnable dolma2 dataset for admission; SFT JSONL is fetched inside the job
edullm check --experiment ckpt-transfer-sft-v3 --dataset math-frontload-100m-v1 --hours 12 --attempts 1
# abort if ceiling > $20
edullm submit --experiment ckpt-transfer-sft-v3 --dataset math-frontload-100m-v1 --hours 12 --attempts 1
```

Size `--hours` so check ceiling stays **≤ $20**. Prefer the smallest hours that still finishes.
