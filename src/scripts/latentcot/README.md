# latentcot scripts

Runnable scripts for the latent chain-of-thought experiments — a CODI
continuous-thought substrate plus a superposition / distributional-shift study.

**Pre-registered design, arms, gates, and build checklist:**
`docs/latent-cot/latent-cot-superposition-prd.md` (start with its TL;DR).
Package code lives in `src/olmo_core/latentcot/`; the GPU run procedure is
`docs/latent-cot/phase8-runbook.md`.

## What this is (one paragraph)

All five arms are fine-tunes of the *same* pretrained "best model" checkpoint on
the same graph-reachability data — only the reasoning/training method differs:
**A0** explicit written-out CoT (the fair baseline), **A1** no-CoT, **A2** CODI
continuous thoughts, **A3** CODI + R1 vocab-manifold regularizer (the hypothesis),
**A4** CODI + matched L2 (the control). Latent arms use **K = 10** continuous
thoughts (≥ the deepest graph, depth 8). Gate A = the depth-vs-advantage slope vs
A0; Gate B = A3 > A2 *and* A3 > A4. See the PRD TL;DR for the full statement.

## Scripts

| Script | Purpose |
|---|---|
| `gen_graph_data.py` | Generate the synthetic directed-graph reachability data (train + held-out, OOD depths 5/8). |
| `publish_dataset.py` | Publish the data to the eduLLM platform as `sft/graph-reachability-depth` (needs AWS creds). |
| `verify_checkpoint.py` | Pre-flight: strict-load the S3 "best model" into `olmo3_370M` and smoke-test the continuous-thought forward/backward. **Run first on the GPU box.** |
| `preflight.py` | Confound gate: per-arm compute report + shared-base / disjoint-seed / arms-differ-only-in checks. |
| `train_codi.py` | Train one arm at one rung (default `olmo3_370M`), save `model.pt` + `metrics.json`. Fork the best model with `--init-checkpoint s3://…`. |
| `eval.py` | Gate A curve + slope and the Gate B table (per-arm accuracy + decodability). |
| `compare_models.py` | Head-to-head: our arms' solve-rate-by-depth vs the A0 baseline, per-depth advantage + slope. |

## Typical GPU run order

```bash
# 1. pre-flight (S3 needs AWS creds; --device auto-detects cuda)
verify_checkpoint.py --init-checkpoint s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/ --model olmo3_370M
preflight.py

# 2. train each arm x seed, all forking the shared best model
train_codi.py --arm A0 --rung olmo3_370M --init-checkpoint s3://… --init-seed 0 --seed 1 ...
#   (repeat for A1..A4; screen peak LR on A0/A2 over {1e-5,2e-5,5e-5,3e-4} first, then fix it)

# 3. gates + head-to-head
eval.py --model olmo3_370M --arm A2=runs/latentcot/A2-seed1/model.pt ...
compare_models.py --model olmo3_370M --baseline A0=… --ours A2=… --ours A3=…
```

Generated data lives under `data/` and run/eval outputs under `runs/` (both gitignored).
