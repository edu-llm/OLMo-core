# Phase 8 runbook — the seeded 370M runs (needs GPU)

All code is in place and unit-tested on tiny CPU models; this is the turnkey GPU procedure.
Everything reuses the same `arm_loss` the tests validate, so behavior matches.

## 0. Prereqs (on the GPU box)
- `pip install -e '.[all]'` + `uv pip install tokenizers` (or ensure the dolma2 tokenizer loads).
- A CUDA GPU (the driver auto-detects `cuda`). 370M with the per-example CODI student is the
  cost driver — expect this to be the slow part; start with a modest `--steps` and scale up.

## 1. Generate the dataset (once)
```bash
.venv/bin/python src/scripts/latentcot/gen_graph_data.py
# -> data/latentcot/graph-reachability-depth/conversations/{train,heldout}-00000.jsonl
```

## 2. Pre-registration gate (once, before training)
```bash
.venv/bin/python src/scripts/latentcot/preflight.py \
  --train-data data/latentcot/graph-reachability-depth/conversations/train-00000.jsonl \
  --test-data  data/latentcot/graph-reachability-depth/conversations/heldout-00000.jsonl
```
Must print PREFLIGHT PASSED (matched config, disjoint seeds). Do not train if it fails.

## 3. Train the arms — matched starts, paired seeds
Screen with 3 seeds, then confirm with 5. **Every arm uses the SAME `--init-seed`** (identical
initial weights = the shared base); only `--seed` (data shuffle) and the arm's whitelisted
fields vary. Arms: `A0` explicit-CoT, `A1` no-CoT, `A2` CODI, `A3` CODI+R1 (the fix),
`A4` CODI+L2 (control).

```bash
DATA=data/latentcot/graph-reachability-depth/conversations
for seed in 1 2 3; do
  for arm in A0 A1 A2 A3 A4; do
    .venv/bin/python src/scripts/latentcot/train_codi.py \
      --arm $arm --rung olmo2_370M --steps 5000 --batch-size 16 \
      --init-seed 0 --seed $seed \
      --train-data $DATA/train-00000.jsonl --test-data $DATA/heldout-00000.jsonl \
      --out runs/latentcot
  done
done
# each writes runs/latentcot/<arm>-seed<seed>/{model.pt, metrics.json}
```
`metrics.json` already carries `overall_acc` + `solve_rate_by_depth` per run.

## 4. Gates + probes
```bash
.venv/bin/python src/scripts/latentcot/eval.py \
  --test-data $DATA/heldout-00000.jsonl --num-continuous-thoughts 8 \
  --arm A0=runs/latentcot/A0-seed1/model.pt \
  --arm A2=runs/latentcot/A2-seed1/model.pt \
  --arm A3=runs/latentcot/A3-seed1/model.pt \
  --arm A4=runs/latentcot/A4-seed1/model.pt
# writes runs/latentcot/eval/report.json (+ gate_a.png if matplotlib present)
```
- **Gate A (superposition):** `report["gate_a"]["slope"]` should be **positive** and the
  `curve` (A2 − A0 by depth) increasing. Aggregate across seeds and report a **paired CI** on
  the slope (compute from the per-seed `solve_rate_by_depth` in each run's metrics.json).
- **Gate B (the fix):** A3 (R1) accuracy + decodability **>** A2 (none), and **A3 > A4** (L2
  control) — that isolates the *vocabulary-space direction*. Paired CIs across seeds.
- **Probes:** use `olmo_core.latentcot.probes` on saved thoughts — logit-lens / linear probe
  (vs shuffled control) / causal ablation, on directed-graph-reachability where theory predicts
  the superposition advantage.

## 4b. Benchmark vs the "best model"
Head-to-head against the general pretrained 370M baseline
(`s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/`, W&B run `f08ey8cm`). The
baseline is **not** run zero-shot — it never saw our graph format and would score ~chance.
Instead every arm forks it as the shared init (§3, `--rung olmo3_370M --init-checkpoint s3://…`),
and **A0** = that best model fine-tuned the *normal* way (explicit CoT, no continuous thoughts,
no vocab reg). Comparing our latent arms against A0 from identical starting weights isolates the
*training method*.
```bash
.venv/bin/python src/scripts/latentcot/compare_models.py \
  --test-data $DATA/heldout-00000.jsonl --num-continuous-thoughts 8 --model olmo3_370M \
  --baseline A0=runs/latentcot/A0-seed1/model.pt \
  --ours A2=runs/latentcot/A2-seed1/model.pt \
  --ours A3=runs/latentcot/A3-seed1/model.pt
# writes runs/latentcot/compare/report.json (+ advantage.png if matplotlib present)
```
Prints solve-rate-by-depth for the baseline and each of our arms side by side with the per-depth
advantage `acc(ours) − acc(baseline)` and its slope vs depth (the superposition signal — positive
& increasing). Pulling the S3 checkpoint needs AWS creds + a GPU; the script's logic is dry-tested
on tiny CPU models. Use `--model olmo3_370M` to match the S3-forked arms.

## 5. Escalation (PRD §3.1)
If, at ≥5 seeds, gate A is null **but** probes show a weak, depth-increasing signal → bump one
rung (`--rung olmo2_600M`, then `760M`, `1B`), re-sweep `--lr`, repeat 3–4. If gate A is absent
even in-distribution **and** A2 never approaches A0 → debug at a reduced-size config first, don't
spend on scale. Escalate at most one rung at a time; stop at 1B (beyond is a separate decision).

## Notes
- `train_codi.py` builds from a seeded init by default (no external base checkpoint needed); pass
  `--init-checkpoint <state_dict.pt | dir | s3://…>` if you have a shared pretrained base (use the
  SAME file for every arm) — e.g. the best model
  `s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/` with `--rung olmo3_370M`.
  `load_checkpoint` handles a plain `.pt` state_dict or a local/S3 OLMo-core checkpoint dir (S3
  needs AWS creds). Either way, keep `--init-seed` identical across arms.
- Publishing the dataset to the platform is separate (`publish_dataset.py`, needs AWS creds) and
  optional for these runs — the driver reads the local shards directly.
