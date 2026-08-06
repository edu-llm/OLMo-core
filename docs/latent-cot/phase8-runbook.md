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
Screen with 3 seeds, then confirm with 5. **Every arm forks the same "best model" via
`--rung olmo3_370M --init-checkpoint s3://…` and uses the SAME `--init-seed`** (identical
starting weights = the shared base); only `--seed` (data shuffle) and the arm's whitelisted
fields vary. Arms: `A0` explicit-CoT (= the best model fine-tuned the normal way, the fair
baseline), `A1` no-CoT, `A2` CODI, `A3` CODI+R1 (the fix), `A4` CODI+L2 (control).

```bash
DATA=data/latentcot/graph-reachability-depth/conversations
BASE=s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/  # needs AWS creds
for seed in 1 2 3; do
  for arm in A0 A1 A2 A3 A4; do
    .venv/bin/python src/scripts/latentcot/train_codi.py \
      --arm $arm --rung olmo3_370M --init-checkpoint $BASE --steps 5000 --batch-size 16 \
      --init-seed 0 --seed $seed \
      --train-data $DATA/train-00000.jsonl --test-data $DATA/heldout-00000.jsonl \
      --out runs/latentcot
  done
done
# each writes runs/latentcot/<arm>-seed<seed>/{model.pt, best.pt, best.json, stepN.pt x2, metrics.json}
```
`metrics.json` already carries `overall_acc` + `solve_rate_by_depth` per run.

**Precision.** `--precision bf16` is the default: bf16 autocast on the training forward, the
in-loop val scoring, and the final gate scoring, plus TF32 for the ops that stay fp32. The
distill and R1 terms are pinned to fp32 internally. Pass `--precision fp32` for a
bit-reproducible (and several-fold slower) run. Keep it the same across every arm — it is
recorded in each `metrics.json`.

**Do not raise `--batch-size` expecting a speedup.** The CODI student is processed one example
at a time, so a bigger batch adds sequential forwards rather than widening a tensor: per-example
step time is flat from batch 2 upward, while total step time grows linearly. At 16 the fixed
per-step optimizer cost is already amortized. Raising it is a gradient-noise choice that costs
wall-clock linearly and would require re-screening peak LR (§ above). Throughput here comes from
packing the per-example loop, not from the batch flag.

**Watch these two tripwires** in `metrics.json` → `train_history` (logged every `--log-every`
steps for the CODI arms A2/A3/A4):

- `thought_rms` — the RMS of the continuous thoughts. Thoughts pass through the LM head's final
  norm before feedback, so this should sit near the token-embedding scale (**≈ 1.0**) and stay
  **flat** as training proceeds. A steady climb means the latent path is drifting off the manifold
  the forked pretrained weights were fit on; unnormalized it reached ~52 by K=10 on this rung.
- `grad_norm` — the pre-clip total gradient norm. A sustained rise is the earliest warning of the
  same problem, and shows up before accuracy moves.

Both are diagnostics, not objective terms, and are identical in definition across arms.

**Checkpoints (crash recovery + best-selection).** Each run saves a rolling checkpoint every
`--save-every` steps (default 500 ≈ ~10 saves; a crash loses <= one interval), keeping the last
`--keep-last` (default 2, oldest deleted) plus `best.pt` — the checkpoint with the highest accuracy
on a **validation split carved off TRAIN** (`--val-fraction` default 0.1, seeded independently of
`--seed`, capped at `--best-eval-size` examples). The gate test set is **never** used to pick best
(that would be selection on the eval data). `model.pt` is still the final last-step weights and
remains what §4/§4b evaluate; point `eval.py`/`compare_models.py` at `best.pt` instead only if you
deliberately want the val-selected checkpoint. Set `--save-every 0` to disable checkpointing (e.g. a
quick smoke run). Note the ~1.9 GB/checkpoint footprint (~7.5 GB per run dir).
GPU is auto-detected (`--device auto`); pass `--device cpu` to force CPU. To sanity-check the
run *from scratch* (no S3/creds), drop `--init-checkpoint` and use `--rung olmo2_370M` — the two
rungs are state-dict-interchangeable at our sequence lengths, but the real sweep forks the base.

**Fine-tune schedule (all arms).** Because every arm forks the pretrained best model, this is a
*fine-tune*: the LR follows warmup-stable-decay (`WSD`, `--warmup-steps 200` default, 10% linear
decay tail) rather than a constant LR. Warmup eases the optimizer into the good pretrained weights
(a full-LR first step can spike the loss and undo the pretraining we forked for). The schedule is
identical across arms — it lives in the shared `train_arm` loop, so it stays confound-clean.

**LR screen (PRD §3.1/§6, do before the confirmatory seeds).** Fine-tuning wants a smaller peak
LR than a from-scratch run. Screen the peak LR on 1 seed of A0/A2 over
`--lr {1e-5, 2e-5, 5e-5, 3e-4}` (pick the highest LR with a stable, decreasing loss curve), then
fix that LR for all arms in the seeded sweep. `--lr`/`--warmup-steps` are recorded in each
`metrics.json`.

## 4. Gates + probes
```bash
.venv/bin/python src/scripts/latentcot/eval.py \
  --test-data $DATA/heldout-00000.jsonl --num-continuous-thoughts 10 --model olmo3_370M \
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
  R1 runs with its **anti-collapse entropy floor OFF by default** so this is the clean,
  one-variable A3-vs-A4 comparison. First inspect A3's `decodability` + the probes below: if the
  continuous thoughts have **collapsed** (near-one-hot logit-lens / degenerate decodability),
  re-run *only A3* with the floor on — `train_codi.py --arm A3 --vocab-reg-entropy-floor 1.0` —
  and note that A3 then carries an extra term the L2 control lacks (interpret Gate B accordingly,
  or add a matched term to A4 for a strict control). The floor is recorded in each `metrics.json`.
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
  --test-data $DATA/heldout-00000.jsonl --num-continuous-thoughts 10 --model olmo3_370M \
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
