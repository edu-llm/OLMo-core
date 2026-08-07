# Phase 8 runbook — the seeded 370M runs (needs GPU)

All code is in place and unit-tested on tiny CPU models; this is the turnkey GPU procedure.
Everything reuses the same `arm_loss` the tests validate, so behavior matches.

## 0. Prereqs (on the GPU box)
- `pip install -e '.[all]'` + `uv pip install tokenizers` (or ensure the dolma2 tokenizer loads).
- A CUDA GPU (the driver auto-detects `cuda`). 370M with the per-example CODI student is the
  cost driver — expect this to be the slow part; start with a modest `--steps` and scale up.

## 0b. Compute requirements (for GPU shopping)

**Shape of the job:** 5 independent single-GPU processes (one per arm), 1 seed, 5,000 steps each.
No multi-node, no multi-GPU-per-job, no interconnect requirement. Estimates below are from
FLOPs × assumed MFU (`local/latentcot_gpu_estimate.py`), using real token counts from the
difficulty grid — **they are estimates, not measurements**; calibrate on the box (see below) before
booking long reservations.

Cost per arm (5,000 steps, batch 16, K=10, mean student sequence ≈ 272 tokens):

| Arm | forwards/step | token-pos/step | train PFLOPs | eval PFLOPs | total |
|---|---|---|---|---|---|
| A0 explicit-CoT | 16 | 4,929 | 70 | **69** | 139 |
| A1 no-CoT | 16 | 4,161 | 59 | 1 | 60 |
| A2/A3/A4 CODI (each) | **192** | 51,453 | 732 | 8 | 740 |

Campaign total ≈ **2,420 PFLOPs**; slowest single arm ≈ **740 PFLOPs**. Note A0's *eval* nearly
equals its training cost — `greedy_generate` has no KV cache, so every generated CoT token is a
full forward. Lower `--best-eval-size` if A0's checkpoint evals dominate.

Wall-clock, dense bf16 peaks, `serial (all 5 arms on one GPU) / parallel (5 GPUs, = slowest arm)`:

| GPU | @2% MFU | @5% MFU | @10% MFU |
|---|---|---|---|
| A100-40GB / 80GB | 108 h / 33 h | 43 h / 13 h | 22 h / 7 h |
| H100-80GB | 34 h / 10 h | 14 h / 4 h | 7 h / 2 h |
| L40S-48GB | 186 h / 57 h | 74 h / 23 h | 37 h / 11 h |
| RTX A6000-48GB | 217 h / 66 h | 87 h / 27 h | 43 h / 13 h |

**Plan against the 5% column** — the loop issues 192 batch-1 forwards per step, which is
launch-latency bound rather than compute bound, so low MFU is expected until the per-example loop
is packed. For the same reason, **do not expect an H100 to deliver its 3.2× FLOPs advantage here**;
at batch 1 clock and memory latency matter more than tensor-core peak. An A100-40GB is the
sweet spot; anything with ≥40 GB and bf16 works.

**Memory (per GPU, one CODI arm):** params fp32 1.9 GB + AdamW states 3.8 GB + grads 1.9 GB +
retained activations ≈ 17 GB → **≈ 25 GB**. The activation term is the interesting one: `codi_loss`
accumulates all 16 examples' graphs and backprops **once**, so every example's whole K-chain is
alive simultaneously, and it scales *linearly* with `--batch-size` (halve the batch to halve it).
40 GB is comfortable; 24 GB cards are marginal and would need a smaller batch. A0/A1 need far less.

**Calibrate before booking (≈10 min on the real box):** run one CODI arm for 20 steps and read the
actual step time and peak memory, then rescale the table:
```bash
.venv/bin/python src/scripts/latentcot/verify_checkpoint.py --model olmo3_370M   # loads + 1 fwd/bwd
.venv/bin/python src/scripts/latentcot/train_codi.py --arm A2 --rung olmo3_370M \
  --steps 20 --batch-size 16 --save-every 0 --log-every 1 \
  --train-data $DATA/train-00000.jsonl --test-data $DATA/heldout-00000.jsonl --out /tmp/cal
# hours per CODI arm ≈ (mean seconds/step) * 5000 / 3600 ; also check nvidia-smi peak memory
```

## 0c. Platform submission (edullm compute)

Branch must live under `edullm/**` — images are only built from that namespace:
```bash
git push -u origin latent-cot-superposition-amy:edullm/latent-cot-superposition-amy
```

Target `gpu-8xa100` (8×40 GB; one arm per card, 3 idle). Five processes on eight cards is refused
unless the command carries `EDULLM_LAUNCH_CHECK=waived` verbatim, and checkpoints must go to
`$EDULLM_CHECKPOINT_DIR` (expanded by the shell at runtime, hence `bash -lc` with single quotes).
Submit with `--attempts 1`: the profile allows two and declares `resume_required: true`, but there
is no `--resume` flag, so a second attempt would silently restart from the base checkpoint.

**The dataset is gitignored — a fresh clone has none.** Generate it in-job before training, and run
the pre-registration gate, or the arms train on nothing:

```bash
bash -lc 'EDULLM_LAUNCH_CHECK=waived
set -uo pipefail
DATA=data/latentcot/graph-reachability-depth/conversations
BASE=s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/
python src/scripts/latentcot/gen_graph_data.py
python src/scripts/latentcot/preflight.py \
  --train-data $DATA/train-00000.jsonl --test-data $DATA/heldout-00000.jsonl || exit 1
pids=()
for i in 0 1 2 3 4; do
  mkdir -p "$EDULLM_CHECKPOINT_DIR/A$i"
  CUDA_VISIBLE_DEVICES=$i python src/scripts/latentcot/train_codi.py \
    --arm A$i --rung olmo3_370M --init-checkpoint "$BASE" \
    --steps 5000 --batch-size 16 --precision bf16 --lr <SCREENED_LR> \
    --init-seed 0 --seed 1 \
    --train-data $DATA/train-00000.jsonl --test-data $DATA/heldout-00000.jsonl \
    --out "$EDULLM_CHECKPOINT_DIR/A$i" > "$EDULLM_CHECKPOINT_DIR/A$i/train.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; exit $rc'
```
Notes on the mapping: the flag is **`--out`**, not `--save-dir`; `A$i` for `i` in 0–4 lines up with
arms A0–A4; `--rung olmo3_370M` is passed explicitly because the script still defaults to
`olmo2_370M`. Waiting on each PID (rather than bare `wait`) is what makes a failed arm fail the job.

**`$EDULLM_CHECKPOINT_DIR` is an `s3://` URI, not a directory.** (`checkpoints.py`: *"a checkpoint
prefix must be an s3:// URI"*; the value is `output_prefix(team, run_id) + "checkpoints/"`.) This
matters more than it looks: `Path("s3://b/k")` is `PosixPath("s3:/b/k")` — a **relative local**
path — so passing the URI to code that assumes `Path` writes into a directory named `s3:` beside
the process, raises nothing, and loses everything when the container exits. `train_codi.py` now
detects a URI in `--out`, stages artifacts in `--staging-dir` (local, default
`runs/latentcot-staging`), and mirrors each one — rolling `stepN.pt`, `best.pt`, `best.json`,
`model.pt`, `metrics.json` — to the URI as it is written. `train_arm` raises if a URI reaches
`save_dir` directly. Remote copies are not pruned (local ones still are): with no `--resume` they
exist for manual recovery, which is exactly what you want if the 24 h cap truncates a run.

The `EDULLM_LAUNCH_CHECK=waived` token is real (`launchers.LAUNCH_CHECK_WAIVER`) and waives
`require_a_process_for_every_device`, which would otherwise refuse 5 processes on an 8-device
machine. It is not silent: `waived_launch_check_note` surfaces a sentence to the approving lead
saying the run bills for 8 devices and starts 5.

```bash
edullm check --json --compute gpu-8xa100 --workload olmo-core-train \
  --experiment latent-cot-pilot --dataset none --attempts 1
```
Fix anything under `refusals` (match on code, not prose), then swap `check` → `submit`.

**Calibrate first on `gpu-1xa10g`** (§0b) — but an **A10G is 24 GB, below the ~25 GB estimate**, so
calibrate at `--batch-size 8` (activations scale linearly ⇒ ~12–13 GB) and double the per-example
step time to project batch 16. That run also *validates the memory model*: if batch 8 lands near
12–13 GB, batch 16 on a 40 GB A100 is safe.

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

## 3. Train the arms — matched starts, **1 seed (pilot)**
**This first campaign is 1 seed per arm.** That makes it a *screen*, not the confirmatory run:
with a single seed there is no seed-level variance, so the pre-registered Gate A/B criteria — which
are **paired-seed 95% CIs** — cannot be *concluded*, only pointed at. See §4 for what you can and
cannot claim, and §6 for what the confirmatory sweep needs.

**Every arm forks the same "best model" via `--rung olmo3_370M --init-checkpoint s3://…` and uses
the SAME `--init-seed`** (identical starting weights = the shared base); only the arm's whitelisted
fields vary. Arms: `A0` explicit-CoT (= the best model fine-tuned the normal way, the fair
baseline), `A1` no-CoT, `A2` CODI, `A3` CODI+R1 (the fix), `A4` CODI+L2 (control).

```bash
DATA=data/latentcot/graph-reachability-depth/conversations
BASE=s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/  # needs AWS creds
SEED=1                       # pilot: one seed, identical for every arm
for arm in A0 A1 A2 A3 A4; do
  .venv/bin/python src/scripts/latentcot/train_codi.py \
    --arm $arm --rung olmo3_370M --init-checkpoint $BASE --steps 5000 --batch-size 16 \
    --init-seed 0 --seed $SEED \
    --train-data $DATA/train-00000.jsonl --test-data $DATA/heldout-00000.jsonl \
    --out runs/latentcot
done
# each writes runs/latentcot/<arm>-seed<seed>/{model.pt, best.pt, best.json, stepN.pt x2, metrics.json}
```
The five arms are **independent processes** — if you have 5 GPUs, run them concurrently (one arm
per GPU, `CUDA_VISIBLE_DEVICES=$i`) and wall-clock becomes the slowest single arm rather than the
sum. There is no multi-GPU parallelism *within* an arm (the direct loop has no DDP/FSDP), so more
GPUs than arms buys nothing here. See §0b for sizing.

**On partial capacity, order the arms `A2` then `A0`.** Gate A is the A2−A0 depth curve, so those
two alone produce the headline signal (~9 h combined at 5% MFU on one A100); A3/A4 (Gate B) and A1
(the floor) can follow on the same GPU or on capacity that frees up later. One ≥40 GB bf16 GPU is
enough to start — don't wait for five.
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
> **1-seed caveat.** Both gates are pre-registered on **paired-seed 95% CIs**. With one seed there
> is no seed-level variance to compute them from, so read everything below as a *point estimate* —
> a screen for "is there a depth-increasing signal at all", not a pass/fail. You may legitimately
> bootstrap a CI over the 960 held-out **items**; that captures test-item noise, not init/data-order
> variance, and does not substitute for the paired-seed criterion. See §6.

- **Gate A (superposition):** `report["gate_a"]["slope"]` should be **positive** and the
  `curve` (A2 − A0 by depth) increasing. With ≥3 seeds, aggregate and report a **paired CI** on
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

## 6. From pilot to confirmatory
The 1-seed run is a screen. What it *can* settle: the harness runs clean at 370M (loss decreases,
`thought_rms` stays ≈1 and flat, `grad_norm` stable, no OOM), the calibrated cost per arm, whether
A2 gets anywhere near A0 at all, and the sign/shape of the depth curve. What it *cannot* settle:
either gate, because both are defined on paired-seed CIs.

To convert it into a result, re-run §3 with `for SEED in 1 2 3` (and 5 for the confirmatory sweep),
keeping `--init-seed 0` and every other flag byte-identical — only `--seed` changes. Cost scales
linearly: ~3× and ~5× the §0b numbers. Nothing else about the procedure changes, and the pilot's
seed-1 runs are reusable as one of the seeds (same code, same flags → same run).

Anything written from the 1-seed pilot must call itself a pilot; presenting it as a gate outcome
would violate the §11 pre-registration terms.

## Notes
- `train_codi.py` builds from a seeded init by default (no external base checkpoint needed); pass
  `--init-checkpoint <state_dict.pt | dir | s3://…>` if you have a shared pretrained base (use the
  SAME file for every arm) — e.g. the best model
  `s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/` with `--rung olmo3_370M`.
  `load_checkpoint` handles a plain `.pt` state_dict or a local/S3 OLMo-core checkpoint dir (S3
  needs AWS creds). Either way, keep `--init-seed` identical across arms.
- Publishing the dataset to the platform is separate (`publish_dataset.py`, needs AWS creds) and
  optional for these runs — the driver reads the local shards directly.
