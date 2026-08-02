# Impl 3 (KL-reweighted SFT) — single-H200 runbook  [TEMPORARY]

Runs the full Impl-3 sweep (variants a+b × T∈{2,4,8,16,32} = 10 runs) as **one batch (`sbatch`)
1×H200 / 4h ORCD job**, then generates the eval outputs. LR stays `2e-4`; checkpoints are
log-spaced (`checkpoint_schedule: log`). Delete this file when done.

> **Why one GPU?** A 1B LoRA uses ~4 GB and <40% of a single H200, so a 2nd GPU bought little but
> caused real pain (SLURM splitting the 2 GPUs across two nodes → a single-node script can only use
> one → idle-GPU emails + priority dings). Single-GPU jobs schedule fastest and never fragment. The
> runs go sequentially and still finish within 4h. The script auto-adapts if it ever gets >1 GPU.

**Always use a batch job (`sbatch`), not an interactive allocation (`salloc`).** `sbatch` queues
the whole pipeline and the scheduler runs it automatically when GPUs free up — you can close your
laptop. `salloc` just gives you a live shell that you then have to babysit and type commands into.

> ⚠️ **First time on a fresh account, go in order: Step 0 → 0.5 → 0.6 → 1.** The batch job needs
> the `p7post` conda env (Step 0.5); if it's missing the job aborts immediately with a clear
> `FATAL:` message (it no longer fakes a "Completed"). Step 0.6 is a ~20-min smoke test that
> catches code/dependency breakage cheaply, before the full ~4h sweep. Once the env exists and
> smoke passes, Step 1 really is submit-and-walk-away.
>
> Honest caveat: these scripts are hardened to **fail fast with clear messages** rather than
> silently misbehave, and deps are pinned — but a cluster you can't pre-run can still surprise you
> (quota, connectivity, preemption). The smoke test in Step 0.6 is what turns "hope it works" into
> "known-good before the big run."

Vanilla Impl-2 is **already trained** — its adapter `checkpoint-923/` is the baseline and the
variant-b reference. Do NOT retrain it.

Time budget once the job **starts** (sequential on one H200): precompute ~0.5h + training ~2h +
eval-gen ~1.5h ≈ **~4h**, matches the `-t 04:00:00` limit (plus however long it waits in the queue
first). Runs are dispatched by a pool scheduler that adapts to however many GPUs are visible.

---

## 0. One time — copy project + baseline onto ORCD (run these ON YOUR MAC)

These are a **push from your Mac → ORCD**, so they must run in a **local Mac terminal**, NOT
inside an SSH session. Check the prompt first:
- `jamesxing@MericBook-Pro …%`  → you're on your Mac ✅ run rsync here.
- `[xing33@login007 ~]$`         → you're on ORCD ❌ rsync will fail with `change_dir … No such file or directory`.

SSH alias is `orcd-login` (from `~/.ssh/config`), not `orcd`. Run the two rsyncs separately:

```bash
rsync -avP --exclude out --exclude '**/__pycache__' ~/Documents/MericXing/MIT/Intern/AlphaAI/Training_Team/post-training orcd-login:~/
```
```bash
rsync -avP ~/Documents/MericXing/MIT/Intern/AlphaAI/Training_Team/checkpoint-923 orcd-login:~/post-training/
```

The second line ships the saved vanilla Impl-2 adapter (the variant-b reference). Success looks
like a streamed file list + a `sent … bytes` line with **no** `change_dir … failed` error.
Re-run the first rsync any time you change code locally (it's incremental — only sends diffs).

> ⚠️ **Both rsyncs are REQUIRED — the second one is easy to forget and fails silently.** If
> `checkpoint-923/` isn't on ORCD, the batch job flips `RUN_VARIANT_B=0` and runs **variant a only**
> (just a `WARNING` in the log; the job still exits `COMPLETED`). This is exactly what bit us on
> 2026-07-29. Verify on ORCD before submitting: `ls checkpoint-923/adapter_*` must list files.

## 0.5 One-time environment setup (REQUIRED — run ONCE on the login node, before Step 1)

The compute nodes have no conda/Python by default; the batch job needs the `p7post` env to exist.
`setup_orcd_env.sh` installs Miniforge into `~/miniforge3`, creates `p7post`, and installs
torch + all deps. Run it on the **login node** (it has internet; this takes ~10–20 min):

```bash
ssh orcd-login
cd ~/post-training && bash clusters/orcd/setup_orcd_env.sh
```

Verify it worked (must print `env OK`):
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate p7post && python -c "import torch, transformers, peft; print('env OK')"
```

Authenticate W&B once (stores the key in `~/.netrc`; avoids pasting it inline). **`wandb` lives
inside the env**, so activate first or you'll get `wandb: command not found`:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate p7post && wandb login
```

**Prestage data (recommended)** — compute nodes may be offline. Snapshot the dataset locally so
training/eval don't depend on the compute node's internet:
```bash
conda activate p7post && python snapshot_hf_dataset.py     # -> data/socrateach_sft_{train,val,test}.jsonl
```
The batch script uses `data/` automatically if present. (If the compute node *does* have internet,
this is just a speedup — the run also streams from the Hub.)

Dependency versions are **pinned** in `requirements.txt` (torch in `setup_orcd_env.sh`) to the
set validated on 2026-07-29, so this env is reproducible and won't drift on a fresh setup.

> ⚠️ **The eval scripts need `sentencepiece` + `tiktoken`** to rebuild the tokenizer from a saved
> checkpoint dir. Training does NOT need them (it uses the base model's cached fast tokenizer), so a
> missing dep stays invisible until the eval stage — where it fails **every** point with
> `You need to have sentencepiece or tiktoken installed ...` and produces an empty eval while the job
> still says `COMPLETED` (the 2026-07-29 bug). Both are now in `requirements.txt` and checked by
> `setup_orcd_env.sh`. On an env built **before** that fix: `pip install sentencepiece tiktoken`.

You only ever do Step 0.5 once. After that, Steps 0.6→1 are the workflow.

## 0.6 Smoke test (recommended — catches breakage in ~20 min, not after a long queue)

Before committing to the full ~4h sweep, run a tiny 1-GPU end-to-end check. It exercises model
load → tokenize → per-token weighted loss → checkpoint save on 32 examples. A short 1-GPU job
backfills almost immediately.

```bash
cd ~/post-training && sbatch --partition mit_preemptable clusters/orcd/smoke.sbatch
squeue --me
```
Watch it (want `SMOKE OK` at the end):
```bash
tail -f logs/p7-smoke_<id>.out
```
- `SMOKE OK` → the code + pinned deps + checkpointing all work; proceed to Step 1.
- `SMOKE FAILED: …` or a Python traceback → **do not launch the full sweep.** Send me the log;
  it's almost always a dependency/API mismatch I can pin, and far cheaper to catch here.

## 1. Submit the batch job — ALWAYS sync-then-submit in ONE command (run ON YOUR MAC)

> ⚠️ **NEVER `sbatch` on ORCD without first rsyncing your local edits up.** On 2026-07-30 we
> burned two jobs because ORCD was running a **stale** `impl3_h200.sbatch` (old `Time Limit: 06:00:00`
> and the old single-GPU logic) — the local fixes were never pushed. The fix below makes that
> impossible: it **pushes the code and submits in one atomic command**, so what runs is always what
> you edited. If you only remember one command in this runbook, remember this one.

Run this in a **local Mac terminal** (prompt `jamesxing@MericBook-Pro …%`). It rsyncs the project,
then SSHes in and submits — you cannot forget the sync because it's the same command:

```bash
rsync -avP --exclude out --exclude '**/__pycache__' ~/Documents/MericXing/MIT/Intern/AlphaAI/Training_Team/post-training orcd-login:~/ && ssh orcd-login 'cd ~/post-training && export WANDB_PROJECT=edullm-p7 && sbatch -p mit_preemptable clusters/orcd/impl3_h200.sbatch'
```

It prints `Submitted batch job <number>`. You can log out now — it runs unattended.

**Confirm the fresh code is what's running** (guards against a stale-file repeat): once it starts,
the log's first lines must show the new scheduler and the 4h limit, not the old 6h job:

```bash
ssh orcd-login "cd ~/post-training && sacct -j <jobid> --format=TimelimitRaw,Partition -n; grep -m1 'scheduling .* GPU' logs/p7-impl3_<jobid>.out"
```
Want: `Timelimit` ≈ `240` min (4h) and a line `scheduling 10 training runs across 1 GPU(s)`. If you
see `06:00:00` or no scheduler line, the sync didn't take — re-run the one-liner above.

- **W&B auth:** run `wandb login` once on ORCD (stores the key in `~/.netrc`). Do NOT paste
  `export WANDB_API_KEY=...` inline — it leaks the key into shell history/logs. If W&B isn't set
  up, the run still trains; add `--no_wandb` in the script's `COMMON` to silence it.
- **HF_HOME:** optional. Your home quota is 200 GB (only ~7 GB used), plenty for a 1B model +
  this dataset, so you can skip it. Only set `HF_HOME=/orcd/pool/<you>/hf_cache` if home fills up.
- **Skip variant b** (no dependency on `checkpoint-923`): submit with
  `sbatch --export=ALL,RUN_VARIANT_B=0 clusters/orcd/impl3_h200.sbatch` → ~5 runs, ~half the time.
- **Variant b only** (a already trained — don't retrain it): submit with
  `sbatch --export=ALL,RUN_VARIANT_A=0 clusters/orcd/impl3_h200.sbatch`. Needs `checkpoint-923/`
  on ORCD. It trains only b, then the eval stage covers b **plus** a's existing `out/impl3-a-*/`
  checkpoints (a is not touched). b runs on GPU0 in this mode, so a 1-GPU alloc works too.
- **Partition:** every `.sbatch` here already targets `mit_preemptable` — that is the standing
  default, not something to add per submission. It has many more free H200s so jobs start far
  sooner, and `--resume auto` plus log-spaced checkpoints means a preemption costs the steps since
  the last checkpoint rather than the run. Override with `-p mit_normal_gpu` only if you have a
  specific reason to want a non-preemptable slot.
- **Queue slow anyway?** `--resume auto` recovers
  from preemption.

### Monitor / cancel (anytime, then log off)

```bash
squeue --me                              # ST: PD = queued (reason in last col), R = running, gone = done
tail -f logs/p7-impl3_<number>.out       # live log once running; final line is "== ALL DONE =="
scancel <number>                         # cancel this job (works whether PD or R)
scancel -u $USER                         # cancel ALL your jobs
```

`clusters/orcd/impl3_h200.sbatch` is the source of truth for what runs; steps 2–5 below just
document its stages (and are also how you'd run it by hand in an interactive `salloc`).

## 2. Env (what the script does; or run once in an interactive shell)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate p7post   # or run setup_orcd_env.sh once
cd ~/post-training
nvidia-smi                                          # expect 2 H200s
```

> Data streams from the Hub (`hf_dataset: meric533/socrateach-sft`). If the compute node has no
> internet, run `python snapshot_hf_dataset.py` on the **login node** first and add `--data_dir data`.

> **Sections 3–4 are the manual/interactive (`salloc`) fallback only** — the batch job in Step 1
> already does precompute + train. They assume a **single GPU** (matching the batch request); on a
> 1-GPU alloc `CUDA_VISIBLE_DEVICES=1` is an invalid ordinal, so everything runs on GPU0 in sequence.

## 3. Precompute the weight signals (once per variant; ~0.5h on one GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python impl3_kl_reweighted_sft/precompute_weights.py \
    --variant a --config impl3_kl_reweighted_sft/config.yaml
CUDA_VISIBLE_DEVICES=0 python impl3_kl_reweighted_sft/precompute_weights.py \
    --variant b --sft_model_id checkpoint-923 --config impl3_kl_reweighted_sft/config.yaml
```

Doing this up front avoids a cache race when the temperature runs start.

## 4. Train the sweep — all runs sequentially on one GPU (~2–3h)

```bash
# H200 knobs: keep EFFECTIVE batch = 32 (per_device 32 × accum 1 × 1 GPU) so #steps (938) and the
# recipe are unchanged vs the POC; drop grad checkpointing (141GB fits it).
COMMON="--config impl3_kl_reweighted_sft/config.yaml --per_device_batch 32 --grad_accum 1 --no_grad_checkpointing --resume auto"

( for V in a b; do
    for T in 2 4 8 16 32; do
      extra=""; [ "$V" = "b" ] && extra="--sft_model_id checkpoint-923"
      CUDA_VISIBLE_DEVICES=0 python impl3_kl_reweighted_sft/train_kl_sft.py \
          --variant $V --temperature $T $extra $COMMON
    done
  done ) > logs_impl3_sweep.txt 2>&1

echo "sweep done; checkpoints under out/impl3-{a,b}-T*/"
```

Outputs land in `out/impl3-a-T2/ … out/impl3-b-T32/`, each with log-spaced
`checkpoint-{1,2,3,4,8,16,32,64,128,256,512,938}` (all kept). Training loss/LR stream to W&B live
under project `edullm-p7`.

## 5. Eval generation (final checkpoint of each run; ~1.5h). See `eval/README.md` for the full flow.

```bash
python snapshot_hf_dataset.py            # -> data/socrateach_sft_{train,val,test}.jsonl (held-out pedagogy)

# collect the final (highest-numbered) checkpoint of every run
CKPTS=$(for d in out/impl3-*/; do echo "$(basename $d)=$d/checkpoint-938"; done)

# (a) KL axis — new-task drift vs base, one shared base load
python run_kl_curve.py --base_model allenai/OLMo-2-0425-1B-Instruct \
    --ckpt $CKPTS impl2=checkpoint-923 \
    --pedagogy_file data/socrateach_sft_val.jsonl --n_prompts 64 \
    --out out/kl_by_checkpoint.json
# ('impl2' = the vanilla baseline point; the Instruct base itself is the KL=0 reference.)

# (b) Math retention — base vs each final checkpoint, both prompt conditions
for d in out/impl3-*/; do tag=$(basename $d);
  python eval/generate_eval.py --prompts eval/math_eval/math_logic_prompts.jsonl \
      --adapter $d/checkpoint-938 --out eval/math_eval/results_$tag.jsonl
  python eval/generate_eval.py --prompts eval/math_eval/math_logic_prompts.jsonl --boxed_hint \
      --adapter $d/checkpoint-938 --out eval/math_eval/results_${tag}_hint.jsonl
done

# (d) Pedagogy generation (NEW-task quality) — one tutor turn per held-out context for
# base + vanilla Impl-2 + each final checkpoint, then blind into judge batches for the frontier judge.
PED="base= impl2=checkpoint-923"
for d in out/impl3-*/; do PED="$PED $(basename $d)=$d/checkpoint-938"; done
python eval/gen_pedagogy.py --test_file data/socrateach_sft_val.jsonl --candidates $PED \
    --n_dialogues 40 --out eval/llm_judge/test_results_instruct.jsonl
( cd eval/llm_judge && python build_batches.py test_results_instruct.jsonl )  # -> judge_batch_*.json + judge_key.json
```

> The batch job (`impl3_h200.sbatch`) already runs all of (a)–(d); this block is the by-hand
> equivalent. `checkpoint-938` is illustrative — the real final step can differ (ours was **923**);
> the sbatch auto-detects the highest-numbered checkpoint per run, so prefer re-running the job over
> hardcoding a step by hand.

The batch job stops here (after *generating* eval outputs). Everything below is **off-GPU** and
runs on the login node.

## 6. Score the eval outputs (after the job finishes — off-GPU, no subagents)

All forgetting-axis scoring is deterministic. Run once the job is done (`squeue --me` shows it gone):

```bash
cd ~/post-training
python eval/math_eval/score_results.py eval/math_eval/results_*.jsonl
```

- **Math:** 250 GSM8K items, integer exact-match, no subagents. Read the hinted and bare columns
  together — the boxing hint makes a tutor-tuned model deflect rather than answer, so `boxed%`
  (commit rate) and `acc|boxed` separate refusal from actual skill loss.
- **General IF (IFEval) was dropped.** At 34 prompts it never separated any two configurations,
  so it cost generation time and yielded nothing. Math is the only prior-task probe now.
- **Pedagogy quality** (new-task y-axis): inherently judge-based (no deterministic metric for *how*
  it tutors). The eval stage generates tutor turns (`eval/gen_pedagogy.py`) and blinds them into
  `eval/llm_judge/judge_batch_*.json` (+ `judge_key.json`). This is the **only** piece that needs
  subagents. Score them with the frontier judge (→ `judge_out_*.json`), then aggregate:
```bash
cd eval/llm_judge && python aggregate.py   # judge_out_*.json + judge_key.json -> judge_summary.json (per-setup 0-1)
```
  `judge_summary.json` gives one pedagogy score per candidate (`base`, `impl2`, each `impl3-*`),
  which is the y-axis for the RL's-Razor Pareto plot.

## 7. Score every checkpoint and plot the RL's-Razor curve

Steps 5–6 cover the final checkpoint of each run. For the full trajectory — which is what the
curve actually needs — one job scores every checkpoint of every run:

```bash
sbatch clusters/orcd/ckpt_sweep_eval.sbatch     # resumable; appends to out/ckpt_sweep_bare_hint250.jsonl
bash eval/make_figures.sh                       # all four KL-condition x math-prompt figures
```

The sweep is safe to re-run: scored checkpoints are skipped, and each row carries a measurement
protocol stamp so a changed probe aborts the job instead of quietly mixing incompatible rows into
one file.

Impl 3 wins if its points sit **left of / below** the vanilla SFT baseline at matched pedagogy —
lower KL, less forgetting. See `RESULTS_192CKPT.md` for how that turned out, including why the
KL must be measured without the system instruction for the comparison to mean anything.

### Cost of per-checkpoint eval (does W&B logging add compute?)

The W&B logging itself is free (post-hoc, tiny). The cost is *evaluating at every checkpoint*
instead of just the final one. With log-spaced checkpoints (~12/run × 10 runs ≈ 120 checkpoints):

- **KL** is cheap: the base continuation depends only on the base model + prompt, so it's cached
  once and reused across all checkpoints → per-checkpoint KL is 2 forward passes, no generation.
- **Math (45 prompts) + IFEval (34 prompts)** are the per-checkpoint generation cost — both small
  and deterministic: together ~3-5 min/checkpoint on an H200 → ~7-10 GPU-h across the full 120-ckpt
  sweep. Fine within budget, and this is the only added cost.

So all deterministic forgetting axes (KL + math + IFEval) can be logged per checkpoint affordably;
only the pedagogy judge (new-task quality) stays final-only or on a subset.

## First-run failures (2026-07-29) — what bit us and how it's now prevented

Both issues below happened on the first end-to-end ORCD run, and the job **still reported
`COMPLETED 0:0`** for each. Lesson: `sacct` state is NOT proof of success — always run the
**post-run sanity check** at the end of this section.

### 1. Only variant a ran (variant b silently skipped)
- **Symptom:** W&B/logs show only `impl3-a-*`, no `impl3-b-*`. The `.out` contains
  `WARNING: vanilla adapter 'checkpoint-923' not found -> running variant a only`.
- **Cause:** only the *first* Step-0 rsync (the `post-training/` folder) was run; the *second* rsync
  that ships `checkpoint-923/` (variant b's reference adapter) was skipped, so it wasn't on ORCD. The
  sbatch guard then sets `RUN_VARIANT_B=0` and continues (variant a doesn't depend on it).
- **Prevention:** run **both** Step-0 rsyncs; before submit `ls checkpoint-923/adapter_*` on ORCD;
  after submit `grep "effective variant_b" logs/p7-impl3_<jobid>.out` must print `=1`.

### 2. Eval stage produced nothing, yet the job said COMPLETED
- **Symptom:** every KL point logs `SKIP (load failed): … You need to have sentencepiece or tiktoken
  installed …`; every math/IFEval line is `[warn] … gen … failed`; `out/kl_by_checkpoint.json` is
  empty and there are no `eval/*/results_*.jsonl`. `sacct` still shows `COMPLETED 0:0`.
- **Cause:** the `p7post` env was missing **`sentencepiece`** (and `tiktoken`), which the eval scripts
  need to rebuild the tokenizer from a checkpoint dir. Training didn't hit it (base model's cached fast
  tokenizer). The eval calls are wrapped in `|| echo "[warn] … failed"`, so they don't fail the job.
  The Step-0.6 smoke test only exercises *training*, so it didn't catch it.
- **Prevention:** `sentencepiece` + `tiktoken` are now pinned in `requirements.txt` and imported by
  `setup_orcd_env.sh`'s verify step. On an older env: `pip install sentencepiece tiktoken`.

### Post-run sanity check (run after EVERY job — "COMPLETED" is not enough)
```bash
J=<jobid>
grep -E "effective variant_b|ALL DONE" logs/p7-impl3_$J.out          # variant_b=1? reached ALL DONE?
python -c "import json; print('KL points:', len(json.load(open('out/kl_by_checkpoint.json'))))"  # >0?
ls -la eval/math_eval/results_impl3-*.jsonl                                                      # exist & non-empty?
```
If KL points = 0 or the `results_*.jsonl` are missing/empty, the eval stage failed silently — fix the
env and re-run **only the eval stage** (it reads the saved checkpoints; **no retrain needed**).

## Pasting gotchas on ORCD (things that bit us)

- **Run rsync on the Mac, not on ORCD.** The source path only exists on your Mac; running it in
  the SSH session gives `change_dir … No such file or directory`. Check the prompt first.
- **One line per paste.** Many SSH terminals accept only a single line at a time. Chain steps with
  `&&` into one line, or type them one at a time.
- **No inline `#` comments.** Interactive `zsh` does NOT treat `#` as a comment by default, so a
  pasted `# note; else …` is parsed as commands → `zsh: parse error near 'else'`. Strip comments.
- **Substitute `<placeholders>`.** `<your_key>` / `<yourpath>` / `<id>` contain `<` `>` which are
  shell redirection operators — replace them with real values (and drop the angle brackets).
- **`salloc` vs `sbatch`.** `salloc` blocks your terminal and dies on disconnect; if a pending
  `salloc` won't cancel via `scancel`, press Ctrl+C in that terminal. Prefer `sbatch` always.
- **Rotate leaked keys.** If you ever `export WANDB_API_KEY=...` inline, rotate it at
  `wandb.ai/settings` afterward and switch to `wandb login`.
- **"Completed" in ~1 second = it failed, not finished.** Check `Time Used`; a 1s "Completed"
  job did no real work. Usual cause: the `p7post` env isn't set up (`conda: command not found` in
  `logs/*.err`). Do Step 0.5. The script now aborts with `FATAL:` + exits non-zero in this case,
  so it will show as `FAILED` rather than a fake "Completed."
- **Activate the env before `wandb` / `python` / `snapshot_hf_dataset.py`.** A fresh SSH session
  is NOT in `p7post` (no `(p7post)` in the prompt) — those tools live in the env:
  `source ~/miniforge3/etc/profile.d/conda.sh && conda activate p7post`. (The batch job activates
  it itself; this only matters for commands you type by hand.)
- **`requirements file not found` during setup** was a CWD bug (fixed): the script now resolves
  its own path up front. If you see it, re-sync (Step 0) and rerun `setup_orcd_env.sh`.

## Notes

- If a run is preempted, re-run the same command / resubmit — `--resume auto` picks up the last checkpoint.
- To recover exact vanilla Impl-2 as a sanity point: `--variant a --temperature 1e9` (T→∞ ⇒ all weights 1.0).
- The sbatch script auto-detects the final (highest-numbered) checkpoint per run, so it's robust
  if the exact step count differs from 938.
