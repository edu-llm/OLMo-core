#!/usr/bin/env python
"""Headless driver. Saturates every GPU on the box without touching a single training number.

## Where the parallelism comes from, and where it deliberately does not

D4 ran on **one** L40S of a ``gpu-4xl40s`` and left three idle. The obvious fix — bigger
micro-batches — is the one option that is not available: ``per_device_batch 8 × grad_accum 4``
is A1's and D4's, and regrouping micro-batches changes each example's contribution to the step
whenever a group's token counts are uneven. That would void the loss-normalisation result Impl
5 inherited from A1 (its acceptance check 4) and put a second variable into a contrast built to
isolate one. Same argument rules out flipping ``gradient_checkpointing`` off for James's "~30%
faster": with LoRA dropout at 0.05, activation recompute is not provably a numerical no-op.

So utilisation comes entirely from **running independent work concurrently**, which changes no
arithmetic anywhere:

============  ==========================  =========================  ==================
stage         serial                      here                       on 4 GPUs
============  ==========================  =========================  ==================
precompute    1 pass over 22,152 rows     4 row-shards + merge       ~4x
train         4 arms back to back         4 arms, 1 per GPU          ~4x
eval/math     4 arms back to back         4 arms, 1 per GPU          ~4x
============  ==========================  =========================  ==================

The arm count is chosen to match the GPU count: three conditions (``bT1``, ``bT2``, ``aT8``)
plus the ``bT451`` control that James recommends and that would otherwise not be affordable.
The fourth GPU is why the control is free.

**The math stage regenerates the base model's answers once per shard.** ``math_only.py`` hoists
them out of its checkpoint loop but keeps them in memory, so four concurrent processes each
redo ~500 completions plus the KL continuations. That is ~4x the GPU-seconds for 1x the wall
clock on GPUs that are otherwise idle, and the alternative is editing their eval code — which
is the one thing that would make these numbers incomparable to Impl 3's and Impl 4's. Left
redundant on purpose.

## Stages

``deps, bundle, fetch, pool, slot, mix, checks_fast, precompute, checks_full, train, bridge,
eval, math``

``fetch`` pulls the two artefacts that cannot be rebuilt on a CPU: D4's distilled pool
(~90 accelerator-minutes to regenerate) and D4's ``ckpt-923``, which is variant b's reference
π_SFT. ``mix`` then rebuilds D4's training file deterministically and ``mix_arm5.py`` asserts
the replay slot reproduces A1 bit-for-bit, so a regenerated mix is a verified mix rather than a
trusted one.

``checks_fast`` runs before the precompute and **decides ``--loss_denom``** (check W1), which
every arm is then passed explicitly rather than left to guess.

    python run_klw.py                                  # everything, all GPUs
    python run_klw.py --stages precompute,train
    python run_klw.py --stages bundle,math --adapters_from s3://<this run>/checkpoints
    python run_klw.py --poc --gpus 1                   # 63-step smoke on one GPU
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC_ROOT = HERE.parent
IMPL5 = POC_ROOT / "impl5_ssd"
IMPL4 = POC_ROOT / "impl4_ssd"
COMPAT = IMPL4 / "impl3_compat"

sys.path.insert(0, str(HERE))
from klw.config_klw import ALL_ARMS, CONDITION_ARMS, CONTROL_ARMS, DATA_ARM  # noqa: E402
from klw.config_klw import REFERENCE_ADAPTER_ARM, REFERENCE_ADAPTER_STEP     # noqa: E402
from klw.config_klw import RUN_PREFIX, variants_needed                       # noqa: E402

PINS = ["transformers==5.14.1", "datasets==5.0.1", "accelerate==1.14.0", "peft==0.20.0",
        "huggingface_hub==1.25.1", "numpy==2.4.6", "langdetect==1.0.9",
        "pyarrow==25.0.0", "matplotlib==3.11.1"]

ALL_STAGES = ("deps", "bundle", "fetch", "pool", "slot", "mix", "checks_fast", "precompute",
              "checks_full", "train", "bridge", "eval", "math")
GPU_STAGES = {"precompute", "train", "eval", "math"}

#: impl4's 12 math steps. D4 is the baseline and a step D4 never measured has no baseline.
MATH_STEPS_DEFAULT = "1,2,4,8,16,32,64,128,256,512,923"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default=",".join(ALL_ARMS),
                   help=f"Comma-separated. Conditions {CONDITION_ARMS}, control {CONTROL_ARMS}.")
    p.add_argument("--stages", default="all")
    p.add_argument("--gpus", type=int, default=0, help="0 = autodetect.")
    p.add_argument("--data_arm", default=DATA_ARM)
    p.add_argument("--runs_root", default=None, help="Where THIS implementation's arms go.")
    p.add_argument("--impl5_runs_root", default=None, help="Where D4's data + ckpt-923 live.")
    p.add_argument("--poc", action="store_true")

    p.add_argument("--per_device_batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--loss_denom", default=None,
                   help="Normally set by checks_fast (W1). Overrides it if given.")
    p.add_argument("--max_batch_tokens", type=int, default=0,
                   help="Precompute forward budget. 0 = size from GPU memory.")

    # Absolute defaults, pointing at where the tarball actually lives in the repo. They used to
    # be Colab's /content paths, inherited from run_impl5.py, which meant every platform
    # submission had to override both -- and a *relative* override is a trap, because `sh()`
    # runs with cwd=HERE while `Path(...).exists()` resolves against the process cwd. On the
    # platform the process cwd is the repo root, so `p7/POC/impl3_handoff.tar.gz` passed the
    # existence check and then failed inside tar. Both are resolved to absolute in main().
    p.add_argument("--bundle", default="/tmp/impl3_handoff")
    p.add_argument("--bundle_tar", default=str(POC_ROOT / "impl3_handoff.tar.gz"))
    p.add_argument("--artifacts", default=None)
    p.add_argument("--checkpoint_dir", default=os.environ.get("EDULLM_CHECKPOINT_DIR") or None)
    p.add_argument("--output_prefix", default=os.environ.get("EDULLM_OUTPUT_PREFIX") or None)
    # Two different prefixes, and conflating them is a trap worth spelling out. `fetch` wants
    # D4's run (the pool and the reference pi_SFT); `math` wants THIS run's own arms. They are
    # different S3 prefixes belonging to different runs, so they get different flags.
    p.add_argument("--pool_from", default=None,
                   help="S3 prefix holding impl5_pool.tar.gz — D4's distilled pool.")
    p.add_argument("--reference_from", default=None,
                   help="S3 prefix holding impl5_D4.tar — variant b's reference pi_SFT. "
                        "D4's TRAINING run, not this one.")
    p.add_argument("--adapters_from", default=None,
                   help="S3 prefix holding impl3x5_<arm>.tar — THIS implementation's arms, for "
                        "a later math job. Not D4's.")
    p.add_argument("--math_steps", default=MATH_STEPS_DEFAULT)
    p.add_argument("--skip_package", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# process plumbing
# --------------------------------------------------------------------------- #
def banner(text: str) -> None:
    print("\n" + "=" * 74 + f"\n== {text}\n" + "=" * 74, flush=True)


def sh(cmd: str, cwd: Path = HERE, check: bool = True, log_path: Path | None = None,
       env_extra: dict | None = None) -> int:
    """Run one command, streaming live.

    Tees in Python rather than shelling out: ``cmd | tee f`` reports *tee's* exit status, so a
    failed training run would come back 0 and the driver would sail on to eval with no
    checkpoints. (Inherited from ``run_impl5.py``, which learned it.)
    """
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    fh = open(log_path, "a", encoding="utf-8") if log_path else None
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONUNBUFFERED="1",
                                     TOKENIZERS_PARALLELISM="false", **(env_extra or {})))
    for line in proc.stdout:
        print(line, end="", flush=True)
        if fh:
            fh.write(line)
    rc = proc.wait()
    if fh:
        fh.close()
    print(f"[exit {rc} in {(time.time() - t0) / 60:.1f} min]", flush=True)
    if check and rc:
        raise SystemExit(f"stage failed (exit {rc}): {cmd}")
    return rc


def sh_parallel(jobs: list[dict], check: bool = True) -> dict[str, int]:
    """Run jobs concurrently, one per GPU, with prefixed interleaved output.

    Each job is ``{"name", "cmd", "gpu", "log"}``. ``CUDA_VISIBLE_DEVICES`` is set per job so
    each process sees exactly one device as ``cuda:0`` — the training and eval scripts then need
    no device argument and no distributed setup, because nothing here is distributed. Four
    independent single-GPU jobs is the whole design.

    Output is line-prefixed with the job name because four training runs interleaving into one
    log is otherwise unreadable. Every job also gets its own log file, which is what to read.
    """
    if not jobs:
        return {}
    lock = threading.Lock()
    rcs: dict[str, int] = {}

    def one(job):
        env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        if job.get("gpu") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        log = Path(job["log"]) if job.get("log") else None
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "a", encoding="utf-8") if log else None
        proc = subprocess.Popen(job["cmd"], shell=True, cwd=job.get("cwd", HERE), text=True,
                                bufsize=1, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env)
        for line in proc.stdout:
            with lock:
                print(f"[{job['name']}] {line}", end="", flush=True)
            if fh:
                fh.write(line)
        rc = proc.wait()
        if fh:
            fh.close()
        with lock:
            rcs[job["name"]] = rc
            print(f"[{job['name']}] [exit {rc}]", flush=True)

    t0 = time.time()
    print(f"\n-- {len(jobs)} jobs in parallel: "
          + ", ".join(f"{j['name']}@gpu{j.get('gpu')}" for j in jobs), flush=True)
    for j in jobs:
        print(f"   $ {j['cmd']}", flush=True)
    threads = [threading.Thread(target=one, args=(j,), daemon=False) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[all {len(jobs)} jobs done in {(time.time() - t0) / 60:.1f} min: {rcs}]", flush=True)
    failed = {k: v for k, v in rcs.items() if v}
    if check and failed:
        raise SystemExit(f"parallel stage failed: {failed}")
    return rcs


def gpu_report(required: bool) -> tuple[int, int]:
    """``(n_gpus, per_gpu_GiB)``."""
    try:
        import torch
        n = torch.cuda.device_count()
        if not n:
            raise RuntimeError("no CUDA device")
        gib = int(torch.cuda.get_device_properties(0).total_memory / 2**30)
        names = {torch.cuda.get_device_name(i) for i in range(n)}
        print(f"GPU: {n} x {'/'.join(sorted(names))}, {gib} GiB each", flush=True)
        return n, gib
    except Exception as exc:                                          # noqa: BLE001
        if required:
            raise SystemExit(f"GPU stages were requested but no CUDA device is usable: {exc}")
        print(f"GPU: none ({exc}) — CPU-only stages", flush=True)
        return 0, 0


def precompute_budget(gib: int, override: int) -> int:
    """Padded-token budget per forward, sized from memory.

    The peak allocation is two ``[n_sel, V]`` bf16 selections plus one full ``[B, L, V]`` logits
    tensor, so it scales linearly in the budget: roughly 0.6 GiB per 1,024 padded tokens at
    OLMo-2's 100k vocab. Two thirds of the card, leaving room for the model and fragmentation.
    """
    if override:
        return override
    if gib <= 0:
        return 4096
    return max(4096, int((gib * 0.66) / 0.6) * 1024 // 1024 * 1024)


# --------------------------------------------------------------------------- #
# S3 (no-ops off-platform, exactly as in run_impl5.py)
# --------------------------------------------------------------------------- #
def s3_split(uri: str) -> tuple[str, str]:
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key.rstrip("/")


def s3_put(local: Path, uri_prefix: str | None, name: str | None = None) -> None:
    if not uri_prefix or not Path(local).exists():
        return
    import boto3
    bucket, key = s3_split(uri_prefix)
    key = f"{key}/{name or Path(local).name}" if key else (name or Path(local).name)
    try:
        boto3.client("s3").upload_file(str(local), bucket, key)
        print(f"  s3 up   s3://{bucket}/{key} "
              f"({Path(local).stat().st_size / 2**20:.1f} MB)", flush=True)
    except Exception as exc:                                          # noqa: BLE001
        print(f"  s3 FAIL s3://{bucket}/{key}: {exc}", flush=True)


def s3_get(uri_prefix: str | None, name: str, local: Path) -> bool:
    if not uri_prefix:
        return False
    import boto3
    bucket, key = s3_split(uri_prefix)
    key = f"{key}/{name}" if key else name
    try:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(bucket, key, str(local))
        print(f"  s3 down s3://{bucket}/{key} -> {local}", flush=True)
        return True
    except Exception:                                                 # noqa: BLE001
        return False


def fetch_d4_assets(args, impl5_runs: Path) -> None:
    """Pull D4's distilled pool and its ckpt-923. Both are prerequisites, for different reasons.

    The pool is ~90 accelerator-minutes of generation and cannot be rebuilt on a CPU. ckpt-923
    is variant b's reference π_SFT — without it ``bT1``, ``bT2`` and ``bT451`` have no signal at
    all, and it is 1 GB of adapters that a training job would otherwise have to reproduce by
    retraining D4.
    """
    pool = IMPL5 / "data" / "distilled_pool.jsonl"
    if pool.exists():
        print(f"  distilled pool already on disk ({pool})", flush=True)
    else:
        src = args.pool_from or args.output_prefix
        tmp = IMPL5 / "data" / "_pool.tar.gz"
        if not s3_get(src, "impl5_pool.tar.gz", tmp):
            raise SystemExit(
                f"no distilled pool at {pool} and impl5_pool.tar.gz is not under "
                f"{src!r}. Pass --pool_from with the prefix of the D4 run that produced it "
                f"(run_019fc3ec-96a2-70fe-8153-21545ef0e908). Regenerating it costs ~90 "
                f"accelerator-minutes and is not something this driver will do silently.")
        sh(f"tar xzf {shlex.quote(str(tmp))} -C {shlex.quote(str(IMPL5))}")
        tmp.unlink(missing_ok=True)
        if not pool.exists():
            raise SystemExit("tarball did not contain data/distilled_pool.jsonl")
        rounds = len(list((IMPL5 / "data" / "distill").glob("round-*.jsonl")))
        print(f"  restored pool + {rounds} round caches (mix_arm5 needs the rounds for "
              f"realised-delta accounting)", flush=True)

    ref = impl5_runs / REFERENCE_ADAPTER_ARM / f"ckpt-{REFERENCE_ADAPTER_STEP}"
    if any(ref.glob("adapter_model*")):
        print(f"  reference pi_SFT already on disk ({ref})", flush=True)
        return
    if "b" not in variants_needed(tuple(a.strip() for a in args.arms.split(","))):
        print("  no variant-b arm requested — reference pi_SFT not needed", flush=True)
        return
    src = args.reference_from or args.pool_from
    tmp = impl5_runs / "_d4.tar"
    impl5_runs.mkdir(parents=True, exist_ok=True)
    if not s3_get(src, f"impl5_{REFERENCE_ADAPTER_ARM}.tar", tmp):
        raise SystemExit(
            f"variant b needs {ref} as pi_SFT and impl5_{REFERENCE_ADAPTER_ARM}.tar is not "
            f"under {src!r}. Pass --reference_from with the prefix D4's TRAINING run wrote its "
            f"checkpoints to (usually <that run's prefix>/checkpoints). Note this is NOT "
            f"--adapters_from, which names this implementation's own arms, and NOT this job's "
            f"EDULLM_CHECKPOINT_DIR, which is derived per run id and empty by construction.")
    sh(f"cd {shlex.quote(str(impl5_runs))} && tar xf {shlex.quote(str(tmp))}")
    tmp.unlink(missing_ok=True)
    if not any(ref.glob("adapter_model*")):
        raise SystemExit(f"tarball did not contain {REFERENCE_ADAPTER_ARM}/ckpt-"
                         f"{REFERENCE_ADAPTER_STEP}")
    print(f"  restored reference pi_SFT -> {ref}", flush=True)


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ALL_ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {', '.join(ALL_ARMS)}")

    want = set(ALL_STAGES) if args.stages == "all" else {
        s.strip() for s in args.stages.split(",")}
    unknown = want - set(ALL_STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}; known: {', '.join(ALL_STAGES)}")

    # Resolved before anything reads them. Every `sh()` runs with an explicit cwd, so a
    # relative path here would mean one thing to `Path.exists()` and another to the
    # subprocess -- which is exactly how the first platform attempt died: the tarball existed
    # relative to the process cwd, passed the guard, then tar could not open it.
    args.bundle = str(Path(args.bundle).expanduser().resolve())
    args.bundle_tar = str(Path(args.bundle_tar).expanduser().resolve())

    runs_root = Path(args.runs_root) if args.runs_root else HERE / "runs"
    impl5_runs = Path(args.impl5_runs_root) if args.impl5_runs_root else IMPL5 / "runs"
    art = Path(args.artifacts) if args.artifacts else HERE / "artifacts"
    runs_root.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)
    poc = "--poc" if args.poc else ""
    rr = shlex.quote(str(runs_root))
    i5 = shlex.quote(str(impl5_runs))

    banner(f"Impl 3x5 — James's weighting on Impl 5's targets | arms: {', '.join(arms)}")
    print(f"stages     : {[s for s in ALL_STAGES if s in want]}")
    print(f"runs root  : {runs_root}")
    print(f"impl5 runs : {impl5_runs}  (data arm {args.data_arm}, shared by every arm)")
    print(f"artifacts  : {art}")

    n_gpu, gib = gpu_report(required=bool(want & GPU_STAGES))
    if args.gpus:
        n_gpu = min(args.gpus, n_gpu) if n_gpu else args.gpus
    n_gpu = max(n_gpu, 1)
    print(f"parallelism: {n_gpu} concurrent job(s)", flush=True)

    if "deps" in want:
        banner("deps")
        sh(f"{sys.executable} -m pip -q install {' '.join(shlex.quote(p) for p in PINS)}",
           check=False)
        sh(f"{sys.executable} -m pip -q uninstall -y torchao", check=False)

    if "bundle" in want:
        banner("bundle — Impl 3's eval assets")
        if not Path(args.bundle, "eval/sweep_ckpt_eval.py").exists():
            if not Path(args.bundle_tar).exists():
                raise SystemExit(f"{args.bundle_tar} not present. Send impl3_handoff.tar.gz "
                                 f"first.")
            sh(f"tar xzf {shlex.quote(args.bundle_tar)} -C "
               f"{shlex.quote(str(Path(args.bundle).parent))}")
        sh(f"{sys.executable} impl3_compat/setup_compat.py --bundle "
           f"{shlex.quote(args.bundle)}", cwd=IMPL4)

    if "fetch" in want:
        banner("fetch — D4's distilled pool and its ckpt-923 (variant b's reference)")
        fetch_d4_assets(args, impl5_runs)

    if "pool" in want:
        banner("pool — the GOLD pedagogy pool (Impl 4's, pinned Hub revision)")
        sh(f"{sys.executable} build_pedagogy_pool.py", cwd=IMPL4)
        src = json.loads((IMPL4 / "data/pedagogy_pool/pool_source.json").read_text())
        if not src.get("comparable_to_impl3"):
            raise SystemExit("pedagogy pool has regenerated SIs — not comparable to Impl 3")

    if "slot" in want:
        banner(f"slot — Tulu-3 gold replay for {args.data_arm} (must reproduce impl4-A1)")
        sh(f"{sys.executable} build_general_slot5.py --arm {args.data_arm} --runs_root {i5} "
           f"{poc}", cwd=IMPL5)

    if "mix" in want:
        banner(f"mix — rebuild {args.data_arm}'s training file (shared by every arm)")
        sh(f"{sys.executable} mix_arm5.py --arm {args.data_arm} --runs_root {i5} {poc}",
           cwd=IMPL5)

    # checks_fast decides --loss_denom, so it has to run before train and it is cheap enough
    # to run before the precompute too.
    denom = args.loss_denom
    if "checks_fast" in want:
        banner("acceptance checks (fast) — no GPU; decides --loss_denom (W1)")
        sh(f"{sys.executable} acceptance_checks_klw.py --stage fast")
        got = json.loads((HERE / "data/acceptance_fast.json").read_text())
        denom = args.loss_denom or got["loss_denom"]
        print(f"  W1 chose --loss_denom {denom}"
              + ("" if got["W1_unit_weights"]["auto_agrees"] else
                 "   (note: 'auto' does NOT agree — passing it explicitly matters here)"),
              flush=True)
    if "train" in want and not denom:
        cached = HERE / "data/acceptance_fast.json"
        if cached.exists():
            denom = json.loads(cached.read_text())["loss_denom"]
            print(f"  reusing --loss_denom {denom} from {cached.name}", flush=True)
        else:
            raise SystemExit("training needs --loss_denom, which acceptance check W1 "
                             "determines. Run --stages checks_fast first, or pass it.")

    if "precompute" in want:
        variants = ",".join(variants_needed(tuple(arms)))
        budget = precompute_budget(gib, args.max_batch_tokens)
        banner(f"precompute — signals for variant(s) {variants} on {n_gpu} GPU(s), "
               f"{budget} padded tokens/forward")
        common = (f"{sys.executable} precompute_signal.py --variants {variants} "
                  f"--impl5_runs_root {i5} --max_batch_tokens {budget}")
        if n_gpu > 1:
            sh_parallel([{
                "name": f"pre{k}", "gpu": k, "log": HERE / f"data/precompute.{k}.log",
                "cmd": f"{common} --shard {k} --n_shards {n_gpu}",
            } for k in range(n_gpu)])
            sh(f"{common} --merge --n_shards {n_gpu}", log_path=HERE / "data/precompute.log")
        else:
            sh(common, log_path=HERE / "data/precompute.log")

    if "checks_full" in want:
        banner("acceptance checks (full) — real mix, real cache, per arm")
        sh(f"{sys.executable} acceptance_checks_klw.py --stage full --impl5_runs_root {i5}")

    if "train" in want:
        banner(f"train — {len(arms)} arm(s) concurrently, one per GPU, loss_denom={denom}")
        jobs = []
        for k, arm in enumerate(arms):
            out_dir = runs_root / arm
            out_dir.mkdir(parents=True, exist_ok=True)
            jobs.append({
                "name": arm, "gpu": k % n_gpu, "log": out_dir / "train.log",
                "cmd": (f"{sys.executable} train_sft_klw.py --arm {arm} --runs_root {rr} "
                        f"--impl5_runs_root {i5} --data_arm {args.data_arm} {poc} "
                        f"--resume auto --loss_denom {denom} "
                        f"--per_device_batch {args.per_device_batch} "
                        f"--grad_accum {args.grad_accum} "
                        f"--save_steps 100 --save_total_limit 1"),
            })
        if len(arms) > n_gpu:
            print(f"  NOTE: {len(arms)} arms on {n_gpu} GPU(s) — jobs will contend for memory. "
                  f"Run in batches of {n_gpu} instead if this OOMs.", flush=True)
        sh_parallel(jobs)
        if not args.skip_package:
            # Uncompressed, immediately. Gzipping ~1 GB of adapter safetensors takes ~15 min
            # and compresses them by almost nothing; a container reclaimed in that window
            # takes the whole run with it. (run_impl5.py learned this.)
            for arm in arms:
                tarball = art / f"impl3x5_{arm}.tar"
                sh(f"cd {rr} && tar cf {shlex.quote(str(tarball))} "
                   f"--exclude='checkpoint-*' --exclude='*.jsonl' {shlex.quote(arm)}",
                   check=False)
                s3_put(tarball, args.checkpoint_dir)

    if "bridge" in want:
        banner("bridge — expose checkpoints in Impl 3's layout")
        sh(f"{sys.executable} impl3_compat/bridge.py --runs_root {rr} "
           f"--prefix {RUN_PREFIX} --arms {' '.join(arms)}", cwd=IMPL4)

    if "eval" in want:
        banner(f"eval — pedagogy NLL, {len(arms)} arm(s) concurrently")
        sh_parallel([{
            "name": f"nll:{arm}", "gpu": k % n_gpu, "cwd": COMPAT,
            "log": HERE / f"data/nll.{arm}.log",
            "cmd": (f"{sys.executable} nll_only.py --runs 'out/{RUN_PREFIX}{arm}' "
                    f"--out out/ped_nll_{RUN_PREFIX}{arm}.jsonl"),
        } for k, arm in enumerate(arms)])
        merged = COMPAT / "work" / "out" / "ped_nll_impl3x5.jsonl"
        merge_jsonl([COMPAT / "work" / "out" / f"ped_nll_{RUN_PREFIX}{a}.jsonl"
                     for a in arms], merged)
        s3_put(merged, args.output_prefix)

    if "math" in want:
        # Self-contained like run_impl5.py's: restores adapters and does its own bridge, so
        # `--stages bundle,fetch,math` is a complete job on a fresh container. Do NOT also
        # request `bridge` in the same invocation — bridge.py skips checkpoint dirs that
        # already exist, so an unfiltered bridge first would leave all 22 steps exposed and
        # the step filter below would silently do nothing.
        banner(f"math — retention + KL on {args.math_steps.count(',') + 1} steps, "
               f"{len(arms)} arm(s) concurrently")
        for arm in arms:
            restore_arm(runs_root, arm, args)
        steps = f" --steps {args.math_steps}" if args.math_steps not in ("all", "") else ""
        sh(f"{sys.executable} impl3_compat/bridge.py --runs_root {rr} --prefix {RUN_PREFIX} "
           f"--arms {' '.join(arms)}{steps}", cwd=IMPL4)
        # --with_kl, not --with-kl: math_only.py's docstring writes it dashed but argparse
        # registered the underscore, and the dashed form is not a prefix of it.
        sh_parallel([{
            "name": f"math:{arm}", "gpu": k % n_gpu, "cwd": COMPAT,
            "log": HERE / f"data/math.{arm}.log",
            "cmd": (f"{sys.executable} math_only.py --runs 'out/{RUN_PREFIX}{arm}' "
                    f"--out out/math_{RUN_PREFIX}{arm}.jsonl --with_kl"),
        } for k, arm in enumerate(arms)])
        merged = COMPAT / "work" / "out" / "math_impl3x5.jsonl"
        merge_jsonl([COMPAT / "work" / "out" / f"math_{RUN_PREFIX}{a}.jsonl" for a in arms],
                    merged, dedupe_base=True)
        s3_put(merged, args.output_prefix)

    if not args.skip_package:
        banner("packaging results")
        # Collected into one staging dir rather than tarred in place: the result rows live under
        # impl4_ssd/impl3_compat/work/out and the acceptance + weighting JSON under ./data, and
        # a single `tar -C` cannot span both.
        stage = art / "results"
        sh(f"rm -rf {shlex.quote(str(stage))} && mkdir -p {shlex.quote(str(stage))}",
           check=False)
        for src in sorted((COMPAT / "work" / "out").glob("*impl3x5*.jsonl")):
            sh(f"cp {shlex.quote(str(src))} {shlex.quote(str(stage))}/", check=False)
        for src in sorted(HERE.glob("data/*.json")):
            sh(f"cp {shlex.quote(str(src))} {shlex.quote(str(stage))}/", check=False)
        for arm in arms:
            mf = runs_root / arm / "manifest.json"
            if mf.exists():
                sh(f"cp {shlex.quote(str(mf))} {shlex.quote(str(stage))}/manifest_{arm}.json",
                   check=False)
            s3_put(mf, args.output_prefix, f"manifest_{arm}.json")
        bundle = art / "impl3x5_results.tar.gz"
        sh(f"cd {shlex.quote(str(art))} && tar czf {shlex.quote(str(bundle))} results",
           check=False)
        s3_put(bundle, args.output_prefix)
    banner("done")


def merge_jsonl(sources, dest: Path, dedupe_base: bool = False) -> None:
    """Concatenate per-arm result files, dropping duplicate ``base`` rows.

    Every math shard scores the base model independently (see the module docstring), so four
    identical ``base`` rows arrive. They are deterministic greedy decodes, so keeping the first
    is not a choice between differing numbers — but leaving four in would make any groupby over
    the file count base four times.
    """
    seen, lines = set(), []
    for src in sources:
        if not Path(src).exists():
            print(f"  merge: {src} missing — skipped", flush=True)
            continue
        for line in open(src, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec.get("run"), rec.get("step"))
            if key in seen:
                if not (dedupe_base and rec.get("run") == "base"):
                    print(f"  merge: duplicate {key} — kept first", flush=True)
                continue
            seen.add(key)
            lines.append(line if line.endswith("\n") else line + "\n")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(lines), encoding="utf-8")
    print(f"  merged {len(lines)} rows -> {dest}", flush=True)


def restore_arm(runs_root: Path, arm: str, args) -> None:
    """Pull one arm's adapters back if the math job started in a fresh container."""
    out = runs_root / arm
    if list(out.glob("ckpt-*")):
        print(f"  {arm}: adapters already on disk", flush=True)
        return
    src = args.adapters_from or args.checkpoint_dir
    tmp = runs_root / f"_restore_{arm}.tar"
    if not s3_get(src, f"impl3x5_{arm}.tar", tmp):
        raise SystemExit(
            f"no {arm} adapters under {runs_root} and impl3x5_{arm}.tar is not under {src!r}. "
            f"The math axis scores the adapters training wrote; it does not retrain. If that "
            f"prefix is this job's own EDULLM_CHECKPOINT_DIR it is empty by construction — "
            f"pass --adapters_from with the TRAINING run's prefix.")
    sh(f"cd {shlex.quote(str(runs_root))} && tar xf {shlex.quote(str(tmp))}")
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
