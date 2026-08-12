#!/usr/bin/env python3
"""Platform entrypoint. Runs one cell per visible GPU.

The platform scaffold generates a command of the form

    bash -lc '<launcher> .edullm/train.py "$EDULLM_RUN_ID" --save-folder "$EDULLM_CHECKPOINT_DIR"'

so this matches that signature: a positional run id and `--save-folder`.

## Why it spawns processes itself

The only provisioned A100 shape is `gpu-8xa100` -- there is no single-GPU A100 --
and a 40M-parameter model has no use for data-parallel across 8 cards. So instead
of one run on 8 GPUs, this puts **one independent arm on each GPU**: 8 cells of the
matrix finish in the wall-clock time of one.

That means GPUs > 1 with no recognised launcher, which the platform's text-reading
guard refuses as `process_per_device`. The sanctioned escape is
`EDULLM_LAUNCH_CHECK=waived` in the command, which is precedent-backed: another
workload does exactly this for a custom entrypoint that execs its own launcher.

On a single-GPU shape (`gpu-1xl40s`, `gpu-1xa10g`) this runs one cell and no waiver
is needed.

## Two details that matter

**The corpus is staged once, by the parent.** Eight children each pulling the same
1.5 GB from S3 would waste bandwidth and race on the same destination path. The
parent stages the union of sidecars the cells need; children then read local files.

**Each cell gets its own checkpoint subdirectory.** All cells share one
`$EDULLM_CHECKPOINT_DIR`, so without `cell-<i>/` they would overwrite each other's
`ckpt.pt` and the ResumeGuard would see a marker describing someone else's state.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_cells(spec: str) -> list[tuple[str, int]]:
    """`"dense:0,split:0"` -> `[("dense", 0), ("split", 0)]`."""
    out: list[tuple[str, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        cond, _, seed = chunk.partition(":")
        out.append((cond.strip(), int(seed or 0)))
    return out


def visible_gpu_count() -> int:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return len([x for x in env.split(",") if x.strip() != ""])
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--data-root", default=os.environ.get("MEMSPLIT_DATA_ROOT"))
    ap.add_argument("--config", default=os.environ.get(
        "MEMSPLIT_CONFIG", "configs/depth_d40m.yaml"))
    ap.add_argument("--cells", default=os.environ.get("MEMSPLIT_CELLS"),
                    help='e.g. "dense:0,split:0,random_contig:0"')
    ap.add_argument("--condition", default=os.environ.get("MEMSPLIT_CONDITION", "dense"))
    ap.add_argument("--seed", default=os.environ.get("MEMSPLIT_SEED", "0"))
    ap.add_argument("--stage-dir", default=os.environ.get(
        "MEMSPLIT_STAGE_DIR", "/tmp/memsplit-corpus"))
    args, extra = ap.parse_known_args()

    cells = parse_cells(args.cells) if args.cells else [(args.condition, int(args.seed))]
    n_gpu = visible_gpu_count()
    print(f"[{args.run_id}] {len(cells)} cell(s), {n_gpu} visible GPU(s)", flush=True)
    if n_gpu and len(cells) > n_gpu:
        raise SystemExit(
            f"{len(cells)} cells requested but only {n_gpu} GPUs visible; "
            "split the matrix across more jobs"
        )

    # Stage once, in the parent, for the union of sidecars the cells need.
    data_root = args.data_root
    if data_root:
        from memsplit import checkpoint_io as cio

        if cio.is_s3(data_root):
            names = ["tokens.bin"] + sorted({f"weights.{c}.bin" for c, _ in cells})
            staged = cio.stage_files(data_root, args.stage_dir, names)
            print(f"[{args.run_id}] staged {names} -> {staged}", flush=True)
            data_root = str(staged)

    procs = []
    for i, (cond, seed) in enumerate(cells):
        env = dict(os.environ)
        if n_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(i)
        # Per-cell checkpoint prefix, or the cells overwrite each other.
        ckpt = args.save_folder.rstrip("/") + (f"/cell-{i}" if len(cells) > 1 else "")
        cmd = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--config", args.config,
            "--condition", cond,
            "--seed", str(seed),
            "--run-id", f"{args.run_id}-{cond}-s{seed}",
            "--checkpoint-dir", ckpt,
            "--out-dir", f"outputs/{args.run_id}/cell-{i}",
            *extra,
        ]
        if data_root:
            cmd += ["--data-root", data_root]
        print(f"[cell {i}] gpu={env.get('CUDA_VISIBLE_DEVICES', '-')} "
              f"{cond} seed={seed}", flush=True)
        procs.append((i, cond, seed, subprocess.Popen(cmd, cwd=ROOT, env=env)))

    failures = []
    for i, cond, seed, proc in procs:
        rc = proc.wait()
        print(f"[cell {i}] {cond} seed={seed}: "
              f"{'ok' if rc == 0 else f'FAILED rc={rc}'}", flush=True)
        if rc != 0:
            failures.append((i, cond, seed, rc))

    if failures:
        # Non-zero, so the attempt is recorded as failed rather than a silent
        # partial success. A matrix with a missing cell is not a matrix.
        print(f"[{args.run_id}] {len(failures)} of {len(cells)} cells failed", flush=True)
        return 1
    print(f"[{args.run_id}] all {len(cells)} cells complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
