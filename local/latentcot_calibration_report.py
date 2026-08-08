"""
Turn a calibration run's metrics.json into the two numbers the compute request guessed at:
measured MFU and measured peak memory -- then say whether 5,000 steps fits the 24 h cap.

The sizing table in phase8-runbook.md 0b assumes 5% MFU. At 2% the campaign is 33 h and busts
the cap, and with --attempts 1 and no --resume that loses the run. This script replaces the
assumption with the measurement.

Usage:
  .venv/bin/python local/latentcot_calibration_report.py <metrics.json | s3://.../metrics.json> [...]

Pass several to compare the LR probes; each prints its own line plus a loss trajectory.
"""

import json
import sys

# Dense bf16 tensor-core peaks, TFLOPS (vendor "with sparsity" figures halved).
PEAK = {"a10g": 125e12, "a100": 312e12, "h100": 989e12, "l40s": 181e12}
PARAMS = 474e6
K = 10
CAP_HOURS = 24.0
TARGET_STEPS = 5_000
ARMS_PARALLEL = 5  # one per GPU on gpu-8xa100


def load(path: str) -> dict:
    """
    Read a metrics.json from the LOCAL filesystem only.

    This deliberately does not reach S3. CLAUDE.md: "Never write a script that calls AWS. No
    boto3, no aws CLI. The credentials live in workflows and a laptop cannot get one." An
    earlier version of this file read s3:// through olmo_core.io (i.e. boto3) and was wrong to.
    Get the file to disk first -- the run's own artifacts are mirrored to
    $EDULLM_CHECKPOINT_DIR, and the container also prints every logged entry, so the numbers
    are readable from the job log without any credential at all.
    """
    if path.startswith(("s3://", "gs://")):
        raise SystemExit(
            f"refusing to read {path}: this script does not call AWS (see CLAUDE.md). "
            "Fetch the metrics.json by whatever sanctioned means, then pass the local path."
        )
    with open(path) as f:
        return json.load(f)


def token_positions(batch: int, prefix=259.0, student=272.0, teacher=308.0) -> float:
    """Token-positions per step for a CODI arm (teacher + K-step loop + final student)."""
    return batch * (teacher + sum(prefix + i for i in range(K)) + student)


def report(path: str) -> None:
    m = load(path)
    hist = m.get("train_history") or []
    if len(hist) < 2:
        print(f"{path}: only {len(hist)} history entries — need >=2 for a rate")
        return

    batch = m.get("batch_size", 16)
    # Skip the first entry: it carries warmup/compile/alloc one-offs.
    a, b = hist[1], hist[-1]
    dt = b["elapsed_s"] - a["elapsed_s"]
    dsteps = b["step"] - a["step"]
    if dt <= 0 or dsteps <= 0:
        print(f"{path}: no usable timing (dt={dt}, dsteps={dsteps})")
        return
    s_per_step = dt / dsteps
    flops_per_step = 6 * PARAMS * token_positions(batch)
    achieved = flops_per_step / s_per_step
    peak_mem = max(h.get("peak_mem_gb", 0.0) for h in hist)

    print(f"\n=== {path}")
    print(f"  arm={m.get('arm')} lr={m.get('lr')} batch={batch} precision={m.get('precision')}")
    print(f"  measured: {s_per_step:.3f} s/step, peak {peak_mem:.1f} GB "
          f"(over {dsteps} steps)")
    for name, peak in PEAK.items():
        print(f"  MFU vs {name:5s} peak: {100 * achieved / peak:5.2f}%")

    # Project the real run: batch 16 is 2x the token-positions of a batch-8 calibration.
    scale = token_positions(16) / token_positions(batch)
    hours = s_per_step * scale * TARGET_STEPS / 3600
    mem16 = peak_mem * (16 / batch) if batch != 16 else peak_mem
    print(f"  --> projected at batch 16: {hours:.1f} h per CODI arm, ~{mem16:.1f} GB peak")
    verdict = "FITS" if hours < CAP_HOURS * 0.8 else ("TIGHT" if hours < CAP_HOURS else "BUSTS")
    print(f"  --> {TARGET_STEPS} steps vs {CAP_HOURS:.0f} h cap: {verdict}")
    if hours >= CAP_HOURS * 0.8:
        safe = int(CAP_HOURS * 0.7 * 3600 / (s_per_step * scale) / 100) * 100
        print(f"      cut --steps to ~{safe} to land at 70% of the cap")
    print(f"  (arms run concurrently on gpu-8xa100, so campaign wall-clock ~= this, "
          f"not x{ARMS_PARALLEL})")

    losses = [(h["step"], round(h["loss"], 4)) for h in hist]
    print(f"  loss: {losses[0]} -> {losses[-1]}"
          f"{'  RISING — LR likely too high' if losses[-1][1] > losses[0][1] else ''}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    for p in sys.argv[1:]:
        report(p)
