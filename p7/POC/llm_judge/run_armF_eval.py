#!/usr/bin/env python
"""Run the P7 multi-turn pedagogy generation for arm F on the edu-llm platform.

Arm F is a LoRA on ``allenai/OLMo-2-1124-7B-Instruct`` — a different and much larger base
than the ``OLMo-2-0425-1B-Instruct`` every other P7 arm sits on. It is too big for the
laptop the rest of this harness runs on, so generation happens on a platform GPU and only
the judging half runs locally.

This is a *thin* wrapper. It fetches the adapter, shells out to ``generate_arms.py``
unmodified, and puts the result back. Reimplementing the generation loop here would mean
arm F was scored by different code than the thirteen arms it is being compared against,
which is the one thing this file exists to avoid.

Two platform facts shape it (both learned the expensive way, see
``p7/POC/impl3x5_klw/submissions/README.md``):

- **The AWS CLI is not in the image.** S3 goes through ``boto3``, same as ``run_klw.py``.
- **The container execs with no shell**, so the caller must wrap in ``bash -lc`` for
  ``$EDULLM_CHECKPOINT_DIR`` to arrive expanded.

The base cell keeps ``generate_arms.py``'s stock ``B_raw_SI`` name even though it is the 7B
base here, not the 1B one. Renaming it remotely would make this file's output differ from a
stock harness run for no gain; ``gen_meta.base_model`` in every record already says which
base produced it, and the output filename carries ``7b``. Do not merge this file into
``arms_multiturn.jsonl`` — two different models would end up under one setup key.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_MODEL = "allenai/OLMo-2-1124-7B-Instruct"
# Every file PeftModel.from_pretrained needs. The tokenizer comes from the base model, so
# the adapter's own tokenizer files are deliberately not required.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def s3_split(uri: str) -> tuple[str, str]:
    """``s3://bucket/a/b`` -> ``("bucket", "a/b")``."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    bucket, _, key = uri[5:].partition("/")
    return bucket, key.rstrip("/")


def fetch_adapter(uri_prefix: str, dest: Path) -> Path:
    import boto3

    bucket, key = s3_split(uri_prefix)
    dest.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3")
    for name in ADAPTER_FILES:
        target = dest / name
        print(f"  s3://{bucket}/{key}/{name} -> {target}", flush=True)
        client.download_file(bucket, f"{key}/{name}", str(target))
    return dest


def put(local: Path, uri_prefix: str) -> None:
    import boto3

    bucket, key = s3_split(uri_prefix)
    print(f"  {local} -> s3://{bucket}/{key}/{local.name}", flush=True)
    boto3.client("s3").upload_file(str(local), bucket, f"{key}/{local.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter_s3", required=True,
                   help="S3 prefix holding adapter_config.json + adapter_model.safetensors.")
    p.add_argument("--arm_name", default="armF",
                   help="Setup key the judge will see for the tuned cell.")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--problems", default=str(HERE / "multiturn_set.jsonl"))
    p.add_argument("--out_name", default="arms_multiturn_armF_7b.jsonl")
    p.add_argument("--gen_max_new", type=int, default=220, help="Harness default; keep it.")
    p.add_argument("--limit", type=int, default=0, help="Cap #problems (0 = all). Smoke only.")
    p.add_argument("--checkpoint_dir", required=True, help='Pass "$EDULLM_CHECKPOINT_DIR".')
    p.add_argument("--output_prefix", default="", help='Pass "$EDULLM_OUTPUT_PREFIX".')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ckpt = Path(args.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    print(f"base     : {args.base_model}")
    print(f"adapter  : {args.adapter_s3}")
    print(f"problems : {args.problems}", flush=True)

    print("\nfetching adapter ...", flush=True)
    adapter_dir = fetch_adapter(args.adapter_s3, ckpt / "armF_adapter")

    out_path = ckpt / args.out_name
    cmd = [
        sys.executable, str(HERE / "generate_arms.py"),
        "--arm", f"{args.arm_name}={adapter_dir}",
        "--base_model", args.base_model,
        "--problems", args.problems,
        "--out", str(out_path),
        "--gen_max_new", str(args.gen_max_new),
        "--si", "canonical",
        "--device", "cuda",
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    print("\n" + " ".join(cmd) + "\n", flush=True)
    # Streamed, not captured: a generation pass this long is only debuggable live, and the
    # platform's log group is the only place the output survives a failed run.
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"generate_arms.py exited {result.returncode}", file=sys.stderr)
        return result.returncode

    if not out_path.exists():
        print(f"expected output missing: {out_path}", file=sys.stderr)
        return 1

    # The output already sits in the checkpoint dir, which the workload contract syncs. The
    # explicit put is so the file lands at a predictable key instead of under a run-scoped
    # checkpoint path a reader would have to go hunting for.
    if args.output_prefix:
        print("\nuploading ...", flush=True)
        put(out_path, args.output_prefix)

    print(f"\ndone: {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
