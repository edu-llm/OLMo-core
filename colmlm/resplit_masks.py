"""Re-split per-worker fact masks onto a promoted corpus's sealed shard boundaries.

``prepare_data.py`` writes one mask shard per annotate worker (19 for the 1B build, 15 for the
750M prefix, which stops mid-worker-14). The eduLLM dataset library re-shards on promotion:
``fineweb-edu-750m/v2`` was sealed as 15 train shards + 1 val shard, with the val tokens carved
off the end of the last train shard. So the per-worker masks no longer pair 1:1 with the sealed
token shards, and ``colmlm/train.py --mode split`` (PR #44) refuses rather than mis-pair.

This concatenates the per-worker masks *in shard-name order* into one stream and re-slices it at
the sealed token-shard boundaries, writing one ``<stem>.mask.bin`` per sealed train shard (the
val shard is training-excluded and only written with ``--with-val``). The result pairs 1:1, by
name and by length, with the sealed tokens, so ``--mask-dir`` points ``--mode split`` at it.

Why this is correct: ``prepare_data.py`` shards sequentially (worker 0 = first docs, worker 1 =
the next, ...), so the concatenation of the per-worker masks is the *same positional stream* as
the sealed tokens -- independent of how either side chose its shard boundaries or how many shards
it made. The sealed corpus is a prefix of that stream, so we slice the first ``sum(sealed counts)``
positions and give each sealed shard its own contiguous run. With ``--verify`` every output length
is checked against the sealed token shard's own byte size, so a stream that does not line up fails
loudly instead of sliding every fact span.

Self-contained: only ``boto3`` (for ``s3://``) and ``numpy`` are needed -- no OLMo-core, no torch.
Paths (``--corpus``, ``--mask-dir``, ``--out``) may be ``s3://`` URIs or local directories.

    pip install boto3 numpy
    python resplit_masks.py \\
        --corpus   s3://edullm-data/pretrain/fineweb-edu-750m/v2 \\
        --mask-dir s3://<your-run-output>/corpus-750m/masks \\
        --out      s3://<your-run-output>/masks-750m-v2 \\
        --profile  sbsandbox --verify

    python resplit_masks.py --self-test    # numpy-only; no S3, no corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

MASK_SUFFIX = ".mask.bin"
EDULLM_DATA_MANIFEST = "tokens/manifest.json"


@dataclass
class SealedShard:
    stem: str      # the token shard's name up to its first suffix, e.g. "train-00000"
    split: str     # "train" | "val" | ...
    count: int     # tokens in the shard
    path: str      # entry path relative to the corpus root, e.g. "tokens/train-00000.u16le.bin"


def _stem(name: str) -> str:
    """Shard name up to its first suffix: ``train-00000.u16le.bin`` -> ``train-00000``."""
    return name.rsplit("/", 1)[-1].split(".", 1)[0]


def parse_sealed(manifest: dict) -> List[SealedShard]:
    """Ordered shards from a sealed edullm-data group manifest, in stream (manifest) order."""
    shards: List[SealedShard] = []
    for entry in manifest["entries"]:
        count = entry.get("count") or {}
        if count.get("unit") != "tokens":
            raise SystemExit(f"entry {entry.get('path')!r} counts {count.get('unit')!r}, not tokens")
        shards.append(
            SealedShard(_stem(entry["path"]), str(entry.get("split")), int(count["value"]), entry["path"])
        )
    return shards


def reslice(stream: np.ndarray, shards: List[SealedShard]) -> Tuple[Dict[str, np.ndarray], int]:
    """Cut ``stream`` into one contiguous run per sealed shard, in order. Pure; unit-tested below.

    Returns ``{stem: mask_slice}`` and the number of trailing positions left over (the part of a
    longer build beyond the sealed prefix). Raises if the stream is shorter than the sealed total.
    """
    total = sum(shard.count for shard in shards)
    if len(stream) < total:
        raise SystemExit(
            f"mask stream is {len(stream):,} tokens but the sealed corpus needs {total:,}; "
            "the masks are not for this corpus (wrong build, wrong prefix, or an interrupted set)"
        )
    out: Dict[str, np.ndarray] = {}
    ptr = 0
    for shard in shards:
        out[shard.stem] = stream[ptr : ptr + shard.count]
        ptr += shard.count
    return out, len(stream) - total


# --------------------------------------------------------------------------------------------------
# I/O. boto3 is imported lazily so --self-test needs neither it nor any credentials.
# --------------------------------------------------------------------------------------------------

_S3 = {"client": None}


def _client(opts):
    if _S3["client"] is None:
        import boto3

        session = boto3.Session(profile_name=opts.profile, region_name=opts.region)
        _S3["client"] = session.client("s3")
    return _S3["client"]


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _split_s3(uri: str) -> Tuple[str, str]:
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def _join(prefix: str, *parts: str) -> str:
    return "/".join([prefix.rstrip("/"), *parts])


def read_bytes(path: str, opts) -> bytes:
    if _is_s3(path):
        bucket, key = _split_s3(path)
        return _client(opts).get_object(Bucket=bucket, Key=key)["Body"].read()
    return Path(path).read_bytes()


def head_size(path: str, opts) -> int:
    if _is_s3(path):
        bucket, key = _split_s3(path)
        return _client(opts).head_object(Bucket=bucket, Key=key)["ContentLength"]
    return Path(path).stat().st_size


def list_masks(mask_dir: str, opts) -> List[str]:
    """Every ``.mask.bin`` under ``mask_dir``, sorted by name (train-00000, train-00001, ...)."""
    if _is_s3(mask_dir):
        bucket, prefix = _split_s3(mask_dir.rstrip("/") + "/")
        keys: List[str] = []
        paginator = _client(opts).get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys += [obj["Key"] for obj in page.get("Contents", []) if obj["Key"].endswith(MASK_SUFFIX)]
        return [f"s3://{bucket}/{key}" for key in sorted(keys)]
    return sorted(str(p) for p in Path(mask_dir).glob("*" + MASK_SUFFIX))


def write_bytes(path: str, data: bytes, opts) -> None:
    if _is_s3(path):
        bucket, key = _split_s3(path)
        _client(opts).put_object(Bucket=bucket, Key=key, Body=data)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)


def run(opts) -> None:
    if opts.sealed_manifest:
        manifest = json.loads(Path(opts.sealed_manifest).read_text(encoding="utf-8"))
    elif opts.corpus:
        manifest = json.loads(read_bytes(_join(opts.corpus, EDULLM_DATA_MANIFEST), opts).decode("utf-8"))
    else:
        raise SystemExit("need --corpus (to read tokens/manifest.json) or --sealed-manifest")
    shards = parse_sealed(manifest)
    trainable = [s for s in shards if s.split == "train"]
    print(f"sealed corpus: {len(shards)} shards ({len(trainable)} train), "
          f"{sum(s.count for s in shards):,} tokens")

    mask_files = list_masks(opts.mask_dir, opts)
    if not mask_files:
        raise SystemExit(f"no {MASK_SUFFIX} files under {opts.mask_dir}")
    print(f"concatenating {len(mask_files)} mask shards from {opts.mask_dir}")
    parts = []
    for path in mask_files:
        part = np.frombuffer(read_bytes(path, opts), dtype=np.bool_)
        parts.append(part)
        print(f"  {path.rsplit('/', 1)[-1]:32s} {len(part):>13,} tokens")
    stream = np.concatenate(parts)

    masks, leftover = reslice(stream, shards)
    if leftover:
        print(f"note: {leftover:,} trailing mask tokens beyond the sealed corpus were dropped "
              "(a longer build re-split onto a shorter promoted prefix)")

    written = 0
    for shard in shards:
        if shard.split != "train" and not opts.with_val:
            continue
        arr = masks[shard.stem]
        if opts.verify:
            if not opts.corpus:
                raise SystemExit("--verify needs --corpus to size-check the sealed token shards")
            token_tokens = head_size(_join(opts.corpus, shard.path), opts) // opts.bytes_per_token
            if token_tokens != shard.count or len(arr) != shard.count:
                raise SystemExit(
                    f"{shard.stem}: sealed tokens={token_tokens:,}, manifest count={shard.count:,}, "
                    f"mask slice={len(arr):,} -- these disagree; not writing a misaligned mask"
                )
        write_bytes(_join(opts.out, shard.stem + MASK_SUFFIX), arr.tobytes(), opts)
        written += 1
        print(f"  wrote {shard.stem}{MASK_SUFFIX}  ({len(arr):,} tokens, split={shard.split})")
    print(f"done: {written} masks under {opts.out}")
    print(f"then run:  --mode split --data-dir <corpus> --mask-dir {opts.out}")


def self_test() -> int:
    # A longer build (120) re-split onto a sealed prefix of 100 = train(30)+train(45)+val(25),
    # the val carved off the end -- the fineweb-edu-750m/v2 shape in miniature.
    rng = np.random.default_rng(0)
    stream = rng.integers(0, 2, size=120, dtype=np.int8).astype(np.bool_)
    shards = [
        SealedShard("train-00000", "train", 30, "tokens/train-00000.u16le.bin"),
        SealedShard("train-00001", "train", 45, "tokens/train-00001.u16le.bin"),
        SealedShard("val-00000", "val", 25, "tokens/val-00000.u16le.bin"),
    ]
    masks, leftover = reslice(stream, shards)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'OK ' if cond else 'ERR'} {name}")

    check("leftover == build - sealed", leftover == 20)
    check("train-00000 == stream[0:30]", np.array_equal(masks["train-00000"], stream[0:30]))
    check("train-00001 == stream[30:75]", np.array_equal(masks["train-00001"], stream[30:75]))
    check("val-00000 == stream[75:100]", np.array_equal(masks["val-00000"], stream[75:100]))
    check("slices are contiguous and cover the sealed prefix",
          np.array_equal(np.concatenate([masks[s.stem] for s in shards]), stream[:100]))
    try:
        reslice(stream[:80], shards)
        check("short stream refuses", False)
    except SystemExit:
        check("short stream refuses", True)
    parsed = parse_sealed({"entries": [
        {"path": "tokens/train-00000.u16le.bin", "split": "train", "count": {"unit": "tokens", "value": 30}},
        {"path": "tokens/val-00000.u16le.bin", "split": "val", "count": {"unit": "tokens", "value": 25}},
    ]})
    check("parse_sealed reads stem/split/count", parsed[0].stem == "train-00000" and parsed[1].split == "val")

    print("\nRESULT:", "RESLICE LOGIC VERIFIED" if ok else "PROBLEMS ABOVE")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="resplit_masks", description=__doc__.splitlines()[0])
    p.add_argument("--self-test", action="store_true", help="Run the numpy-only reslice tests and exit.")
    p.add_argument("--corpus", help="Sealed corpus root (s3:// or local); reads tokens/manifest.json.")
    p.add_argument("--sealed-manifest", help="Local tokens/manifest.json, if --corpus is not readable.")
    p.add_argument("--mask-dir", help="Your per-worker .mask.bin shards (s3:// or local).")
    p.add_argument("--out", help="Where to write the resliced masks (s3:// or local).")
    p.add_argument("--with-val", action="store_true", help="Also write the held-out val shard's mask.")
    p.add_argument("--verify", action="store_true",
                   help="Check each output length against the sealed token shard's byte size (needs --corpus).")
    p.add_argument("--bytes-per-token", type=int, default=2, help="uint16 = 2.")
    p.add_argument("--profile", default=None, help="AWS profile name (e.g. sbsandbox).")
    p.add_argument("--region", default=None, help="AWS region (else the profile/default).")
    return p


def main() -> None:
    opts = build_parser().parse_args()
    if opts.self_test:
        sys.exit(self_test())
    missing = [f for f in ("mask_dir", "out") if not getattr(opts, f)]
    if missing:
        raise SystemExit("--" + ", --".join(m.replace("_", "-") for m in missing) + " are required")
    run(opts)


if __name__ == "__main__":
    main()
