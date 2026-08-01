"""Write header-free copies of the FineWeb-Edu token stream for OLMo-core.

WHY THIS EXISTS
---------------
OLMo-core does not read ``.npy`` files. ``NumpyFSLDataset`` reads token ``i`` from raw byte
offset ``i * itemsize`` (``data/utils.py:load_array_slice`` -> ``io.get_bytes_range``), and
``_get_file_size_and_length`` derives the token count as ``filesize / itemsize``. There is no
``.npy`` header parsing anywhere in the package.

The corpus at ``kda/lm/data/`` was written by ``np.save``, so it carries a **128-byte header**
= 64 uint16 slots. Pointed at it directly, OLMo-core would:

  * report 1,200,000,064 tokens instead of 1,200,000,000;
  * return header bytes as the first 64 "tokens" (they decode to 20115, 19797, 22864, ...);
  * shift every subsequent read by 64 tokens.

**It would not crash.** The header's largest uint16 is 32,032, comfortably inside a 50,304
embedding, so training would run, log a falling loss, and look completely healthy while reading
a misaligned stream. That is the failure mode worth spending two minutes to remove.

The original files are left untouched; this only writes new ``*_raw.bin`` siblings.
"""

import hashlib
import json
import os
import sys

import numpy as np

SRC = "/scratch/users/ericrcwu/kda/lm/data"
DST = "/scratch/users/ericrcwu/liv/data"
CHUNK = 64 * 1024 * 1024  # tokens per copy step, ~128 MiB of uint16

os.makedirs(DST, exist_ok=True)
report = {}

for split in ("train", "val"):
    src = f"{SRC}/{split}.npy"
    dst = f"{DST}/{split}_raw.bin"

    a = np.load(src, mmap_mode="r")
    assert a.dtype == np.uint16, a.dtype
    assert a.ndim == 1, a.shape
    header = os.path.getsize(src) - a.nbytes
    print(f"[{split}] {len(a):,} tokens, {header}-byte header -> {dst}", flush=True)

    # Stream rather than load: train.npy is 2.4 GB and the login node is shared.
    running = hashlib.sha256()
    with open(dst, "wb") as fh:
        for start in range(0, len(a), CHUNK):
            block = np.ascontiguousarray(a[start : start + CHUNK])
            fh.write(block.tobytes())
            running.update(block.tobytes())

    # Verify by re-reading the way OLMo-core will: raw memmap, no header.
    raw = np.memmap(dst, mode="r", dtype=np.uint16)
    assert len(raw) == len(a), f"{split}: length {len(raw):,} != {len(a):,}"
    assert os.path.getsize(dst) % 2 == 0
    for probe in (0, 1, 63, 64, 65, len(a) // 2, len(a) - 1):
        assert int(raw[probe]) == int(a[probe]), f"{split}: mismatch at {probe}"
    vmax = int(raw[:: max(1, len(raw) // 20_000_000)].max())
    print(f"[{split}] verified: len ok, spot-checks ok, sampled max id {vmax}", flush=True)

    report[split] = {
        "path": dst,
        "tokens": int(len(raw)),
        "bytes": int(os.path.getsize(dst)),
        "sha256": running.hexdigest(),
        "sampled_max_token_id": vmax,
    }

# The whole point of the pilot's vocab choice: every id must be indexable.
train_max = int(np.memmap(report["train"]["path"], mode="r", dtype=np.uint16).max())
val_max = int(np.memmap(report["val"]["path"], mode="r", dtype=np.uint16).max())
report["exact_max_token_id"] = {"train": train_max, "val": val_max}
print(f"\nEXACT max token id -- train {train_max}, val {val_max}")
assert max(train_max, val_max) < 50304, "vocab 50,304 does not cover the corpus"
print("vocab 50,304 covers the corpus (GPT-2 EOS 50,256 is the true ceiling)")

with open(f"{DST}/meta.json", "w") as fh:
    json.dump(report, fh, indent=2)
print(f"\nwrote {DST}/meta.json")
