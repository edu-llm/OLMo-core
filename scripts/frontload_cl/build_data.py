#!/usr/bin/env python3
"""
Build frontload-cl token shards and SFT JSONL from Hugging Face.

Tokenizer: allenai/dolma2-tokenizer (same as OLMo 2 / tokenizer/dolma2-bpe)
Output PT shards: uint32 little-endian .u32le.bin (NO .npy header), ~1GiB each
Layout: <out>/tokens/<source>/{train,val}-NNNNN.u32le.bin

Published artifacts (already on ``s3://edullm-data``):
  - ``pretrain/frontload-cl-10b/v1``
  - ``sft/frontload-cl-chat-sft/v1``
Re-run this only to rebuild local staging; then use ``scripts/frontload_cl/publish_datasets.py``
(which allocates a new version). Training reads from ``edullm-data``, not this tree.

Fast path:
  - Rust `tokenizers` + encode_batch
  - Optional multiprocessing (shard by doc index % workers)
  - FineWeb single-pass: main + anneal writers together

Examples:
  python scripts/frontload_cl/build_data.py tokenize-hf \\
    --repo HuggingFaceTB/finemath --config finemath-4plus \\
    --source finemath-4plus --token-budget 60000000 --workers 4

  python scripts/frontload_cl/build_data.py tokenize-fineweb \\
    --main-budget 8360000000 --anneal-budget 950000000 --workers 8
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import struct
import time
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, Optional

SEED = 42069666
EOS_ID = 100257
PAD_ID = 100277
SHARD_BYTES = 1 << 30  # 1 GiB on disk per shard file
TOKENS_PER_SHARD = SHARD_BYTES // 4
# Keep only a small in-memory write buffer (4 MiB of uint32), not a full shard.
WRITE_BUFFER_TOKENS = 1 << 20  # 1,048,576 tokens == 4 MiB
DEFAULT_BATCH = 128
DEFAULT_MAX_RSS_GB = 20.0

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "frontload-cl"


def load_fast_tokenizer(name_or_path: str = "allenai/dolma2-tokenizer"):
    """Load Rust tokenizers backend (much faster than transformers.AutoTokenizer)."""
    from tokenizers import Tokenizer

    path = Path(name_or_path)
    tok_json = path / "tokenizer.json" if path.is_dir() else None
    if tok_json is not None and tok_json.exists():
        return Tokenizer.from_file(str(tok_json))
    # HF hub id or direct json path
    if path.suffix == ".json" and path.exists():
        return Tokenizer.from_file(str(path))
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(name_or_path, "tokenizer.json")
        return Tokenizer.from_file(local)
    except Exception:
        # last resort: transformers save then reload
        from transformers import AutoTokenizer

        tmp = Path(os.environ.get("TMP", "/tmp")) / "dolma2_tok_cache"
        AutoTokenizer.from_pretrained(name_or_path).save_pretrained(tmp)
        return Tokenizer.from_file(str(tmp / "tokenizer.json"))


def encode_docs_batch(tok, texts: list[str]) -> list[list[int]]:
    encs = tok.encode_batch(texts)
    out: list[list[int]] = []
    for e in encs:
        ids = list(e.ids)
        if not ids:
            out.append([])
            continue
        if ids[-1] != EOS_ID:
            ids.append(EOS_ID)
        out.append(ids)
    return out


def encode_doc(tok, text: str) -> list[int]:
    return encode_docs_batch(tok, [text])[0]


# Back-compat alias used by older one-off scripts
def load_tokenizer(name_or_path: str = "allenai/dolma2-tokenizer"):
    return load_fast_tokenizer(name_or_path)


def _shard_index_from_name(path: Path, split: str) -> Optional[int]:
    # train-00007.u32le.bin -> 7
    name = path.name
    prefix = f"{split}-"
    suffix = ".u32le.bin"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    mid = name[len(prefix) : -len(suffix)]
    if mid.isdigit():
        return int(mid)
    return None


def _rss_gb() -> float:
    """Current process RSS in GiB (0.0 if unavailable)."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return counters.WorkingSetSize / (1024**3)
        except Exception:
            return 0.0
    try:
        import resource

        # Linux: ru_maxrss is KiB.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    except Exception:
        return 0.0


def _available_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return 999.0


def _job_tree_rss_gb() -> float:
    """RSS for parent process + all children (best-effort)."""
    try:
        import psutil

        me = psutil.Process(os.getpid())
        parent = me.parent() or me
        total = parent.memory_info().rss
        for c in parent.children(recursive=True):
            try:
                total += c.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / (1024**3)
    except Exception:
        return _rss_gb()


def _wait_for_memory_headroom(
    max_rss_gb: float,
    workers: int,
    min_available_gb: float = 3.0,
) -> None:
    """
    Soft cap that protects the machine without false-throttling HF workers.

    Sleep only when:
      - system available RAM is below ``min_available_gb``, or
      - the whole job tree is over ``max_rss_gb`` *and* available RAM is getting
        tight, or
      - this single worker is a runaway (> max(8 GiB, 60% of the tree cap)).

    Do NOT use fair-share (cap/workers): each HF streaming client often needs
    ~2-3 GiB, so a 20/8=2.5 GiB share causes constant sleep under a healthy tree.
    """
    del workers  # kept for call-site compatibility
    if max_rss_gb <= 0:
        return
    runaway_limit = max(8.0, max_rss_gb * 0.6)
    while True:
        self_rss = _rss_gb()
        avail = _available_ram_gb()
        tree = _job_tree_rss_gb()
        low_avail = avail < min_available_gb
        runaway = self_rss > runaway_limit
        # Allow a little slack over the nominal tree cap when the OS still has room.
        over_tree_and_tight = tree > max_rss_gb and avail < (min_available_gb + 4.0)
        if not low_avail and not runaway and not over_tree_and_tight:
            return
        print(
            f"memory pressure tree={tree:.1f}/{max_rss_gb:.1f}GiB "
            f"avail={avail:.1f}GiB self={self_rss:.1f}GiB; sleeping",
            flush=True,
        )
        time.sleep(5.0)


def _progress_path(out: Path, worker: int) -> Path:
    return Path(out) / f"fineweb-progress-w{worker}.json"


def write_worker_progress(
    out: Path,
    worker: int,
    *,
    docs: int,
    main_tokens: int,
    anneal_tokens: int,
    stream_i: int,
    workers: int,
) -> None:
    path = _progress_path(out, worker)
    path.write_text(
        json.dumps(
            {
                "worker": worker,
                "workers": workers,
                "docs": docs,
                "main_tokens": main_tokens,
                "anneal_tokens": anneal_tokens,
                "stream_i": stream_i,
                "updated_at": time.time(),
            }
        ),
        encoding="utf-8",
    )


def read_worker_progress(out: Path, worker: int) -> Optional[dict]:
    path = _progress_path(out, worker)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


class ShardWriter:
    """
    Stream uint32 tokens to ~1GiB shard files with a tiny in-memory buffer.

    Important: never hold a full shard as Python ``list[int]`` (that was multi-GB).
    """

    def __init__(
        self,
        dir_path: Path,
        split: str,
        max_tokens: int = TOKENS_PER_SHARD,
        shard_offset: int = 0,
        shard_stride: int = 1,
        *,
        resume: bool = False,
        write_buffer_tokens: int = WRITE_BUFFER_TOKENS,
    ):
        self.dir_path = dir_path
        self.split = split
        self.max_tokens = max_tokens
        self.shard_stride = max(1, shard_stride)
        self.shard_idx = shard_offset
        self.write_buffer_tokens = max(64 * 1024, write_buffer_tokens)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.buf = array("I")
        self.total_tokens = 0
        self.file_tokens = 0  # tokens already on disk in the open shard
        self._fh = None
        if resume:
            self._open_resume(shard_offset)
        else:
            self._open_next(overwrite=True)

    def _path(self, idx: Optional[int] = None) -> Path:
        i = self.shard_idx if idx is None else idx
        return self.dir_path / f"{self.split}-{i:05d}.u32le.bin"

    def _owned_shards(self, shard_offset: int) -> list[tuple[int, Path]]:
        owned: list[tuple[int, Path]] = []
        for p in self.dir_path.glob(f"{self.split}-*.u32le.bin"):
            idx = _shard_index_from_name(p, self.split)
            if idx is None:
                continue
            if idx % self.shard_stride == shard_offset % self.shard_stride:
                owned.append((idx, p))
        owned.sort(key=lambda t: t[0])
        return owned

    def _open_resume(self, shard_offset: int) -> None:
        owned = self._owned_shards(shard_offset)
        if not owned:
            self.shard_idx = shard_offset
            self._open_next(overwrite=True)
            return
        total = 0
        for _, p in owned:
            total += p.stat().st_size // 4
        last_idx, last_path = owned[-1]
        last_tokens = last_path.stat().st_size // 4
        self.total_tokens = total
        if last_tokens < self.max_tokens:
            self.shard_idx = last_idx
            self.file_tokens = last_tokens
            self._fh = open(last_path, "ab")
        else:
            self.shard_idx = last_idx + self.shard_stride
            self._open_next(overwrite=True)

    def _open_next(self, overwrite: bool = True) -> None:
        if self._fh is not None:
            self._fh.close()
        mode = "wb" if overwrite else "ab"
        self._fh = open(self._path(), mode)
        self.file_tokens = 0 if overwrite else (self._path().stat().st_size // 4)

    def _flush_buf(self) -> None:
        assert self._fh is not None
        if not self.buf:
            return
        self.buf.tofile(self._fh)
        n = len(self.buf)
        self.total_tokens += n
        self.file_tokens += n
        self.buf = array("I")

    def _rotate_shard(self) -> None:
        self._flush_buf()
        self.shard_idx += self.shard_stride
        self._open_next(overwrite=True)

    def write_ids(self, ids: list[int]) -> None:
        if not ids:
            return
        i = 0
        n = len(ids)
        while i < n:
            room_in_shard = self.max_tokens - self.file_tokens - len(self.buf)
            if room_in_shard <= 0:
                self._rotate_shard()
                continue
            room_in_buf = self.write_buffer_tokens - len(self.buf)
            take = min(room_in_shard, room_in_buf, n - i)
            if take <= 0:
                self._flush_buf()
                continue
            self.buf.fromlist(ids[i : i + take])
            i += take
            if self.file_tokens + len(self.buf) >= self.max_tokens:
                self._rotate_shard()
            elif len(self.buf) >= self.write_buffer_tokens:
                self._flush_buf()

    def close(self) -> int:
        assert self._fh is not None
        self._flush_buf()
        self._fh.close()
        empty = self._path()
        if empty.exists() and empty.stat().st_size == 0:
            empty.unlink()
        return self.total_tokens


def iter_hf_rows(
    repo: str,
    config: Optional[str],
    split: str,
    streaming: bool = True,
):
    from datasets import load_dataset

    kwargs = {"path": repo, "split": split, "streaming": streaming}
    if config:
        kwargs["name"] = config
    return load_dataset(**kwargs)


def tokenize_text_stream(
    texts: Iterable[str],
    out_source_dir: Path,
    split_name: str,
    token_budget: int,
    tok,
    batch_size: int = DEFAULT_BATCH,
    shard_offset: int = 0,
    shard_stride: int = 1,
) -> int:
    writer = ShardWriter(out_source_dir, split_name, shard_offset=shard_offset, shard_stride=shard_stride)
    n_docs = 0
    t0 = time.time()
    batch: list[str] = []

    def flush_batch() -> bool:
        nonlocal n_docs
        if not batch:
            return True
        id_lists = encode_docs_batch(tok, batch)
        batch.clear()
        for ids in id_lists:
            if not ids:
                continue
            if writer.total_tokens + len(writer.buf) + len(ids) > token_budget:
                return False
            writer.write_ids(ids)
            n_docs += 1
            if (writer.total_tokens + len(writer.buf)) >= token_budget:
                return False
        return True

    try:
        for text in texts:
            if not text or not text.strip():
                continue
            batch.append(text)
            if len(batch) >= batch_size:
                if not flush_batch():
                    break
                have = writer.total_tokens + len(writer.buf)
                elapsed = max(time.time() - t0, 1e-6)
                rate = have / elapsed / 1e6
                print(
                    f"  docs={n_docs} tokens={have}/{token_budget} ({100*have/token_budget:.2f}%) "
                    f"{rate:.2f}M tok/s",
                    flush=True,
                )
        else:
            flush_batch()
    finally:
        total = writer.close()
    elapsed = max(time.time() - t0, 1e-6)
    print(
        f"Wrote {total} tokens ({n_docs} docs) under {out_source_dir} "
        f"in {elapsed/60:.1f} min ({total/elapsed/1e6:.2f}M tok/s)"
    )
    return total


def _worker_tokenize_hf(payload: dict) -> dict:
    tok = load_fast_tokenizer(payload["tokenizer"])
    worker = payload["worker"]
    workers = payload["workers"]
    budget = payload["token_budget"]  # per-worker budget
    text_field = payload["text_field"]

    def texts():
        ds = iter_hf_rows(payload["repo"], payload.get("config"), payload["split"], streaming=True)
        for i, row in enumerate(ds):
            if i % workers != worker:
                continue
            text = row.get(text_field)
            if isinstance(text, str) and text.strip():
                yield text

    out_dir = Path(payload["out"]) / "tokens" / payload["source"]
    total = tokenize_text_stream(
        texts(),
        out_dir,
        payload["split_name"],
        budget,
        tok,
        batch_size=payload["batch_size"],
        shard_offset=worker,
        shard_stride=workers,
    )
    return {"worker": worker, "tokens": total}


def cmd_tokenize_hf(args: argparse.Namespace) -> None:
    workers = max(1, args.workers)
    per_worker = args.token_budget // workers
    extras = args.token_budget - per_worker * workers
    print(
        f"Streaming {args.repo} -> tokens/{args.source} budget={args.token_budget} "
        f"workers={workers} batch={args.batch_size}"
    )
    if workers == 1:
        tok = load_fast_tokenizer(args.tokenizer)
        texts = (
            row.get(args.text_field)
            for row in iter_hf_rows(args.repo, args.config, args.split, streaming=True)
            if isinstance(row.get(args.text_field), str) and row.get(args.text_field).strip()
        )
        tokenize_text_stream(
            texts,
            Path(args.out) / "tokens" / args.source,
            args.split_name,
            args.token_budget,
            tok,
            batch_size=args.batch_size,
        )
        return

    payloads = []
    for w in range(workers):
        budget = per_worker + (extras if w == 0 else 0)
        payloads.append(
            {
                "worker": w,
                "workers": workers,
                "token_budget": budget,
                "repo": args.repo,
                "config": args.config,
                "split": args.split,
                "text_field": args.text_field,
                "source": args.source,
                "split_name": args.split_name,
                "out": str(args.out),
                "tokenizer": args.tokenizer,
                "batch_size": args.batch_size,
            }
        )
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker_tokenize_hf, p) for p in payloads]
        for fut in as_completed(futs):
            print("worker done:", fut.result(), flush=True)


def _worker_fineweb(payload: dict) -> dict:
    # Keep HF/datasets from buffering large in-memory tables.
    os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tok = load_fast_tokenizer(payload["tokenizer"])
    worker = payload["worker"]
    workers = payload["workers"]
    main_budget = payload["main_budget"]
    anneal_budget = payload["anneal_budget"]
    batch_size = payload["batch_size"]
    out = Path(payload["out"])
    resume = bool(payload.get("resume", False))
    max_rss_gb = float(payload.get("max_rss_gb", DEFAULT_MAX_RSS_GB))
    # Empirically ~1k tokens/doc on FineWeb-Edu with Dolma2; used only to skip
    # already-written docs when resuming after a crash.
    avg_tok_per_doc = int(payload.get("avg_tok_per_doc", 1050))

    main_w = ShardWriter(
        out / "tokens" / "fineweb-edu-main",
        "train",
        shard_offset=worker,
        shard_stride=workers,
        resume=resume,
    )
    anneal_w = ShardWriter(
        out / "tokens" / "fineweb-edu-anneal",
        "train",
        shard_offset=worker,
        shard_stride=workers,
        resume=resume,
    )

    already = main_w.total_tokens + anneal_w.total_tokens
    # Prefer exact doc cursors from progress files (written during prior runs).
    # Fall back to token/avg estimate. Per-worker skip avoids gaps from using
    # max(docs)*workers as a global stream cut.
    skip_docs = 0
    if resume:
        prog = read_worker_progress(out, worker)
        if prog and int(prog.get("workers", workers)) == workers:
            skip_docs = int(prog.get("docs", 0))
        if skip_docs <= 0 and already:
            skip_docs = already // avg_tok_per_doc
        print(
            f"[w{worker}] resume main={main_w.total_tokens} anneal={anneal_w.total_tokens} "
            f"skip_docs={skip_docs}",
            flush=True,
        )
        if main_w.total_tokens >= main_budget and anneal_w.total_tokens >= anneal_budget:
            return {
                "worker": worker,
                "main": main_w.close(),
                "anneal": anneal_w.close(),
                "skipped": True,
            }

    ds = iter_hf_rows("HuggingFaceFW/fineweb-edu", None, "train", streaming=True)
    batch_rows: list[dict] = []
    n = 0
    skipped = 0
    t0 = time.time()
    base_tokens = already
    last_progress_save = 0

    def score_of(row: dict) -> int:
        s = row.get("int_score")
        if s is None and row.get("score") is not None:
            s = int(round(float(row["score"])))
        return int(s) if s is not None else 0

    def maybe_save_progress(stream_i: int, docs_done: int) -> None:
        nonlocal last_progress_save
        if docs_done - last_progress_save < 2000:
            return
        last_progress_save = docs_done
        try:
            write_worker_progress(
                out,
                worker,
                docs=docs_done,
                main_tokens=main_w.total_tokens + len(main_w.buf),
                anneal_tokens=anneal_w.total_tokens + len(anneal_w.buf),
                stream_i=stream_i,
                workers=workers,
            )
        except Exception as exc:
            print(f"[w{worker}] progress save failed: {exc}", flush=True)

    def flush() -> None:
        nonlocal n
        if not batch_rows:
            return
        _wait_for_memory_headroom(max_rss_gb, workers)
        texts = [r["text"] for r in batch_rows]
        scores = [r["_score"] for r in batch_rows]
        id_lists = encode_docs_batch(tok, texts)
        batch_rows.clear()
        for ids, sc in zip(id_lists, scores):
            if not ids:
                continue
            # Prefer filling anneal from score>=4; otherwise main.
            if sc >= 4 and (anneal_w.total_tokens + len(anneal_w.buf)) < anneal_budget:
                if anneal_w.total_tokens + len(anneal_w.buf) + len(ids) <= anneal_budget:
                    anneal_w.write_ids(ids)
                    n += 1
                    continue
            if (main_w.total_tokens + len(main_w.buf)) < main_budget:
                if main_w.total_tokens + len(main_w.buf) + len(ids) <= main_budget:
                    main_w.write_ids(ids)
                    n += 1
        del id_lists, texts, scores

    try:
        for i, row in enumerate(ds):
            if i % workers != worker:
                continue
            # Resume catch-up: count this worker's assigned rows only. Avoid
            # strip/score/tokenize until past the already-written doc cursor.
            if skipped < skip_docs:
                skipped += 1
                if skipped % 20_000 == 0 or skipped == skip_docs:
                    print(
                        f"[w{worker}] skip_docs {skipped}/{skip_docs} "
                        f"({100.0 * skipped / max(skip_docs, 1):.1f}%) "
                        f"rss={_rss_gb():.1f}GiB",
                        flush=True,
                    )
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            batch_rows.append({"text": text, "_score": score_of(row)})
            if len(batch_rows) >= batch_size:
                flush()
                docs_done = n + skip_docs
                maybe_save_progress(i, docs_done)
                if (main_w.total_tokens + len(main_w.buf)) >= main_budget and (
                    anneal_w.total_tokens + len(anneal_w.buf)
                ) >= anneal_budget:
                    break
                if n % (batch_size * 4) == 0:
                    elapsed = max(time.time() - t0, 1e-6)
                    mt = main_w.total_tokens + len(main_w.buf)
                    at = anneal_w.total_tokens + len(anneal_w.buf)
                    new_tok = max(mt + at - base_tokens, 0)
                    print(
                        f"[w{worker}] docs={docs_done} main={mt}/{main_budget} "
                        f"anneal={at}/{anneal_budget} "
                        f"({new_tok / elapsed / 1e6:.2f}M tok/s) rss={_rss_gb():.1f}GiB",
                        flush=True,
                    )
        flush()
        maybe_save_progress(0, n + skip_docs)
    finally:
        mt = main_w.close()
        at = anneal_w.close()
        try:
            write_worker_progress(
                out,
                worker,
                docs=n + skip_docs,
                main_tokens=mt,
                anneal_tokens=at,
                stream_i=0,
                workers=workers,
            )
        except Exception:
            pass
    return {"worker": worker, "main": mt, "anneal": at}


def cmd_tokenize_fineweb(args: argparse.Namespace) -> None:
    workers = max(1, args.workers)
    # Split budgets across workers (approx).
    main_each = args.main_budget // workers
    anneal_each = args.anneal_budget // workers
    main_extra = args.main_budget - main_each * workers
    anneal_extra = args.anneal_budget - anneal_each * workers
    resume = bool(getattr(args, "resume", False))
    max_rss_gb = float(getattr(args, "max_rss_gb", DEFAULT_MAX_RSS_GB))
    print(
        f"FineWeb single-pass main={args.main_budget} anneal={args.anneal_budget} "
        f"workers={workers} batch={args.batch_size} resume={resume} max_rss_gb={max_rss_gb}"
    )
    payloads = []
    for w in range(workers):
        payloads.append(
            {
                "worker": w,
                "workers": workers,
                "main_budget": main_each + (main_extra if w == 0 else 0),
                "anneal_budget": anneal_each + (anneal_extra if w == 0 else 0),
                "out": str(args.out),
                "tokenizer": args.tokenizer,
                "batch_size": args.batch_size,
                "resume": resume,
                "max_rss_gb": max_rss_gb,
            }
        )
    if workers == 1:
        print(_worker_fineweb(payloads[0]), flush=True)
        return

    def _tree_rss_gb(parent_pid: int) -> float:
        try:
            import psutil

            p = psutil.Process(parent_pid)
            total = p.memory_info().rss
            for c in p.children(recursive=True):
                try:
                    total += c.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total / (1024**3)
        except Exception:
            return 0.0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker_fineweb, p) for p in payloads]
        main_total = anneal_total = 0
        pending = set(futs)
        parent_pid = os.getpid()
        while pending:
            done, pending = concurrent_wait(pending, timeout=30.0)
            rss = _tree_rss_gb(parent_pid)
            if rss > 0:
                print(f"[mem] process-tree rss={rss:.1f}GiB / cap={max_rss_gb:.1f}GiB", flush=True)
                if max_rss_gb > 0 and rss > max_rss_gb:
                    print(
                        f"[mem] OVER CAP ({rss:.1f}>{max_rss_gb:.1f}GiB); "
                        "workers should pause via soft throttle",
                        flush=True,
                    )
            for fut in done:
                r = fut.result()
                print("worker done:", r, flush=True)
                main_total += r["main"]
                anneal_total += r["anneal"]
        print(f"TOTAL main={main_total} anneal={anneal_total}", flush=True)


def concurrent_wait(futs, timeout: float):
    """Split completed futures out of a pending set (stdlib-friendly)."""
    from concurrent.futures import wait

    if not futs:
        return set(), set()
    done, not_done = wait(futs, timeout=timeout)
    return done, not_done


def messages_from_ultrachat_or_norobots(row: dict) -> Optional[list]:
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        return msgs
    return None


def cmd_build_sft(args: argparse.Namespace) -> None:
    """Download SFT sources and write conversations JSONL (not tokenized)."""
    from datasets import load_dataset

    rng = random.Random(args.seed)
    out = Path(args.out) / "conversations"
    out.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    val_rows: list[dict] = []

    nr_train = load_dataset("HuggingFaceH4/no_robots", split="train")
    nr_test = load_dataset("HuggingFaceH4/no_robots", split="test")
    for row in nr_train:
        msgs = messages_from_ultrachat_or_norobots(row)
        if msgs:
            train_rows.append({"messages": msgs, "source": "no_robots"})
    for row in nr_test:
        msgs = messages_from_ultrachat_or_norobots(row)
        if msgs:
            val_rows.append({"messages": msgs, "source": "no_robots"})
    print(f"no_robots train={len(nr_train)} val={len(nr_test)}")

    uc_train = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    uc_test = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    for row in uc_train:
        msgs = messages_from_ultrachat_or_norobots(row)
        if msgs:
            train_rows.append({"messages": msgs, "source": "ultrachat"})
    for row in uc_test:
        msgs = messages_from_ultrachat_or_norobots(row)
        if msgs:
            val_rows.append({"messages": msgs, "source": "ultrachat"})
    print(f"ultrachat train={len(uc_train)} val={len(uc_test)}")

    numina = load_dataset("AI-MO/NuminaMath-1.5", split="train")
    idxs = list(range(len(numina)))
    rng.shuffle(idxs)
    val_idx = set(idxs[:5000])
    train_idx = idxs[5000 : 5000 + 250000]
    for i in train_idx:
        row = numina[i]
        problem = row.get("problem") or row.get("question") or ""
        solution = row.get("solution") or row.get("answer") or ""
        if not problem or not solution:
            continue
        train_rows.append(
            {
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ],
                "source": "numina",
            }
        )
    for i in val_idx:
        row = numina[i]
        problem = row.get("problem") or row.get("question") or ""
        solution = row.get("solution") or row.get("answer") or ""
        if not problem or not solution:
            continue
        val_rows.append(
            {
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ],
                "source": "numina",
            }
        )
    print(f"numina train~={len(train_idx)} val=5000")

    oh = load_dataset("brahmairesearch/OpenHermes-2.5-Formatted", split="train")
    oh_idxs = list(range(len(oh)))
    rng.shuffle(oh_idxs)
    oh_train = oh_idxs[:100000]
    id_path = Path(args.out) / "openhermes_sft_ids.json"
    id_path.write_text(json.dumps(oh_train), encoding="utf-8")
    print(f"Wrote OpenHermes SFT ids -> {id_path}")
    for i in oh_train:
        row = oh[i]
        msgs = messages_from_ultrachat_or_norobots(row)
        if not msgs:
            conv = row.get("conversations")
            if conv:
                msgs = [
                    {
                        "role": ("user" if c.get("from") == "human" else "assistant"),
                        "content": c.get("value") or "",
                    }
                    for c in conv
                    if c.get("value")
                ]
        if msgs:
            train_rows.append({"messages": msgs, "source": "openhermes"})

    train_path = out / "train-00000.jsonl.gz"
    val_path = out / "val-00000.jsonl.gz"
    with gzip.open(train_path, "wt", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with gzip.open(val_path, "wt", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"SFT train rows={len(train_rows)} -> {train_path}")
    print(f"SFT val rows={len(val_rows)} -> {val_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tokenize-hf", help="Stream HF dataset and write .u32le.bin shards")
    t.add_argument("--repo", required=True)
    t.add_argument("--config", default=None)
    t.add_argument("--split", default="train")
    t.add_argument("--text-field", default="text")
    t.add_argument("--source", required=True, help="tokens/<source>/ folder name")
    t.add_argument("--split-name", default="train", choices=["train", "val"])
    t.add_argument("--token-budget", type=int, required=True)
    t.add_argument("--out", type=Path, default=DEFAULT_OUT)
    t.add_argument("--tokenizer", default=str(DEFAULT_OUT / "tokenizer" / "dolma2-bpe"))
    t.add_argument("--seed", type=int, default=SEED)
    t.add_argument("--keep-prob", type=float, default=1.0)
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    t.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    t.set_defaults(func=cmd_tokenize_hf)

    fw = sub.add_parser(
        "tokenize-fineweb",
        help="Single-pass FineWeb-Edu -> fineweb-edu-main + fineweb-edu-anneal",
    )
    fw.add_argument("--main-budget", type=int, default=8_360_000_000)
    fw.add_argument("--anneal-budget", type=int, default=950_000_000)
    fw.add_argument("--out", type=Path, default=DEFAULT_OUT)
    fw.add_argument("--tokenizer", default=str(DEFAULT_OUT / "tokenizer" / "dolma2-bpe"))
    fw.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    fw.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    fw.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing FineWeb shards and skip already-written docs",
    )
    fw.add_argument(
        "--max-rss-gb",
        type=float,
        default=DEFAULT_MAX_RSS_GB,
        help="Soft cap on process-tree RSS in GiB (workers pause if over fair share)",
    )
    fw.set_defaults(func=cmd_tokenize_fineweb)

    s = sub.add_parser("build-sft", help="Download SFT mixes into conversations/*.jsonl.gz")
    s.add_argument("--out", type=Path, default=DEFAULT_OUT)
    s.add_argument("--seed", type=int, default=SEED)
    s.set_defaults(func=cmd_build_sft)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    # Windows ProcessPoolExecutor needs this guard; also help spawn find the module.
    main()
