#!/usr/bin/env python3
"""
Carve a tiny multi-domain smoke pack from Hugging Face allenai/dolma (streaming).

Writes:
  <out>/raw/<domain>/*.jsonl
  <out>/tokenized/<domain>/shard-00000.npy   (uint32, OLMo-core style)
  <out>/manifests/docs.jsonl
  <out>/manifests/domain_index.json
  <out>/manifests/mixes/natural.json

Does not modify any OLMo-core library code.

Example:
  python prepare_smoke_dolma.py --out-dir ./data/smoke_dolma_v0 --docs-per-domain 200
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_domain_map(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def infer_domain(source: str, domain_cfg: dict) -> str | None:
    s = (source or "").lower()
    for domain, meta in domain_cfg["domains"].items():
        for key in meta.get("hf_source_substrings", []):
            if key.lower() in s:
                return domain
    return None


def try_load_streaming(dataset_name: str, name: str | None):
    from datasets import load_dataset

    kwargs = {"split": "train", "streaming": True}
    if name:
        kwargs["name"] = name
    try:
        return load_dataset(dataset_name, **kwargs)
    except Exception:
        # dolma often needs DATA_DIR / config; fall back to config-less
        return load_dataset(dataset_name, split="train", streaming=True)


def tokenize_docs(texts: list[str], tokenizer_name: str) -> list[list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    out = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=True)
        if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
            if not ids or ids[-1] != tok.eos_token_id:
                ids.append(tok.eos_token_id)
        out.append(ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--docs-per-domain", type=int, default=200)
    ap.add_argument("--max-scan", type=int, default=50_000, help="Max HF rows to scan")
    ap.add_argument("--tokenizer", type=str, default="allenai/dolma2-tokenizer")
    ap.add_argument(
        "--domains-json",
        type=Path,
        default=Path(__file__).with_name("domains.json"),
    )
    ap.add_argument("--dataset", type=str, default="allenai/dolma")
    ap.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional HF config / dolma version name if required by your datasets version",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    domain_cfg = load_domain_map(args.domains_json)
    domains = list(domain_cfg["domains"].keys())
    buckets: dict[str, list[dict]] = {d: [] for d in domains}

    print(f"Streaming {args.dataset} (config={args.dataset_config}) …")
    ds = try_load_streaming(args.dataset, args.dataset_config)

    scanned = 0
    for row in tqdm(ds, total=args.max_scan, desc="scan"):
        scanned += 1
        if scanned > args.max_scan:
            break
        if all(len(buckets[d]) >= args.docs_per_domain for d in domains):
            break

        text = row.get("text") or row.get("content") or ""
        if not isinstance(text, str) or len(text.strip()) < 64:
            continue
        source = str(row.get("source") or row.get("metadata", {}).get("source") or "")
        # sometimes source nested
        if not source and isinstance(row.get("metadata"), dict):
            source = str(row["metadata"].get("source", ""))

        domain = infer_domain(source, domain_cfg)
        if domain is None:
            # crude fallback from path-like ids
            blob = json.dumps(row.get("id", "")) + source
            domain = infer_domain(blob, domain_cfg)
        if domain is None or len(buckets[domain]) >= args.docs_per_domain:
            continue

        doc_id = f"{domain}_{len(buckets[domain]):06d}"
        buckets[domain].append(
            {
                "doc_id": doc_id,
                "domain": domain,
                "source": source or "unknown",
                "text": text.strip()[:50_000],
            }
        )

    # report fill
    for d in domains:
        print(f"  {d}: {len(buckets[d])} docs")

    out = args.out_dir
    (out / "manifests" / "mixes").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(parents=True, exist_ok=True)

    # write raw jsonl
    for d, docs in buckets.items():
        raw_dir = out / "raw" / d
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / "docs.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # tokenize + manifests
    manifest_rows = []
    domain_index = {
        "substrate_version": "smoke_dolma_v0",
        "tokenizer": args.tokenizer,
        "domains": {},
        "total_tokens": 0,
    }

    for d, docs in buckets.items():
        if not docs:
            continue
        texts = [x["text"] for x in docs]
        print(f"Tokenizing {d} ({len(texts)} docs) with {args.tokenizer} …")
        try:
            tokenized = tokenize_docs(texts, args.tokenizer)
        except Exception as e:
            print(f"WARNING: tokenizer failed ({e}); writing raw-only pack (no .npy).")
            tokenized = None

        tok_dir = out / "tokenized" / d
        tok_dir.mkdir(parents=True, exist_ok=True)
        shard_rel = f"tokenized/{d}/shard-00000.npy"
        n_tokens = 0
        offset = 0

        if tokenized is not None:
            flat: list[int] = []
            for doc, ids in zip(docs, tokenized):
                flat.extend(ids)
                manifest_rows.append(
                    {
                        "doc_id": doc["doc_id"],
                        "domain": d,
                        "source": doc["source"],
                        "text_path": f"raw/{d}/docs.jsonl",
                        "token_path": shard_rel,
                        "token_offset": offset,
                        "token_length": len(ids),
                        "tokenizer": args.tokenizer,
                        "substrate_version": "smoke_dolma_v0",
                    }
                )
                offset += len(ids)
            arr = np.asarray(flat, dtype=np.uint32)
            shard_path = out / shard_rel
            mm = np.memmap(shard_path, mode="w+", dtype=np.uint32, shape=(len(arr),))
            mm[:] = arr
            mm.flush()
            n_tokens = int(arr.size)
        else:
            for doc in docs:
                manifest_rows.append(
                    {
                        "doc_id": doc["doc_id"],
                        "domain": d,
                        "source": doc["source"],
                        "text_path": f"raw/{d}/docs.jsonl",
                        "token_path": None,
                        "token_offset": None,
                        "token_length": None,
                        "tokenizer": args.tokenizer,
                        "substrate_version": "smoke_dolma_v0",
                    }
                )

        domain_index["domains"][d] = {
            "shards": [shard_rel] if tokenized is not None else [],
            "n_docs": len(docs),
            "n_tokens": n_tokens,
        }
        domain_index["total_tokens"] += n_tokens

    with (out / "manifests" / "docs.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    with (out / "manifests" / "domain_index.json").open("w", encoding="utf-8") as f:
        json.dump(domain_index, f, indent=2)

    # natural mix = proportional to tokens (or docs if no tokens)
    weights = {}
    denom = domain_index["total_tokens"] or sum(
        domain_index["domains"][d]["n_docs"] for d in domain_index["domains"]
    )
    for d, meta in domain_index["domains"].items():
        num = meta["n_tokens"] or meta["n_docs"]
        weights[d] = (num / denom) if denom else 0.0
    # renormalize
    s = sum(weights.values()) or 1.0
    weights = {k: v / s for k, v in weights.items()}
    with (out / "manifests" / "mixes" / "natural.json").open("w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    print(f"Done. Wrote smoke pack → {out}")
    print("Next: skill_dag/run_smoke.sh or curriculum/run_smoke.sh")


if __name__ == "__main__":
    main()
