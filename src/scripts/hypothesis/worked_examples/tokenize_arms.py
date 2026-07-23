#!/usr/bin/env python3
"""Tokenize each arm's docs.jsonl into a single uint32 .npy shard (OLMo-core style).

Uses OLMo's dolma2 tokenizer (TokenizerConfig.dolma2 → allenai/dolma2-tokenizer),
matching edu-llm/OLMo-core training defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ARMS = ["bare", "complete", "fade_ordered", "fade_shuffled"]


# Mirrors olmo_core.data.tokenizer.TokenizerConfig.dolma2() when olmo_core can't import.
_DOLMA2 = {
    "identifier": "allenai/dolma2-tokenizer",
    "eos_token_id": 100257,
    "vocab_size": 100278,
}
_DOLMA2_SIGDIG = {
    "identifier": "allenai/dolma2-tokenizer-sigdig",
    "eos_token_id": 100257,
    "vocab_size": 100278,
}


def resolve_olmo_tokenizer(name: str | None):
    """Resolve to HF id + EOS using OLMo TokenizerConfig when possible."""
    key = (name or "dolma2").strip()
    aliases = {
        "dolma2": _DOLMA2,
        "allenai/dolma2-tokenizer": _DOLMA2,
        "dolma2_sigdig": _DOLMA2_SIGDIG,
        "allenai/dolma2-tokenizer-sigdig": _DOLMA2_SIGDIG,
    }

    try:
        repo_src = Path(__file__).resolve().parents[3]
        if str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))
        from olmo_core.data.tokenizer import TokenizerConfig, TokenizerName

        if key in ("dolma2", TokenizerName.dolma2, "allenai/dolma2-tokenizer"):
            cfg = TokenizerConfig.dolma2()
        elif key in ("dolma2_sigdig", TokenizerName.dolma2_sigdig, "allenai/dolma2-tokenizer-sigdig"):
            cfg = TokenizerConfig.dolma2_sigdig()
        else:
            cfg = TokenizerConfig.from_hf(key)
        identifier = str(cfg.identifier)
        eos_token_id = int(cfg.eos_token_id)
        print(f"OLMo TokenizerConfig: identifier={identifier} eos={eos_token_id} vocab={cfg.vocab_size}")
        return identifier, eos_token_id
    except Exception as e:
        if key in aliases:
            cfg = aliases[key]
            print(
                f"Using OLMo dolma2 constants (olmo_core import failed: {e}): "
                f"identifier={cfg['identifier']} eos={cfg['eos_token_id']}"
            )
            return cfg["identifier"], cfg["eos_token_id"]
        print(f"olmo_core unavailable ({e}); treating --tokenizer as HF id={key}")
        return key, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", type=Path, required=True)
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="dolma2",
        help="dolma2 (OLMo default), dolma2_sigdig, or a HuggingFace tokenizer id",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    identifier, cfg_eos = resolve_olmo_tokenizer(args.tokenizer)
    tok = AutoTokenizer.from_pretrained(identifier, trust_remote_code=True)
    eos_id = cfg_eos if cfg_eos is not None else tok.eos_token_id
    pack = args.pack_dir
    summary = {
        "tokenizer_id": identifier,
        "eos_token_id": eos_id,
        "arms": {},
    }

    for arm in ARMS:
        docs_path = pack / "arms" / arm / "docs.jsonl"
        docs = []
        with docs_path.open(encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))

        ids: list[int] = []
        mask_bits: list[bool] = []
        for doc in docs:
            text = doc["text"]
            enc = tok.encode(text, add_special_tokens=True)
            if eos_id is not None and (not enc or enc[-1] != eos_id):
                enc.append(eos_id)

            # Fade arms: train only after loss_start_char (shown scaffold is context).
            # Bare/complete: train on the full document.
            loss_start = doc.get("loss_start_char")
            if loss_start is None:
                doc_mask = [True] * len(enc)
            else:
                context = text[: int(loss_start)]
                ctx_enc = tok.encode(context, add_special_tokens=True)
                n_ctx = min(len(ctx_enc), len(enc))
                doc_mask = [False] * n_ctx + [True] * (len(enc) - n_ctx)
                if not any(doc_mask):
                    doc_mask[-1] = True

            ids.extend(enc)
            mask_bits.extend(doc_mask)

        out_dir = pack / "tokenized" / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        shard = out_dir / "shard-00000.npy"
        mask_path = out_dir / "label_mask-00000.npy"
        arr = np.asarray(ids, dtype=np.uint32)
        mm = np.memmap(shard, mode="w+", dtype=np.uint32, shape=(len(arr),))
        mm[:] = arr
        mm.flush()
        mask_arr = np.asarray(mask_bits, dtype=np.bool_)
        if mask_arr.shape[0] != arr.shape[0]:
            raise SystemExit(f"{arm}: token/mask length mismatch {arr.size} vs {mask_arr.size}")
        mask_mm = np.memmap(mask_path, mode="w+", dtype=np.bool_, shape=(len(mask_arr),))
        mask_mm[:] = mask_arr
        mask_mm.flush()
        summary["arms"][arm] = {
            "n_docs": len(docs),
            "n_tokens": int(arr.size),
            "tokens_per_doc_approx": round(arr.size / max(len(docs), 1), 1),
            "shard": str(shard),
            "label_mask": str(mask_path),
            "label_mask_true_frac": round(float(mask_arr.mean()) if mask_arr.size else 0.0, 4),
        }
        print(
            f"{arm}: {len(docs)} docs, {arr.size} tokens, "
            f"mask_true={summary['arms'][arm]['label_mask_true_frac']} -> {shard}"
        )

    (pack / "reports" / "tokenize_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Note: bare has fewer tokens/doc; match CPT budget with more passes over that shard.")


if __name__ == "__main__":
    main()
