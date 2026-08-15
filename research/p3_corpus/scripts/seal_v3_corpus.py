#!/usr/bin/env python3
"""Freeze the six final train/eval JSONL splits into a v3 sealed corpus manifest.

This produces the ``p3-sealed-corpus-manifest-v1`` JSON consumed by
``tokenize_corpus.py --sealed-corpus-manifest``. It binds the *exact* final bytes
of every family/split by SHA-256 without requiring a committed corpus-generation
transaction, and it is fail-closed: a missing family, a wrong row schema, a row
without a single ``text`` separator, or an unexpected eval-row total aborts the
seal before any tokenization can occur.

Hashing helpers and the canonical family/schema constants are imported directly
from ``tokenize_corpus`` so the manifest root SHA-256 written here is byte-for-byte
identical to the one the tokenizer recomputes when it loads the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_P3_TRAIN_RELPATH = Path("src/scripts/train/p3_math_split")
_OLMO_P3_CANDIDATES = (
    # This archive lives inside OLMo-core itself.
    REPO_ROOT.parents[1] / _P3_TRAIN_RELPATH,
    # Legacy layout: the archive sat beside a separate OLMo-core checkout.
    REPO_ROOT.parent / "eduLLM" / "OLMo-core" / _P3_TRAIN_RELPATH,
)
OLMO_P3 = next(
    (path for path in _OLMO_P3_CANDIDATES if (path / "tokenize_corpus.py").is_file()),
    _OLMO_P3_CANDIDATES[0],
)
if str(OLMO_P3) not in sys.path:
    sys.path.insert(0, str(OLMO_P3))

import tokenize_corpus as tc  # noqa: E402  (path injected above)

SEPARATOR_SEARCH = tc.SEPARATOR_SEARCH
FAMILIES = tc.FAMILIES
P3_SOURCE_SCHEMAS = tc.P3_SOURCE_SCHEMAS
SCHEMA_VERSION = tc.SEALED_CORPUS_MANIFEST_SCHEMA_VERSION
EXPECTED_EVAL_ROWS_DEFAULT = 4191

# Fixed vendored Qwen tokenizer directory (informational tokenizer-root record).
TOKENIZER_DIR_DEFAULT = REPO_ROOT / "tokenizers" / "qwen25-vendored"


ORIGINAL_BUILDERS_RELPATH = (
    "generation-work-persistent/p3-p3-full13-repaired.y13zg7ik/builders"
)


def _default_sources(
    work_root: Path, builders_root: Path | None = None
) -> dict[str, dict[str, Path]]:
    mml = work_root / "mml-semantic-holdout-v7"
    gen = builders_root if builders_root is not None else work_root / ORIGINAL_BUILDERS_RELPATH
    return {
        "mizar": {"train": mml / "shards/mizar.jsonl", "eval": mml / "eval/mizar.jsonl"},
        "thproofs": {
            "train": mml / "shards/thproofs.jsonl",
            "eval": mml / "eval/thproofs.jsonl",
        },
        "prf2": {"train": mml / "shards/prf2.jsonl", "eval": mml / "eval/prf2.jsonl"},
        "enigma": {"train": mml / "shards/enigma.jsonl", "eval": mml / "eval/enigma.jsonl"},
        "metamath": {
            "train": gen / "metamath/normalized-resume/train.jsonl",
            "eval": gen / "metamath/normalized-resume/eval.jsonl",
        },
        "isabelle": {
            "train": gen / "isabelle/normalized-resume/train.jsonl",
            "eval": gen / "isabelle/normalized-resume/eval.jsonl",
        },
    }


def _seal_file(path: Path, *, family: str, role: str) -> dict:
    if path.is_symlink():
        sys.exit(f"sealed source must not be a symlink: {path}")
    if not path.is_file():
        sys.exit(f"sealed source JSONL is missing: {path}")
    expected_schema = P3_SOURCE_SCHEMAS[family]
    digest = hashlib.sha256()
    rows = 0
    first_line = last_line = None
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            if raw.strip():
                rows += 1
                if first_line is None:
                    first_line = raw
                last_line = raw
    if rows == 0:
        sys.exit(f"sealed source is empty: {family}/{role} -> {path}")
    # Validate the boundary rows: correct schema and exactly one separator in text.
    for label, raw in (("first", first_line), ("last", last_line)):
        row = json.loads(raw)
        if row.get("schema_version") != expected_schema:
            sys.exit(
                f"{family}/{role} {label} row schema {row.get('schema_version')!r} "
                f"!= expected {expected_schema!r}"
            )
        text = row.get("text")
        if not isinstance(text, str) or text.count(SEPARATOR_SEARCH) != 1:
            sys.exit(
                f"{family}/{role} {label} row lacks a single {SEPARATOR_SEARCH!r} separator"
            )
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _tokenizer_record(tokenizer_dir: Path) -> dict:
    record = {"path": str(tokenizer_dir.resolve()), "files": {}}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        p = tokenizer_dir / name
        if not p.is_file():
            sys.exit(f"fixed tokenizer file is missing: {p}")
        record["files"][name] = tc.file_sha256(p)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--work-root",
        default=str(REPO_ROOT / ".p3-work" / "full13"),
        help="persistent full13 work root holding the final splits",
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / ".p3-work" / "full13" / "sealed-corpus-v3" / "sealed-corpus-manifest.json"),
        help="destination path for the sealed corpus manifest JSON",
    )
    ap.add_argument(
        "--builders-root",
        default=None,
        help=(
            "directory holding <family>/normalized-resume/ for metamath and isabelle; "
            f"defaults to <work-root>/{ORIGINAL_BUILDERS_RELPATH}, which names the "
            "original build's generation id and does not exist in a fresh rebuild"
        ),
    )
    ap.add_argument("--tokenizer-dir", default=str(TOKENIZER_DIR_DEFAULT))
    ap.add_argument("--expected-eval-rows", type=int, default=EXPECTED_EVAL_ROWS_DEFAULT)
    args = ap.parse_args()

    work_root = Path(args.work_root).expanduser().resolve()
    builders_root = (
        Path(args.builders_root).expanduser().resolve() if args.builders_root else None
    )
    sources = _default_sources(work_root, builders_root)
    if set(sources) != set(FAMILIES):
        sys.exit("source map does not cover exactly the ordered P3 family set")

    families: dict[str, dict] = {}
    total_train = total_eval = 0
    for family in FAMILIES:
        train = _seal_file(sources[family]["train"], family=family, role="train")
        eval_ = _seal_file(sources[family]["eval"], family=family, role="eval")
        families[family] = {
            "schema": P3_SOURCE_SCHEMAS[family],
            "train": train,
            "eval": eval_,
        }
        total_train += train["rows"]
        total_eval += eval_["rows"]

    if total_eval != args.expected_eval_rows:
        sys.exit(
            f"total eval rows {total_eval} != expected {args.expected_eval_rows}; "
            "refusing to seal a corpus whose held-out inventory drifted"
        )

    body = {"schema_version": SCHEMA_VERSION, "families": families}
    manifest_root_sha256 = tc.fingerprint_dict(body)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "families": families,
        "manifest_root_sha256": manifest_root_sha256,
        "total_train_rows": total_train,
        "total_eval_rows": total_eval,
        "tokenizer": _tokenizer_record(Path(args.tokenizer_dir)),
    }

    # Re-validate through the exact loader the tokenizer uses (fail-closed round trip).
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    reloaded = tc.load_sealed_corpus_manifest(tmp)
    if reloaded["manifest_root_sha256"] != manifest_root_sha256:
        sys.exit("round-trip manifest root mismatch; refusing to publish seal")
    tmp.replace(out_path)

    print(f"sealed {len(families)} families -> {out_path}")
    print(f"  manifest_root_sha256 = {manifest_root_sha256}")
    print(f"  total train rows = {total_train:,}")
    print(f"  total eval rows  = {total_eval:,}")
    for family in FAMILIES:
        rec = families[family]
        print(
            f"  {family:<9} train {rec['train']['rows']:>7,} rows "
            f"({rec['train']['bytes'] / 1e6:8.1f} MB)  "
            f"eval {rec['eval']['rows']:>5,} rows  schema={rec['schema']}"
        )


if __name__ == "__main__":
    main()
