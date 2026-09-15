#!/usr/bin/env python3
"""Map every kept document to its difficulty decile, per metric (v4 section 5.1).

The corpus rebuild re-chunks in difficulty order, so chunk index *is*
difficulty rank and the training loader passes an identity permutation. Deciles
are therefore equal-chunk-mass ranges of the sorted chunk array, obtained from
`split_equal_mass(chunk_count)`.

To place a *document* in a decile we need where its tokens landed in the
metric's chunk sequence. That follows from the per-metric document ranks in
`gated_docs.jsonl.gz` plus the token accounting:

    chunk_start(d) = cumulative_tokens_before(d) // 2048
    cumulative counts (n_tokens + 1) per document -- one EOS each

The EOS term is not an assumption. Summing `kept_tokens_by_domain` from
gate_and_rank_manifest.json gives 9,973,904,776 while the corpus manifest
records total_tokens = 9,979,762,663, and the difference is exactly
5,857,887 = kept_docs.

Deciles differ per metric because the three corpora differ in chunk count
(4,872,915 / 4,872,916 / 4,872,917), which shifts the equal-mass boundaries:
mtld is 5x487,292 then 5x487,291, flesch 6x487,292 then 4x487,291,
compression_ratio 7x487,292 then 3x487,291. A long document can span a decile
boundary, so a start and end decile are both recorded.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np

_EDULLM = Path(__file__).resolve().parents[1]
if str(_EDULLM) not in sys.path:
    sys.path.insert(0, str(_EDULLM))
from curriculum_pacing import split_equal_mass  # noqa: E402

log = logging.getLogger("build_decile_map")

SEQUENCE_LENGTH = 2048
METRICS = ("mtld", "flesch", "compression_ratio")
RANK_FIELD = {
    "mtld": "rank_mtld",
    "flesch": "rank_flesch",
    "compression_ratio": "rank_compression_ratio",
}
# Domains are read in this fixed order so the emitted row index is stable and
# reproducible across runs; the scan reports (domain, source_doc) and joins on
# the same ordering.
DOMAINS = (
    "algebraic-stack",
    "arxiv",
    "dclm",
    "nemotron-hqdqa",
    "open-web-math",
    "pes2o",
    "starcoder",
    "wiki",
)


def load_gated(path: Path) -> dict[str, np.ndarray]:
    """Read gated_docs.jsonl.gz into flat arrays."""
    domain_ids: list[int] = []
    source_docs: list[int] = []
    n_tokens: list[int] = []
    ranks: dict[str, list[int]] = {m: [] for m in METRICS}
    domain_index = {d: i for i, d in enumerate(DOMAINS)}

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            rec = json.loads(line)
            domain = rec["domain"]
            if domain not in domain_index:
                raise RuntimeError(f"line {line_no}: unexpected domain {domain!r}")
            domain_ids.append(domain_index[domain])
            source_docs.append(int(rec["source_doc"]))
            n_tokens.append(int(rec["n_tokens"]))
            for metric in METRICS:
                ranks[metric].append(int(rec[RANK_FIELD[metric]]))

    out = {
        "domain_id": np.asarray(domain_ids, dtype=np.int8),
        "source_doc": np.asarray(source_docs, dtype=np.int64),
        "n_tokens": np.asarray(n_tokens, dtype=np.int64),
    }
    for metric in METRICS:
        out[f"rank_{metric}"] = np.asarray(ranks[metric], dtype=np.int64)
    return out


def deciles_for_metric(
    rank: np.ndarray, n_tokens: np.ndarray, chunk_count: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (start_decile, end_decile, n_straddling) in input row order."""
    n_docs = rank.shape[0]
    if not np.array_equal(np.sort(rank), np.arange(n_docs)):
        raise RuntimeError(
            "document ranks are not a complete permutation of 0..n_docs-1; "
            "the decile map would be silently wrong"
        )

    order = np.argsort(rank, kind="stable")
    tokens_in_rank_order = n_tokens[order] + 1  # one EOS per document
    ends = np.cumsum(tokens_in_rank_order)
    starts = ends - tokens_in_rank_order

    # Chunk containing each document's first and last token. The last token
    # index is ends-1; a document whose tail falls in the truncated remainder
    # is clamped to the final chunk.
    start_chunk = np.minimum(starts // SEQUENCE_LENGTH, chunk_count - 1)
    end_chunk = np.minimum((ends - 1) // SEQUENCE_LENGTH, chunk_count - 1)

    bounds = split_equal_mass(chunk_count)
    edges = np.asarray([b[1] for b in bounds], dtype=np.int64)  # exclusive ends
    start_dec = np.searchsorted(edges, start_chunk, side="right").astype(np.int8)
    end_dec = np.searchsorted(edges, end_chunk, side="right").astype(np.int8)

    # Undo the sort so results align with the input rows.
    out_start = np.empty(n_docs, dtype=np.int8)
    out_end = np.empty(n_docs, dtype=np.int8)
    out_start[order] = start_dec
    out_end[order] = end_dec
    return out_start, out_end, int((start_dec != end_dec).sum())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gated", required=True, help="gated_docs.jsonl.gz")
    parser.add_argument("--corpus-root", required=True, help="dir holding <metric>/manifest.json")
    parser.add_argument("--out", required=True, help="output .npz")
    parser.add_argument("--summary", required=True, help="output summary .json")
    args = parser.parse_args()

    log.info("reading %s", args.gated)
    data = load_gated(Path(args.gated))
    n_docs = data["n_tokens"].shape[0]
    log.info("documents: %d", n_docs)

    arrays: dict[str, np.ndarray] = {
        "domain_id": data["domain_id"],
        "source_doc": data["source_doc"],
        "n_tokens": data["n_tokens"],
    }
    summary: dict[str, object] = {"n_docs": n_docs, "metrics": {}}

    for metric in METRICS:
        manifest = json.loads(
            (Path(args.corpus_root) / metric / "manifest.json").read_text(encoding="utf-8")
        )
        chunk_count = int(manifest["chunk_count"])
        total_docs = int(manifest["total_docs"])
        total_tokens = int(manifest["total_tokens"])
        if total_docs != n_docs:
            raise RuntimeError(
                f"{metric}: manifest says {total_docs} docs, gated list has {n_docs}"
            )
        # The EOS accounting must reconcile or every chunk position is off.
        expected = int(data["n_tokens"].sum()) + n_docs
        if expected != total_tokens:
            raise RuntimeError(
                f"{metric}: sum(n_tokens)+n_docs = {expected} but manifest "
                f"total_tokens = {total_tokens}; the chunk positions this "
                f"script computes would be wrong"
            )

        start_dec, end_dec, straddling = deciles_for_metric(
            data[f"rank_{metric}"], data["n_tokens"], chunk_count
        )
        arrays[f"start_decile_{metric}"] = start_dec
        arrays[f"end_decile_{metric}"] = end_dec

        counts = np.bincount(start_dec, minlength=10).tolist()
        tok_by_dec = [
            int(data["n_tokens"][start_dec == d].sum()) + int((start_dec == d).sum())
            for d in range(10)
        ]
        summary["metrics"][metric] = {  # type: ignore[index]
            "chunk_count": chunk_count,
            "decile_widths_chunks": [b[1] - b[0] for b in split_equal_mass(chunk_count)],
            "docs_per_decile": counts,
            "tokens_per_decile": tok_by_dec,
            "straddling_docs": straddling,
        }
        log.info(
            "%-18s chunks=%d straddling=%6d docs/decile=%s",
            metric,
            chunk_count,
            straddling,
            counts,
        )
        log.info("%-18s tokens/decile=%s", "", [f"{t/1e6:.1f}M" for t in tok_by_dec])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    Path(args.summary).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("wrote %s", out)
    log.info("wrote %s", args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
