#!/usr/bin/env python3
"""Assert a run scored the same eval items the scan indexed (v4 section 4b).

Section 4b had the evaluator record, per label, the number of scored items, a
digest of the sorted doc_id list, and normalized-text hashes for a sample of
doc_ids. Nothing compared them to anything. That made it a record rather than
a verification -- which is the weaker half, because an off-by-one join between
a contamination hit and a per-item bpb number produces entirely plausible
numbers and no error.

This is the comparison. It joins the section 4a dump against a task-loss
payload and refuses on any disagreement:

  * the same set of labels on both sides
  * the same item count per label
  * the same doc_id digest per label, which catches a reordering that leaves
    the count intact
  * identical context and gold hashes for every sampled doc_id, which catches
    doc_ids that line up while the underlying oe-eval text has changed
  * the same normalizer fingerprint, since every hash is taken after
    normalization and a normalizer change silently invalidates all of them

Stdlib only, so it runs anywhere and is testable without torch.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

_EDULLM = Path(__file__).resolve().parents[1]
if str(_EDULLM) not in sys.path:
    sys.path.insert(0, str(_EDULLM))
from task_loss.item_identity import normalizer_fingerprint, sha256  # noqa: E402

log = logging.getLogger("verify_item_identity")


class IdentityMismatch(RuntimeError):
    """The run scored a different item set than the scan indexed."""


def load_dump(path: str | Path) -> dict[str, dict[int, dict[str, str]]]:
    """label -> doc_id -> {context hash, gold hash} from the 4a dump."""
    out: dict[str, dict[int, dict[str, str]]] = {}
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = rec["label"]
            doc_id = int(rec["doc_id"])
            per_label = out.setdefault(label, {})
            if doc_id in per_label:
                raise IdentityMismatch(
                    f"{label}: duplicate doc_id {doc_id} in the dump; one row "
                    f"per item is the invariant the join depends on"
                )
            per_label[doc_id] = {
                "full_context_sha256": rec["full_context_sha256"],
                "gold_sha256": rec["gold_sha256"],
            }
    if not out:
        raise IdentityMismatch(f"{path}: dump is empty")
    return out


def doc_id_digest(doc_ids: list[int]) -> str:
    """Must match `capture_item_identity`'s construction exactly."""
    return sha256(",".join(str(d) for d in sorted(doc_ids)))


def verify_identity(
    dump: dict[str, dict[int, dict[str, str]]],
    identity: dict[str, dict],
    *,
    payload_fingerprint: str | None,
    expected_fingerprint: str | None = None,
) -> dict[str, object]:
    """Compare a task-loss payload's identity block against the dump.

    :raises IdentityMismatch: on any disagreement, naming the label and the
        first offending doc_id rather than only that something differed.
    """
    expected_fingerprint = expected_fingerprint or normalizer_fingerprint()
    if payload_fingerprint is None:
        raise IdentityMismatch(
            "the payload carries no normalizer_fingerprint; every recorded "
            "hash is taken after normalization, so an unpinned normalizer "
            "makes them uncomparable"
        )
    if payload_fingerprint != expected_fingerprint:
        raise IdentityMismatch(
            f"normalizer fingerprint differs: payload {payload_fingerprint} "
            f"vs {expected_fingerprint}. The hashes on the two sides were "
            f"taken under different normalizers and cannot be compared."
        )

    dump_labels = set(dump)
    run_labels = set(identity)
    if dump_labels != run_labels:
        only_dump = sorted(dump_labels - run_labels)
        only_run = sorted(run_labels - dump_labels)
        raise IdentityMismatch(
            f"label sets differ: {len(only_dump)} only in the dump "
            f"(first {only_dump[:3]}), {len(only_run)} only in the run "
            f"(first {only_run[:3]})"
        )

    checked_items = 0
    for label in sorted(run_labels):
        block = identity[label]
        dump_items = dump[label]
        n_run = int(block["n_docs"])
        if n_run != len(dump_items):
            raise IdentityMismatch(
                f"{label}: the run scored {n_run} items, the dump holds "
                f"{len(dump_items)}"
            )
        expected_digest = doc_id_digest(list(dump_items))
        if block["doc_id_digest"] != expected_digest:
            raise IdentityMismatch(
                f"{label}: doc_id digest differs ({block['doc_id_digest'][:12]} "
                f"vs {expected_digest[:12]}) while both sides report {n_run} "
                f"items -- the item set is reordered or substituted, and a "
                f"positional join would silently pair the wrong rows"
            )
        for doc_id_str, hashes in sorted(block.get("sampled", {}).items()):
            doc_id = int(doc_id_str)
            if doc_id not in dump_items:
                raise IdentityMismatch(
                    f"{label}: the run scored doc_id {doc_id}, which the dump "
                    f"does not contain"
                )
            for field in ("full_context_sha256", "gold_sha256"):
                if hashes[field] != dump_items[doc_id][field]:
                    raise IdentityMismatch(
                        f"{label} doc_id {doc_id}: {field} differs "
                        f"({hashes[field][:12]} vs "
                        f"{dump_items[doc_id][field][:12]}); the doc_ids line "
                        f"up but the underlying text does not, so the oe-eval "
                        f"data changed between the dump and the run"
                    )
            checked_items += 1

    return {
        "labels": len(run_labels),
        "items_total": sum(len(v) for v in dump.values()),
        "items_hash_checked": checked_items,
        "normalizer_fingerprint": expected_fingerprint,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, help="4a dump, eval_items.jsonl.gz")
    parser.add_argument("--task-loss", required=True, help="stepN_task_loss.json")
    args = parser.parse_args()

    payload = json.loads(Path(args.task_loss).read_text(encoding="utf-8"))
    identity = payload.get("item_identity")
    if not isinstance(identity, dict) or not identity:
        raise IdentityMismatch(
            f"{args.task_loss} carries no item_identity block; it predates "
            f"section 4b and its per-item rows cannot be joined to the scan"
        )
    summary = verify_identity(
        load_dump(args.items),
        identity,
        payload_fingerprint=payload.get("normalizer_fingerprint"),
    )
    log.info(
        "identity OK: %d labels, %d items, %d hash-checked, fingerprint %s",
        summary["labels"],
        summary["items_total"],
        summary["items_hash_checked"],
        str(summary["normalizer_fingerprint"])[:12],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
