"""Build evaluation-paraphrase author packets (channel B) for the confirmation catalog.

Run on FarmShare from the catalog root (which holds src/fict_qa.parquet and the source_catalog
output in packets/): python -I -B build_paraphrase_packets.py
"""

import hashlib
import json
import pathlib

import pyarrow.parquet as pq


def seal(obj):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    body["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return body


rows = {r["question_id"]: r for r in pq.read_table("src/fict_qa.parquet").to_pylist()}
probe_map = json.load(open("packets/private-probe-id-map.json"))
out = pathlib.Path("paraphrase-packets")
out.mkdir()
total = 0
for n in range(1, 9):
    author_packet = json.load(open(f"packets/author-packets/chunk-{n:02d}.json"))
    events = set(author_packet["events"])
    probes = []
    for probe_id, entry in sorted(probe_map["probes"].items()):
        if entry["event"] not in events:
            continue
        row = rows[entry["original_probe_id"]]
        probes.append(
            {
                "opaque_probe_id": probe_id,
                "event": entry["event"],
                "canonical_question": row["question"],
                "canonical_answer": row["natural_answer"],
            }
        )
    probes.sort(key=lambda p: (p["event"], p["opaque_probe_id"]))
    total += len(probes)
    packet = seal(
        {
            "schema": "spacing-eval-paraphrase-packet-v1",
            "channel": "evaluation_paraphrase",
            "chunk_index": n,
            "events": author_packet["events"],
            "may_see": ["canonical_evaluation_question", "canonical_answer_label"],
            "excluded_inputs": [
                "source-QA catalog",
                "grounded declarations",
                "linked_fictions",
                "model_outputs",
                "MCQ_choices",
            ],
            "private_probe_id_map_sha256": probe_map["sha256"],
            "probes": probes,
        }
    )
    (out / f"chunk-{n:02d}.json").write_text(json.dumps(packet, indent=1, ensure_ascii=False))
print("probes", total)
