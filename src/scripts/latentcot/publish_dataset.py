"""
Publish the graph-reachability dataset to the eduLLM platform (`edullm-datasets` skill).

Uploads the compliant source dir (produced by ``gen_graph_data.py``) to
``s3://edullm-landing``; the validator recomputes it and, on pass, promotes it to
``s3://edullm-data/sft/graph-reachability-depth/<version>/`` and writes the catalog entry +
generated README. You never write ``edullm-data`` directly.

Requires the ``edullm-data`` package and AWS credentials that can write ``edullm-landing``
(in this project, the sb-aws broker; elsewhere ordinary AWS creds). Run this on the machine
that holds the data and can reach AWS — NOT needed on a GPU box.

Usage::

    .venv/bin/python src/scripts/latentcot/publish_dataset.py \
        --source data/latentcot/graph-reachability-depth
"""

import argparse
import datetime
from pathlib import Path

from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3

DATASET_ID = "sft/graph-reachability-depth"
GROUP = "conversations"

PURPOSE = (
    "Synthetic directed-graph reachability conversations (question + edge list -> BFS reasoning "
    "+ yes/no) for the latent chain-of-thought (CODI) experiment: trains the reasoning arms and "
    "evaluates reachability accuracy by graph depth."
)

ABOUT = (
    "Layered directed graphs where every edge advances exactly one level, so a reachable "
    "source->target distance equals the requested depth D (no shortcuts) and difficulty scales "
    "cleanly with D. Each conversation's user turn states the reachability query and the edge "
    "list; the assistant turn gives the breadth-first-search frontier expansion followed by the "
    "yes/no answer. Reachable/unreachable are balanced and expand to matched frontier depth so "
    "the label cannot be read off frontier depth. The held-out split uses disjoint seeds and "
    "includes out-of-distribution depths (5, 8) unseen in train. Fully synthetic and "
    "contamination-free; each row also carries the raw graph (edges, source, target, depth, BFS "
    "frontiers, shortest path) as metadata for the reasoning experiment."
)

# The sft-conversations/v1 group contract (the validator recomputes the leakage itself).
GROUP_META = {
    GROUP: {
        "record_schema": {"type": "object", "required": ["messages"]},
        "partitions": [
            {"name": "train", "by": "path", "glob": "train-*.jsonl"},
            {"name": "heldout", "by": "path", "glob": "heldout-*.jsonl"},
        ],
        "dedup": {"key": "messages", "method": "sha256-of-message-contents"},
        "leakage": {"train_vs_heldout": "recomputed-by-validator", "max_leakage": 0},
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("data/latentcot/graph-reachability-depth")
    )
    args = parser.parse_args()

    plan = publish(
        str(args.source),
        dataset_id=DATASET_ID,
        purpose=PURPOSE,
        profile="sft-conversations/v1",
        s3=Boto3S3.default(),
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        group_meta=GROUP_META,
        about=ABOUT,
        license={"id": "ODC-By-1.0", "basis": "declared"},
        sources=[{"name": "synthetic-graph-generator", "scope": "this-dataset"}],
    )
    print("Uploaded to landing; validator will promote on pass.")
    print(plan)


if __name__ == "__main__":
    main()
